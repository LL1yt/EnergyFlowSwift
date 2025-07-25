"""
Flow Processor - механизм распространения энергии
=================================================

Управляет параллельной обработкой всех энергетических потоков.
Координирует взаимодействие между SimpleNeuron и EnergyCarrier.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import time

from ..utils.logging import get_logger, log_memory_state
from ..config import get_energy_config, create_debug_config, set_energy_config
from ..utils.device_manager import get_device_manager
from .simple_neuron import SimpleNeuron, create_simple_neuron
from .energy_carrier import EnergyCarrier, create_energy_carrier
from .energy_lattice import EnergyLattice, create_energy_lattice

logger = get_logger(__name__)


class FlowProcessor(nn.Module):
    """
    Механизм распространения энергии через решетку
    
    Координирует:
    - Параллельную обработку всех активных потоков
    - Взаимодействие SimpleNeuron и EnergyCarrier
    - Создание новых потоков
    - Управление жизненным циклом потоков
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: EnergyConfig с настройками
        """
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        
        # Основные компоненты
        self.lattice = create_energy_lattice(config)
        self.neuron = create_simple_neuron(config)
        self.carrier = create_energy_carrier(config)
        
        # Переносим на устройство
        self.lattice = self.lattice.to(self.device)
        self.neuron = self.neuron.to(self.device)
        self.carrier = self.carrier.to(self.device)
        
        # Статистика производительности
        self.perf_stats = {
            'step_times': [],
            'flows_per_step': [],
            'gpu_memory_usage': []
        }
        
        logger.info(f"FlowProcessor initialized on {self.device}")
        logger.info(f"Components: Lattice {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}, "
                   f"SimpleNeuron, EnergyCarrier")
    
    def forward(self, input_embeddings: torch.Tensor, max_steps: Optional[int] = None) -> torch.Tensor:
        """
        Полный проход энергии через решетку
        
        Args:
            input_embeddings: [batch, 768] - входные эмбеддинги
            max_steps: Максимальное количество шагов (если None - depth решетки)
            
        Returns:
            output_embeddings: [batch, 768] - выходные эмбеддинги
        """
        batch_size = input_embeddings.shape[0]
        
        # Размещаем входную энергию
        self.lattice.reset()
        flow_ids = self.lattice.place_initial_energy(input_embeddings)
        
        # Определяем количество шагов
        if max_steps is None:
            max_steps = self.config.lattice_depth
        
        logger.info(f"Starting energy propagation: {len(flow_ids)} initial flows, max {max_steps} steps")
        
        # Основной цикл распространения
        for step in range(max_steps):
            active_flows = self.lattice.get_active_flows()
            buffered_count = self.lattice.get_buffered_flows_count()
            
            # Проверяем условие завершения: нет активных потоков И нет потоков в буфере
            if not active_flows and buffered_count == 0:
                logger.info(f"No active flows and empty buffer at step {step}, stopping")
                break
            
            # Если есть активные потоки - обрабатываем их
            if active_flows:
                self.step(active_flows)
            
            # Логирование прогресса
            if step % self.config.log_interval == 0:
                stats = self.lattice.get_statistics()
                buffered_count = self.lattice.get_buffered_flows_count()
                logger.info(f"Step {step}: {stats['current_active']} active flows, "
                          f"{stats['total_completed']} completed, {buffered_count} buffered")
        
        # Собираем выходную энергию из буфера (гибридный подход)
        output_embeddings, completed_flows = self._collect_final_output()
        
        # Если нет выходов, возвращаем нули
        if output_embeddings.shape[0] == 0:
            logger.warning("No flows reached output, returning zero embeddings")
            output_embeddings = torch.zeros(batch_size, self.config.input_embedding_dim_from_teacher, 
                                          device=self.device)
        
        # Приводим к размеру батча
        if output_embeddings.shape[0] < batch_size:
            # Дополняем нулями
            padding = torch.zeros(batch_size - output_embeddings.shape[0], 
                                output_embeddings.shape[1], device=self.device)
            output_embeddings = torch.cat([output_embeddings, padding], dim=0)
        elif output_embeddings.shape[0] > batch_size:
            # Обрезаем
            output_embeddings = output_embeddings[:batch_size]
        
        # Финальная статистика
        final_stats = self.lattice.get_statistics()
        logger.info(f"Energy propagation complete: {final_stats['total_completed']} flows reached output, "
                   f"{final_stats['total_died']} died")
        
        return output_embeddings
    
    def _collect_final_output(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Гибридный сбор финальной энергии
        
        Проверяет активные потоки и буфер, собирает энергию когда все готово.
        
        Returns:
            output_embeddings: [batch, embedding_dim] - собранные эмбеддинги
            completed_flows: ID завершенных потоков
        """
        active_flows = self.lattice.get_active_flows()
        buffered_count = self.lattice.get_buffered_flows_count()
        
        logger.debug(f"Final collection: {len(active_flows)} active flows, {buffered_count} buffered flows")
        
        # Если есть активные потоки на выходной стороне - добавляем их в буфер
        active_at_output = 0
        for flow in active_flows:
            z_pos = flow.position[2].item()
            if z_pos >= self.config.lattice_depth - 1:
                # Поток достиг выхода но еще не буферизован
                self.lattice._buffer_output_flow(flow.id)
                active_at_output += 1
        
        if active_at_output > 0:
            logger.debug(f"Moved {active_at_output} remaining flows to output buffer")
        
        # Теперь собираем все из буфера
        output_embeddings, completed_flows = self.lattice.collect_buffered_energy()
        
        # Очищаем буфер после сбора (FlowProcessor координирует жизненный цикл)
        if completed_flows:
            self.lattice.clear_output_buffer()
            logger.info(f"Collected and cleared {len(completed_flows)} flows from output buffer")
        
        return output_embeddings, completed_flows
    
    def step(self, active_flows: Optional[List] = None):
        """
        Один шаг распространения энергии для всех активных потоков
        
        Args:
            active_flows: Список активных потоков (если None - получаем из lattice)
        """
        start_time = time.time()
        
        if active_flows is None:
            active_flows = self.lattice.get_active_flows()
        
        if not active_flows:
            return
        
        # Батчевая обработка для эффективности
        batch_size = min(len(active_flows), self.config.batch_size)
        
        for i in range(0, len(active_flows), batch_size):
            batch_flows = active_flows[i:i + batch_size]
            self._process_flow_batch(batch_flows)
        
        # Статистика
        step_time = time.time() - start_time
        self.perf_stats['step_times'].append(step_time)
        self.perf_stats['flows_per_step'].append(len(active_flows))
        
        if self.device.type == 'cuda':
            memory_info = self.device_manager.get_memory_info()
            self.perf_stats['gpu_memory_usage'].append(memory_info['gpu_allocated_gb'])
    
    def _process_flow_batch(self, flows):
        """Обрабатывает батч потоков параллельно"""
        batch_size = len(flows)
        
        # Собираем данные потоков
        positions = torch.stack([f.position for f in flows])  # [batch, 3]
        energies = torch.stack([f.energy for f in flows])     # [batch, embedding_dim]
        hidden_states = torch.stack([f.hidden_state for f in flows])  # [batch, layers, hidden]
        
        # Транспонируем hidden states для GRU и делаем непрерывными
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # [layers, batch, hidden]
        
        # 1. SimpleNeuron обрабатывает позиции и энергии
        neuron_output = self.neuron(positions, energies)  # [batch, neuron_output_dim]
        
        # 2. EnergyCarrier генерирует следующее состояние
        carrier_output, new_hidden = self.carrier(
            neuron_output, 
            energies,
            hidden_states,
            positions
        )
        
        # Транспонируем обратно и делаем непрерывными
        new_hidden = new_hidden.transpose(0, 1).contiguous()  # [batch, layers, hidden]
        
        # 3. Обрабатываем результаты для каждого потока
        for idx, flow in enumerate(flows):
            # Проверяем уровень энергии
            is_alive = self.carrier.check_energy_level(carrier_output.energy_value[idx:idx+1])
            
            if not is_alive[0]:
                self.lattice.deactivate_flow(flow.id, "energy_depleted")
                continue
            
            # Обновляем состояние потока
            self.lattice.update_flow(
                flow.id,
                carrier_output.next_position[idx],
                carrier_output.energy_value[idx],
                new_hidden[idx]
            )
            
            # Обрабатываем порождение новых потоков
            if idx < len(carrier_output.spawn_energies):
                # Получаем энергии для этого потока
                flow_spawn_count = min(
                    len(carrier_output.spawn_energies) - idx,
                    self.config.max_spawn_per_step
                )
                
                if flow_spawn_count > 0:
                    spawn_energies = carrier_output.spawn_energies[idx:idx+flow_spawn_count]
                    self.lattice.spawn_flows(flow.id, spawn_energies)
    
    def get_performance_stats(self) -> Dict:
        """Возвращает статистику производительности"""
        if not self.perf_stats['step_times']:
            return {}
        
        import numpy as np
        
        return {
            'avg_step_time': np.mean(self.perf_stats['step_times']),
            'max_step_time': np.max(self.perf_stats['step_times']),
            'avg_flows_per_step': np.mean(self.perf_stats['flows_per_step']),
            'max_flows_per_step': np.max(self.perf_stats['flows_per_step']),
            'avg_gpu_memory_gb': np.mean(self.perf_stats['gpu_memory_usage']) if self.perf_stats['gpu_memory_usage'] else 0,
            'lattice_stats': self.lattice.get_statistics()
        }
    
    def visualize_flow_state(self) -> Dict:
        """Возвращает данные для визуализации текущего состояния потоков"""
        active_flows = self.lattice.get_active_flows()
        
        # Собираем позиции и энергии
        positions = []
        energies = []
        ages = []
        
        for flow in active_flows:
            positions.append(flow.position.cpu().numpy())
            energy_norm = torch.norm(flow.energy).item()
            energies.append(energy_norm)
            ages.append(flow.age)
        
        return {
            'positions': positions,
            'energies': energies,
            'ages': ages,
            'total_flows': len(active_flows),
            'lattice_dims': (self.config.lattice_width, 
                           self.config.lattice_height,
                           self.config.lattice_depth)
        }


def create_flow_processor(config=None) -> FlowProcessor:
    """Фабричная функция для создания FlowProcessor"""
    return FlowProcessor(config)