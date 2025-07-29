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
        
        # Embedding mapper - ОБЯЗАТЕЛЬНО для архитектуры
        from .embedding_mapper import EnergyFlowMapper
        self.mapper = EnergyFlowMapper(config)
        # Переносим mapper компоненты на устройство
        self.mapper.input_mapper = self.mapper.input_mapper.to(self.device)
        self.mapper.output_collector = self.mapper.output_collector.to(self.device)
        
        # Переносим на устройство
        self.lattice = self.lattice.to(self.device)
        self.neuron = self.neuron.to(self.device)
        self.carrier = self.carrier.to(self.device)
        # Mapper уже инициализируется на правильном устройстве
        
        # Статистика производительности
        self.perf_stats = {
            'step_times': [],
            'flows_per_step': [],
            'gpu_memory_usage': []
        }
        
        # Статистика убийства потоков
        self.stats = {
            'flows_killed_backward': 0,
            'flows_killed_bounds': 0,
            'flows_killed_energy': 0
        }
        
        # Статистика конвергенции
        self.convergence_stats = {
            'completed_count_history': [],
            'no_improvement_steps': 0,
            'best_completed_count': 0
        }
        
        logger.info(f"FlowProcessor initialized on {self.device}")
        logger.info(f"Components: Lattice {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}, "
                   f"SimpleNeuron, EnergyCarrier")
        
        if config.convergence_enabled:
            logger.info(f"Adaptive convergence enabled: threshold={config.convergence_threshold:.2f}, "
                       f"min_steps={config.convergence_min_steps}, patience={config.convergence_patience}")
    
    def forward(self, input_embeddings: torch.Tensor, max_steps: Optional[int] = None, 
                global_training_step: Optional[int] = None) -> torch.Tensor:
        """
        Полный проход энергии через решетку
        
        Args:
            input_embeddings: [batch, 768] - входные эмбеддинги
            max_steps: Максимальное количество шагов (если None - depth решетки)
            global_training_step: Глобальный шаг обучения для curriculum learning
            
        Returns:
            output_embeddings: [batch, 768] - выходные эмбеддинги
        """
        batch_size = input_embeddings.shape[0]
        
        # Размещаем входную энергию с использованием маппера
        self.lattice.reset()
        flow_ids = self.lattice.place_initial_energy(input_embeddings, self.mapper)
        
        # Определяем количество шагов
        if max_steps is None:
            max_steps = self.config.lattice_depth
        
        logger.info(f"Starting energy propagation: {len(flow_ids)} initial flows, max {max_steps} steps")
        
        # Сбрасываем статистику конвергенции
        initial_flows_count = len(flow_ids)
        self.convergence_stats = {
            'completed_count_history': [],
            'no_improvement_steps': 0,
            'best_completed_count': 0
        }
        
        # Основной цикл распространения с adaptive convergence
        actual_steps = 0
        for step in range(max_steps):
            actual_steps = step + 1
            active_flows = self.lattice.get_active_flows()
            buffered_count = self.lattice.get_buffered_flows_count()
            
            # Проверяем условие завершения: нет активных потоков И нет потоков в буфере
            if not active_flows and buffered_count == 0:
                logger.info(f"No active flows and empty buffer at step {step}, stopping")
                break
            
            # Проверяем конвергенцию (adaptive max_steps)
            if self._check_convergence(step, initial_flows_count):
                logger.log(20, f"Early convergence at step {step}/{max_steps}")
                break
            
            # Если есть активные потоки - обрабатываем их
            if active_flows:
                self.step(active_flows, global_training_step=global_training_step)
            
            # Логирование прогресса
            if step % self.config.log_interval == 0:
                stats = self.lattice.get_statistics()
                buffered_count = self.lattice.get_buffered_flows_count()
                completion_rate = stats['total_completed'] / initial_flows_count if initial_flows_count > 0 else 0
                logger.info(f"Step {step}: {stats['current_active']} active flows, "
                          f"{stats['total_completed']} completed ({completion_rate:.2f}), {buffered_count} buffered")
        
        # Собираем выходную энергию из буфера (БЕЗ преобразования в 768D!)
        output_surface_embeddings, completed_flows = self._collect_final_surface_output()
        
        # Если нет выходов, возвращаем нули (surface размерность!)
        if output_surface_embeddings.shape[0] == 0:
            logger.warning("No flows reached output, returning zero surface embeddings")
            surface_dim = self.config.lattice_width * self.config.lattice_height
            output_surface_embeddings = torch.zeros(batch_size, surface_dim, device=self.device)
        
        # Приводим к размеру батча
        if output_surface_embeddings.shape[0] < batch_size:
            # Дополняем нулями
            padding = torch.zeros(batch_size - output_surface_embeddings.shape[0], 
                                output_surface_embeddings.shape[1], device=self.device)
            output_surface_embeddings = torch.cat([output_surface_embeddings, padding], dim=0)
        elif output_surface_embeddings.shape[0] > batch_size:
            # Обрезаем
            output_surface_embeddings = output_surface_embeddings[:batch_size]
        
        # Финальная статистика
        final_stats = self.lattice.get_statistics()
        killed_backward = self.stats['flows_killed_backward']
        killed_bounds = self.stats['flows_killed_bounds']
        killed_energy = self.stats['flows_killed_energy']
        
        # Статистика adaptive max_steps
        steps_saved = max_steps - actual_steps
        if self.config.convergence_enabled and steps_saved > 0:
            speedup = max_steps / actual_steps if actual_steps > 0 else 1.0
            logger.log(20, f"Adaptive convergence saved {steps_saved} steps ({speedup:.2f}x speedup)")
        
        logger.info(f"Energy propagation complete ({actual_steps}/{max_steps} steps): "
                   f"{final_stats['total_completed']} flows reached output, "
                   f"{final_stats['total_died']} died "
                   f"(energy: {killed_energy}, backward: {killed_backward}, bounds: {killed_bounds})")
        
        return output_surface_embeddings
    
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
        
        # Теперь собираем все из буфера используя маппер
        output_embeddings, completed_flows = self.lattice.collect_output_energy(self.mapper)
        
        # Очищаем буфер после сбора (FlowProcessor координирует жизненный цикл)
        if completed_flows:
            self.lattice.clear_output_buffer()
            logger.info(f"Collected and cleared {len(completed_flows)} flows from output buffer")
        
        return output_embeddings, completed_flows
    
    def _collect_final_surface_output(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает surface embeddings БЕЗ преобразования в 768D
        
        Returns:
            output_surface_embeddings: [batch, surface_dim] - surface embeddings
            completed_flows: ID завершенных потоков
        """
        active_flows = self.lattice.get_active_flows()
        buffered_count = self.lattice.get_buffered_flows_count()
        
        logger.debug(f"Surface collection: {len(active_flows)} active flows, {buffered_count} buffered flows")
        
        # Если есть активные потоки на выходной стороне - добавляем их в буфер
        active_at_output = 0
        for flow in active_flows:
            z_pos = flow.position[2].item()
            if z_pos >= self.config.lattice_depth - 1:
                self.lattice._buffer_output_flow(flow.id)
                active_at_output += 1
        
        if active_at_output > 0:
            logger.debug(f"Moved {active_at_output} remaining flows to output buffer")
        
        # Собираем surface embeddings из буфера напрямую
        output_surface_embeddings, completed_flows = self.lattice.collect_buffered_surface_energy()
        
        # Очищаем буфер после сбора (FlowProcessor координирует жизненный цикл)
        if completed_flows:
            self.lattice.clear_output_buffer()
            logger.info(f"Collected and cleared {len(completed_flows)} flows from output buffer")
        
        return output_surface_embeddings, completed_flows
    
    def step(self, active_flows: Optional[List] = None, global_training_step: Optional[int] = None):
        """
        Один шаг распространения энергии для всех активных потоков
        ПОЛНАЯ ПАРАЛЛЕЛИЗАЦИЯ: все потоки обрабатываются одновременно вместо sequential batches
        
        Args:
            active_flows: Список активных потоков (если None - получаем из lattice)
        """
        start_time = time.time()
        
        if active_flows is None:
            active_flows = self.lattice.get_active_flows()
        
        if not active_flows:
            return
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: убираем Sequential Processing Bottleneck!
        # Вместо цикла с маленькими batch'ами - обрабатываем ВСЕ потоки сразу
        # Это позволяет GPU cores работать параллельно с 1000+ потоками одновременно
        
        flows_count = len(active_flows)
        max_flows_per_step = self.config.max_active_flows  # RTX 5090 может обработать все сразу
        
        if flows_count <= max_flows_per_step:
            # Оптимальный случай: обрабатываем ВСЕ потоки одним большим batch'ем
            self._process_flow_batch(active_flows, global_training_step=global_training_step)
        else:
            # Fallback: если слишком много потоков (>200K), делим на крупные chunk'и
            optimal_chunk_size = max_flows_per_step // 2  # 100K потоков за раз
            
            for i in range(0, flows_count, optimal_chunk_size):
                chunk_flows = active_flows[i:i + optimal_chunk_size]
                self._process_flow_batch(chunk_flows, global_training_step=global_training_step)
        
        # Статистика
        step_time = time.time() - start_time
        self.perf_stats['step_times'].append(step_time)
        self.perf_stats['flows_per_step'].append(flows_count)
        
        if self.device.type == 'cuda':
            memory_info = self.device_manager.get_memory_info()
            self.perf_stats['gpu_memory_usage'].append(memory_info['gpu_allocated_gb'])
    
    def _process_flow_batch(self, flows, global_training_step: Optional[int] = None):
        """Обрабатывает батч потоков с векторизованными операциями"""
        batch_size = len(flows)
        
        # Собираем данные потоков и ID
        positions = torch.stack([f.position for f in flows])  # [batch, 3]
        energies = torch.stack([f.energy for f in flows])     # [batch, embedding_dim]
        hidden_states = torch.stack([f.hidden_state for f in flows])  # [batch, layers, hidden]
        flow_ids = torch.tensor([f.id for f in flows], dtype=torch.long, device=positions.device)
        
        # Транспонируем hidden states для GRU и делаем непрерывными
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # [layers, batch, hidden]
        
        # 1. SimpleNeuron обрабатывает позиции и энергии
        neuron_output = self.neuron(positions, energies)  # [batch, neuron_output_dim]
        
        # Собираем возраста потоков для progressive bias
        ages = torch.tensor([flow.age for flow in flows], dtype=torch.float32, device=positions.device)
        
        # 2. EnergyCarrier генерирует следующее состояние с curriculum learning
        carrier_output, new_hidden = self.carrier(
            neuron_output, 
            energies,
            hidden_states,
            positions,
            flow_age=ages,
            global_training_step=global_training_step  # Передаем глобальный шаг
        )
        
        # Транспонируем обратно и делаем непрерывными
        new_hidden = new_hidden.transpose(0, 1).contiguous()  # [batch, layers, hidden]
        
        # 3. ВЕКТОРИЗОВАННАЯ обработка результатов
        self._process_results_vectorized(flows, flow_ids, positions, carrier_output, new_hidden)
    
    def _process_results_vectorized(self, flows, flow_ids, current_positions, carrier_output, new_hidden):
        """Векторизованная обработка результатов carrier_output"""
        batch_size = len(flows)
        device = current_positions.device
        
        # Векторизованные проверки выживания потоков
        energy_alive_mask = self.carrier.check_energy_level(carrier_output.energy_value)  # [batch]
        
        # Проверка движения вперед по Z
        z_forward_mask = carrier_output.next_position[:, 2] > current_positions[:, 2]  # [batch]
        
        # Проверка границ X,Y
        next_pos = carrier_output.next_position
        bounds_mask = (
            (next_pos[:, 0] >= 0) & (next_pos[:, 0] < self.config.lattice_width) &
            (next_pos[:, 1] >= 0) & (next_pos[:, 1] < self.config.lattice_height)
        )  # [batch]
        
        # Комбинированная маска выживших потоков
        alive_mask = energy_alive_mask & z_forward_mask & bounds_mask  # [batch]
        
        # Подсчет статистики БЕЗ .item() для избежания CPU-GPU синхронизации
        energy_dead_count = (~energy_alive_mask).sum()
        backward_dead_count = (energy_alive_mask & ~z_forward_mask).sum()
        bounds_dead_count = (energy_alive_mask & z_forward_mask & ~bounds_mask).sum()
        
        # Обновляем статистику с detach() для безопасности, но без синхронизации
        self.stats['flows_killed_energy'] += energy_dead_count.detach().cpu().numpy().item()
        self.stats['flows_killed_backward'] += backward_dead_count.detach().cpu().numpy().item()
        self.stats['flows_killed_bounds'] += bounds_dead_count.detach().cpu().numpy().item()
        
        # ПОЛНАЯ ВЕКТОРИЗАЦИЯ: batch deactivation dead flows БЕЗ циклов
        dead_mask = ~alive_mask
        if dead_mask.any():
            dead_flow_ids = flow_ids[dead_mask]
            # Создаем причины векторизованно
            energy_dead_mask = dead_mask & (~energy_alive_mask)
            backward_dead_mask = dead_mask & energy_alive_mask & (~z_forward_mask)
            bounds_dead_mask = dead_mask & energy_alive_mask & z_forward_mask & (~bounds_mask)
            
            # Batch deactivation with vectorized reasons
            self.lattice.batch_deactivate_flows(
                dead_flow_ids,
                energy_dead_mask[dead_mask],
                backward_dead_mask[dead_mask], 
                bounds_dead_mask[dead_mask]
            )
        
        # ПОЛНАЯ ВЕКТОРИЗАЦИЯ: batch update alive flows БЕЗ циклов
        if alive_mask.any():
            alive_flow_ids = flow_ids[alive_mask]
            alive_positions = carrier_output.next_position[alive_mask]
            alive_energies = carrier_output.energy_value[alive_mask]
            alive_hidden = new_hidden[alive_mask]
            
            # Batch update all alive flows at once
            self.lattice.batch_update_flows(
                alive_flow_ids,
                alive_positions,
                alive_energies,
                alive_hidden
            )
        
        # Обработка spawn потоков (оптимизированная)
        self._process_spawns_optimized(flows, carrier_output, alive_mask)
    
    def _process_spawns_optimized(self, flows, carrier_output, alive_mask):
        """Оптимизированная обработка spawn потоков"""
        if not carrier_output.spawn_info:
            return
        
        # Создаем индекс spawn_info по parent_batch_idx для O(1) поиска
        spawn_by_idx = {}
        for spawn_info in carrier_output.spawn_info:
            spawn_by_idx[spawn_info.parent_batch_idx] = spawn_info
        
        # Обрабатываем spawn'ы только для живых потоков
        alive_indices = torch.where(alive_mask)[0]
        for idx in alive_indices:
            idx_val = idx.item()
            if idx_val in spawn_by_idx:
                spawn_info = spawn_by_idx[idx_val]
                if spawn_info.energies:
                    # Ограничиваем количество spawn'ов конфигом
                    spawn_energies = spawn_info.energies[:self.config.max_spawn_per_step]
                    flow_id = flows[idx_val].id
                    self.lattice.spawn_flows(flow_id, spawn_energies)
    
    def _check_convergence(self, step: int, initial_flows_count: int) -> bool:
        """
        Проверяет условия конвергенции для adaptive max_steps
        
        Args:
            step: Текущий шаг
            initial_flows_count: Количество начальных потоков
            
        Returns:
            True если следует остановить обучение
        """
        if not self.config.convergence_enabled:
            return False
        
        # Минимальное количество шагов
        if step < self.config.convergence_min_steps:
            return False
        
        # Получаем текущую статистику
        stats = self.lattice.get_statistics()
        completed_count = stats['total_completed']
        
        # Добавляем в историю
        self.convergence_stats['completed_count_history'].append(completed_count)
        
        # Проверяем порог конвергенции
        completion_rate = completed_count / initial_flows_count if initial_flows_count > 0 else 0
        
        logger.log(20, f"Convergence check step {step}: {completed_count}/{initial_flows_count} "
                      f"flows completed ({completion_rate:.2f})")
        
        # Условие 1: Достигнут порог конвергенции
        if completion_rate >= self.config.convergence_threshold:
            logger.log(20, f"Convergence threshold reached: {completion_rate:.2f} >= {self.config.convergence_threshold:.2f}")
            return True
        
        # Условие 2: Patience - нет улучшения в течение N шагов
        if completed_count > self.convergence_stats['best_completed_count']:
            self.convergence_stats['best_completed_count'] = completed_count
            self.convergence_stats['no_improvement_steps'] = 0
        else:
            self.convergence_stats['no_improvement_steps'] += 1
        
        if self.convergence_stats['no_improvement_steps'] >= self.config.convergence_patience:
            logger.log(20, f"Convergence patience exceeded: {self.convergence_stats['no_improvement_steps']} "
                          f">= {self.config.convergence_patience}")
            return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Возвращает статистику производительности"""
        if not self.perf_stats['step_times']:
            return {}
        
        import numpy as np
        
        stats = {
            'avg_step_time': np.mean(self.perf_stats['step_times']),
            'max_step_time': np.max(self.perf_stats['step_times']),
            'avg_flows_per_step': np.mean(self.perf_stats['flows_per_step']),
            'max_flows_per_step': np.max(self.perf_stats['flows_per_step']),
            'avg_gpu_memory_gb': np.mean(self.perf_stats['gpu_memory_usage']) if self.perf_stats['gpu_memory_usage'] else 0,
            'lattice_stats': self.lattice.get_statistics()
        }
        
        # Добавляем статистику конвергенции
        if self.config.convergence_enabled and self.convergence_stats['completed_count_history']:
            stats['convergence_stats'] = {
                'best_completion_count': self.convergence_stats['best_completed_count'],
                'final_completion_count': self.convergence_stats['completed_count_history'][-1] if self.convergence_stats['completed_count_history'] else 0,
                'completion_trend': len(self.convergence_stats['completed_count_history'])
            }
        
        return stats
    
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