"""
Energy Lattice - 3D решетка для управления энергетическими потоками
===================================================================

Управляет активными потоками, их позициями и взаимодействием с решеткой.
Работает только с входной и выходной сторонами куба.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
import numpy as np

from ..utils.logging import get_logger, log_memory_state
from ..config import get_energy_config, create_debug_config, set_energy_config
from ..utils.device_manager import get_device_manager

logger = get_logger(__name__)


class Position3D(NamedTuple):
    """3D позиция в решетке"""
    x: int
    y: int
    z: int


@dataclass
class EnergyFlow:
    """Представление одного энергетического потока"""
    id: int
    position: torch.Tensor  # [3] - текущая позиция
    energy: torch.Tensor    # [embedding_dim] - энергия/эмбеддинг
    hidden_state: torch.Tensor  # [num_layers, hidden_size] - состояние GRU
    batch_idx: int = 0  # Индекс в батче
    parent_id: Optional[int] = None
    age: int = 0  # Количество шагов жизни потока
    is_active: bool = True


class EnergyLattice(nn.Module):
    """
    3D решетка для управления энергетическими потоками
    
    Основные функции:
    - Размещение входных эмбеддингов на входной стороне
    - Управление активными потоками
    - Сбор выходных эмбеддингов с выходной стороны
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
        
        # Размеры решетки
        self.width = config.lattice_width
        self.height = config.lattice_height
        self.depth = config.lattice_depth
        
        # Параметры потоков
        self.max_active_flows = config.max_active_flows
        self.energy_dim = 1  # Скалярная энергия от mapper'а
        
        # Активные потоки
        self.active_flows: Dict[int, EnergyFlow] = {}
        self.next_flow_id = 0
        
        # Буфер для выходных потоков (буферизованный сбор)
        self.output_buffer: Dict[Tuple[int, int], List[EnergyFlow]] = {}  # (x,y) -> [flows]
        
        # Статистика
        self.stats = {
            'total_created': 0,
            'total_completed': 0,
            'total_died': 0,
            'max_concurrent': 0
        }
        
        logger.info(f"EnergyLattice initialized: {self.width}x{self.height}x{self.depth}")
        logger.info(f"Input/output cells: {self.width * self.height}, max flows: {self.max_active_flows}")
    
    def place_initial_energy(self, embeddings: torch.Tensor, mapper=None) -> List[int]:
        """
        Размещает входные эмбеддинги на входной стороне куба (z=0)
        
        Args:
            embeddings: [batch, embedding_dim] - входные эмбеддинги (768D)
            mapper: EnergyFlowMapper для проекции (опционально)
            
        Returns:
            flow_ids: Список ID созданных потоков
        """
        batch_size = embeddings.shape[0]
        
        # Проверяем размерность
        expected_dim = self.config.input_embedding_dim_from_teacher
        if embeddings.shape[1] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {embeddings.shape[1]}")
        
        # Очищаем неактивные потоки
        self._cleanup_inactive_flows()
        
        flow_ids = []
        
        if mapper is None:
            raise ValueError("EnergyFlowMapper is required! No fallback logic allowed.")
        
        # Используем маппер для проекции 768D -> surface_dim
        cell_energies = mapper.map_to_surface(embeddings)
        
        for (x, y), energy, batch_idx in cell_energies:
            if len(self.active_flows) >= self.max_active_flows:
                logger.warning(f"Reached max active flows limit: {self.max_active_flows}")
                break
            
            # Создаем поток с энергией из маппера
            position = torch.tensor([x, y, 0], dtype=torch.float32, device=self.device)
            flow_id = self._create_flow(position, energy, batch_idx=batch_idx)
            flow_ids.append(flow_id)
        
        logger.info(f"Created {len(flow_ids)} initial flows on input surface")
        return flow_ids
    
    def _create_flow(self, position: torch.Tensor, energy: torch.Tensor, 
                    batch_idx: int = 0,
                    parent_id: Optional[int] = None,
                    hidden_state: Optional[torch.Tensor] = None) -> int:
        """Создает новый энергетический поток"""
        flow_id = self.next_flow_id
        self.next_flow_id += 1
        
        # Создаем пустое скрытое состояние если не передано
        if hidden_state is None:
            # Размеры берем из конфига
            num_layers = self.config.carrier_num_layers
            hidden_size = self.config.carrier_hidden_size
            hidden_state = torch.zeros(num_layers, hidden_size, device=self.device)
        
        flow = EnergyFlow(
            id=flow_id,
            position=position,
            energy=energy,
            hidden_state=hidden_state,
            batch_idx=batch_idx,
            parent_id=parent_id,
            age=0,
            is_active=True
        )
        
        self.active_flows[flow_id] = flow
        self.stats['total_created'] += 1
        self.stats['max_concurrent'] = max(self.stats['max_concurrent'], len(self.active_flows))
        
        return flow_id
    
    def get_active_flows(self) -> List[EnergyFlow]:
        """Возвращает список активных потоков"""
        return [flow for flow in self.active_flows.values() if flow.is_active]
    
    def update_flow(self, flow_id: int, 
                   new_position: torch.Tensor,
                   new_energy: torch.Tensor,
                   new_hidden: torch.Tensor):
        """Обновляет состояние потока"""
        if flow_id not in self.active_flows:
            return
        
        flow = self.active_flows[flow_id]
        flow.position = new_position
        flow.energy = new_energy
        flow.hidden_state = new_hidden
        flow.age += 1
        
        # Буферизуем потоки при достижении выхода
        if new_position[2] >= self.depth - 1:
            self._buffer_output_flow(flow_id)
            logger.debug(f"Flow {flow_id} reached output side at age {flow.age} (buffered for collection)")
    
    def spawn_flows(self, parent_id: int, spawn_energies: List[torch.Tensor]) -> List[int]:
        """
        Создает новые потоки от родительского
        
        Args:
            parent_id: ID родительского потока
            spawn_energies: Список энергий для новых потоков
            
        Returns:
            new_flow_ids: ID созданных потоков
        """
        if parent_id not in self.active_flows:
            return []
        
        parent = self.active_flows[parent_id]
        new_flow_ids = []
        
        for energy in spawn_energies:
            if len(self.active_flows) >= self.max_active_flows:
                logger.warning("Cannot spawn: max flows reached")
                break
            
            # Новые потоки начинают с позиции родителя
            flow_id = self._create_flow(
                parent.position.clone(),
                energy,
                batch_idx=parent.batch_idx,  # Наследуем batch_idx от родителя
                parent_id=parent_id,
                hidden_state=parent.hidden_state.clone()
            )
            new_flow_ids.append(flow_id)
        
        if new_flow_ids:
            logger.debug(f"Spawned {len(new_flow_ids)} flows from parent {parent_id}")
        
        return new_flow_ids
    
    def deactivate_flow(self, flow_id: int, reason: str = "energy_depleted"):
        """Деактивирует поток"""
        if flow_id in self.active_flows:
            self.active_flows[flow_id].is_active = False
            if reason != "reached_output":
                self.stats['total_died'] += 1
            logger.debug(f"Flow {flow_id} deactivated: {reason}")
    
    def _buffer_output_flow(self, flow_id: int):
        """Помещает поток в буфер выходных потоков"""
        if flow_id not in self.active_flows:
            return
        
        flow = self.active_flows[flow_id]
        
        # Корректируем позицию если поток вышел за пределы
        if flow.position[2] > self.depth - 1:
            corrected_flow = EnergyFlow(
                id=flow.id,
                position=torch.tensor([
                    flow.position[0], 
                    flow.position[1], 
                    self.depth - 1  # Устанавливаем на выходную сторону
                ], device=self.device),
                energy=flow.energy,
                hidden_state=flow.hidden_state,
                parent_id=flow.parent_id,
                age=flow.age,
                is_active=flow.is_active
            )
            buffered_flow = corrected_flow
        else:
            buffered_flow = flow
        
        # Определяем клетку на выходной стороне
        x = int(torch.clamp(buffered_flow.position[0], 0, self.width - 1).item())
        y = int(torch.clamp(buffered_flow.position[1], 0, self.height - 1).item())
        key = (x, y)
        
        # Добавляем в буфер
        if key not in self.output_buffer:
            self.output_buffer[key] = []
        self.output_buffer[key].append(buffered_flow)
        
        # Деактивируем поток после буферизации
        flow.is_active = False
        self.stats['total_completed'] += 1
        
        logger.debug(f"Flow {flow_id} buffered to output cell ({x}, {y})")
    
    def get_buffered_flows_count(self) -> int:
        """Возвращает количество потоков в выходном буфере"""
        return sum(len(flows) for flows in self.output_buffer.values())
    
    def clear_output_buffer(self):
        """Очищает буфер выходных потоков"""
        cleared_count = self.get_buffered_flows_count()
        self.output_buffer.clear()
        logger.debug(f"Cleared output buffer ({cleared_count} flows)")
    
    def get_all_buffered_flows(self) -> List[EnergyFlow]:
        """Возвращает все потоки из буфера"""
        all_flows = []
        for flows in self.output_buffer.values():
            all_flows.extend(flows)
        return all_flows
    
    def collect_buffered_energy(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает энергию из буфера выходных потоков
        
        Returns:
            output_embeddings: [batch, embedding_dim] - собранные эмбеддинги
            flow_ids: ID потоков в буфере
        """
        if not self.output_buffer:
            logger.debug("No flows in output buffer")
            return torch.zeros(0, self.embedding_dim, device=self.device), []
        
        flow_ids = []
        # Используем существующую логику взвешенного усреднения
        output_embeddings = []
        aggregation_stats = []
        
        logger.debug(f"Collecting from buffer: {len(self.output_buffer)} cells with flows")
        
        for y in range(self.height):
            for x in range(self.width):
                key = (x, y)
                if key in self.output_buffer:
                    flows_in_cell = self.output_buffer[key]
                    
                    # Собираем ID всех потоков в этой клетке
                    for flow in flows_in_cell:
                        flow_ids.append(flow.id)
                    
                    if len(flows_in_cell) == 1:
                        # Один поток - просто берем его энергию
                        aggregated = flows_in_cell[0].energy
                        stats = f"single_flow(id={flows_in_cell[0].id})"
                    else:
                        # Несколько потоков - взвешенное усреднение
                        energies = torch.stack([flow.energy for flow in flows_in_cell])
                        
                        # Вычисляем веса (энергия * возраст)
                        weights = []
                        for flow in flows_in_cell:
                            energy_magnitude = torch.norm(flow.energy).item()
                            age_factor = 1.0 + flow.age * 0.1
                            weight = energy_magnitude * age_factor
                            weights.append(weight)
                        
                        weights = torch.tensor(weights, device=self.device)
                        weights = weights / weights.sum()  # Нормализуем
                        
                        # Взвешенное усреднение
                        aggregated = (energies * weights.unsqueeze(-1)).sum(dim=0)
                        
                        # Статистика
                        flow_ids_cell = [flow.id for flow in flows_in_cell]
                        avg_age = sum(flow.age for flow in flows_in_cell) / len(flows_in_cell)
                        stats = f"weighted_avg({len(flows_in_cell)}_flows, ids={flow_ids_cell}, avg_age={avg_age:.1f})"
                        
                        logger.debug(f"Cell ({x},{y}): {stats}")
                    
                    aggregation_stats.append(stats)
                else:
                    # Пустая клетка
                    aggregated = torch.zeros(self.embedding_dim, device=self.device)
                
                output_embeddings.append(aggregated)
        
        # Собираем в тензор
        output_embeddings = torch.stack(output_embeddings)  # [width*height, embedding_dim]
        
        # Восстанавливаем исходную размерность эмбеддинга
        output_embeddings = output_embeddings.view(-1)[:self.config.input_embedding_dim_from_teacher]
        output_embeddings = output_embeddings.unsqueeze(0)  # [1, 768]
        
        # Статистика
        cells_with_flows = len([stats for stats in aggregation_stats if 'flow' in stats])
        multi_flow_cells = len([stats for stats in aggregation_stats if 'weighted_avg' in stats])
        
        logger.info(f"Collected energy from {len(flow_ids)} buffered flows")
        logger.info(f"Output grid: {cells_with_flows}/{self.width*self.height} cells with flows, "
                   f"{multi_flow_cells} cells with multiple flows")
        
        return output_embeddings, flow_ids
    
    def collect_output_energy(self, mapper=None) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает энергию с выходной поверхности
        
        Args:
            mapper: EnergyFlowMapper для восстановления эмбеддингов
            
        Returns:
            output_embeddings: [batch, embedding_dim] - собранные эмбеддинги
            flow_ids: ID потоков, достигших выхода
        """
        if mapper is None:
            raise ValueError("EnergyFlowMapper is required! No fallback logic allowed.")
        
        # Собираем энергию из буфера по клеткам
        surface_energy = {}
        flow_ids = []
        
        for (x, y), flows in self.output_buffer.items():
            if flows:
                # Усредняем энергию в клетке с весами
                energies = []
                weights = []
                
                for flow in flows:
                    energy_magnitude = torch.norm(flow.energy).item()
                    age_factor = 1 + flow.age * 0.1
                    weight = energy_magnitude * age_factor
                    
                    energies.append(flow.energy)
                    weights.append(weight)
                    flow_ids.append(flow.id)
                
                # Взвешенное усреднение
                weights = torch.tensor(weights, device=self.device)
                weights = weights / weights.sum()
                
                avg_energy = sum(e * w for e, w in zip(energies, weights))
                surface_energy[(x, y)] = avg_energy
        
        # Определяем размер батча из потоков
        all_flows = list(self.active_flows.values()) + list(self.get_all_buffered_flows())
        batch_indices = {flow.batch_idx for flow in all_flows} if all_flows else {0}
        batch_size = max(batch_indices) + 1 if batch_indices else 1
        
        # Восстанавливаем эмбеддинги через маппер
        output_embeddings = mapper.collect_from_surface(surface_energy, batch_size)
        
        logger.info(f"Collected energy from {len(surface_energy)} cells using mapper")
        return output_embeddings, flow_ids
    
    def _cleanup_inactive_flows(self):
        """Удаляет неактивные потоки из памяти"""
        inactive_ids = [fid for fid, flow in self.active_flows.items() if not flow.is_active]
        
        for flow_id in inactive_ids:
            del self.active_flows[flow_id]
        
        if inactive_ids:
            logger.debug(f"Cleaned up {len(inactive_ids)} inactive flows")
    
    def get_statistics(self) -> Dict[str, any]:
        """Возвращает статистику решетки"""
        active_count = len(self.get_active_flows())
        
        return {
            **self.stats,
            'current_active': active_count,
            'utilization': active_count / self.max_active_flows if self.max_active_flows > 0 else 0
        }
    
    def reset(self):
        """Сбрасывает состояние решетки"""
        self.active_flows.clear()
        self.output_buffer.clear()  # Очищаем буфер
        self.next_flow_id = 0
        self.stats = {
            'total_created': 0,
            'total_completed': 0,
            'total_died': 0,
            'max_concurrent': 0
        }
        logger.info("EnergyLattice reset")


def create_energy_lattice(config=None) -> EnergyLattice:
    """Фабричная функция для создания EnergyLattice"""
    return EnergyLattice(config)