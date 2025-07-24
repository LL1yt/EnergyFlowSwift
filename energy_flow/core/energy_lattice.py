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
        self.embedding_dim = config.embedding_per_cell
        
        # Активные потоки
        self.active_flows: Dict[int, EnergyFlow] = {}
        self.next_flow_id = 0
        
        # Статистика
        self.stats = {
            'total_created': 0,
            'total_completed': 0,
            'total_died': 0,
            'max_concurrent': 0
        }
        
        logger.info(f"EnergyLattice initialized: {self.width}x{self.height}x{self.depth}")
        logger.info(f"Input/output cells: {self.width * self.height}, max flows: {self.max_active_flows}")
    
    def place_initial_energy(self, embeddings: torch.Tensor) -> List[int]:
        """
        Размещает входные эмбеддинги на входной стороне куба (z=0)
        
        Args:
            embeddings: [batch, embedding_dim] - входные эмбеддинги (768D)
            
        Returns:
            flow_ids: Список ID созданных потоков
        """
        batch_size = embeddings.shape[0]
        
        # Проверяем размерность
        expected_dim = self.config.input_embedding_dim
        if embeddings.shape[1] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim}, got {embeddings.shape[1]}")
        
        # Очищаем неактивные потоки
        self._cleanup_inactive_flows()
        
        # Распределяем эмбеддинги по входным клеткам
        input_cells = self.width * self.height
        embedding_per_cell = embeddings.shape[1] // input_cells
        
        flow_ids = []
        
        # Создаем потоки для каждой входной клетки
        for y in range(self.height):
            for x in range(self.width):
                cell_idx = y * self.width + x
                
                # Извлекаем часть эмбеддинга для этой клетки
                start_idx = cell_idx * embedding_per_cell
                end_idx = start_idx + embedding_per_cell
                
                for batch_idx in range(batch_size):
                    if len(self.active_flows) >= self.max_active_flows:
                        logger.warning(f"Reached max active flows limit: {self.max_active_flows}")
                        break
                    
                    # Создаем энергию для клетки
                    if end_idx <= embeddings.shape[1]:
                        cell_energy = embeddings[batch_idx, start_idx:end_idx]
                    else:
                        # Padding для последних клеток
                        cell_energy = torch.zeros(embedding_per_cell, device=self.device)
                        available = embeddings.shape[1] - start_idx
                        if available > 0:
                            cell_energy[:available] = embeddings[batch_idx, start_idx:]
                    
                    # Создаем поток
                    position = torch.tensor([x, y, 0], dtype=torch.float32, device=self.device)
                    flow_id = self._create_flow(position, cell_energy)
                    flow_ids.append(flow_id)
        
        logger.info(f"Created {len(flow_ids)} initial flows on input surface")
        return flow_ids
    
    def _create_flow(self, position: torch.Tensor, energy: torch.Tensor, 
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
        
        # Проверяем достижение выходной стороны
        if new_position[2] >= self.depth - 1:
            flow.is_active = False
            self.stats['total_completed'] += 1
            logger.debug(f"Flow {flow_id} reached output side at age {flow.age}")
    
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
    
    def collect_output_energy(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает энергию с выходной стороны (z=depth-1)
        
        Returns:
            output_embeddings: [num_outputs, embedding_dim] - собранные эмбеддинги
            flow_ids: ID потоков, достигших выхода
        """
        output_flows = []
        flow_ids = []
        
        # Собираем потоки на выходной стороне
        for flow_id, flow in self.active_flows.items():
            if not flow.is_active:
                continue
                
            # Проверяем, достиг ли поток выходной стороны
            z_pos = flow.position[2].item()
            if z_pos >= self.depth - 1:
                output_flows.append(flow)
                flow_ids.append(flow_id)
        
        if not output_flows:
            # Возвращаем пустой тензор правильной размерности
            return torch.zeros(0, self.embedding_dim, device=self.device), []
        
        # Группируем по выходным клеткам
        output_grid = {}  # (x, y) -> List[energy]
        
        for flow in output_flows:
            x = int(torch.clamp(flow.position[0], 0, self.width - 1).item())
            y = int(torch.clamp(flow.position[1], 0, self.height - 1).item())
            
            key = (x, y)
            if key not in output_grid:
                output_grid[key] = []
            output_grid[key].append(flow.energy)
        
        # Агрегируем энергии в клетках
        output_embeddings = []
        
        for y in range(self.height):
            for x in range(self.width):
                key = (x, y)
                if key in output_grid:
                    # Усредняем энергии в клетке
                    cell_energies = torch.stack(output_grid[key])
                    aggregated = cell_energies.mean(dim=0)
                else:
                    # Пустая клетка
                    aggregated = torch.zeros(self.embedding_dim, device=self.device)
                
                output_embeddings.append(aggregated)
        
        # Собираем в один тензор
        output_embeddings = torch.stack(output_embeddings)  # [width*height, embedding_dim]
        
        # Объединяем части эмбеддингов обратно в 768D
        # Reshape и flatten для восстановления исходного эмбеддинга
        output_embeddings = output_embeddings.view(-1)[:self.config.input_embedding_dim]
        output_embeddings = output_embeddings.unsqueeze(0)  # [1, 768]
        
        logger.info(f"Collected energy from {len(output_flows)} flows at output side")
        
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