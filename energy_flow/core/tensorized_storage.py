"""
Tensorized Flow Storage - эффективное хранение потоков в едином тензоре
========================================================================

Хранит все данные потоков в предаллоцированных тензорах для максимальной
параллелизации GPU операций. Поддерживает динамическое создание/удаление потоков.
"""

import torch
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger
from ..utils.device_manager import get_device_manager

logger = get_logger(__name__)


@dataclass
class TensorizedFlowStorage:
    """
    Тензорное хранилище для всех данных потоков
    
    Использует предаллоцированные тензоры и пул свободных индексов
    для эффективного управления памятью при spawn/death потоков.
    """
    
    def __init__(self, max_flows: int, embedding_dim: int, hidden_layers: int, 
                 hidden_size: int, device: torch.device):
        """
        Args:
            max_flows: Максимальное количество потоков
            embedding_dim: Размерность эмбеддингов
            hidden_layers: Количество слоев GRU
            hidden_size: Размер скрытого состояния
            device: Устройство для тензоров
        """
        self.max_flows = max_flows
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.device = device
        
        # Предаллоцированные тензоры для всех данных потоков
        self.positions = torch.zeros(max_flows, 3, device=device)
        self.energies = torch.zeros(max_flows, embedding_dim, device=device)
        self.hidden_states = torch.zeros(max_flows, hidden_layers, hidden_size, device=device)
        
        # Метаданные потоков (скаляры)
        self.batch_indices = torch.zeros(max_flows, dtype=torch.long, device=device)
        self.parent_ids = torch.full((max_flows,), -1, dtype=torch.long, device=device)
        self.ages = torch.zeros(max_flows, dtype=torch.long, device=device)
        self.steps_taken = torch.zeros(max_flows, dtype=torch.long, device=device)
        self.distances_to_surface = torch.zeros(max_flows, device=device)
        
        # Маска активности и тип поверхности
        self.is_active = torch.zeros(max_flows, dtype=torch.bool, device=device)
        self.is_completed = torch.zeros(max_flows, dtype=torch.bool, device=device)
        self.projected_surface = torch.zeros(max_flows, dtype=torch.long, device=device)  # 0=unknown, 1=z0, 2=zdepth
        
        # Mapping: flow_id -> tensor_index
        self.id_to_index: Dict[int, int] = {}
        self.index_to_id: Dict[int, int] = {}
        
        # Пул свободных индексов для эффективного переиспользования
        self.free_indices: Set[int] = set(range(max_flows))
        self.active_count = 0
        
        # Статистика
        self.total_created = 0
        self.total_completed = 0
        self.total_died = 0
        
        logger.info(f"TensorizedFlowStorage initialized: max_flows={max_flows}, "
                   f"embedding_dim={embedding_dim}, device={device}")
    
    def allocate_flows(self, num_flows: int, initial_positions: torch.Tensor,
                       initial_energies: torch.Tensor, batch_indices_list: List[int],
                       parent_ids: Optional[List[int]] = None) -> List[int]:
        """
        Аллоцирует место для новых потоков
        
        Args:
            num_flows: Количество новых потоков
            initial_positions: [num_flows, 3] начальные позиции
            initial_energies: [num_flows, embedding_dim] начальные энергии
            batch_indices_list: Список batch индексов
            parent_ids: Опциональные ID родительских потоков
            
        Returns:
            flow_ids: Список созданных ID потоков
        """
        if len(self.free_indices) < num_flows:
            available = len(self.free_indices)
            logger.warning(f"Cannot allocate {num_flows} flows, only {available} slots available")
            num_flows = available
            if num_flows == 0:
                return []
        
        # Получаем свободные индексы
        indices = []
        for _ in range(num_flows):
            idx = self.free_indices.pop()
            indices.append(idx)
        
        indices_tensor = torch.tensor(indices, device=self.device)
        
        # Заполняем тензоры данными новых потоков
        self.positions[indices_tensor] = initial_positions[:num_flows]
        self.energies[indices_tensor] = initial_energies[:num_flows]
        self.hidden_states[indices_tensor] = 0  # Инициализируем нулями
        
        # Метаданные
        self.batch_indices[indices_tensor] = torch.tensor(batch_indices_list[:num_flows], 
                                                          device=self.device)
        if parent_ids:
            self.parent_ids[indices_tensor] = torch.tensor(parent_ids[:num_flows], 
                                                          device=self.device)
        else:
            self.parent_ids[indices_tensor] = -1
        
        self.ages[indices_tensor] = 0
        self.steps_taken[indices_tensor] = 0
        self.distances_to_surface[indices_tensor] = 0
        
        # Активация
        self.is_active[indices_tensor] = True
        self.is_completed[indices_tensor] = False
        self.projected_surface[indices_tensor] = 0  # unknown
        
        # Создаем ID и mapping
        flow_ids = []
        for i, idx in enumerate(indices):
            flow_id = self.total_created + i
            flow_ids.append(flow_id)
            self.id_to_index[flow_id] = idx
            self.index_to_id[idx] = flow_id
        
        self.total_created += num_flows
        self.active_count += num_flows
        
        logger.debug(f"Allocated {num_flows} flows, active_count={self.active_count}")
        return flow_ids
    
    def deallocate_flows(self, flow_ids: List[int]):
        """Освобождает место от завершенных/умерших потоков"""
        indices_to_free = []
        
        for flow_id in flow_ids:
            if flow_id in self.id_to_index:
                idx = self.id_to_index[flow_id]
                indices_to_free.append(idx)
                
                # Очистка mapping
                del self.id_to_index[flow_id]
                if idx in self.index_to_id:
                    del self.index_to_id[idx]
                
                # Возвращаем индекс в пул
                self.free_indices.add(idx)
                
                # Деактивация
                self.is_active[idx] = False
        
        if indices_to_free:
            indices_tensor = torch.tensor(indices_to_free, device=self.device)
            # Опционально: очистка данных для экономии памяти
            self.positions[indices_tensor] = 0
            self.energies[indices_tensor] = 0
            self.hidden_states[indices_tensor] = 0
            
            self.active_count -= len(indices_to_free)
            logger.debug(f"Deallocated {len(indices_to_free)} flows, active_count={self.active_count}")
    
    def get_active_mask(self) -> torch.Tensor:
        """Возвращает маску активных потоков"""
        return self.is_active
    
    def get_active_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает данные только активных потоков
        
        Returns:
            positions: [num_active, 3]
            energies: [num_active, embedding_dim] 
            hidden_states: [num_active, layers, hidden]
            flow_ids: [num_active] - ID активных потоков
        """
        active_mask = self.is_active
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return (torch.empty(0, 3, device=self.device),
                   torch.empty(0, self.embedding_dim, device=self.device),
                   torch.empty(0, self.hidden_layers, self.hidden_size, device=self.device),
                   torch.empty(0, dtype=torch.long, device=self.device))
        
        # Собираем ID для активных потоков
        flow_ids = []
        for idx in active_indices.cpu().tolist():
            if idx in self.index_to_id:
                flow_ids.append(self.index_to_id[idx])
        
        flow_ids_tensor = torch.tensor(flow_ids, device=self.device)
        
        return (self.positions[active_indices],
               self.energies[active_indices],
               self.hidden_states[active_indices],
               flow_ids_tensor)
    
    def batch_update(self, flow_ids: torch.Tensor, 
                     new_positions: Optional[torch.Tensor] = None,
                     new_energies: Optional[torch.Tensor] = None,
                     new_hidden: Optional[torch.Tensor] = None,
                     increment_age: bool = True):
        """
        Векторизованное обновление множества потоков
        
        Args:
            flow_ids: [batch] ID потоков для обновления
            new_positions: [batch, 3] новые позиции
            new_energies: [batch, embedding_dim] новые энергии
            new_hidden: [batch, layers, hidden] новые скрытые состояния
            increment_age: Увеличить возраст потоков
        """
        # Конвертируем flow_ids в индексы
        indices = []
        for fid in flow_ids.cpu().tolist():
            if fid in self.id_to_index:
                indices.append(self.id_to_index[fid])
        
        if not indices:
            return
        
        indices_tensor = torch.tensor(indices, device=self.device)
        
        # Обновляем данные
        if new_positions is not None:
            self.positions[indices_tensor] = new_positions[:len(indices)]
        
        if new_energies is not None:
            self.energies[indices_tensor] = new_energies[:len(indices)]
        
        if new_hidden is not None:
            self.hidden_states[indices_tensor] = new_hidden[:len(indices)]
        
        if increment_age:
            self.ages[indices_tensor] += 1
            self.steps_taken[indices_tensor] += 1
    
    def mark_completed(self, flow_ids: List[int], surface_type: str):
        """
        Помечает потоки как завершенные
        
        Args:
            flow_ids: ID потоков
            surface_type: 'z0' или 'zdepth'
        """
        surface_code = 1 if surface_type == 'z0' else 2
        
        for flow_id in flow_ids:
            if flow_id in self.id_to_index:
                idx = self.id_to_index[flow_id]
                self.is_completed[idx] = True
                self.is_active[idx] = False
                self.projected_surface[idx] = surface_code
                self.total_completed += 1
    
    def get_statistics(self) -> Dict[str, any]:
        """Возвращает статистику хранилища"""
        return {
            'total_created': self.total_created,
            'total_completed': self.total_completed,
            'total_died': self.total_died,
            'active_count': self.active_count,
            'free_slots': len(self.free_indices),
            'utilization': self.active_count / self.max_flows if self.max_flows > 0 else 0
        }
    
    def spawn_flows(self, parent_flow_ids: torch.Tensor, num_spawns_per_parent: torch.Tensor,
                   spawn_energies: torch.Tensor) -> List[int]:
        """
        Создает новые потоки от родительских (векторизованно)
        
        Args:
            parent_flow_ids: [num_parents] ID родительских потоков
            num_spawns_per_parent: [num_parents] количество spawn'ов для каждого родителя
            spawn_energies: [total_spawns, embedding_dim] энергии новых потоков
            
        Returns:
            new_flow_ids: ID созданных потоков
        """
        total_spawns = num_spawns_per_parent.sum().item()
        if total_spawns == 0:
            return []
        
        # Собираем позиции и метаданные родителей
        parent_positions = []
        parent_hidden = []
        parent_batch_indices = []
        spawn_parent_ids = []
        
        spawn_idx = 0
        for i, parent_id in enumerate(parent_flow_ids.cpu().tolist()):
            if parent_id in self.id_to_index:
                idx = self.id_to_index[parent_id]
                num_spawns = num_spawns_per_parent[i].item()
                
                for _ in range(num_spawns):
                    parent_positions.append(self.positions[idx])
                    parent_hidden.append(self.hidden_states[idx])
                    parent_batch_indices.append(self.batch_indices[idx].item())
                    spawn_parent_ids.append(parent_id)
                    spawn_idx += 1
        
        if not parent_positions:
            return []
        
        # Создаем тензоры
        initial_positions = torch.stack(parent_positions)
        initial_hidden = torch.stack(parent_hidden)
        
        # Аллоцируем новые потоки
        new_flow_ids = self.allocate_flows(
            len(parent_positions),
            initial_positions,
            spawn_energies[:len(parent_positions)],
            parent_batch_indices,
            spawn_parent_ids
        )
        
        # Копируем hidden states от родителей
        if new_flow_ids:
            new_indices = [self.id_to_index[fid] for fid in new_flow_ids]
            new_indices_tensor = torch.tensor(new_indices, device=self.device)
            self.hidden_states[new_indices_tensor] = initial_hidden
        
        return new_flow_ids
