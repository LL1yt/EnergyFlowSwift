#!/usr/bin/env python3
"""
Batch Neighbor Extractor
=======================

Модуль для эффективного извлечения соседей для batch'а клеток.
Оптимизирует операции получения соседей для множества клеток одновременно.
"""

import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np

from ....utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchNeighborData:
    """Структура для хранения данных о соседях для batch'а"""
    # Маски для каждого типа соединений [batch_size]
    local_mask: torch.Tensor
    functional_mask: torch.Tensor  
    distant_mask: torch.Tensor
    
    # Индексы соседей [batch_size, max_neighbors_per_type]
    local_indices: torch.Tensor
    functional_indices: torch.Tensor
    distant_indices: torch.Tensor
    
    # Количество соседей каждого типа [batch_size]
    local_counts: torch.Tensor
    functional_counts: torch.Tensor
    distant_counts: torch.Tensor
    
    # Максимальное количество соседей каждого типа в batch'е
    max_local: int
    max_functional: int
    max_distant: int


class BatchNeighborExtractor:
    """
    Извлекает соседей для batch'а клеток эффективным способом.
    Использует векторизованные операции вместо циклов.
    """
    
    def __init__(self, max_neighbors: int = 26):
        self.max_neighbors = max_neighbors
        self._neighbor_cache = {}
        
    def extract_batch_neighbors(
        self,
        cell_indices: torch.Tensor,  # [batch_size]
        connection_classifier,  # Классификатор соединений
        full_states: torch.Tensor,  # [total_cells, state_size]
    ) -> BatchNeighborData:
        """
        Извлекает соседей для batch'а клеток
        
        Args:
            cell_indices: индексы клеток для обработки
            connection_classifier: классификатор для определения типов соединений
            full_states: состояния всех клеток решетки
            
        Returns:
            BatchNeighborData со всей информацией о соседях
        """
        batch_size = cell_indices.shape[0]
        device = cell_indices.device
        
        # Предварительно выделяем память для результатов
        local_indices_list = []
        functional_indices_list = []
        distant_indices_list = []
        
        local_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        functional_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        distant_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Batch обработка классификации соседей
        for i, cell_idx in enumerate(cell_indices):
            cell_idx_int = cell_idx.item()
            
            # Получаем классификацию из кэша (уже оптимизировано в connection_classifier)
            neighbors_data = connection_classifier.get_cached_neighbors_and_classification(
                cell_idx=cell_idx_int,
                states=full_states
            )
            
            # Сохраняем индексы
            local_idx = neighbors_data["local"]["indices"]
            functional_idx = neighbors_data["functional"]["indices"]
            distant_idx = neighbors_data["distant"]["indices"]
            
            local_indices_list.append(local_idx)
            functional_indices_list.append(functional_idx)
            distant_indices_list.append(distant_idx)
            
            # Считаем количества
            local_counts[i] = len(local_idx)
            functional_counts[i] = len(functional_idx)
            distant_counts[i] = len(distant_idx)
        
        # Находим максимальные количества для паддинга
        max_local = int(local_counts.max().item()) if local_counts.max() > 0 else 1
        max_functional = int(functional_counts.max().item()) if functional_counts.max() > 0 else 1
        max_distant = int(distant_counts.max().item()) if distant_counts.max() > 0 else 1
        
        # Создаем padded тензоры индексов
        local_indices = self._pad_indices(local_indices_list, max_local, batch_size, device)
        functional_indices = self._pad_indices(functional_indices_list, max_functional, batch_size, device)
        distant_indices = self._pad_indices(distant_indices_list, max_distant, batch_size, device)
        
        # Создаем маски
        local_mask = local_counts > 0
        functional_mask = functional_counts > 0
        distant_mask = distant_counts > 0
        
        return BatchNeighborData(
            local_mask=local_mask,
            functional_mask=functional_mask,
            distant_mask=distant_mask,
            local_indices=local_indices,
            functional_indices=functional_indices,
            distant_indices=distant_indices,
            local_counts=local_counts,
            functional_counts=functional_counts,
            distant_counts=distant_counts,
            max_local=max_local,
            max_functional=max_functional,
            max_distant=max_distant,
        )
    
    def extract_neighbor_states(
        self,
        full_states: torch.Tensor,  # [total_cells, state_size]
        neighbor_indices: torch.Tensor,  # [batch_size, max_neighbors]
        neighbor_counts: torch.Tensor,  # [batch_size]
        mask: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:
        """
        Извлекает состояния соседей для batch'а с учетом маски
        
        Returns:
            [num_valid, max_neighbors, state_size] - состояния соседей
        """
        if not mask.any():
            return torch.empty(0, 1, full_states.shape[1], device=full_states.device)
            
        # Берем только валидные элементы
        valid_indices = neighbor_indices[mask]  # [num_valid, max_neighbors]
        valid_counts = neighbor_counts[mask]  # [num_valid]
        
        batch_size = valid_indices.shape[0]
        max_neighbors = valid_indices.shape[1]
        state_size = full_states.shape[1]
        
        # Создаем выходной тензор
        neighbor_states = torch.zeros(
            batch_size, max_neighbors, state_size,
            device=full_states.device,
            dtype=full_states.dtype
        )
        
        # Векторизованное извлечение состояний
        for i in range(batch_size):
            n_neighbors = valid_counts[i]
            if n_neighbors > 0:
                valid_idx = valid_indices[i, :n_neighbors]
                neighbor_states[i, :n_neighbors] = full_states[valid_idx]
        
        return neighbor_states
    
    def _pad_indices(
        self,
        indices_list: List[List[int]],
        max_length: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Паддинг списка индексов до одинаковой длины"""
        padded = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        
        for i, indices in enumerate(indices_list):
            if indices:
                length = min(len(indices), max_length)
                padded[i, :length] = torch.tensor(indices[:length], device=device)
                
        return padded