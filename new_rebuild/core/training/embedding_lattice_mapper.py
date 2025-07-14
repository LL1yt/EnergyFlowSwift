#!/usr/bin/env python3
"""
Маппинг эмбедингов в 3D решетку и обратно
=========================================

Правильная интеграция teacher эмбедингов с 3D cellular neural network:
1. EmbeddingToLatticeMapper - размещение эмбедингов на поверхности куба
2. LatticeToEmbeddingExtractor - извлечение эмбедингов с поверхности
3. VolumeStateInitializer - инициализация внутренних клеток

Архитектурный поток:
Teacher Embeddings (768D) → Surface (8×8=64D) → Lattice States → MoE Processing → 
Surface States → Teacher Embeddings (768D)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from ...config.simple_config import SimpleProjectConfig
from ...utils.logging import get_logger
from ..lattice.position import Position3D

logger = get_logger(__name__)


@dataclass
class EmbeddingLatticeSettings:
    """Настройки интеграции эмбедингов с решеткой"""
    
    # Размещение на поверхности
    placement_strategy: str = "faces"  # "faces", "edges", "corners", "uniform"
    surface_coverage: float = 0.8  # Покрытие поверхности
    
    # Динамика решетки
    lattice_steps: int = 5  # Количество шагов эмерджентной динамики
    convergence_threshold: float = 1e-4  # Порог сходимости
    
    # Извлечение результатов
    extraction_strategy: str = "surface_mean"  # "surface_mean", "weighted_surface", "volume_projection"
    
    # Обучение
    lattice_loss_weight: float = 0.1  # Вес loss'а внутренней динамики
    spatial_consistency_weight: float = 0.05  # Вес пространственной согласованности


class EmbeddingToLatticeMapper(nn.Module):
    """
    Размещает эмбединги на поверхности 3D куба
    
    Преобразует поверхностные эмбединги в полноценные состояния 3D решетки,
    где поверхностные клетки получают входные данные, а внутренние 
    инициализируются на основе поверхностных.
    """
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.lattice_dims = config.lattice.dimensions  # (8, 8, 8)
        self.surface_dim = config.cube_embedding_dim  # Вычисляется автоматически
        self.state_size = config.model.state_size  # 32
        
        # Настройки маппинга
        self.lattice_settings = EmbeddingLatticeSettings()
        
        logger.info(f"EmbeddingToLatticeMapper: {self.lattice_dims} → state_size={self.state_size}")
        
        # Проекция эмбедингов в состояния клеток
        self.embedding_to_state = nn.Sequential(
            nn.Linear(self.surface_dim, self.state_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.state_size * 2),
            nn.Linear(self.state_size * 2, self.state_size)
        )
        
        # Инициализация внутренних клеток
        self.volume_initializer = VolumeStateInitializer(config)
        
        # Позиционное кодирование для лучшего размещения
        self.positional_encoding = SurfacePositionalEncoding(
            self.lattice_dims, self.state_size
        )
        
        # Кэш индексов поверхности
        self._surface_indices = None
        self._volume_indices = None
    
    def forward(self, surface_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Маппинг эмбедингов в состояния решетки
        
        Args:
            surface_embeddings: [batch, surface_dim] - поверхностные эмбединги
        Returns:
            lattice_states: [batch, total_cells, state_size] - состояния всех клеток
        """
        batch_size = surface_embeddings.shape[0]
        total_cells = np.prod(self.lattice_dims)
        
        # Инициализируем все состояния нулями
        lattice_states = torch.zeros(
            batch_size, total_cells, self.state_size, 
            device=surface_embeddings.device, dtype=surface_embeddings.dtype
        )
        
        # Получаем индексы поверхности и объема
        surface_indices = self.get_surface_indices()
        volume_indices = self.get_volume_indices()
        
        # Преобразуем эмбединги в состояния клеток
        surface_states = self.embedding_to_state(surface_embeddings)  # [batch, state_size]
        
        # Добавляем позиционное кодирование
        surface_states = self.positional_encoding(surface_states, surface_indices)
        
        # Размещаем на поверхностных клетках
        # Дублируем состояние на все поверхностные клетки
        num_surface_cells = len(surface_indices)
        
        # Расширяем состояния для всех поверхностных клеток
        surface_states_expanded = surface_states.unsqueeze(1).expand(
            batch_size, num_surface_cells, self.state_size
        )  # [batch, num_surface_cells, state_size]
        
        # Размещаем на поверхности решетки
        lattice_states[:, surface_indices] = surface_states_expanded
        
        # Инициализируем внутренние клетки
        volume_states = self.volume_initializer(
            lattice_states[:, surface_indices], volume_indices, surface_indices
        )
        lattice_states[:, volume_indices] = volume_states
        
        logger.debug(f"Маппинг завершен: {surface_embeddings.shape} → {lattice_states.shape}")
        
        return lattice_states
    
    def get_surface_indices(self) -> List[int]:
        """Получение индексов поверхностных клеток"""
        if self._surface_indices is None:
            self._surface_indices = self._compute_surface_indices()
        return self._surface_indices
    
    def get_volume_indices(self) -> List[int]:
        """Получение индексов внутренних клеток"""
        if self._volume_indices is None:
            surface_set = set(self.get_surface_indices())
            total_cells = np.prod(self.lattice_dims)
            self._volume_indices = [i for i in range(total_cells) if i not in surface_set]
        return self._volume_indices
    
    def _compute_surface_indices(self) -> List[int]:
        """Вычисление индексов клеток на поверхности куба"""
        x_max, y_max, z_max = self.lattice_dims
        surface_indices = []
        
        if self.lattice_settings.placement_strategy == "faces":
            # Векторизованное размещение на всех 6 гранях куба
            # Создаем сетку координат
            x_coords, y_coords, z_coords = torch.meshgrid(
                torch.arange(x_max), torch.arange(y_max), torch.arange(z_max), indexing='ij'
            )
            
            # Маска для граней (векторизованная проверка)
            surface_mask = (
                (x_coords == 0) | (x_coords == x_max-1) |
                (y_coords == 0) | (y_coords == y_max-1) |
                (z_coords == 0) | (z_coords == z_max-1)
            )
            
            # Получаем индексы поверхностных клеток
            surface_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)[surface_mask]
            surface_indices = (surface_coords[:, 0] * y_max * z_max + 
                             surface_coords[:, 1] * z_max + 
                             surface_coords[:, 2]).tolist()
        
        elif self.lattice_settings.placement_strategy == "corners":
            # Векторизованное размещение углов куба
            corners = torch.tensor([
                [0, 0, 0], [0, 0, z_max-1], [0, y_max-1, 0], [0, y_max-1, z_max-1],
                [x_max-1, 0, 0], [x_max-1, 0, z_max-1], [x_max-1, y_max-1, 0], [x_max-1, y_max-1, z_max-1]
            ])
            surface_indices = (corners[:, 0] * y_max * z_max + 
                             corners[:, 1] * z_max + 
                             corners[:, 2]).tolist()
        
        elif self.lattice_settings.placement_strategy == "uniform":
            # Векторизованное равномерное распределение
            step = max(1, int(np.cbrt(np.prod(self.lattice_dims) / 64)))  # Примерно 64 точки
            
            # Создаем векторизованную сетку с шагом
            x_range = torch.arange(0, x_max, step)
            y_range = torch.arange(0, y_max, step)
            z_range = torch.arange(0, z_max, step)
            
            x_coords, y_coords, z_coords = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            coords = torch.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], dim=-1)
            
            surface_indices = (coords[:, 0] * y_max * z_max + 
                             coords[:, 1] * z_max + 
                             coords[:, 2]).tolist()
        
        logger.debug(f"Surface indices ({self.lattice_settings.placement_strategy}): {len(surface_indices)} клеток")
        return surface_indices
    
    def _interpolate_to_surface(self, surface_states: torch.Tensor, num_surface_cells: int) -> torch.Tensor:
        """Интерполяция состояний на поверхностные клетки"""
        batch_size, input_dim, state_size = surface_states.shape
        
        if input_dim >= num_surface_cells:
            # Если состояний достаточно, берем первые
            return surface_states[:, :num_surface_cells, :]
        else:
            # Интерполируем с повторением
            repeat_factor = (num_surface_cells + input_dim - 1) // input_dim
            interpolated = surface_states.repeat(1, repeat_factor, 1)
            return interpolated[:, :num_surface_cells, :]


class VolumeStateInitializer(nn.Module):
    """Инициализация состояний внутренних клеток на основе поверхностных"""
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.state_size = config.model.state_size
        self.lattice_dims = config.lattice.dimensions
        
        # Сеть для генерации внутренних состояний
        self.state_generator = nn.Sequential(
            nn.Linear(self.state_size, self.state_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_size * 2, self.state_size),
            nn.Tanh()  # Ограничиваем диапазон начальных состояний
        )
        
        # Позиционная зависимость
        self.position_encoder = nn.Linear(3, self.state_size // 4)  # x, y, z координаты
    
    def forward(self, surface_states: torch.Tensor, 
                volume_indices: List[int], 
                surface_indices: List[int]) -> torch.Tensor:
        """
        Генерация состояний для внутренних клеток
        
        Args:
            surface_states: [batch, num_surface, state_size] - состояния поверхности
            volume_indices: индексы внутренних клеток
            surface_indices: индексы поверхностных клеток
        Returns:
            volume_states: [batch, num_volume, state_size] - состояния внутренних клеток
        """
        batch_size = surface_states.shape[0]
        num_volume = len(volume_indices)
        
        # Усредняем поверхностные состояния как базу
        surface_mean = surface_states.mean(dim=1, keepdim=True)  # [batch, 1, state_size]
        
        # Генерируем состояния для каждой внутренней клетки
        volume_states_list = []
        
        for i, vol_idx in enumerate(volume_indices):
            # Получаем координаты клетки
            coords = self._index_to_coords(vol_idx)
            
            # Кодируем позицию
            pos_encoding = self.position_encoder(
                torch.tensor(coords, dtype=surface_states.dtype, device=surface_states.device)
            )
            
            # Генерируем состояние на основе поверхности и позиции
            base_state = self.state_generator(surface_mean.squeeze(1))  # [batch, state_size]
            
            # Добавляем позиционную информацию (избегаем in-place операции)
            if pos_encoding.shape[-1] <= self.state_size:
                # Create a new tensor instead of modifying in-place
                modified_state = base_state.clone()
                modified_state[:, :pos_encoding.shape[-1]] = modified_state[:, :pos_encoding.shape[-1]] + pos_encoding
                base_state = modified_state
            
            volume_states_list.append(base_state.unsqueeze(1))  # [batch, 1, state_size]
        
        # Stack all volume states
        if volume_states_list:
            volume_states = torch.cat(volume_states_list, dim=1)  # [batch, num_volume, state_size]
        else:
            volume_states = torch.zeros(
                batch_size, 0, self.state_size,
                device=surface_states.device, dtype=surface_states.dtype
            )
        
        return volume_states
    
    def _index_to_coords(self, index: int) -> Tuple[float, float, float]:
        """Преобразование линейного индекса в 3D координаты (нормализованные)"""
        x_max, y_max, z_max = self.lattice_dims
        
        z = index % z_max
        y = (index // z_max) % y_max
        x = index // (y_max * z_max)
        
        # Нормализуем координаты в диапазон [-1, 1]
        x_norm = (x / (x_max - 1)) * 2 - 1 if x_max > 1 else 0
        y_norm = (y / (y_max - 1)) * 2 - 1 if y_max > 1 else 0
        z_norm = (z / (z_max - 1)) * 2 - 1 if z_max > 1 else 0
        
        return (x_norm, y_norm, z_norm)


class SurfacePositionalEncoding(nn.Module):
    """Позиционное кодирование для поверхностных клеток"""
    
    def __init__(self, lattice_dims: Tuple[int, int, int], state_size: int):
        super().__init__()
        self.lattice_dims = lattice_dims
        self.state_size = state_size
        
        # Позиционные эмбединги
        self.pos_embedding = nn.Parameter(
            torch.randn(np.prod(lattice_dims), state_size) * 0.1
        )
    
    def forward(self, states: torch.Tensor, surface_indices: List[int]) -> torch.Tensor:
        """Добавление позиционного кодирования к состояниям"""
        batch_size = states.shape[0]
        
        if states.dim() == 2:  # [batch, state_size]
            # Один эмбединг для всех поверхностных клеток
            return states  # Позиционное кодирование будет добавлено при размещении
        else:  # [batch, num_surface, state_size]
            # Добавляем позиционные эмбединги
            for i, surf_idx in enumerate(surface_indices[:states.shape[1]]):
                states[:, i] += self.pos_embedding[surf_idx]
        
        return states


class LatticeToEmbeddingExtractor(nn.Module):
    """
    Извлечение эмбедингов с поверхности 3D решетки
    
    Обратная операция к EmbeddingToLatticeMapper - извлекает
    состояния с поверхности и преобразует их в эмбединги.
    """
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.lattice_dims = config.lattice.dimensions
        self.surface_dim = config.cube_embedding_dim  # Вычисляется автоматически
        self.state_size = config.model.state_size  # 32
        
        # Настройки извлечения
        self.lattice_settings = EmbeddingLatticeSettings()
        
        logger.info(f"LatticeToEmbeddingExtractor: state_size={self.state_size} → {self.surface_dim}")
        
        # Проекция состояний клеток в эмбединги
        self.state_to_embedding = nn.Sequential(
            nn.Linear(self.state_size, self.state_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.state_size * 2),
            nn.Dropout(0.1),
            nn.Linear(self.state_size * 2, self.surface_dim)
        )
        
        # Агрегация поверхностных состояний
        if self.lattice_settings.extraction_strategy == "weighted_surface":
            self.attention_weights = nn.MultiheadAttention(
                embed_dim=self.state_size, num_heads=4, batch_first=True
            )
    
    def forward(self, lattice_states: torch.Tensor) -> torch.Tensor:
        """
        Извлечение эмбедингов из состояний решетки
        
        Args:
            lattice_states: [batch, total_cells, state_size] - состояния всех клеток
        Returns:
            surface_embeddings: [batch, surface_dim] - эмбединги поверхности
        """
        # Получаем индексы поверхности
        surface_indices = self._get_surface_indices()
        
        # Извлекаем состояния поверхностных клеток
        surface_states = lattice_states[:, surface_indices, :]  # [batch, num_surface, state_size]
        
        # Агрегируем поверхностные состояния
        if self.lattice_settings.extraction_strategy == "surface_mean":
            aggregated_states = surface_states.mean(dim=1)  # [batch, state_size]
        
        elif self.lattice_settings.extraction_strategy == "weighted_surface":
            # Используем attention для взвешенной агрегации
            attended_states, _ = self.attention_weights(
                surface_states, surface_states, surface_states
            )
            aggregated_states = attended_states.mean(dim=1)
        
        elif self.lattice_settings.extraction_strategy == "volume_projection":
            # Используем всю решетку с весами
            all_states = lattice_states.mean(dim=1)  # [batch, state_size]
            surface_contribution = surface_states.mean(dim=1)
            aggregated_states = 0.7 * surface_contribution + 0.3 * all_states
        
        else:
            aggregated_states = surface_states.mean(dim=1)
        
        # Преобразуем в эмбединги
        surface_embeddings = self.state_to_embedding(aggregated_states)
        
        logger.debug(f"Извлечение завершено: {lattice_states.shape} → {surface_embeddings.shape}")
        
        return surface_embeddings
    
    def _get_surface_indices(self) -> List[int]:
        """Получение индексов поверхностных клеток (аналогично EmbeddingToLatticeMapper)"""
        x_max, y_max, z_max = self.lattice_dims
        surface_indices = []
        
        for x in range(x_max):
            for y in range(y_max):
                for z in range(z_max):
                    if (x == 0 or x == x_max-1 or 
                        y == 0 or y == y_max-1 or 
                        z == 0 or z == z_max-1):
                        idx = x * y_max * z_max + y * z_max + z
                        surface_indices.append(idx)
        
        return surface_indices


# === ФАБРИЧНЫЕ ФУНКЦИИ ===

def create_embedding_lattice_mapper(config: SimpleProjectConfig) -> EmbeddingToLatticeMapper:
    """Фабричная функция для создания маппера эмбедингов в решетку"""
    return EmbeddingToLatticeMapper(config)


def create_lattice_embedding_extractor(config: SimpleProjectConfig) -> LatticeToEmbeddingExtractor:
    """Фабричная функция для создания экстрактора эмбедингов из решетки"""
    return LatticeToEmbeddingExtractor(config)