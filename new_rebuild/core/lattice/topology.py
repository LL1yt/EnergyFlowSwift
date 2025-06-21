"""
Модуль Топологии Соседства
===========================

Содержит класс NeighborTopology, отвечающий за определение и поиск
соседей для каждой клетки в 3D-решетке. Поддерживает различные
стратегии поиска, включая локальные, случайные и многоуровневые
гибридные подходы. Все операции для производительности проводятся
с использованием линейных индексов клеток.

Адаптирован для работы с ProjectConfig из new_rebuild.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import logging
import torch
from datetime import datetime
import json

from ...config import get_project_config
from .enums import BoundaryCondition, NeighborStrategy
from .position import Position3D, Coordinates3D
from .spatial_hashing import SpatialHashGrid


class NeighborTopology:
    """
    Система управления соседством клеток в 3D решетке.

    Реализует различные типы граничных условий и предоставляет
    эффективные, основанные на индексах, методы для получения
    соседей каждой клетки.

    Поддерживает разные стратегии поиска соседей:
    - local: стандартные 6 соседей (фон Нейман)
    - random_sample: случайная выборка N соседей со всей решетки
    - hybrid: комбинация локальных и случайных соседей
    - tiered: трехуровневая стратегия (локальные+функциональные+дальние)
    """

    _LOCAL_NEIGHBOR_DIRECTIONS = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    def __init__(self, all_coords: List[Coordinates3D]):
        """
        Инициализация системы соседства.

        Args:
            all_coords: Полный список всех координат в решетке.
        """
        self.config = get_project_config()

        # Инициализация из ProjectConfig
        self.dimensions = self.config.lattice_dimensions
        self.boundary_conditions = BoundaryCondition.PERIODIC  # по умолчанию
        self.pos_helper = Position3D(self.dimensions)

        # Стратегия поиска соседей - используем hybrid по умолчанию
        self.strategy = NeighborStrategy.HYBRID

        # Количество соседей - синхронизированное значение
        if self.config.architecture_type == "nca":
            self.num_neighbors = self.config.nca_neighbor_count
        elif self.config.architecture_type == "gmlp":
            self.num_neighbors = self.config.gmlp_neighbor_count
        else:  # hybrid
            # Для hybrid берем максимальное значение
            self.num_neighbors = max(
                self.config.nca_neighbor_count, self.config.gmlp_neighbor_count
            )

        # Конфигурация стратегии (можно расширить в будущем)
        self.strategy_config = {
            "local_count": 6,
            "local_tier": {"radius": 5.0, "ratio": 0.7},
            "functional_tier": {"ratio": 0.2},
            "local_grid_cell_size": 5.0,
        }

        self._all_indices_set = set(range(self.pos_helper.total_positions))

        self._spatial_grid: Optional[SpatialHashGrid] = None
        if self.strategy == NeighborStrategy.TIERED:
            grid_cell_size = self.strategy_config.get("local_grid_cell_size", 5.0)
            self._spatial_grid = SpatialHashGrid(self.dimensions, grid_cell_size)
            for i, c in enumerate(all_coords):
                self._spatial_grid.insert(c, i)

        # Кэширование - включено по умолчанию для производительности
        self.neighbor_cache: Optional[Dict[int, List[int]]] = {}
        self._build_neighbor_cache()

        self.device = torch.device(self.config.device)

        # Логирование инициализации с централизованной системой
        from ...utils.logging import log_init

        if self.config.debug_mode:
            log_init(
                "NeighborTopology",
                dimensions=self.dimensions,
                neighbors=self.num_neighbors,
                strategy=self.strategy.value,
                boundary=self.boundary_conditions.value,
            )

    def get_neighbor_indices(self, linear_index: int) -> List[int]:
        """
        Получает список линейных индексов соседей.
        Использует кэш или вычисляет "на лету" в зависимости от настроек.
        """
        if self.neighbor_cache is not None:
            return self.neighbor_cache.get(linear_index, [])

        if self.strategy == NeighborStrategy.LOCAL:
            return self._get_local_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.RANDOM_SAMPLE:
            return self._get_random_sample_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.HYBRID:
            return self._get_hybrid_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.TIERED:
            return self._get_tiered_neighbor_indices(linear_index)
        else:
            raise ValueError(f"Unknown neighbor finding strategy: {self.strategy}")

    def _get_local_neighbor_indices(self, linear_index: int) -> List[int]:
        """Возвращает до 6 локальных соседей в виде линейных индексов."""
        neighbors = []
        coords = self.pos_helper.to_3d_coordinates(linear_index)
        for direction in self._LOCAL_NEIGHBOR_DIRECTIONS:
            neighbor_coords = (
                coords[0] + direction[0],
                coords[1] + direction[1],
                coords[2] + direction[2],
            )
            valid_coords = self._apply_boundary_conditions(neighbor_coords)
            if valid_coords:
                neighbors.append(self.pos_helper.to_linear_index(valid_coords))
        return neighbors

    def _get_random_sample_neighbor_indices(self, linear_index: int) -> List[int]:
        """Возвращает случайную выборку N соседей со всей решетки в виде индексов."""
        possible_neighbors = list(self._all_indices_set - {linear_index})
        num_to_sample = min(self.num_neighbors, len(possible_neighbors))
        if num_to_sample == 0:
            return []

        indices = np.random.choice(possible_neighbors, num_to_sample, replace=False)
        return list(indices)

    def _get_hybrid_neighbor_indices(self, linear_index: int) -> List[int]:
        """Комбинирует локальных и случайных соседей, возвращая их индексы."""
        local_count = self.strategy_config.get("local_count", 6)
        random_count = max(0, self.num_neighbors - local_count)

        local_neighbors = self._get_local_neighbor_indices(linear_index)
        if len(local_neighbors) > local_count:
            local_neighbors = list(
                np.random.choice(local_neighbors, local_count, replace=False)
            )

        exclude_set = {linear_index}.union(local_neighbors)
        possible_random = list(self._all_indices_set - exclude_set)

        num_to_sample = min(random_count, len(possible_random))
        random_neighbors = []
        if num_to_sample > 0:
            indices = np.random.choice(possible_random, num_to_sample, replace=False)
            random_neighbors = list(indices)

        return local_neighbors + random_neighbors

    def _get_tiered_neighbor_indices(self, cell_idx: int) -> List[int]:
        """
        Реализует трехуровневую гибридную стратегию, возвращая индексы.
        """
        if self._spatial_grid is None:
            raise RuntimeError(
                "SpatialHashGrid не инициализирован для 'tiered' стратегии."
            )

        current_coords_3d = self.pos_helper.to_3d_coordinates(cell_idx)

        # 1. Local Tier (через Spatial Hashing)
        local_config = self.strategy_config.get("local_tier", {})
        local_radius = local_config.get("radius", 5.0)
        local_ratio = local_config.get("ratio", 0.7)
        local_count = int(self.num_neighbors * local_ratio)

        local_indices = self._spatial_grid.query_radius(current_coords_3d, local_radius)
        local_indices_set = set(local_indices)
        local_indices_set.discard(cell_idx)

        if len(local_indices_set) > local_count:
            final_local_indices = list(
                np.random.choice(list(local_indices_set), local_count, replace=False)
            )
        else:
            final_local_indices = list(local_indices_set)

        # 2. Functional Tier (случайные соседи)
        functional_config = self.strategy_config.get("functional_tier", {})
        functional_ratio = functional_config.get("ratio", 0.2)
        functional_count = int(self.num_neighbors * functional_ratio)

        exclude_indices = set(final_local_indices) | {cell_idx}
        possible_functional = list(self._all_indices_set - exclude_indices)

        num_to_sample_func = min(functional_count, len(possible_functional))
        functional_indices = []
        if num_to_sample_func > 0:
            functional_indices = list(
                np.random.choice(possible_functional, num_to_sample_func, replace=False)
            )

        # 3. Long-range Tier (дальние связи с весами от расстояния)
        long_range_count = (
            self.num_neighbors - len(final_local_indices) - len(functional_indices)
        )

        exclude_indices.update(functional_indices)
        possible_long_range = list(self._all_indices_set - exclude_indices)
        long_range_indices = []

        if long_range_count > 0 and possible_long_range:
            distances = np.array(
                [
                    self.pos_helper.euclidean_distance(
                        current_coords_3d, self.pos_helper.to_3d_coordinates(idx)
                    )
                    for idx in possible_long_range
                ]
            )
            probabilities = 1.0 / (distances + 1e-6)
            probabilities_sum = np.sum(probabilities)
            if probabilities_sum > 0:
                probabilities /= probabilities_sum
            else:  # Fallback to uniform if all probabilities are zero
                probabilities = np.ones(len(possible_long_range)) / len(
                    possible_long_range
                )

            num_to_sample_lr = min(long_range_count, len(possible_long_range))
            long_range_indices = list(
                np.random.choice(
                    possible_long_range,
                    num_to_sample_lr,
                    replace=False,
                    p=probabilities,
                )
            )

        final_indices = final_local_indices + functional_indices + long_range_indices
        return final_indices

    def _apply_boundary_conditions(
        self, coords: Coordinates3D
    ) -> Optional[Coordinates3D]:
        """
        Применяет граничные условия к координатам, возвращая новые
        координаты или None, если клетка находится за пределами "стены".
        """
        if self.boundary_conditions == BoundaryCondition.WALLS:
            if self.pos_helper.is_valid_coordinates(coords):
                return coords
            return None
        elif self.boundary_conditions == BoundaryCondition.PERIODIC:
            x, y, z = coords
            return (
                x % self.dimensions[0],
                y % self.dimensions[1],
                z % self.dimensions[2],
            )
        elif self.boundary_conditions == BoundaryCondition.ABSORBING:
            # Для absorbing, клетки за границей "поглощаются"
            if self.pos_helper.is_valid_coordinates(coords):
                return coords
            return None
        elif self.boundary_conditions == BoundaryCondition.REFLECTING:
            # Для reflecting, координаты отражаются от границ
            x, y, z = coords
            x = max(0, min(x, self.dimensions[0] - 1))
            y = max(0, min(y, self.dimensions[1] - 1))
            z = max(0, min(z, self.dimensions[2] - 1))
            return (x, y, z)
        else:
            raise ValueError(
                f"Неподдерживаемые граничные условия: {self.boundary_conditions}"
            )

    def _build_neighbor_cache(self):
        """Предварительно строит кэш всех соседей для быстрого доступа."""
        if self.neighbor_cache is None:
            return

        for linear_index in range(self.pos_helper.total_positions):
            neighbors = self._compute_neighbors_without_cache(linear_index)
            self.neighbor_cache[linear_index] = neighbors

    def _compute_neighbors_without_cache(self, linear_index: int) -> List[int]:
        """Вычисляет соседей без использования кэша."""
        if self.strategy == NeighborStrategy.LOCAL:
            return self._get_local_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.RANDOM_SAMPLE:
            return self._get_random_sample_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.HYBRID:
            return self._get_hybrid_neighbor_indices(linear_index)
        elif self.strategy == NeighborStrategy.TIERED:
            return self._get_tiered_neighbor_indices(linear_index)
        else:
            raise ValueError(f"Unknown neighbor finding strategy: {self.strategy}")

    def get_all_neighbor_indices_batched(self) -> torch.Tensor:
        """
        Возвращает тензор всех соседей для batch обработки.
        Форма: [total_cells, max_neighbors]
        """
        max_neighbors = self.num_neighbors
        total_cells = self.pos_helper.total_positions

        neighbor_tensor = torch.full(
            (total_cells, max_neighbors),
            fill_value=-1,  # -1 для отсутствующих соседей
            dtype=torch.long,
            device=self.device,
        )

        for cell_idx in range(total_cells):
            neighbors = self.get_neighbor_indices(cell_idx)
            num_actual_neighbors = min(len(neighbors), max_neighbors)
            if num_actual_neighbors > 0:
                neighbor_tensor[cell_idx, :num_actual_neighbors] = torch.tensor(
                    neighbors[:num_actual_neighbors], dtype=torch.long
                )

        return neighbor_tensor

    def validate_topology(self) -> Dict[str, Any]:
        """Валидирует топологию и возвращает статистику."""
        stats = {
            "total_cells": self.pos_helper.total_positions,
            "target_neighbors": self.num_neighbors,
            "strategy": self.strategy.value,
            "boundary_conditions": self.boundary_conditions.value,
            "dimensions": self.dimensions,
        }

        # Проверяем несколько случайных клеток
        sample_size = min(10, self.pos_helper.total_positions)
        sample_indices = np.random.choice(
            self.pos_helper.total_positions, sample_size, replace=False
        )

        neighbor_counts = []
        for idx in sample_indices:
            neighbors = self.get_neighbor_indices(idx)
            neighbor_counts.append(len(neighbors))

        stats.update(
            {
                "sample_neighbor_counts": neighbor_counts,
                "avg_neighbors": np.mean(neighbor_counts),
                "min_neighbors": np.min(neighbor_counts),
                "max_neighbors": np.max(neighbor_counts),
            }
        )

        return stats
