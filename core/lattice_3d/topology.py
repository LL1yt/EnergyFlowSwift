"""
Модуль Топологии Соседства
===========================

Содержит класс NeighborTopology, отвечающий за определение и поиск
соседей для каждой клетки в 3D-решетке. Поддерживает различные
стратегии поиска, включая локальные, случайные и многоуровневые
гибридные подходы. Все операции для производительности проводятся
с использованием линейных индексов клеток.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import logging
import torch
from datetime import datetime
import json

from .config import LatticeConfig
from .enums import BoundaryCondition, NeighborStrategy
from .position import Position3D, Coordinates3D
from .spatial_hashing import SpatialHashGrid
from ..log_utils import _get_caller_info


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

    def __init__(self, config: LatticeConfig, all_coords: List[Coordinates3D]):
        """
        Инициализация системы соседства.

        Args:
            config: Конфигурация решетки.
            all_coords: Полный список всех координат в решетке.
        """
        # --- Enhanced Initialization Logging ---
        caller_info = _get_caller_info()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger = logging.getLogger(__name__)
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = {"error": "Failed to serialize config"}

        logger.info(
            f"🚀 INIT NeighborTopology @ {timestamp}\n"
            f"     FROM: {caller_info}\n"
            f"     WITH_CONFIG: {json.dumps(config_dict, indent=2, default=str)}"
        )
        # --- End of Logging ---

        self.config = config
        self.dimensions = config.dimensions
        self.boundary_conditions = config.boundary_conditions
        self.pos_helper = Position3D(self.dimensions)

        strategy_value = getattr(
            config, "neighbor_finding_strategy", NeighborStrategy.HYBRID
        )
        # Преобразуем строку в enum, если необходимо
        if isinstance(strategy_value, str):
            strategy_mapping = {
                "local": NeighborStrategy.LOCAL,
                "random_sample": NeighborStrategy.RANDOM_SAMPLE,
                "hybrid": NeighborStrategy.HYBRID,
                "tiered": NeighborStrategy.TIERED,
            }
            self.strategy = strategy_mapping.get(strategy_value, NeighborStrategy.LOCAL)
        else:
            self.strategy = strategy_value
        self.num_neighbors = config.neighbors
        self.strategy_config = getattr(config, "neighbor_strategy_config", {})

        self._all_indices_set = set(range(self.pos_helper.total_positions))

        self._spatial_grid: Optional[SpatialHashGrid] = None
        if self.strategy == NeighborStrategy.TIERED:
            grid_cell_size = self.strategy_config.get("local_grid_cell_size", 5.0)
            self._spatial_grid = SpatialHashGrid(self.dimensions, grid_cell_size)
            for i, c in enumerate(all_coords):
                self._spatial_grid.insert(c, i)

        self.neighbor_cache: Optional[Dict[int, List[int]]] = (
            {} if config.cache_neighbors else None
        )
        if config.cache_neighbors:
            self._build_neighbor_cache()

        self.device = torch.device(
            "cuda" if config.gpu_enabled and torch.cuda.is_available() else "cpu"
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
        # TODO: Реализовать другие типы граничных условий
        elif self.boundary_conditions in [
            BoundaryCondition.ABSORBING,
            BoundaryCondition.REFLECTING,
        ]:
            if self.pos_helper.is_valid_coordinates(coords):
                return coords
            return None
        return coords

    def _build_neighbor_cache(self):
        """Кэширует списки соседей (в виде линейных индексов) для каждой клетки."""
        if self.neighbor_cache is None:
            return

        logging.info(
            f"Building neighbor cache with strategy '{self.strategy.value}'..."
        )
        for i in range(self.pos_helper.total_positions):
            self.neighbor_cache[i] = self.get_neighbor_indices(i)
        logging.info(f"Neighbor cache built: {len(self.neighbor_cache)} entries")

    def get_all_neighbor_indices_batched(self) -> torch.Tensor:
        """
        Возвращает индексы соседей для ВСЕХ клеток в решетке в виде единого тензора.
        Оптимизировано для пакетной обработки.

        Returns:
            torch.Tensor: Тензор с индексами соседей. Shape: (total_cells, num_neighbors).
                          Если у клетки меньше соседей, чем num_neighbors, остаток
                          добивается индексом самой клетки (self-loop).
        """
        total_cells = self.pos_helper.total_positions
        all_indices = torch.full(
            (total_cells, self.num_neighbors), -1, dtype=torch.long
        )

        for i in range(total_cells):
            neighbors = self.get_neighbor_indices(i)
            # Если соседей больше чем надо, берем случайную выборку
            if len(neighbors) > self.num_neighbors:
                neighbors = list(
                    np.random.choice(neighbors, self.num_neighbors, replace=False)
                )

            num_found = len(neighbors)
            all_indices[i, :num_found] = torch.tensor(neighbors, dtype=torch.long)

            # Если соседей меньше, добиваем индексом самой клетки
            if num_found < self.num_neighbors:
                all_indices[i, num_found:] = i

        return all_indices.to(self.device)

    def validate_topology(self) -> Dict[str, Any]:
        """
        Запускает серию проверок для валидации топологии соседства.
        """
        stats: Dict[str, Any] = {
            "total_cells": self.pos_helper.total_positions,
            "boundary_conditions": self.boundary_conditions.value,
            "neighbor_counts": {},
            "symmetric": True,
            "self_loops": 0,
        }

        neighbor_counts = []
        for i in range(self.pos_helper.total_positions):
            neighbors = self.get_neighbor_indices(i)
            neighbor_counts.append(len(neighbors))
            if i in neighbors:
                stats["self_loops"] += 1

        if neighbor_counts:
            stats["neighbor_counts"] = {
                "min": int(np.min(neighbor_counts)),
                "max": int(np.max(neighbor_counts)),
                "mean": float(np.mean(neighbor_counts)),
                "std": float(np.std(neighbor_counts)),
            }
        else:
            stats["neighbor_counts"] = {"min": 0, "max": 0, "mean": 0.0, "std": 0.0}

        # Проверка симметрии (дорогая операция)
        if self.config.validate_connections:
            is_symmetric = True
            for i in range(self.pos_helper.total_positions):
                neighbors = self.get_neighbor_indices(i)
                for neighbor_idx in neighbors:
                    # Для случайных стратегий симметрия не гарантируется
                    # Проверяем только если обратная связь тоже существует
                    if i not in self.get_neighbor_indices(neighbor_idx):
                        is_symmetric = False
                        break
                if not is_symmetric:
                    break
            stats["symmetric"] = is_symmetric
        else:
            stats["symmetric"] = "not_checked"

        return stats
