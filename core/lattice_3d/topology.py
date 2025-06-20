"""
Модуль Топологии Соседства
===========================

Содержит класс NeighborTopology, отвечающий за определение и поиск
соседей для каждой клетки в 3D-решетке. Поддерживает различные
стратегии поиска, включая локальные, случайные и многоуровневые
гибридные подходы.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import logging

from .config import LatticeConfig
from .enums import BoundaryCondition
from .position import Position3D, Coordinates3D
from .spatial_hashing import SpatialHashGrid


class NeighborTopology:
    """
    Система управления соседством клеток в 3D решетке.

    Реализует различные типы граничных условий и предоставляет
    эффективные методы для получения соседей каждой клетки.

    Поддерживает разные стратегии поиска соседей:
    - local: стандартные 6 соседей (фон Нейман)
    - random_sample: случайная выборка N соседей со всей решетки
    - hybrid: комбинация локальных и случайных соседей
    - tiered: трехуровневая стратегия (локальные+функциональные+дальние)
    """

    # Направления к 6 локальным соседям в 3D пространстве
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
        self.config = config
        self.dimensions = config.dimensions
        self.boundary_conditions = config.boundary_conditions
        self.pos_helper = Position3D(self.dimensions)

        self.strategy = getattr(config, "neighbor_finding_strategy", "local")
        self.num_neighbors = config.neighbors
        self.strategy_config = getattr(config, "neighbor_strategy_config", {})

        self._all_coords_list = all_coords
        self._all_coords_set = set(all_coords)

        # Инициализация SpatialHashGrid для стратегий, которые его используют
        self._spatial_grid: Optional[SpatialHashGrid] = None
        if self.strategy == "tiered":
            grid_cell_size = self.strategy_config.get("local_grid_cell_size", 5)
            self._spatial_grid = SpatialHashGrid(self.dimensions, grid_cell_size)
            # Заполняем решетку всеми клетками
            for i, c in enumerate(all_coords):
                self._spatial_grid.insert(c, i)

        self.neighbor_cache: Optional[Dict[int, List[int]]] = (
            {} if config.cache_neighbors else None
        )
        if config.cache_neighbors:
            self._build_neighbor_cache()

    def get_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """
        Получает список координат соседей для заданной клетки в зависимости от стратегии.
        Этот метод не использует кэш и вычисляет соседей "на лету".
        """
        if self.strategy == "local":
            return self._get_local_neighbors(coords)
        elif self.strategy == "random_sample":
            return self._get_random_sample_neighbors(coords)
        elif self.strategy == "hybrid":
            return self._get_hybrid_neighbors(coords)
        elif self.strategy == "tiered":
            return self._get_tiered_neighbors(coords)
        else:
            raise ValueError(f"Unknown neighbor finding strategy: {self.strategy}")

    def _get_local_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """Возвращает до 6 локальных соседей."""
        neighbors = []
        for direction in self._LOCAL_NEIGHBOR_DIRECTIONS:
            neighbor_coords = (
                coords[0] + direction[0],
                coords[1] + direction[1],
                coords[2] + direction[2],
            )
            valid_coords = self._apply_boundary_conditions(neighbor_coords)
            if valid_coords:
                neighbors.append(valid_coords)
        return neighbors

    def _get_random_sample_neighbors(
        self, coords: Coordinates3D
    ) -> List[Coordinates3D]:
        """Возвращает случайную выборку N соседей со всей решетки."""
        possible_neighbors = list(self._all_coords_set - {coords})
        num_to_sample = min(self.num_neighbors, len(possible_neighbors))
        if num_to_sample == 0:
            return []

        indices = np.random.choice(
            len(possible_neighbors), num_to_sample, replace=False
        )
        return [possible_neighbors[i] for i in indices]

    def _get_hybrid_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """Комбинирует локальных и случайных соседей."""
        local_count = self.strategy_config.get("local_count", 6)
        random_count = max(0, self.num_neighbors - local_count)

        local_neighbors = self._get_local_neighbors(coords)
        if len(local_neighbors) > local_count:
            local_neighbors = local_neighbors[:local_count]

        exclude_set = {coords}.union(local_neighbors)
        possible_random = list(self._all_coords_set - exclude_set)

        num_to_sample = min(random_count, len(possible_random))

        if num_to_sample > 0:
            indices = np.random.choice(
                len(possible_random), num_to_sample, replace=False
            )
            random_neighbors = [possible_random[i] for i in indices]
        else:
            random_neighbors = []

        return local_neighbors + random_neighbors

    def _get_tiered_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """
        Реализует трехуровневую гибридную стратегию.
        """
        if self._spatial_grid is None:
            raise RuntimeError(
                "SpatialHashGrid не инициализирован для 'tiered' стратегии."
            )

        local_config = self.strategy_config.get("local_tier", {})
        local_radius = local_config.get("radius", 5.0)
        local_ratio = local_config.get("ratio", 0.7)
        local_count = int(self.num_neighbors * local_ratio)

        local_indices = self._spatial_grid.query_radius(coords, local_radius)
        self_index = self.pos_helper.to_linear_index(coords)

        local_indices_set = set(local_indices)
        local_indices_set.discard(self_index)

        if len(local_indices_set) > local_count:
            final_local_indices = list(
                np.random.choice(list(local_indices_set), local_count, replace=False)
            )
        else:
            final_local_indices = list(local_indices_set)

        functional_config = self.strategy_config.get("functional_tier", {})
        functional_ratio = functional_config.get("ratio", 0.2)
        functional_count = int(self.num_neighbors * functional_ratio)

        exclude_indices = set(final_local_indices) | {self_index}
        all_indices_set = set(range(self.pos_helper.total_positions))
        possible_functional = list(all_indices_set - exclude_indices)

        num_to_sample_func = min(functional_count, len(possible_functional))
        functional_indices = (
            list(
                np.random.choice(possible_functional, num_to_sample_func, replace=False)
            )
            if num_to_sample_func > 0
            else []
        )

        long_range_config = self.strategy_config.get("long_range_tier", {})
        long_range_count = (
            self.num_neighbors - len(final_local_indices) - len(functional_indices)
        )

        exclude_indices.update(functional_indices)
        possible_long_range = list(all_indices_set - exclude_indices)

        if long_range_count > 0 and possible_long_range:
            distances = np.array(
                [
                    self.pos_helper.euclidean_distance(
                        coords, self.pos_helper.to_3d_coordinates(idx)
                    )
                    for idx in possible_long_range
                ]
            )
            probabilities = 1.0 / (distances + 1e-6)
            probabilities /= np.sum(probabilities)

            num_to_sample_lr = min(long_range_count, len(possible_long_range))
            long_range_indices = list(
                np.random.choice(
                    possible_long_range,
                    num_to_sample_lr,
                    replace=False,
                    p=probabilities,
                )
            )
        else:
            long_range_indices = []

        final_indices = final_local_indices + functional_indices + long_range_indices
        return [self.pos_helper.to_3d_coordinates(idx) for idx in final_indices]

    def get_neighbor_indices(self, linear_index: int) -> List[int]:
        """
        Получает список линейных индексов соседей.
        Использует кэш, если он доступен и был построен.
        """
        if self.neighbor_cache is not None:
            return self.neighbor_cache.get(linear_index, [])

        coords = self.pos_helper.to_3d_coordinates(linear_index)
        neighbor_coords = self.get_neighbors(coords)
        return [self.pos_helper.to_linear_index(nc) for nc in neighbor_coords]

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
            return None  # Заглушка

        return coords

    def _build_neighbor_cache(self):
        """Кэширует списки соседей (в виде линейных индексов) для каждой клетки."""
        if self.neighbor_cache is None:
            return

        logging.info(f"Building neighbor cache with strategy '{self.strategy}'...")
        for i, coords in enumerate(self._all_coords_list):
            neighbors_coords = self.get_neighbors(coords)
            self.neighbor_cache[i] = [
                self.pos_helper.to_linear_index(nc) for nc in neighbors_coords
            ]
        logging.info(f"Neighbor cache built: {len(self.neighbor_cache)} entries")

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

        stats["neighbor_counts"] = {
            "min": int(np.min(neighbor_counts)),
            "max": int(np.max(neighbor_counts)),
            "mean": float(np.mean(neighbor_counts)),
            "std": float(np.std(neighbor_counts)),
        }

        # Проверка симметрии (дорогая операция)
        if self.config.validate_connections:
            for i in range(self.pos_helper.total_positions):
                neighbors = self.get_neighbor_indices(i)
                for neighbor_idx in neighbors:
                    if i not in self.get_neighbor_indices(neighbor_idx):
                        stats["symmetric"] = False
                        break
                if not stats["symmetric"]:
                    break
        else:
            stats["symmetric"] = "not_checked"

        return stats
