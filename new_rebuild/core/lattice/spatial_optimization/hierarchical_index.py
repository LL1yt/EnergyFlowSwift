#!/usr/bin/env python3
"""
Многоуровневое пространственное индексирование
============================================

HierarchicalSpatialIndex для эффективного поиска соседей
в очень больших пространствах (1M+ клеток).
"""

from typing import List, Set
from ..spatial_hashing import SpatialHashGrid, Coordinates3D
from ....config.project_config import get_project_config
from ....utils.logging import get_logger

logger = get_logger(__name__)


class HierarchicalSpatialIndex:
    """
    Многоуровневое пространственное индексирование

    Создает иерархию spatial hash grid'ов для эффективного поиска
    в очень больших пространствах (1M+ клеток).
    """

    def __init__(self, dimensions: Coordinates3D, config: dict = None):
        self.dimensions = dimensions
        self.config = config or get_project_config().get_spatial_optim_config()

        # Создаем иерархию spatial grid'ов
        self.levels: List[SpatialHashGrid] = []
        self._build_hierarchy()

        logger.info(f"🏗️ HierarchicalSpatialIndex созданы {len(self.levels)} уровней")

    def _build_hierarchy(self):
        """Строит иерархию spatial grid'ов"""
        max_dim = max(self.dimensions)

        for level in range(self.config.spatial_levels):
            # Размер ячейки увеличивается с каждым уровнем
            cell_size = max(1, max_dim // (4 ** (level + 1)))

            grid = SpatialHashGrid(self.dimensions, cell_size)
            self.levels.append(grid)

            logger.debug(f"   Level {level}: cell_size={cell_size}")

    def insert_batch(self, coords_list: List[Coordinates3D], indices_list: List[int]):
        """Вставляет batch координат во все уровни иерархии"""
        for coords, idx in zip(coords_list, indices_list):
            for grid in self.levels:
                grid.insert(coords, idx)

    def query_hierarchical(self, coords: Coordinates3D, radius: float) -> Set[int]:
        """
        Иерархический поиск соседей

        Начинает с крупных ячеек, затем уточняет в мелких
        """
        # Начинаем с самого крупного уровня
        candidates = set()

        for level_idx, grid in enumerate(self.levels):
            level_radius = radius * (
                2**level_idx
            )  # Увеличиваем радиус для крупных уровней
            level_candidates = set(grid.query_radius(coords, level_radius))

            if level_idx == 0:
                candidates = level_candidates
            else:
                # Пересечение с предыдущим уровнем для уточнения
                candidates = candidates.intersection(level_candidates)

            # Если кандидатов мало, можно остановиться раньше
            if len(candidates) < self.config.min_cells_per_node:
                break

        return candidates
