#!/usr/bin/env python3
"""
Distance Calculator - вычисление расстояний в 3D решетке
======================================================

Оптимизированные алгоритмы для вычисления различных типов расстояний
между клетками в трехмерной решетке. Поддержка batch operations.
"""

import torch
from typing import Tuple

from ...utils.logging import get_logger

logger = get_logger(__name__)


class DistanceCalculator:
    """Оптимизированный вычислитель расстояний в 3D решетке"""

    def __init__(self, lattice_dimensions: Tuple[int, int, int]):
        self.width, self.height, self.depth = lattice_dimensions
        self.total_cells = self.width * self.height * self.depth

        logger.debug(f"DistanceCalculator initialized: {lattice_dimensions}")

    def linear_to_3d(self, linear_idx: int) -> Tuple[int, int, int]:
        """Преобразование линейного индекса в 3D координаты"""
        z = linear_idx // (self.width * self.height)
        remainder = linear_idx % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width
        return x, y, z

    def batch_linear_to_3d(self, linear_indices: torch.Tensor) -> torch.Tensor:
        """Batch преобразование линейных индексов в 3D координаты

        Args:
            linear_indices: [batch] - линейные индексы

        Returns:
            coords_3d: [batch, 3] - 3D координаты (x, y, z)
        """
        z = linear_indices // (self.width * self.height)
        remainder = linear_indices % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width

        return torch.stack([x, y, z], dim=1).float()

    def euclidean_distance_batch(
        self, idx1: torch.Tensor, idx2: torch.Tensor
    ) -> torch.Tensor:
        """Batch вычисление евклидовых расстояний

        Args:
            idx1: [batch] - индексы первых клеток
            idx2: [batch] - индексы вторых клеток

        Returns:
            distances: [batch] - евклидовы расстояния
        """
        coords1 = self.batch_linear_to_3d(idx1)  # [batch, 3]
        coords2 = self.batch_linear_to_3d(idx2)  # [batch, 3]

        diff = coords1 - coords2  # [batch, 3]
        distances = torch.norm(diff, dim=1)  # [batch]

        return distances

    def manhattan_distance_batch(
        self, idx1: torch.Tensor, idx2: torch.Tensor
    ) -> torch.Tensor:
        """Batch вычисление манхэттенских расстояний"""
        coords1 = self.batch_linear_to_3d(idx1)  # [batch, 3]
        coords2 = self.batch_linear_to_3d(idx2)  # [batch, 3]

        diff = torch.abs(coords1 - coords2)  # [batch, 3]
        distances = torch.sum(diff, dim=1)  # [batch]

        return distances

    def euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Единичное вычисление евклидова расстояния (backward compatibility)"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

    def manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Единичное вычисление манхэттенского расстояния (backward compatibility)"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)
        return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
