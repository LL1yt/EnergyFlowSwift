"""
Модуль для работы с 3D позициями и координатами
=================================================

Содержит вспомогательный класс Position3D для преобразования
между 3D-координатами и линейными индексами, а также для
выполнения базовых операций с координатами.
"""

from typing import Tuple, List
import numpy as np

# Типы для координат и размеров
Coordinates3D = Tuple[int, int, int]
Dimensions3D = Tuple[int, int, int]


class Position3D:
    """
    Вспомогательный класс для управления 3D-координатами и индексами.

    Предоставляет методы для преобразования между 3D-координатами
    и линейными (1D) индексами, а также для валидации и вычисления
    расстояний.
    """

    def __init__(self, dimensions: Dimensions3D):
        """
        Инициализация.

        Args:
            dimensions: Размеры решетки (X, Y, Z).
        """
        self.dimensions = dimensions
        self.width, self.height, self.depth = dimensions
        self.total_positions = self.width * self.height * self.depth

    def to_linear_index(self, coords: Coordinates3D) -> int:
        """
        Преобразует 3D-координаты в линейный (1D) индекс.
        Используется row-major порядок (Z-индекс меняется быстрее всего).
        """
        x, y, z = coords
        self._validate_coordinates(coords)
        return z + y * self.depth + x * self.depth * self.height

    def to_3d_coordinates(self, linear_index: int) -> Coordinates3D:
        """Преобразует линейный (1D) индекс в 3D-координаты."""
        if not (0 <= linear_index < self.total_positions):
            raise IndexError(f"Linear index {linear_index} is out of bounds.")

        x = linear_index // (self.height * self.depth)
        remainder = linear_index % (self.height * self.depth)
        y = remainder // self.depth
        z = remainder % self.depth
        return int(x), int(y), int(z)

    def _validate_coordinates(self, coords: Coordinates3D) -> None:
        """Внутренняя проверка корректности координат (вызывает исключение)."""
        x, y, z = coords
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise IndexError(
                f"Coordinates {coords} are out of bounds for dimensions {self.dimensions}"
            )

    def is_valid_coordinates(self, coords: Coordinates3D) -> bool:
        """Проверяет, находятся ли координаты в пределах решетки."""
        x, y, z = coords
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def get_all_coordinates(self) -> List[Coordinates3D]:
        """Возвращает список всех 3D-координат в решетке."""
        return [self.to_3d_coordinates(i) for i in range(self.total_positions)]

    def manhattan_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> int:
        """Вычисляет манхэттенское расстояние между двумя точками."""
        return (
            abs(coord1[0] - coord2[0])
            + abs(coord1[1] - coord2[1])
            + abs(coord1[2] - coord2[2])
        )

    def euclidean_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> float:
        """Вычисляет евклидово расстояние между двумя точками."""
        return float(
            np.sqrt(
                (coord1[0] - coord2[0]) ** 2
                + (coord1[1] - coord2[1]) ** 2
                + (coord1[2] - coord2[2]) ** 2
            )
        )
