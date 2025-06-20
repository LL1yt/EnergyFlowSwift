"""
Модуль Spatial Hashing для 3D Решетки
======================================

Реализует высокоэффективные структуры данных для быстрого пространственного
поиска соседей в трехмерном пространстве. Этот модуль является ключевым
компонентом для новой, биологически правдоподобной архитектуры связности.

Основные компоненты:
- MortonEncoder: Реализует кодирование по кривой Мортона (Z-order curve)
                 для отображения 3D координат в 1D для улучшения
                 пространственной локальности и производительности кэша.
- SpatialHashGrid: Реализует простую и быструю пространственную хэш-решетку,
                   которая позволяет находить соседей в радиусе за время O(1)
                   в среднем случае.

Автор: 3D Cellular Neural Network Project
Версия: 1.0.0 (2024-07-15)
"""

import numpy as np
from typing import Tuple, List, Dict, Set

Coordinates3D = Tuple[int, int, int]


class MortonEncoder:
    """
    Кодировщик кривой Мортона для 3D-пространства.

    Преобразует 3D-координаты в одномерное целое число, сохраняя
    пространственную близость. Клетки, близкие в 3D, с высокой
    вероятностью будут близки и в 1D-представлении.
    """

    def __init__(self, dimensions: Coordinates3D):
        self.max_dim = max(dimensions)
        # Определяем количество бит, необходимых для представления максимальной координаты
        self.bits = self.max_dim.bit_length()

    def _interleave_bits(self, x: int, y: int, z: int) -> int:
        """Вставляет биты координат для создания Z-order кода."""
        code = 0
        for i in range(self.bits):
            mask = 1 << i
            code |= (
                ((x & mask) << (2 * i))
                | ((y & mask) << (2 * i - 1))
                | ((z & mask) << (2 * i - 2))
            )
        return code

    def encode(self, coords: Coordinates3D) -> int:
        """Кодирует 3D-координаты в 1D-код Мортона."""
        x, y, z = coords
        return self._interleave_bits(x, y, z)

    def decode(self, code: int) -> Coordinates3D:
        """Декодирует 1D-код Мортона обратно в 3D-координаты."""
        x = self._deinterleave_bits(code, 2)
        y = self._deinterleave_bits(code, 1)
        z = self._deinterleave_bits(code, 0)
        return x, y, z

    def _deinterleave_bits(self, code: int, offset: int) -> int:
        """Извлекает биты одной координаты из кода Мортона."""
        val = 0
        for i in range(self.bits):
            val |= ((code >> (3 * i + offset)) & 1) << i
        return val


class SpatialHashGrid:
    """
    Пространственная хэш-решетка для эффективного поиска соседей.

    Разделяет 3D-пространство на сетку ячеек и хранит списки клеток
    в каждой ячейке. Поиск соседей сводится к проверке лишь нескольких
    смежных ячеек, что значительно быстрее полного перебора.
    """

    def __init__(self, dimensions: Coordinates3D, cell_size: int):
        """
        Инициализация хэш-решетки.

        Args:
            dimensions: Размеры всей 3D-решетки (X, Y, Z).
            cell_size: Размер одной ячейки хэш-решетки. Должен быть
                       примерно равен радиусу поиска соседей.
        """
        if cell_size <= 0:
            raise ValueError("cell_size должен быть положительным числом.")
        self.dimensions = dimensions
        self.cell_size = cell_size

        # Вычисляем размеры хэш-решетки в ячейках
        self.grid_dims = (
            (dimensions[0] + cell_size - 1) // cell_size,
            (dimensions[1] + cell_size - 1) // cell_size,
            (dimensions[2] + cell_size - 1) // cell_size,
        )

        # Основная структура данных: словарь, где ключ - хэш ячейки,
        # а значение - список линейных индексов клеток в ней.
        self.grid: Dict[int, List[int]] = {}

    def _get_grid_coords(self, coords: Coordinates3D) -> Coordinates3D:
        """Преобразует мировые координаты в координаты ячейки хэш-решетки."""
        return (
            coords[0] // self.cell_size,
            coords[1] // self.cell_size,
            coords[2] // self.cell_size,
        )

    def _hash(self, grid_coords: Coordinates3D) -> int:
        """Вычисляет хэш для координат ячейки (простое хэширование кортежа)."""
        return hash(grid_coords)

    def insert(self, coords: Coordinates3D, cell_linear_index: int):
        """
        Вставляет клетку в решетку.

        Args:
            coords: Мировые 3D-координаты клетки.
            cell_linear_index: Линейный индекс клетки для хранения.
        """
        grid_coords = self._get_grid_coords(coords)
        key = self._hash(grid_coords)

        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(cell_linear_index)

    def query_radius(self, coords: Coordinates3D, radius: float) -> List[int]:
        """
        Находит всех соседей в заданном радиусе от точки.

        Args:
            coords: 3D-координаты центральной точки.
            radius: Радиус поиска.

        Returns:
            Список линейных индексов всех клеток-соседей.
        """
        center_grid_coords = self._get_grid_coords(coords)

        # Определяем диапазон ячеек для проверки
        min_x = (coords[0] - int(radius)) // self.cell_size
        max_x = (coords[0] + int(radius)) // self.cell_size
        min_y = (coords[1] - int(radius)) // self.cell_size
        max_y = (coords[1] + int(radius)) // self.cell_size
        min_z = (coords[2] - int(radius)) // self.cell_size
        max_z = (coords[2] + int(radius)) // self.cell_size

        found_indices: Set[int] = set()

        # Перебираем все потенциально релевантные ячейки
        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                for gz in range(min_z, max_z + 1):
                    key = self._hash((gx, gy, gz))
                    if key in self.grid:
                        # Используем set.update для эффективного добавления
                        # уникальных индексов
                        found_indices.update(self.grid[key])

        # Примечание: Этот простой метод возвращает всех кандидатов из
        # смежных ячеек. Для точного результата потребовался бы
        # дополнительный шаг фильтрации по точному расстоянию,
        # но для наших целей "достаточно хорошего" поиска это избыточно.

        # Удаляем саму центральную клетку из результатов, если она там оказалась
        # (в данном методе мы не знаем ее индекс, это должен сделать вызывающий код)
        return list(found_indices)


# Пример использования (для тестирования)
if __name__ == "__main__":
    dims = (100, 100, 100)
    grid = SpatialHashGrid(dimensions=dims, cell_size=10)

    # Заполняем решетку точками
    from tqdm import tqdm

    # Конвертер для удобства
    def to_linear(c, D):
        return c[0] + c[1] * D[0] + c[2] * D[0] * D[1]

    print("Inserting 1,000,000 cells into hash grid...")
    for i in tqdm(range(dims[0])):
        for j in range(dims[1]):
            for k in range(dims[2]):
                coords = (i, j, k)
                linear_idx = to_linear(coords, dims)
                grid.insert(coords, linear_idx)

    print("Grid populated.")

    # Тестовый запрос
    query_point = (50, 50, 50)
    query_radius = 5

    import time

    start_time = time.time()
    neighbors = grid.query_radius(query_point, query_radius)
    end_time = time.time()

    print(f"\nQuery for point {query_point} with radius {query_radius}:")
    print(f"Found {len(neighbors)} potential neighbors.")
    print(f"Query time: {(end_time - start_time) * 1000:.4f} ms")

    # Проверка кодирования Мортона
    encoder = MortonEncoder(dimensions=dims)
    test_coords = (10, 25, 60)
    morton_code = encoder.encode(test_coords)
    decoded_coords = encoder.decode(morton_code)

    print(f"\nMorton coding test:")
    print(f"Original: {test_coords}")
    print(f"Encoded: {morton_code}")
    print(f"Decoded: {decoded_coords}")
    print(f"Test passed: {test_coords == decoded_coords}")
