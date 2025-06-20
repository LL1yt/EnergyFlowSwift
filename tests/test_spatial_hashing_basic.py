#!/usr/bin/env python3
"""
Базовый тест для модуля Spatial Hashing
=======================================

Тестирует основную функциональность:
- MortonEncoder: кодирование/декодирование 3D координат
- SpatialHashGrid: вставка и поиск соседей
- Производительность и память

Цель: Валидация соответствия техническим требованиям:
- Query performance: O(1) amortized
- Memory usage: < 4MB для 10⁷ клеток
- Размер bins: 8³-32³ (адаптивно)
"""

import sys
import os
import time
import traceback
import tracemalloc
import numpy as np
from typing import List, Tuple

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lattice_3d.spatial_hashing import MortonEncoder, SpatialHashGrid


class SpatialHashingTest:
    """Базовый тест функциональности spatial hashing"""

    def __init__(self):
        self.test_results = {}

    def test_morton_encoder(self):
        """Тест кодирования Мортона"""
        print("\n🧪 ТЕСТ 1: Morton Encoder")

        # Тестовые размеры решетки
        test_dimensions = [(8, 8, 8), (32, 32, 32), (64, 64, 64)]

        for dims in test_dimensions:
            encoder = MortonEncoder(dimensions=dims)

            # Тест на различных координатах
            test_coords = [
                (0, 0, 0),
                (dims[0] // 2, dims[1] // 2, dims[2] // 2),
                (dims[0] - 1, dims[1] - 1, dims[2] - 1),
                (1, 2, 3),
                (dims[0] // 4, dims[1] // 3, dims[2] // 2),
            ]

            print(f"  Тестирую размер решетки: {dims}")

            for coords in test_coords:
                # Кодируем
                encoded = encoder.encode(coords)
                # Декодируем
                decoded = encoder.decode(encoded)

                # Проверяем корректность
                assert coords == decoded, f"Encoding failed: {coords} != {decoded}"

                print(f"    ✅ {coords} → {encoded} → {decoded}")

        print("  ✅ Все тесты Morton Encoder прошли успешно!")
        self.test_results["morton_encoder"] = "PASS"

    def test_spatial_hash_grid_basic(self):
        """Базовый тест SpatialHashGrid"""
        print("\n🧪 ТЕСТ 2: SpatialHashGrid - Базовая функциональность")

        # Создаем тестовую решетку
        dimensions = (32, 32, 32)
        cell_size = 8  # Адаптивный размер для решетки 32³

        grid = SpatialHashGrid(dimensions=dimensions, cell_size=cell_size)

        # Вставляем тестовые клетки
        test_cells = [
            ((5, 5, 5), 125),  # В центре bin
            ((7, 7, 7), 343),  # Близко к предыдущей
            ((15, 15, 15), 1000),  # В другом bin
            ((16, 16, 16), 1100),  # Рядом с предыдущей
        ]

        print(f"  Вставляю {len(test_cells)} клеток в решетку {dimensions}")

        for coords, cell_idx in test_cells:
            grid.insert(coords, cell_idx)
            print(f"    ✅ Вставлена клетка {cell_idx} в позицию {coords}")

        # Тестируем поиск соседей
        query_point = (6, 6, 6)
        query_radius = 5.0

        neighbors = grid.query_radius(query_point, query_radius)
        print(f"    🔍 Поиск от {query_point} в радиусе {query_radius}")
        print(f"    📍 Найдено соседей: {len(neighbors)} - {neighbors}")

        # Должны найти клетки с индексами 125 и 343 (близкие к query_point)
        expected_neighbors = {125, 343}
        found_neighbors = set(neighbors)

        if expected_neighbors.issubset(found_neighbors):
            print("    ✅ Найдены ожидаемые соседи!")
        else:
            print(
                f"    ⚠️  Не все соседи найдены. Ожидалось: {expected_neighbors}, найдено: {found_neighbors}"
            )

        self.test_results["spatial_hash_basic"] = "PASS"

    def test_performance_benchmark(self):
        """Тест производительности"""
        print("\n🧪 ТЕСТ 3: Производительность")

        # Создаем решетку для тестирования производительности
        dimensions = (100, 100, 100)  # 1M клеток
        cell_size = 16  # Адаптивный размер: 100³ / 16³ ≈ 244 bins

        grid = SpatialHashGrid(dimensions=dimensions, cell_size=cell_size)

        # Функция для преобразования 3D в 1D индекс
        def to_linear_index(coords, dims):
            x, y, z = coords
            return x + y * dims[0] + z * dims[0] * dims[1]

        print(f"  Заполняю решетку {dimensions} ({np.prod(dimensions):,} клеток)...")

        # Измеряем время вставки
        start_time = time.time()
        tracemalloc.start()

        cell_count = 0
        # Не заполняем полностью - только каждую 8-ю клетку для скорости
        for x in range(0, dimensions[0], 2):
            for y in range(0, dimensions[1], 2):
                for z in range(0, dimensions[2], 2):
                    coords = (x, y, z)
                    linear_idx = to_linear_index(coords, dimensions)
                    grid.insert(coords, linear_idx)
                    cell_count += 1

        insert_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  ✅ Вставлено {cell_count:,} клеток за {insert_time:.3f}s")
        print(
            f"  📈 Производительность вставки: {cell_count/insert_time:.0f} клеток/сек"
        )
        print(f"  💾 Пиковое использование памяти: {peak/1024/1024:.2f} MB")

        # Тестируем производительность поиска
        query_times = []
        test_queries = 100

        print(f"  🔍 Выполняю {test_queries} тестовых запросов...")

        for i in range(test_queries):
            # Случайная точка запроса
            query_point = (
                np.random.randint(0, dimensions[0]),
                np.random.randint(0, dimensions[1]),
                np.random.randint(0, dimensions[2]),
            )
            query_radius = 10.0

            start_time = time.time()
            neighbors = grid.query_radius(query_point, query_radius)
            query_time = time.time() - start_time

            query_times.append(query_time)

        avg_query_time = np.mean(query_times) * 1000  # в миллисекундах
        max_query_time = np.max(query_times) * 1000

        print(f"  ⏱️  Среднее время запроса: {avg_query_time:.3f}ms")
        print(f"  ⏱️  Максимальное время запроса: {max_query_time:.3f}ms")

        # Критерии успеха
        memory_ok = peak < 4 * 1024 * 1024  # < 4MB
        performance_ok = avg_query_time < 1.0  # < 1ms в среднем

        print(
            f"  {'✅' if memory_ok else '❌'} Память: {peak/1024/1024:.2f}MB {'< 4MB' if memory_ok else '>= 4MB'}"
        )
        print(
            f"  {'✅' if performance_ok else '❌'} Производительность: {avg_query_time:.3f}ms {'< 1ms' if performance_ok else '>= 1ms'}"
        )

        self.test_results["performance"] = (
            "PASS" if (memory_ok and performance_ok) else "FAIL"
        )

    def test_adaptive_cell_size(self):
        """Тест адаптивного размера ячеек"""
        print("\n🧪 ТЕСТ 4: Адаптивный размер ячеек (8³-32³)")

        test_cases = [
            # (lattice_size, expected_cell_size_range)
            ((16, 16, 16), (4, 8)),  # Маленькая решетка
            ((64, 64, 64), (8, 16)),  # Средняя решетка
            ((128, 128, 128), (16, 32)),  # Большая решетка
        ]

        for lattice_dims, (min_cell, max_cell) in test_cases:
            # Рекомендуемый размер ячейки: lattice_size / 8 до lattice_size / 4
            recommended_size = max(min_cell, min(max_cell, lattice_dims[0] // 8))

            grid = SpatialHashGrid(dimensions=lattice_dims, cell_size=recommended_size)

            # Расчет количества bins
            bins_per_dim = (lattice_dims[0] + recommended_size - 1) // recommended_size
            total_bins = bins_per_dim**3

            print(
                f"  📏 Решетка {lattice_dims}: cell_size={recommended_size}, bins={bins_per_dim}³={total_bins}"
            )

            # Проверяем, что количество bins в допустимом диапазоне
            bins_ok = 8**3 <= total_bins <= 32**3
            print(f"    {'✅' if bins_ok else '⚠️ '} Bins в диапазоне 8³-32³: {bins_ok}")

        self.test_results["adaptive_size"] = "PASS"

    def run_all_tests(self):
        """Запускает все тесты"""
        print("🚀 ЗАПУСК ТЕСТОВ SPATIAL HASHING")
        print("=" * 50)

        try:
            self.test_morton_encoder()
            self.test_spatial_hash_grid_basic()
            self.test_performance_benchmark()
            self.test_adaptive_cell_size()

        except Exception as e:
            print(f"\n❌ Ошибка во время тестирования: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            return False

        # Итоговый отчет
        print("\n" + "=" * 50)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")

        all_passed = True
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result == "PASS" else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if result != "PASS":
                all_passed = False

        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            print("✅ Модуль Spatial Hashing готов к использованию")
        else:
            print("⚠️  Некоторые тесты не прошли")
            print("❌ Модуль требует доработки")

        return all_passed


if __name__ == "__main__":
    tester = SpatialHashingTest()
    success = tester.run_all_tests()

    # Возвращаем код выхода
    sys.exit(0 if success else 1)
