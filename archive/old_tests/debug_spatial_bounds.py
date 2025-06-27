#!/usr/bin/env python3
"""
Отладочный скрипт для проверки bounds checking
===========================================

Проверяет правильность работы find_neighbors_by_radius_safe
и выявляет источник проблемы с индексами.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "new_rebuild"))

from new_rebuild.core.lattice.spatial_optimization.moe_spatial_optimizer import (
    MoESpatialOptimizer,
)
from new_rebuild.core.lattice.position import Position3D
from new_rebuild.config import get_project_config


def debug_neighbor_search():
    """Отладка поиска соседей"""

    print("🔧 Отладка поиска соседей")

    # Создаем маленькую решетку для отладки
    dimensions = (5, 5, 5)
    total_cells = 5 * 5 * 5  # 125 клеток

    print(f"   📐 Размеры решетки: {dimensions}")
    print(f"   📊 Всего клеток: {total_cells}")
    print(f"   📍 Валидные индексы: 0-{total_cells-1}")

    # Проверяем adaptive radius
    config = get_project_config()
    adaptive_radius = config.calculate_adaptive_radius()
    print(f"   📏 Adaptive radius: {adaptive_radius}")

    # Создаем spatial optimizer
    moe_optimizer = MoESpatialOptimizer(dimensions, device="cpu")
    pos_helper = Position3D(dimensions)

    # Тестируем несколько клеток
    test_cells = [0, 62, 124]  # начало, центр, конец

    for cell_idx in test_cells:
        print(f"\n🧪 Тестируем клетку {cell_idx}")

        # Получаем координаты
        coords = pos_helper.to_3d_coordinates(cell_idx)
        print(f"   📍 Координаты: {coords}")

        # Вычисляем bounds как в find_neighbors_by_radius_safe
        search_radius = adaptive_radius
        print(f"   📏 Search radius: {search_radius}")

        x_min = max(0, coords[0] - int(search_radius))
        x_max = min(dimensions[0], coords[0] + int(search_radius) + 1)
        y_min = max(0, coords[1] - int(search_radius))
        y_max = min(dimensions[1], coords[1] + int(search_radius) + 1)
        z_min = max(0, coords[2] - int(search_radius))
        z_max = min(dimensions[2], coords[2] + int(search_radius) + 1)

        print(
            f"   📦 Bounds: x=[{x_min}, {x_max}), y=[{y_min}, {y_max}), z=[{z_min}, {z_max})"
        )
        print(
            f"   🔢 Размеры итерации: x={x_max-x_min}, y={y_max-y_min}, z={z_max-z_min}"
        )
        print(f"   🔄 Всего итераций: {(x_max-x_min) * (y_max-y_min) * (z_max-z_min)}")

        # Ищем соседей
        neighbors = moe_optimizer.find_neighbors_by_radius_safe(cell_idx)
        print(f"   👥 Найдено соседей: {len(neighbors)}")

        # Проверяем валидность всех найденных соседей
        invalid_neighbors = [n for n in neighbors if not (0 <= n < total_cells)]
        if invalid_neighbors:
            print(f"   ❌ НЕВАЛИДНЫЕ СОСЕДИ: {invalid_neighbors}")
        else:
            print(f"   ✅ Все соседи валидны")

        # Показываем первые несколько соседей
        print(f"   📋 Первые 5 соседей: {neighbors[:5]}")


def debug_position_helper():
    """Отладка Position3D helper"""

    print("\n🔧 Отладка Position3D helper")

    dimensions = (5, 5, 5)
    pos_helper = Position3D(dimensions)
    total_cells = 5 * 5 * 5

    # Тестируем граничные случаи
    test_coords = [
        (0, 0, 0),  # начало
        (2, 2, 2),  # центр
        (4, 4, 4),  # конец
        (5, 5, 5),  # за границей - ДОЛЖЕН БЫТЬ НЕВАЛИДЕН
        (-1, 0, 0),  # отрицательный - ДОЛЖЕН БЫТЬ НЕВАЛИДЕН
    ]

    for coords in test_coords:
        is_valid = pos_helper.is_valid_coordinates(coords)
        print(f"   📍 Координаты {coords}: валидны={is_valid}")

        if is_valid:
            linear_idx = pos_helper.to_linear_index(coords)
            print(f"      📊 Линейный индекс: {linear_idx}")

            # Проверяем что индекс в пределах
            if 0 <= linear_idx < total_cells:
                print(f"      ✅ Индекс валиден")
            else:
                print(f"      ❌ Индекс {linear_idx} вне пределов [0, {total_cells-1}]")


if __name__ == "__main__":
    try:
        debug_position_helper()
        debug_neighbor_search()
        print("🎉 Отладка завершена!")
    except Exception as e:
        print(f"❌ Ошибка в отладке: {e}")
        import traceback

        traceback.print_exc()
        raise
