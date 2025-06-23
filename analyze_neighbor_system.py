"""
Анализ системы соседей в 3D Cellular Neural Network
====================================================

Исследуем как определяются и вычисляются соседи в проекте:
1. Координатная система vs числовая
2. Различные стратегии поиска соседей
3. Влияние размера решетки на количество соседей
4. Spatial hashing vs локальные соседи
"""

import sys

sys.path.append(".")

from new_rebuild.config import get_project_config
from new_rebuild.core.lattice.topology import NeighborTopology
from new_rebuild.core.lattice.position import Position3D
from new_rebuild.core.lattice.enums import NeighborStrategy
import numpy as np


def analyze_neighbor_strategies():
    """Анализируем различные стратегии поиска соседей"""

    print("=== АНАЛИЗ СТРАТЕГИЙ ПОИСКА СОСЕДЕЙ ===\n")

    config = get_project_config()
    print(f"Архитектура: {config.architecture_type}")
    print(f"Размеры решетки: {config.lattice_dimensions}")
    print(f"Effective neighbors: {config.effective_neighbors}")
    print(f"Total cells: {config.total_cells}")
    print()

    # Создаем небольшую решетку для анализа
    pos_helper = Position3D(config.lattice_dimensions)
    all_coords = pos_helper.get_all_coordinates()

    # Создаем топологию соседей
    topology = NeighborTopology(all_coords)

    print(f"Используемая стратегия: {topology.strategy}")
    print(f"Количество соседей: {topology.num_neighbors}")
    print()

    return topology, pos_helper


def analyze_local_neighbors(topology, pos_helper):
    """Анализ локальных соседей (Von Neumann vs Moore)"""

    print("=== 1. ЛОКАЛЬНЫЕ СОСЕДИ (LOCAL STRATEGY) ===\n")

    # Посмотрим на направления локальных соседей
    print("Направления локальных соседей (Von Neumann):")
    for i, direction in enumerate(topology._LOCAL_NEIGHBOR_DIRECTIONS):
        print(f"  {i}: {direction}")
    print()

    # Проверим центральную клетку
    center_coords = (3, 3, 3) if pos_helper.dimensions == (6, 6, 6) else (13, 13, 13)
    center_linear = pos_helper.to_linear_index(center_coords)

    # Получим только локальных соседей
    local_neighbors = topology._get_local_neighbor_indices(center_linear)

    print(f"Центральная клетка: coords={center_coords}, linear_index={center_linear}")
    print(f"Локальных соседей найдено: {len(local_neighbors)}")
    print("Локальные соседи:")

    for i, neighbor_idx in enumerate(local_neighbors):
        neighbor_coords = pos_helper.to_3d_coordinates(neighbor_idx)
        distance = pos_helper.euclidean_distance(center_coords, neighbor_coords)
        direction = (
            neighbor_coords[0] - center_coords[0],
            neighbor_coords[1] - center_coords[1],
            neighbor_coords[2] - center_coords[2],
        )
        print(
            f"  {i}: idx={neighbor_idx}, coords={neighbor_coords}, direction={direction}, distance={distance:.1f}"
        )
    print()


def analyze_current_strategy(topology, pos_helper):
    """Анализ текущей стратегии (HYBRID/TIERED)"""

    print(f"=== 2. ТЕКУЩАЯ СТРАТЕГИЯ: {topology.strategy} ===\n")

    # Проверим ту же центральную клетку
    center_coords = (3, 3, 3) if pos_helper.dimensions == (6, 6, 6) else (13, 13, 13)
    center_linear = pos_helper.to_linear_index(center_coords)

    all_neighbors = topology.get_neighbor_indices(center_linear)

    print(f"Центральная клетка: coords={center_coords}, linear_index={center_linear}")
    print(f"Всего соседей: {len(all_neighbors)} (цель: {topology.num_neighbors})")
    print()

    # Анализ расстояний
    distances = []
    for neighbor_idx in all_neighbors:
        neighbor_coords = pos_helper.to_3d_coordinates(neighbor_idx)
        distance = pos_helper.euclidean_distance(center_coords, neighbor_coords)
        distances.append(distance)

    distances = np.array(distances)
    print(f"Статистика расстояний:")
    print(f"  Минимальное: {distances.min():.2f}")
    print(f"  Максимальное: {distances.max():.2f}")
    print(f"  Среднее: {distances.mean():.2f}")
    print(f"  Медиана: {np.median(distances):.2f}")
    print()

    # Показать первых 10 соседей
    print("Первые 10 соседей:")
    for i, neighbor_idx in enumerate(all_neighbors[:10]):
        neighbor_coords = pos_helper.to_3d_coordinates(neighbor_idx)
        distance = pos_helper.euclidean_distance(center_coords, neighbor_coords)
        print(
            f"  {i}: idx={neighbor_idx}, coords={neighbor_coords}, distance={distance:.2f}"
        )
    print()


def analyze_coordinate_vs_numerical():
    """Анализ: используются координаты или просто числа?"""

    print("=== 3. КООРДИНАТЫ VS ЧИСЛА ===\n")

    print("ОТВЕТ: Система использует КООРДИНАТЫ, но оптимизированно:")
    print()
    print("1. 🎯 КООРДИНАТНАЯ ОСНОВА:")
    print("   - Все расчеты базируются на 3D координатах (x, y, z)")
    print("   - Position3D класс управляет преобразованием")
    print("   - Расстояния вычисляются в 3D пространстве")
    print()

    print("2. 🚀 ОПТИМИЗАЦИЯ ЧЕРЕЗ ЛИНЕЙНЫЕ ИНДЕКСЫ:")
    print("   - 3D координаты → линейный индекс для производительности")
    print("   - Формула: z + y*depth + x*depth*height")
    print("   - Все вычисления в векторизованном виде на GPU")
    print()

    print("3. 🔍 SPATIAL HASHING (для больших решеток):")
    print("   - Разбивает 3D пространство на ячейки")
    print("   - Быстрый поиск соседей в радиусе O(1)")
    print("   - Morton encoding для улучшения пространственной локальности")
    print()

    # Демонстрация преобразований
    config = get_project_config()
    pos_helper = Position3D(config.lattice_dimensions)

    test_coords = [(1, 2, 3), (0, 0, 0), (5, 5, 5)]
    print("4. 📊 ДЕМОНСТРАЦИЯ ПРЕОБРАЗОВАНИЙ:")
    for coords in test_coords:
        if pos_helper.is_valid_coordinates(coords):
            linear = pos_helper.to_linear_index(coords)
            back_coords = pos_helper.to_3d_coordinates(linear)
            print(f"   3D: {coords} → Linear: {linear} → 3D: {back_coords}")
    print()


def analyze_effective_neighbors():
    """Анализ effective_neighbors - как определяется количество соседей"""

    print("=== 4. EFFECTIVE NEIGHBORS - ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ===\n")

    config = get_project_config()

    print("ЛОГИКА ВЫЧИСЛЕНИЯ:")
    print(f"  Текущие размеры: {config.lattice_dimensions}")
    print(f"  Total cells: {config.total_cells}")
    print(f"  Effective neighbors: {config.effective_neighbors}")
    print()

    # Показать логику из ProjectConfig
    print("АЛГОРИТМ (из ProjectConfig.effective_neighbors):")
    print("  if total_cells <= 216:    # 6x6x6")
    print("      return 26             # Базовое соседство")
    print("  elif total_cells <= 1000: # небольшие решетки")
    print("      return min(100, total_cells // 10)")
    print("  elif total_cells <= 19683: # 27x27x27")
    print("      return 2000           # Средние решетки")
    print("  elif total_cells <= 125000: # 50x50x50")
    print("      return 5000           # Большие решетки")
    print("  else:")
    print("      return min(max_neighbors, 19683) # Крупные решетки")
    print()

    # Проверим разные размеры
    test_sizes = [
        ((6, 6, 6), 216),
        ((10, 10, 10), 1000),
        ((27, 27, 27), 19683),
        ((50, 50, 50), 125000),
        ((100, 100, 100), 1000000),
    ]

    print("ПРИМЕРЫ ДЛЯ РАЗНЫХ РАЗМЕРОВ:")
    for dims, total in test_sizes:
        if total <= 216:
            neighbors = 26
        elif total <= 1000:
            neighbors = min(100, total // 10)
        elif total <= 19683:
            neighbors = 2000
        elif total <= 125000:
            neighbors = 5000
        else:
            neighbors = min(10000, 19683)

        percentage = (neighbors / total) * 100
        print(f"  {dims}: {total} клеток → {neighbors} соседей ({percentage:.1f}%)")
    print()


def main():
    """Основная функция анализа"""

    print("🔍 АНАЛИЗ СИСТЕМЫ СОСЕДЕЙ В 3D CELLULAR NEURAL NETWORK")
    print("=" * 60)
    print()

    try:
        # 1. Анализ стратегий
        topology, pos_helper = analyze_neighbor_strategies()

        # 2. Локальные соседи
        analyze_local_neighbors(topology, pos_helper)

        # 3. Текущая стратегия
        analyze_current_strategy(topology, pos_helper)

        # 4. Координаты vs числа
        analyze_coordinate_vs_numerical()

        # 5. Effective neighbors
        analyze_effective_neighbors()

        print("✅ ЗАКЛЮЧЕНИЕ:")
        print("  - Система использует КООРДИНАТЫ (не просто числа)")
        print("  - Von Neumann для локальных (6 соседей)")
        print("  - Spatial hashing для дальних соседей")
        print("  - Динамическое количество соседей по размеру решетки")
        print("  - Эффективные линейные индексы для производительности")
        print()

    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
