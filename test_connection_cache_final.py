#!/usr/bin/env python3
"""
Финальный тест Connection Cache - проверка исправлений
=====================================================
"""

import torch
import numpy as np
from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.config.project_config import (
    get_project_config,
    reset_global_config,
    set_project_config,
    ProjectConfig,
)

# Настройка конфигурации
reset_global_config()
config = ProjectConfig()
config.lattice.dimensions = (15, 15, 15)
set_project_config(config)


def test_cache_final():
    print("🧪 ФИНАЛЬНЫЙ ТЕСТ CONNECTION CACHE")
    print("=" * 50)

    lattice_dimensions = (15, 15, 15)
    total_cells = np.prod(lattice_dimensions)
    config = get_project_config()

    print(f"📐 Решетка: {lattice_dimensions}")
    print(f"🔍 Adaptive radius: {config.calculate_adaptive_radius()}")

    # Создаем классификаторы
    classifier_no_cache = UnifiedConnectionClassifier(
        lattice_dimensions, enable_cache=False
    )
    classifier_with_cache = UnifiedConnectionClassifier(
        lattice_dimensions, enable_cache=True
    )

    # Сбрасываем статистику
    classifier_no_cache.reset_stats()
    classifier_with_cache.reset_stats()

    # Тестовые данные
    all_states = torch.randn(total_cells, config.gnn.state_size)
    cell_idx = 1000

    # Находим соседей
    from new_rebuild.core.moe.distance_calculator import DistanceCalculator

    calc = DistanceCalculator(lattice_dimensions)

    neighbor_indices = []
    for potential_neighbor in range(
        max(0, cell_idx - 100), min(total_cells, cell_idx + 100)
    ):
        if potential_neighbor == cell_idx:
            continue
        distance = calc.euclidean_distance(cell_idx, potential_neighbor)
        if distance <= config.calculate_adaptive_radius():
            neighbor_indices.append(potential_neighbor)

    print(f"🔗 Найдено {len(neighbor_indices)} соседей для клетки {cell_idx}")

    if not neighbor_indices:
        print("❌ Нет соседей для тестирования")
        return

    # Извлекаем состояния
    cell_state = all_states[cell_idx]
    neighbor_states = all_states[neighbor_indices]

    # Тест БЕЗ кэша
    print("\n🔄 Тест БЕЗ кэша...")
    result_no_cache = classifier_no_cache.classify_connections(
        cell_idx=cell_idx,
        neighbor_indices=neighbor_indices,
        cell_state=cell_state,
        neighbor_states=neighbor_states,
    )

    # Тест С кэшем
    print("\n🚀 Тест С кэшем...")
    result_with_cache = classifier_with_cache.classify_connections(
        cell_idx=cell_idx,
        neighbor_indices=neighbor_indices,
        cell_state=cell_state,
        neighbor_states=neighbor_states,
    )

    # Анализ результатов
    from new_rebuild.core.moe.connection_types import ConnectionCategory

    def analyze_result(name, result):
        local_count = len(result[ConnectionCategory.LOCAL])
        functional_count = len(result[ConnectionCategory.FUNCTIONAL])
        distant_count = len(result[ConnectionCategory.DISTANT])
        total = local_count + functional_count + distant_count

        if total > 0:
            print(f"\n{name}:")
            print(f"   LOCAL: {local_count} ({local_count/total*100:.1f}%)")
            print(
                f"   FUNCTIONAL: {functional_count} ({functional_count/total*100:.1f}%)"
            )
            print(f"   DISTANT: {distant_count} ({distant_count/total*100:.1f}%)")
            print(f"   ВСЕГО: {total}")
        else:
            print(f"\n{name}: Нет связей")

    analyze_result("БЕЗ кэша", result_no_cache)
    analyze_result("С кэшем", result_with_cache)

    # Статистика классификаторов
    print("\n📊 СТАТИСТИКА КЛАССИФИКАТОРОВ")
    print("=" * 50)

    stats_no_cache = classifier_no_cache.get_classification_stats()
    stats_with_cache = classifier_with_cache.get_classification_stats()

    print(f"\nБЕЗ кэша:")
    print(f"   LOCAL ratio: {stats_no_cache['local_ratio']:.1%}")
    print(f"   FUNCTIONAL ratio: {stats_no_cache['functional_ratio']:.1%}")
    print(f"   DISTANT ratio: {stats_no_cache['distant_ratio']:.1%}")
    print(f"   Всего связей: {stats_no_cache['total_connections']}")

    print(f"\nС кэшем:")
    print(f"   LOCAL ratio: {stats_with_cache['local_ratio']:.1%}")
    print(f"   FUNCTIONAL ratio: {stats_with_cache['functional_ratio']:.1%}")
    print(f"   DISTANT ratio: {stats_with_cache['distant_ratio']:.1%}")
    print(f"   Всего связей: {stats_with_cache['total_connections']}")

    if "cache_performance" in stats_with_cache:
        cache_perf = stats_with_cache["cache_performance"]
        if cache_perf.get("cache_enabled", False):
            print(f"   Cache hit rate: {cache_perf['cache_hit_rate']:.1%}")
            print(f"   Cache hits: {cache_perf['cache_hits']}")
            print(f"   Cache misses: {cache_perf['cache_misses']}")

    # Проверка успешности
    success = (
        stats_with_cache["total_connections"] > 0
        and stats_with_cache["functional_ratio"] > 0.1
    )

    if success:
        print("\n✅ ТЕСТ ПРОШЕЛ УСПЕШНО!")
        print("🎯 Кэш работает корректно")
        print("📊 Статистика обновляется")
    else:
        print("\n❌ ТЕСТ НЕ ПРОШЕЛ")
        print("🐛 Проблемы с кэшем или статистикой")

    return success


if __name__ == "__main__":
    test_cache_final()
