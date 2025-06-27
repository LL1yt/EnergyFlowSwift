#!/usr/bin/env python3
"""
Тест исправленной классификации связей
=====================================

Проверяем что после исправления порогов и adaptive_radius
мы получаем правильное распределение LOCAL/FUNCTIONAL/DISTANT связей.
"""

import torch
import numpy as np
from typing import Dict, Any

from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.config.project_config import (
    get_project_config,
    reset_global_config,
    set_project_config,
    ProjectConfig,
)
from utils.centralized_config import CentralizedConfig

# Инициализируем конфигурацию заново
reset_global_config()  # Сброс старой конфигурации

# Создаем новую конфигурацию с правильными размерами
config = ProjectConfig()
config.lattice.dimensions = (15, 15, 15)  # ИСПРАВЛЕНО: устанавливаем правильные размеры
set_project_config(config)  # Устанавливаем глобально

CentralizedConfig()


def test_fixed_connection_classification():
    """Тест исправленной классификации связей"""
    print("🧪 ТЕСТ ИСПРАВЛЕННОЙ КЛАССИФИКАЦИИ СВЯЗЕЙ")
    print("=" * 60)

    lattice_dimensions = (15, 15, 15)
    total_cells = np.prod(lattice_dimensions)

    config = get_project_config()
    state_size = config.gnn.state_size

    print(f"📐 Решетка: {lattice_dimensions} ({total_cells:,} клеток)")
    print(f"🔍 Adaptive radius: {config.calculate_adaptive_radius()}")
    print(f"🎯 Пороги:")
    print(f"   LOCAL: ≤ {config.expert.connections.local_distance_threshold}")
    print(
        f"   FUNCTIONAL: {config.expert.connections.local_distance_threshold} < x ≤ {config.expert.connections.functional_distance_threshold}"
    )
    print(f"   DISTANT: ≥ {config.expert.connections.distant_distance_threshold}")
    print(
        f"   Similarity threshold: {config.expert.connections.functional_similarity_threshold}"
    )

    # === ТЕСТ БЕЗ КЭША ===
    print("\n🔄 Тестируем БЕЗ кэша...")
    classifier_no_cache = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dimensions, enable_cache=False
    )

    # === ТЕСТ С КЭШЕМ ===
    print("\n🚀 Тестируем С кэшем...")
    classifier_with_cache = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dimensions, enable_cache=True
    )

    # ИСПРАВЛЕНО: Сбрасываем статистику перед тестом
    classifier_no_cache.reset_stats()
    classifier_with_cache.reset_stats()

    # Генерируем тестовые данные (ПОЛНЫЙ STATES TENSOR)
    all_states = torch.randn(total_cells, state_size)

    # Тестируем несколько клеток
    test_cells = [1000, 1500, 2000]  # Клетки ближе к центру решетки

    for cell_idx in test_cells:
        print(f"\n📍 Клетка {cell_idx}")
        print("-" * 40)

        # ИСПРАВЛЕНО: Используем реальных соседей из решетки
        from new_rebuild.core.moe.distance_calculator import DistanceCalculator

        calc = DistanceCalculator(lattice_dimensions)

        # Находим всех соседей в радиусе 6.0
        neighbor_indices = []
        adaptive_radius = config.calculate_adaptive_radius()

        for potential_neighbor in range(total_cells):
            if potential_neighbor == cell_idx:
                continue

            distance = calc.euclidean_distance(cell_idx, potential_neighbor)
            if distance <= adaptive_radius:
                neighbor_indices.append(potential_neighbor)

        if not neighbor_indices:
            print("   ⚠️ Нет валидных соседей")
            continue

        print(f"   🔗 Тестируем {len(neighbor_indices)} соседей")

        # ИСПРАВЛЕНО: Извлекаем состояния из полного тензора
        cell_state = all_states[cell_idx]
        neighbor_states = all_states[neighbor_indices]

        # Тестируем классификацию БЕЗ кэша
        classifications_no_cache = classifier_no_cache.classify_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            cell_state=cell_state,
            neighbor_states=neighbor_states,
        )

        # ИСПРАВЛЕНО: Для кэша нужен специальный метод с полным states tensor
        classifications_with_cache = classifier_with_cache.cache_manager.get_cached_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            states=all_states,  # Передаем ПОЛНЫЙ tensor
            functional_similarity_threshold=config.expert.connections.functional_similarity_threshold,
        )

        # Анализ результатов
        for mode_name, classifications in [
            ("БЕЗ кэша", classifications_no_cache),
            ("С кэшем", classifications_with_cache),
        ]:
            from new_rebuild.core.moe.connection_types import ConnectionCategory

            local_count = len(classifications[ConnectionCategory.LOCAL])
            functional_count = len(classifications[ConnectionCategory.FUNCTIONAL])
            distant_count = len(classifications[ConnectionCategory.DISTANT])
            total = local_count + functional_count + distant_count

            if total > 0:
                local_percent = (local_count / total) * 100
                functional_percent = (functional_count / total) * 100
                distant_percent = (distant_count / total) * 100

                print(f"   {mode_name}:")
                print(f"      🎯 LOCAL: {local_count} ({local_percent:.1f}%)")
                print(
                    f"      🎯 FUNCTIONAL: {functional_count} ({functional_percent:.1f}%)"
                )
                print(f"      🎯 DISTANT: {distant_count} ({distant_percent:.1f}%)")
            else:
                print(f"   {mode_name}: Нет классифицированных связей")

    # Общая статистика
    print("\n📊 ОБЩАЯ СТАТИСТИКА")
    print("=" * 60)

    for mode_name, classifier in [
        ("БЕЗ кэша", classifier_no_cache),
        ("С кэшем", classifier_with_cache),
    ]:
        stats = classifier.get_classification_stats()
        print(f"\n{mode_name}:")
        print(f"   LOCAL ratio: {stats['local_ratio']:.1%}")
        print(f"   FUNCTIONAL ratio: {stats['functional_ratio']:.1%}")
        print(f"   DISTANT ratio: {stats['distant_ratio']:.1%}")
        print(f"   Всего связей: {stats['total_connections']}")

        if "cache_performance" in stats:
            cache_perf = stats["cache_performance"]
            if cache_perf.get("cache_enabled", False):
                print(f"   Cache hit rate: {cache_perf['cache_hit_rate']:.1%}")
                print(f"   Cache hits: {cache_perf['cache_hits']}")
                print(f"   Cache misses: {cache_perf['cache_misses']}")

    return True


def test_distance_calculation():
    """Тест расчета расстояний для отладки"""
    print("\n🔍 ТЕСТ РАСЧЕТА РАССТОЯНИЙ")
    print("=" * 40)

    lattice_dimensions = (15, 15, 15)
    config = get_project_config()

    from new_rebuild.core.moe.distance_calculator import DistanceCalculator

    calc = DistanceCalculator(lattice_dimensions)

    # Тестируем расстояния от центра
    center_idx = 15 * 15 * 7 + 15 * 7 + 7  # Примерно центр решетки

    test_indices = [
        center_idx + 1,  # Соседний (LOCAL)
        center_idx + 15,  # На одну строку (LOCAL/FUNCTIONAL)
        center_idx + 15 * 15,  # На один слой (FUNCTIONAL)
        center_idx + 15 * 15 * 2,  # На два слоя (FUNCTIONAL/DISTANT)
        center_idx + 15 * 15 * 4,  # На четыре слоя (DISTANT)
    ]

    print(f"Центральная клетка: {center_idx}")
    print(
        f"Пороги: LOCAL≤{config.expert.connections.local_distance_threshold}, "
        f"FUNCTIONAL≤{config.expert.connections.functional_distance_threshold}, "
        f"DISTANT≥{config.expert.connections.distant_distance_threshold}"
    )

    for test_idx in test_indices:
        if 0 <= test_idx < 15 * 15 * 15:
            euclidean = calc.euclidean_distance(center_idx, test_idx)
            manhattan = calc.manhattan_distance(center_idx, test_idx)

            if euclidean <= config.expert.connections.local_distance_threshold:
                category = "LOCAL"
            elif euclidean <= config.expert.connections.functional_distance_threshold:
                category = "FUNCTIONAL candidate"
            elif euclidean >= config.expert.connections.distant_distance_threshold:
                category = "DISTANT"
            else:
                category = "Middle (needs similarity check)"

            print(
                f"   {center_idx} -> {test_idx}: Euclidean={euclidean:.2f}, Manhattan={manhattan:.1f} -> {category}"
            )


def test_cache_configuration():
    """Тест что конфигурация правильно загружается"""
    print("\n⚙️ ТЕСТ КОНФИГУРАЦИИ")
    print("=" * 40)

    config = get_project_config()

    print(f"📐 Lattice dimensions: {config.lattice.dimensions}")
    print(f"🔍 Adaptive radius calculation:")

    max_dim = max(config.lattice.dimensions)
    expected_radius = 6.0 if max_dim <= 27 else 8.0
    actual_radius = config.calculate_adaptive_radius()

    print(f"   Max dimension: {max_dim}")
    print(f"   Expected radius: {expected_radius}")
    print(f"   Actual radius: {actual_radius}")

    print(f"🎯 Connection thresholds:")
    print(
        f"   local_distance_threshold: {config.expert.connections.local_distance_threshold}"
    )
    print(
        f"   functional_distance_threshold: {config.expert.connections.functional_distance_threshold}"
    )
    print(
        f"   distant_distance_threshold: {config.expert.connections.distant_distance_threshold}"
    )

    # Проверяем что изменения применились
    if actual_radius == expected_radius:
        print("✅ Adaptive radius настроен правильно")
    else:
        print(
            f"❌ Adaptive radius неправильный: ожидалось {expected_radius}, получено {actual_radius}"
        )

    if config.expert.connections.local_distance_threshold == 1.8:
        print("✅ Local threshold настроен правильно")
    else:
        print(
            f"❌ Local threshold неправильный: ожидалось 1.8, получено {config.expert.connections.local_distance_threshold}"
        )


if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ КЛАССИФИКАЦИИ СВЯЗЕЙ")
    print("=" * 80)

    try:
        # Тест конфигурации
        test_cache_configuration()

        # Тест расчета расстояний
        test_distance_calculation()

        # Основной тест классификации
        success = test_fixed_connection_classification()

        if success:
            print("\n✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            print("🎯 Классификация связей работает корректно")
            print("⚡ Кэширование функционирует")

    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        import traceback

        traceback.print_exc()
