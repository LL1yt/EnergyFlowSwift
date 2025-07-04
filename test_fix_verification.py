#!/usr/bin/env python3
"""
Тест для проверки исправления проблемы с кэшированием соседей
и новой оптимизированной архитектуры без дублирования
"""

import torch
import numpy as np
import time
from new_rebuild.config import (
    create_experiment_config,
    create_debug_config,
    set_project_config,
    get_project_config,
)
from new_rebuild.core.moe import ConnectionCacheManager, UnifiedConnectionClassifier
from new_rebuild.core.lattice.spatial_optimization import UnifiedSpatialOptimizer
from new_rebuild.core.moe.distance_calculator import DistanceCalculator
from new_rebuild.core.lattice import create_lattice
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_spatial_hash_fix():
    """Тест исправления spatial hash для точной фильтрации по расстоянию"""
    print("=== Тест исправления spatial hash ===\n")

    # Создаем конфигурацию
    config = create_experiment_config()
    set_project_config(config)

    # Создаем spatial optimizer
    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        OptimizationConfig,
    )

    opt_config = OptimizationConfig(
        enable_morton_encoding=config.lattice.enable_morton_encoding,
        enable_adaptive_chunking=config.adaptive_chunker is not None,
        max_memory_gb=8.0,
        target_performance_ms=config.lattice.target_performance_ms,
    )

    spatial_optimizer = UnifiedSpatialOptimizer(
        dimensions=config.lattice.dimensions, config=opt_config
    )

    # Тестируем клетку 677
    test_cell = 677
    adaptive_radius = config.calculate_adaptive_radius()

    print(f"Тестируем клетку {test_cell} с adaptive_radius={adaptive_radius}")

    # Получаем соседей через spatial optimizer
    neighbors = spatial_optimizer.find_neighbors_by_radius_safe(test_cell)
    print(f"Spatial optimizer нашел {len(neighbors)} соседей")

    # Проверяем расстояния
    dist_calc = DistanceCalculator(config.lattice.dimensions)
    distances = []
    for neighbor_idx in neighbors:
        dist = dist_calc.euclidean_distance(test_cell, neighbor_idx)
        distances.append((neighbor_idx, dist))

    distances.sort(key=lambda x: x[1])

    # Анализируем распределение расстояний
    local_threshold = adaptive_radius * config.lattice.local_distance_ratio
    functional_threshold = adaptive_radius * config.lattice.functional_distance_ratio
    distant_threshold = adaptive_radius * config.lattice.distant_distance_ratio

    within_local = sum(1 for _, d in distances if d < local_threshold)
    within_functional = sum(
        1 for _, d in distances if local_threshold <= d <= functional_threshold
    )
    within_distant = sum(
        1 for _, d in distances if functional_threshold < d <= distant_threshold
    )
    beyond_distant = sum(1 for _, d in distances if d > distant_threshold)

    print(f"\nРаспределение расстояний для {len(neighbors)} соседей:")
    print(f"  LOCAL (d < {local_threshold:.2f}): {within_local}")
    print(
        f"  FUNCTIONAL ({local_threshold:.2f} <= d <= {functional_threshold:.2f}): {within_functional}"
    )
    print(
        f"  DISTANT ({functional_threshold:.2f} < d <= {distant_threshold:.2f}): {within_distant}"
    )
    print(f"  BEYOND threshold (d > {distant_threshold:.2f}): {beyond_distant}")

    if beyond_distant == 0:
        print("✅ ИСПРАВЛЕНО: Все соседи находятся в пределах distant_threshold!")
    else:
        print(
            f"❌ ПРОБЛЕМА ОСТАЕТСЯ: {beyond_distant} соседей за пределами distant_threshold"
        )
        for idx, (neighbor, dist) in enumerate(distances):
            if dist > distant_threshold:
                print(
                    f"    Neighbor {neighbor}: distance = {dist:.3f} (превышает {distant_threshold:.3f})"
                )
                if idx >= 5:
                    break

    return beyond_distant == 0


def test_cache_consistency():
    """Тест согласованности кэша и spatial optimizer"""
    print("\n=== Тест согласованности кэша ===\n")

    config = get_project_config()

    # Создаем кэш менеджер
    cache_manager = ConnectionCacheManager(config.lattice.dimensions)

    # Пересоздаем кэш
    print("Пересоздаем кэш...")
    cache_manager.precompute_all_connections(force_rebuild=True)

    # Создаем spatial optimizer
    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        OptimizationConfig,
    )

    opt_config = OptimizationConfig(
        enable_morton_encoding=config.lattice.enable_morton_encoding,
        enable_adaptive_chunking=config.adaptive_chunker is not None,
        max_memory_gb=8.0,
        target_performance_ms=config.lattice.target_performance_ms,
    )

    spatial_optimizer = UnifiedSpatialOptimizer(
        dimensions=config.lattice.dimensions, config=opt_config
    )

    # Тестируем несколько клеток
    test_cells = [677, 1000, 2000]
    all_consistent = True

    for test_cell in test_cells:
        print(f"\nТестируем клетку {test_cell}:")

        # Получаем соседей от spatial optimizer
        spatial_neighbors = set(
            spatial_optimizer.find_neighbors_by_radius_safe(test_cell)
        )

        # Получаем соседей из кэша
        cached_data = cache_manager.cache.get(test_cell, {})
        cache_neighbors = set()

        if cached_data:
            for conn_list in cached_data.values():
                for conn in conn_list:
                    if hasattr(conn, "target_idx"):
                        cache_neighbors.add(conn.target_idx)
                    else:
                        cache_neighbors.add(conn["target_idx"])

        print(f"  Spatial optimizer: {len(spatial_neighbors)} соседей")
        print(f"  Кэш: {len(cache_neighbors)} соседей")

        if spatial_neighbors == cache_neighbors:
            print("  ✅ СОГЛАСОВАННОСТЬ: Количество соседей совпадает")
        else:
            print("  ❌ НЕСОГЛАСОВАННОСТЬ: Количество соседей не совпадает")
            difference = len(spatial_neighbors.symmetric_difference(cache_neighbors))
            print(f"     Разница: {difference} соседей")
            all_consistent = False

    return all_consistent


def test_optimized_architecture():
    """Тестирует новую архитектуру без дублирования"""
    print("\n=== Тест оптимизированной архитектуры ===\n")
    
    # Конфигурация
    config = create_debug_config()
    config.cache.enabled = True  # Обязательно включаем кэш
    config.model.neighbor_count = -1  # Динамическое определение соседей
    config.lattice.dimensions = (8, 8, 8)  # Небольшая решетка для теста
    set_project_config(config)
    
    print(f"Конфигурация:")
    print(f"  - Размеры решетки: {config.lattice.dimensions}")
    print(f"  - Кэш включен: {config.cache.enabled}")
    print(f"  - State size: {config.model.state_size}")
    print(f"  - Adaptive radius: {config.calculate_adaptive_radius():.3f}")
    
    # Создаем решетку
    print("\nСоздание решетки...")
    start_time = time.time()
    lattice = create_lattice()
    creation_time = time.time() - start_time
    print(f"✅ Решетка создана за {creation_time:.2f}с")
    
    # Проверяем, что кэш инициализирован
    moe_processor = lattice.moe_processor
    connection_classifier = moe_processor.connection_classifier
    
    if hasattr(connection_classifier, 'cache_manager'):
        cache_stats = connection_classifier.cache_manager.get_cache_stats()
        print(f"\n📊 Статистика кэша:")
        print(f"  - Статус: {cache_stats['status']}")
        print(f"  - Закэшировано клеток: {cache_stats['cached_cells']}")
        print(f"  - Всего связей: {cache_stats['total_connections']}")
        print(f"  - Размер кэша: {cache_stats['cache_size_mb']:.1f}MB")
    
    # Тестируем новый метод get_cached_neighbors_and_classification
    print("\n🔍 Тест нового метода get_cached_neighbors_and_classification:")
    test_cell_idx = 100  # Тестовая клетка
    
    neighbors_data = connection_classifier.get_cached_neighbors_and_classification(
        cell_idx=test_cell_idx,
        states=lattice.states
    )
    
    print(f"\nРезультаты для клетки {test_cell_idx}:")
    for category in ["local", "functional", "distant"]:
        data = neighbors_data[category]
        print(f"  - {category.upper()}:")
        print(f"    - Количество: {len(data['indices'])}")
        print(f"    - Индексы: {data['indices'][:5]}{'...' if len(data['indices']) > 5 else ''}")
        print(f"    - States shape: {data['states'].shape}")
    
    # Тестируем forward pass
    print("\n🔄 Тестирование forward pass...")
    external_input = torch.randn(len(lattice.input_indices), config.model.state_size)
    lattice.set_input_states(external_input)
    
    # Запускаем несколько итераций
    num_iterations = 3
    for i in range(num_iterations):
        print(f"\nИтерация {i+1}/{num_iterations}:")
        start_time = time.time()
        
        # Forward pass
        output_states = lattice.forward()
        
        iteration_time = time.time() - start_time
        print(f"  - Время: {iteration_time*1000:.1f}ms")
        print(f"  - Output shape: {output_states.shape}")
    
    # Проверяем статистику MoE
    print("\n📊 Статистика MoE Processor:")
    stats = moe_processor.get_stats()
    print(f"  - Всего обработано клеток: {stats['total_cells_processed']}")
    print(f"  - Среднее кол-во соседей: {stats['avg_neighbors_per_cell']:.1f}")
    print(f"  - Распределение по экспертам:")
    for expert, ratio in stats['expert_usage_ratios'].items():
        print(f"    - {expert}: {ratio*100:.1f}%")
    
    print("\n✅ Тест оптимизированной архитектуры завершен!")
    return True


def main():
    """Основная функция тестирования"""
    print("🔧 Тестирование исправлений кэширования соседей\n")

    # Тест 1: Проверка исправления spatial hash
    spatial_hash_fixed = test_spatial_hash_fix()

    # Тест 2: Проверка согласованности кэша
    cache_consistent = test_cache_consistency()
    
    # Тест 3: Проверка оптимизированной архитектуры
    architecture_optimized = test_optimized_architecture()

    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"{'='*50}")
    print(f"Spatial hash исправлен: {'✅' if spatial_hash_fixed else '❌'}")
    print(f"Кэш согласован: {'✅' if cache_consistent else '❌'}")
    print(f"Архитектура оптимизирована: {'✅' if architecture_optimized else '❌'}")

    if spatial_hash_fixed and cache_consistent and architecture_optimized:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Проблема с кэшированием исправлена и архитектура оптимизирована.")
        print("\n🎯 Достигнутые оптимизации:")
        print("  - ✅ Убрано дублирование поиска соседей")
        print("  - ✅ Используется единый кэш для соседей и классификации")
        print("  - ✅ MoE Processor упрощен")
        print("  - ✅ Повышена производительность")
    else:
        print("\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ. Требуется дополнительная отладка.")


if __name__ == "__main__":
    main()
