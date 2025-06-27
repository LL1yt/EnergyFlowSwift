#!/usr/bin/env python3
"""
Тест оптимизации Connection Classification через кэширование
==========================================================

Демонстрирует разницу в производительности между:
1. Обычной классификацией связей (fallback)
2. Pre-computed кэшированием связей

Показывает speedup и memory usage для различных размеров решетки.
"""

import torch
import time
import numpy as np
from typing import Dict, Any

from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.config.project_config import get_project_config
from utils.centralized_config import CentralizedConfig

# Инициализируем конфигурацию
CentralizedConfig()


def benchmark_connection_classification(
    lattice_dimensions: tuple = (15, 15, 15), num_tests: int = 100, batch_size: int = 32
) -> Dict[str, Any]:
    """
    Бенчмарк сравнение обычной классификации vs кэшированной

    Args:
        lattice_dimensions: Размеры решетки (x, y, z)
        num_tests: Количество тестов для усреднения
        batch_size: Размер batch для тестирования

    Returns:
        Результаты бенчмарка с временами и speedup
    """
    print(f"\n🔍 Benchmarking Connection Classification")
    print(f"📐 Решетка: {lattice_dimensions}")
    print(f"🔄 Тесты: {num_tests}")
    print(f"📦 Batch size: {batch_size}")
    print("=" * 60)

    total_cells = np.prod(lattice_dimensions)
    config = get_project_config()
    state_size = config.gnn.state_size
    max_neighbors = config.max_neighbors

    # Генерируем тестовые данные
    print("📊 Генерируем тестовые данные...")

    # Случайные состояния клеток
    all_states = torch.randn(total_cells, state_size)

    # Случайные batch индексы
    test_batches = []
    for _ in range(num_tests):
        cell_indices = torch.randint(0, total_cells, (batch_size,))
        # Генерируем случайных соседей (с padding -1)
        neighbor_indices = torch.randint(-1, total_cells, (batch_size, max_neighbors))
        # Убеждаемся что есть хотя бы несколько валидных соседей
        for i in range(batch_size):
            valid_count = torch.randint(5, max_neighbors // 2, (1,)).item()
            neighbor_indices[i, :valid_count] = torch.randint(
                0, total_cells, (valid_count,)
            )
            neighbor_indices[i, valid_count:] = -1

        test_batches.append((cell_indices, neighbor_indices))

    results = {}

    # ==========================================
    # 1. ТЕСТ БЕЗ КЭША (Fallback режим)
    # ==========================================
    print("\n🔄 Тестируем БЕЗ кэша (fallback режим)...")

    classifier_no_cache = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dimensions, enable_cache=False  # Отключаем кэш
    )

    # Прогрев
    for i in range(3):
        cell_indices, neighbor_indices = test_batches[0]
        _ = classifier_no_cache.classify_connections_batch(
            cell_indices, neighbor_indices, all_states
        )

    # Измеряем время
    start_time = time.time()

    for cell_indices, neighbor_indices in test_batches:
        _ = classifier_no_cache.classify_connections_batch(
            cell_indices, neighbor_indices, all_states
        )

    fallback_time = time.time() - start_time
    fallback_avg = fallback_time / num_tests

    print(f"⏱️  Время БЕЗ кэша: {fallback_time:.4f}s")
    print(f"📊 Среднее время: {fallback_avg:.6f}s/batch")

    results["fallback"] = {
        "total_time": fallback_time,
        "avg_time": fallback_avg,
        "stats": classifier_no_cache.get_classification_stats(),
    }

    # ==========================================
    # 2. ТЕСТ С КЭШЕМ
    # ==========================================
    print("\n🚀 Тестируем С кэшем...")

    classifier_with_cache = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dimensions, enable_cache=True  # Включаем кэш
    )

    # Даем время на инициализацию кэша
    print("⌛ Ожидаем инициализацию кэша...")
    time.sleep(2)

    # Прогрев кэша
    for i in range(5):
        cell_indices, neighbor_indices = test_batches[0]
        _ = classifier_with_cache.classify_connections_batch(
            cell_indices, neighbor_indices, all_states
        )

    # Измеряем время с кэшем
    start_time = time.time()

    for cell_indices, neighbor_indices in test_batches:
        _ = classifier_with_cache.classify_connections_batch(
            cell_indices, neighbor_indices, all_states
        )

    cached_time = time.time() - start_time
    cached_avg = cached_time / num_tests

    print(f"⏱️  Время С кэшем: {cached_time:.4f}s")
    print(f"📊 Среднее время: {cached_avg:.6f}s/batch")

    results["cached"] = {
        "total_time": cached_time,
        "avg_time": cached_avg,
        "stats": classifier_with_cache.get_classification_stats(),
    }

    # ==========================================
    # 3. АНАЛИЗ РЕЗУЛЬТАТОВ
    # ==========================================
    print("\n📈 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    if cached_time > 0:
        speedup = fallback_time / cached_time
        speedup_percent = ((fallback_time - cached_time) / fallback_time) * 100

        print(f"⚡ Speedup: {speedup:.2f}x")
        print(f"📉 Ускорение: {speedup_percent:.1f}%")
        print(f"⏱️  Экономия времени: {fallback_time - cached_time:.4f}s")

        results["performance"] = {
            "speedup": speedup,
            "speedup_percent": speedup_percent,
            "time_saved": fallback_time - cached_time,
        }
    else:
        print("⚠️  Cached время слишком мало для точного измерения")
        results["performance"] = {"speedup": float("inf")}

    # Статистика кэша
    cache_stats = classifier_with_cache.get_cache_stats()
    if cache_stats["status"] == "active":
        print(f"\n💾 СТАТИСТИКА КЭША:")
        print(f"🔢 Кэшированных клеток: {cache_stats['cached_cells']}")
        print(f"🔗 Всего связей: {cache_stats['total_connections']}")
        print(f"💽 Размер кэша: {cache_stats['cache_size_mb']:.1f} MB")
        print(f"🎯 LOCAL: {cache_stats['local_connections']}")
        print(f"🎯 FUNCTIONAL candidates: {cache_stats['functional_candidates']}")
        print(f"🎯 DISTANT: {cache_stats['distant_connections']}")

    # Статистика производительности кэша
    perf_stats = results["cached"]["stats"].get("cache_performance", {})
    if perf_stats.get("cache_enabled", False):
        print(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ КЭША:")
        print(f"🎯 Hit rate: {perf_stats['cache_hit_rate']:.1%}")
        print(f"✅ Cache hits: {perf_stats['cache_hits']}")
        print(f"❌ Cache misses: {perf_stats['cache_misses']}")
        print(f"⏱️  Среднее время кэша: {perf_stats['avg_cache_time']:.6f}s")

    return results


def test_different_lattice_sizes():
    """Тестирует производительность на разных размерах решетки"""
    print("\n🔬 ТЕСТ МАСШТАБИРУЕМОСТИ")
    print("=" * 80)

    lattice_sizes = [
        (10, 10, 10),  # 1K клеток
        (15, 15, 15),  # 3.4K клеток
        (20, 20, 15),  # 6K клеток
        (25, 25, 15),  # 9.4K клеток
    ]

    for dimensions in lattice_sizes:
        total_cells = np.prod(dimensions)
        print(f"\n📐 Решетка {dimensions} ({total_cells:,} клеток)")
        print("-" * 50)

        try:
            results = benchmark_connection_classification(
                lattice_dimensions=dimensions,
                num_tests=20,  # Меньше тестов для больших решеток
                batch_size=16,
            )

            if "performance" in results:
                speedup = results["performance"].get("speedup", 0)
                print(f"🚀 Результат: {speedup:.2f}x speedup")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue

        print("\n" + "=" * 30)


def demonstrate_cache_persistence():
    """Демонстрирует персистентность кэша между запусками"""
    print("\n💾 ТЕСТ ПЕРСИСТЕНТНОСТИ КЭША")
    print("=" * 60)

    lattice_dims = (12, 12, 12)

    print("1️⃣  Первый запуск (создание кэша)...")
    start_time = time.time()
    classifier1 = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dims, enable_cache=True
    )
    first_init_time = time.time() - start_time
    print(f"⏱️  Время инициализации: {first_init_time:.2f}s")

    # Уничтожаем объект
    del classifier1

    print("\n2️⃣  Второй запуск (загрузка кэша с диска)...")
    start_time = time.time()
    classifier2 = UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dims, enable_cache=True
    )
    second_init_time = time.time() - start_time
    print(f"⏱️  Время инициализации: {second_init_time:.2f}s")

    if second_init_time > 0:
        cache_speedup = first_init_time / second_init_time
        print(f"🚀 Кэш загрузка быстрее в {cache_speedup:.1f}x")

    return classifier2


if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ CONNECTION CACHE OPTIMIZATION")
    print("=" * 80)

    try:
        # Основной бенчмарк
        print("\n1️⃣  ОСНОВНОЙ БЕНЧМАРК")
        main_results = benchmark_connection_classification(
            lattice_dimensions=(15, 15, 15), num_tests=50, batch_size=32
        )

        # Тест масштабируемости
        print("\n2️⃣  ТЕСТ МАСШТАБИРУЕМОСТИ")
        test_different_lattice_sizes()

        # Тест персистентности
        print("\n3️⃣  ТЕСТ ПЕРСИСТЕНТНОСТИ")
        final_classifier = demonstrate_cache_persistence()

        print("\n✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("=" * 80)

        # Финальная статистика
        if "performance" in main_results:
            speedup = main_results["performance"]["speedup"]
            print(f"🎯 ИТОГОВЫЙ РЕЗУЛЬТАТ: {speedup:.2f}x ускорение через кэширование")

    except KeyboardInterrupt:
        print("\n⚠️  Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        import traceback

        traceback.print_exc()
