#!/usr/bin/env python3
"""
Тест GPU Spatial Optimization Components
========================================

Демонстрирует возможности новых GPU-accelerated компонентов:
- GPU Spatial Hashing
- Adaptive Chunking
- Integrated Spatial Processor

Этот тест показывает производительность и возможности системы.
"""

import sys
import os
import torch
import numpy as np
import time
import logging
from typing import List, Tuple

# Добавляем путь к корневой папке проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импортируем наши компоненты
try:
    from core.lattice.gpu_spatial_hashing import (
        AdaptiveGPUSpatialHash,
        GPUSpatialHashGrid,
        GPUMortonEncoder,
    )
    from core.lattice.spatial_optimization.adaptive_chunker import AdaptiveGPUChunker
    from core.lattice.spatial_optimization.gpu_spatial_processor import (
        GPUSpatialProcessor,
    )

    from config.project_config import get_project_config
    from utils.device_manager import get_device_manager
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете тест из папки new_rebuild")
    sys.exit(1)


def test_gpu_morton_encoder():
    """Тестирует GPU Morton Encoder"""
    print("\n" + "=" * 60)
    print("🔢 Тестирование GPU Morton Encoder")
    print("=" * 60)

    dimensions = (32, 32, 32)
    encoder = GPUMortonEncoder(dimensions)

    # Создаем тестовые координаты
    test_coords = torch.tensor(
        [[0, 0, 0], [1, 1, 1], [15, 15, 15], [31, 31, 31]],
        device=encoder.device,
        dtype=torch.long,
    )

    print(f"Тестовые координаты:\n{test_coords}")

    # Кодируем
    start_time = time.time()
    morton_codes = encoder.encode_batch(test_coords)
    encode_time = (time.time() - start_time) * 1000

    print(f"Morton коды: {morton_codes}")
    print(f"Время кодирования: {encode_time:.2f}ms")

    # Декодируем обратно
    start_time = time.time()
    decoded_coords = encoder.decode_batch(morton_codes)
    decode_time = (time.time() - start_time) * 1000

    print(f"Декодированные координаты:\n{decoded_coords}")
    print(f"Время декодирования: {decode_time:.2f}ms")

    # Проверяем корректность
    matches = torch.allclose(test_coords.float(), decoded_coords.float())
    print(f"✅ Корректность кодирования/декодирования: {matches}")

    return matches


def test_gpu_spatial_hash():
    """Тестирует GPU Spatial Hash Grid"""
    print("\n" + "=" * 60)
    print("🏎️ Тестирование GPU Spatial Hash Grid")
    print("=" * 60)

    dimensions = (64, 64, 64)
    cell_size = 8
    hash_grid = GPUSpatialHashGrid(dimensions, cell_size)

    # Генерируем случайные координаты и индексы
    num_cells = 10000
    coordinates = torch.randint(
        0, 64, (num_cells, 3), device=hash_grid.device, dtype=torch.long
    )
    indices = torch.arange(num_cells, device=hash_grid.device, dtype=torch.long)

    print(f"Вставляем {num_cells} клеток...")

    # Вставляем данные
    start_time = time.time()
    hash_grid.insert_batch(coordinates, indices)
    insert_time = (time.time() - start_time) * 1000

    print(f"Время вставки: {insert_time:.2f}ms")
    print(f"Производительность: {num_cells / insert_time * 1000:.0f} клеток/сек")

    # Тестируем поиск
    query_points = torch.randint(
        0, 64, (100, 3), device=hash_grid.device, dtype=torch.float32
    )
    radius = 5.0

    print(f"\nВыполняем {len(query_points)} запросов с радиусом {radius}...")

    start_time = time.time()
    results = hash_grid.query_radius_batch(query_points, radius)
    query_time = (time.time() - start_time) * 1000

    print(f"Время поиска: {query_time:.2f}ms")
    print(
        f"Производительность: {len(query_points) / query_time * 1000:.0f} запросов/сек"
    )

    # Анализируем результаты
    neighbor_counts = [len(neighbors) for neighbors in results]
    avg_neighbors = np.mean(neighbor_counts)
    max_neighbors = max(neighbor_counts)

    print(f"Среднее количество соседей: {avg_neighbors:.1f}")
    print(f"Максимальное количество соседей: {max_neighbors}")

    # Статистика памяти
    memory_stats = hash_grid.get_memory_usage()
    print(f"\nИспользование памяти:")
    print(f"  GPU память: {memory_stats['total_gpu_mb']:.1f}MB")
    print(f"  Записи в кэше: {memory_stats['cache_entries']}")
    print(f"  Hash buckets: {memory_stats['grid_buckets']}")

    return True


def test_adaptive_spatial_hash():
    """Тестирует Adaptive GPU Spatial Hash"""
    print("\n" + "=" * 60)
    print("🎯 Тестирование Adaptive GPU Spatial Hash")
    print("=" * 60)

    dimensions = (100, 100, 100)
    target_memory_mb = 512.0  # 512MB целевая память

    adaptive_hash = AdaptiveGPUSpatialHash(dimensions, target_memory_mb)

    print(f"Оптимальный размер ячейки: {adaptive_hash.optimal_cell_size}")

    # Генерируем данные в несколько этапов для тестирования адаптации
    stages = [1000, 5000, 10000, 20000]

    for stage_idx, num_cells in enumerate(stages):
        print(f"\n--- Этап {stage_idx + 1}: {num_cells} клеток ---")

        # Генерируем координаты (кластеризованные для реалистичности)
        center = np.random.randint(20, 80, 3)
        spread = 15

        coordinates = []
        indices = []

        for i in range(num_cells):
            coord = center + np.random.normal(0, spread, 3)
            coord = np.clip(coord, 0, 99).astype(int)
            coordinates.append(coord)
            indices.append(len(coordinates) - 1)

        coords_tensor = torch.tensor(
            coordinates, device=adaptive_hash.device, dtype=torch.float32
        )
        indices_tensor = torch.tensor(
            indices, device=adaptive_hash.device, dtype=torch.long
        )

        # Вставляем данные
        start_time = time.time()
        adaptive_hash.insert_batch(coords_tensor, indices_tensor)
        insert_time = (time.time() - start_time) * 1000

        print(f"Время вставки: {insert_time:.2f}ms")

        # Тестируем поиск
        query_points = coords_tensor[:50]  # первые 50 точек как запросы
        radius = 10.0

        start_time = time.time()
        results = adaptive_hash.query_radius_batch(query_points, radius)
        query_time = (time.time() - start_time) * 1000

        neighbor_counts = [len(neighbors) for neighbors in results]
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0

        print(f"Время поиска: {query_time:.2f}ms")
        print(f"Среднее количество соседей: {avg_neighbors:.1f}")

        # Статистика
        stats = adaptive_hash.get_comprehensive_stats()
        print(f"Память: {stats['memory']['total_gpu_mb']:.1f}MB")
        print(f"Запросы: {stats['spatial_hash']['queries']}")
        print(f"Cache hit rate: {stats['spatial_hash']['cache_hit_rate']:.2f}")

    return True


def test_adaptive_chunker():
    """Тестирует Adaptive GPU Chunker"""
    print("\n" + "=" * 60)
    print("🧩 Тестирование Adaptive GPU Chunker")
    print("=" * 60)

    dimensions = (80, 80, 80)
    chunker = AdaptiveGPUChunker(dimensions)

    print(f"Создано {len(chunker.adaptive_chunks)} chunk'ов")

    # Анализируем chunk'и
    chunk_sizes = [len(chunk.cell_indices) for chunk in chunker.adaptive_chunks]
    total_cells = sum(chunk_sizes)
    avg_chunk_size = np.mean(chunk_sizes)

    print(f"Общее количество клеток: {total_cells}")
    print(f"Средний размер chunk'а: {avg_chunk_size:.1f}")

    # Тестируем поиск chunk'а по координатам
    test_coordinates = [(10, 10, 10), (40, 40, 40), (70, 70, 70)]

    print(f"\nТестируем поиск chunk'ов по координатам:")
    for coord in test_coordinates:
        try:
            chunk = chunker.get_chunk_by_coords(coord)
            print(
                f"  {coord} -> Chunk {chunk.chunk_id} "
                f"(приоритет: {chunk.processing_priority}, "
                f"память: {chunk.memory_size_mb:.1f}MB)"
            )
        except ValueError as e:
            print(f"  {coord} -> Ошибка: {e}")

    # Генерируем adaptive schedule
    print(f"\nГенерируем adaptive расписание...")
    schedule = chunker.get_adaptive_processing_schedule()

    print(f"Создано {len(schedule)} batch'ей:")
    for i, batch in enumerate(schedule[:5]):  # показываем первые 5
        print(f"  Batch {i+1}: {len(batch)} chunk'ов")

    # Статистика памяти
    memory_stats = chunker.get_memory_stats()
    print(f"\nСтатистика памяти:")
    print(f"  Общая память chunk'ов: {memory_stats['total_chunks_memory_mb']:.1f}MB")
    print(f"  Активная память: {memory_stats['active_chunks_memory_mb']:.1f}MB")
    print(f"  Эффективность памяти: {memory_stats['memory_efficiency']:.2f}")

    return True


def test_integrated_spatial_processor():
    """Тестирует интегрированный GPU Spatial Processor"""
    print("\n" + "=" * 60)
    print("🚀 Тестирование Integrated GPU Spatial Processor")
    print("=" * 60)

    dimensions = (50, 50, 50)
    processor = GPUSpatialProcessor(dimensions)

    # Подготавливаем тестовые данные
    num_queries = 10
    coordinates = torch.randint(0, 50, (num_queries, 3), dtype=torch.float32)
    radius = 8.0

    print(f"Тестируем синхронный поиск: {num_queries} запросов с радиусом {radius}")

    try:
        # Синхронный поиск
        start_time = time.time()
        result = processor.query_neighbors_sync(coordinates, radius, timeout=30.0)
        sync_time = (time.time() - start_time) * 1000

        print(f"✅ Синхронный поиск завершен за {sync_time:.2f}ms")
        print(f"Query ID: {result.query_id}")
        print(f"Время обработки: {result.processing_time_ms:.2f}ms")
        print(f"Использование памяти: {result.memory_usage_mb:.2f}MB")
        print(f"Cache hit rate: {result.cache_hit_rate:.2f}")
        print(f"Затронутые chunk'и: {len(result.chunks_accessed)}")

        # Анализируем результаты
        neighbor_counts = [len(neighbors) for neighbors in result.neighbor_lists]
        if neighbor_counts:
            print(f"Среднее количество соседей: {np.mean(neighbor_counts):.1f}")
            print(f"Максимальное количество соседей: {max(neighbor_counts)}")

    except Exception as e:
        print(f"❌ Ошибка синхронного поиска: {e}")
        return False

    # Тестируем асинхронный поиск
    print(f"\nТестируем асинхронный поиск...")

    async_results = []
    query_ids = []

    # Запускаем несколько асинхронных запросов
    for i in range(5):
        coords = torch.randint(0, 50, (5, 3), dtype=torch.float32)
        query_id = processor.query_neighbors_async(coords, radius=6.0, priority=i * 10)
        query_ids.append(query_id)
        print(f"  Запущен запрос {query_id}")

    # Ждем результаты
    print(f"Ожидаем результаты...")
    start_wait = time.time()
    completed_queries = 0

    while completed_queries < len(query_ids) and (time.time() - start_wait) < 30:
        for query_id in query_ids:
            if processor.is_query_complete(query_id):
                if query_id not in [r.query_id for r in async_results]:
                    result = processor.get_query_result(query_id)
                    async_results.append(result)
                    completed_queries += 1
                    print(
                        f"  ✅ Завершен {query_id}: {result.processing_time_ms:.1f}ms"
                    )

        time.sleep(0.1)

    # Статистика производительности
    print(f"\nСтатистика производительности:")
    perf_stats = processor.get_performance_stats()

    processor_stats = perf_stats["processor"]
    print(f"  Общее количество запросов: {processor_stats['total_queries']}")
    print(f"  Среднее время обработки: {processor_stats['avg_query_time_ms']:.2f}ms")
    print(f"  Эффективность памяти: {processor_stats['memory_efficiency']:.2f}")
    print(f"  Cache hit rate: {processor_stats['cache_hit_rate']:.2f}")

    chunker_stats = perf_stats["chunker"]
    print(f"  Chunk'и - всего: {chunker_stats['chunks']['total_chunks']}")
    print(
        f"  Chunk'и - высокое давление: {chunker_stats['chunks']['high_pressure_chunks']}"
    )

    # Очистка
    processor.shutdown()

    return True


def benchmark_performance():
    """Бенчмарк производительности всех компонентов"""
    print("\n" + "=" * 60)
    print("⚡ Бенчмарк производительности")
    print("=" * 60)

    device_manager = get_device_manager()
    device = device_manager.get_device()

    print(f"Устройство: {device}")

    # Разные размеры решеток для тестирования
    test_sizes = [(32, 32, 32), (64, 64, 64), (100, 100, 100)]

    results = []

    for dimensions in test_sizes:
        print(f"\n--- Тестирование размера {dimensions} ---")

        total_cells = np.prod(dimensions)
        print(f"Общее количество клеток: {total_cells:,}")

        # Тестируем GPU Spatial Hash
        target_memory = 256.0  # 256MB
        adaptive_hash = AdaptiveGPUSpatialHash(dimensions, target_memory)

        # Генерируем данные
        num_test_cells = min(10000, total_cells // 4)
        coordinates = torch.randint(
            0, max(dimensions), (num_test_cells, 3), device=device, dtype=torch.float32
        )
        indices = torch.arange(num_test_cells, device=device, dtype=torch.long)

        # Время вставки
        start_time = time.time()
        adaptive_hash.insert_batch(coordinates, indices)
        insert_time = (time.time() - start_time) * 1000

        # Время поиска
        query_points = coordinates[:100]
        radius = max(dimensions) * 0.1

        start_time = time.time()
        query_results = adaptive_hash.query_radius_batch(query_points, radius)
        query_time = (time.time() - start_time) * 1000

        # Статистика
        stats = adaptive_hash.get_comprehensive_stats()
        memory_usage = stats["memory"]["total_gpu_mb"]

        result = {
            "dimensions": dimensions,
            "total_cells": total_cells,
            "test_cells": num_test_cells,
            "insert_time_ms": insert_time,
            "query_time_ms": query_time,
            "memory_usage_mb": memory_usage,
            "insert_rate": num_test_cells / insert_time * 1000,
            "query_rate": len(query_points) / query_time * 1000,
        }

        results.append(result)

        print(
            f"  Вставка: {insert_time:.1f}ms ({result['insert_rate']:.0f} клеток/сек)"
        )
        print(f"  Поиск: {query_time:.1f}ms ({result['query_rate']:.0f} запросов/сек)")
        print(f"  Память: {memory_usage:.1f}MB")

    # Сводная таблица
    print(f"\n📊 Сводная таблица производительности:")
    print(f"{'Размер':<15} {'Клетки':<10} {'Вставка':<12} {'Поиск':<12} {'Память':<10}")
    print("-" * 70)

    for result in results:
        dim_str = f"{result['dimensions'][0]}³"
        cells_str = f"{result['total_cells']:,}"
        insert_str = f"{result['insert_rate']:.0f}/сек"
        query_str = f"{result['query_rate']:.0f}/сек"
        memory_str = f"{result['memory_usage_mb']:.1f}MB"

        print(
            f"{dim_str:<15} {cells_str:<10} {insert_str:<12} {query_str:<12} {memory_str:<10}"
        )

    return results


def main():
    """Главная функция тестирования"""
    print("🧪 Тестирование GPU Spatial Optimization Components")
    print("=" * 80)

    try:
        # Получаем информацию об устройстве
        device_manager = get_device_manager()
        device_stats = device_manager.get_memory_stats()

        print(f"Устройство: {device_manager.get_device()}")
        print(f"CUDA доступен: {device_manager.is_cuda()}")

        if device_manager.is_cuda():
            print(f"GPU память - выделено: {device_stats.get('allocated_mb', 0):.1f}MB")
            print(
                f"GPU память - зарезервировано: {device_stats.get('reserved_mb', 0):.1f}MB"
            )

        # Запускаем тесты
        test_results = []

        print("\n🔍 Запуск индивидуальных тестов...")

        # Индивидуальные тесты
        test_results.append(("Morton Encoder", test_gpu_morton_encoder()))
        test_results.append(("Spatial Hash Grid", test_gpu_spatial_hash()))
        test_results.append(("Adaptive Spatial Hash", test_adaptive_spatial_hash()))
        test_results.append(("Adaptive Chunker", test_adaptive_chunker()))
        test_results.append(("Spatial Processor", test_integrated_spatial_processor()))

        # Бенчмарк производительности
        print("\n⚡ Запуск бенчмарка производительности...")
        benchmark_results = benchmark_performance()

        # Итоговый отчет
        print("\n" + "=" * 80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 80)

        print("Результаты тестов:")
        for test_name, result in test_results:
            status = "✅ ПРОШЕЛ" if result else "❌ ПРОВАЛЕН"
            print(f"  {test_name:<25} {status}")

        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)

        print(f"\nОбщий результат: {passed_tests}/{total_tests} тестов прошли")

        if passed_tests == total_tests:
            print("🎉 Все тесты успешно пройдены!")
            print("🚀 GPU Spatial Optimization готов к использованию!")
        else:
            print("⚠️ Некоторые тесты провалены. Проверьте конфигурацию.")

        return passed_tests == total_tests

    except Exception as e:
        print(f"❌ Критическая ошибка в тестировании: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
