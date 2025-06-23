#!/usr/bin/env python3
"""
Пример использования GPU Spatial Optimization
============================================

Демонстрирует основные возможности интегрированных компонентов:
- GPU Morton Encoder для пространственного кодирования
- GPU Spatial Hash Grid для быстрого поиска соседей
- Adaptive GPU Spatial Hash для самооптимизирующегося поиска
- Adaptive GPU Chunker для работы с большими решетками
- GPU Spatial Processor как интегрированное решение

Основано на new_rebuild/GPU_SPATIAL_OPTIMIZATION_GUIDE.md
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time

from config.project_config import get_project_config
from utils.device_manager import get_device_manager
from utils.logging import get_logger

# Импорт компонентов GPU Spatial Optimization
from core.lattice.spatial_optimization import (
    GPUSpatialProcessor,
    AdaptiveGPUChunker,
    GPUMortonEncoder,
    GPUSpatialHashGrid,
    AdaptiveGPUSpatialHash,
)

logger = get_logger(__name__)


def example_basic_usage():
    """Базовый пример использования GPUSpatialProcessor"""
    print("📍 Базовый пример использования")

    # Создаем processor для решетки 100x100x100
    processor = GPUSpatialProcessor((100, 100, 100))

    # Подготавливаем запрос
    query_coordinates = torch.tensor(
        [[25, 25, 25], [50, 50, 50], [75, 75, 75]], dtype=torch.float32
    )

    # Выполняем поиск соседей
    result = processor.query_neighbors_sync(
        coordinates=query_coordinates, radius=8.0, timeout=30.0
    )

    # Анализируем результаты
    print(f"Найдено соседей: {[len(neighbors) for neighbors in result.neighbor_lists]}")
    print(f"Время обработки: {result.processing_time_ms:.2f}ms")
    print(f"Использование памяти: {result.memory_usage_mb:.2f}MB")

    # Получаем статистику производительности
    stats = processor.get_performance_stats()
    print(f"Всего запросов: {stats['processor']['total_queries']}")
    print(f"Среднее время: {stats['processor']['avg_query_time_ms']:.2f}ms")

    # Завершаем работу
    processor.shutdown()
    print("✅ Базовый пример завершен\n")


async def example_advanced_async():
    """Продвинутый пример с async обработкой"""
    print("🚀 Продвинутый пример с async")

    processor = GPUSpatialProcessor((200, 200, 200))

    # Функция обработки результата
    def process_result(result):
        print(
            f"Query {result.query_id} completed: {len(result.neighbor_lists)} results"
        )

    # Запускаем несколько асинхронных запросов
    query_ids = []
    for i in range(5):
        coords = torch.randint(0, 200, (10, 3), dtype=torch.float32)
        query_id = processor.query_neighbors_async(
            coordinates=coords, radius=12.0, priority=i * 10, callback=process_result
        )
        query_ids.append(query_id)

    # Ждем завершения всех запросов
    completed = 0
    max_wait = 50  # 5 секунд
    wait_count = 0

    while completed < len(query_ids) and wait_count < max_wait:
        new_completed = 0
        for query_id in query_ids:
            if processor.is_query_complete(query_id):
                new_completed += 1

        if new_completed > completed:
            completed = new_completed
            print(f"Завершено запросов: {completed}/{len(query_ids)}")

        time.sleep(0.1)
        wait_count += 1

    processor.shutdown()
    print("✅ Продвинутый пример завершен\n")


def example_morton_encoding():
    """Пример использования GPU Morton Encoder"""
    print("🔢 Пример Morton Encoding")

    device_manager = get_device_manager()
    device = device_manager.get_device()

    # Создаем encoder
    encoder = GPUMortonEncoder((128, 128, 128))

    # Тестовые координаты
    coordinates = torch.tensor(
        [[10, 20, 30], [64, 32, 96], [100, 50, 75]], dtype=torch.long, device=device
    )

    # Кодируем в Morton коды
    morton_codes = encoder.encode_batch(coordinates)
    print(f"Координаты: {coordinates.tolist()}")
    print(f"Morton коды: {morton_codes.tolist()}")

    # Декодируем обратно
    decoded = encoder.decode_batch(morton_codes)
    print(f"Декодированные: {decoded.tolist()}")

    # Проверяем точность
    matches = torch.allclose(coordinates.float(), decoded.float())
    print(f"Кодирование точное: {'✅' if matches else '❌'}")
    print("✅ Morton encoding пример завершен\n")


def example_spatial_hash_grid():
    """Пример использования GPU Spatial Hash Grid"""
    print("🏎️ Пример Spatial Hash Grid")

    device_manager = get_device_manager()
    device = device_manager.get_device()

    # Создаем hash grid
    hash_grid = GPUSpatialHashGrid((64, 64, 64), cell_size=8)

    # Генерируем случайные точки
    num_points = 200
    coordinates = torch.randint(0, 64, (num_points, 3), dtype=torch.long, device=device)
    indices = torch.arange(num_points, device=device)

    # Вставляем данные
    print(f"Вставляем {num_points} точек...")
    hash_grid.insert_batch(coordinates, indices)

    # Выполняем запросы
    query_points = torch.tensor(
        [
            [32, 32, 32],  # Центр
            [16, 16, 16],  # Первая четверть
            [48, 48, 48],  # Третья четверть
        ],
        dtype=torch.float32,
        device=device,
    )

    neighbors = hash_grid.query_radius_batch(query_points, radius=10.0)

    for i, neighbors_list in enumerate(neighbors):
        print(f"Точка {query_points[i].tolist()}: {len(neighbors_list)} соседей")

    # Получаем статистику
    stats = hash_grid.get_stats()
    print(
        f"Статистика: {stats.total_queries} запросов, "
        f"{stats.avg_query_time_ms:.2f}ms среднее время"
    )
    print("✅ Spatial Hash Grid пример завершен\n")


def example_adaptive_chunker():
    """Пример использования Adaptive GPU Chunker"""
    print("📦 Пример Adaptive Chunker")

    # Создаем chunker для большой решетки
    chunker = AdaptiveGPUChunker((80, 80, 80))

    print(f"Создано chunk'ов: {len(chunker.chunks)}")

    # Получаем информацию о конкретном chunk'е
    test_coords = (40, 40, 40)
    chunk_info = chunker.get_chunk_by_coords(test_coords)

    if chunk_info:
        print(f"Chunk для {test_coords}:")
        print(f"  ID: {chunk_info.chunk_id}")
        print(f"  Размеры: {chunk_info.start_coords} -> {chunk_info.end_coords}")
        print(f"  Клеток: {len(chunk_info.cell_indices)}")
        print(f"  Память: {chunk_info.gpu_memory_usage_mb:.2f}MB")

    # Получаем расписание обработки
    schedule = chunker.get_adaptive_processing_schedule()
    print(f"Расписание обработки: {len(schedule)} этапов")

    for i, stage in enumerate(schedule[:3]):  # Показываем первые 3 этапа
        print(f"  Этап {i+1}: {len(stage)} chunk'ов")

    # Тестируем асинхронную обработку
    if chunk_info:
        future = chunker.process_chunk_async(chunk_info.chunk_id, "load")
        print(f"Запущена асинхронная загрузка chunk {chunk_info.chunk_id}")

    # Получаем статистику
    stats = chunker.get_comprehensive_stats()
    print(f"Статистика chunker: {stats}")

    print("✅ Adaptive Chunker пример завершен\n")


def example_performance_comparison():
    """Сравнение производительности разных подходов"""
    print("⚡ Сравнение производительности")

    # Тестовые данные
    dimensions = (50, 50, 50)
    num_queries = 20
    radius = 8.0

    device_manager = get_device_manager()
    device = device_manager.get_device()

    # Генерируем тестовые координаты
    test_coordinates = torch.rand(num_queries, 3, device=device) * 50

    results = {}

    # 1. GPU Spatial Hash Grid
    print("  Тестирование GPU Spatial Hash Grid...")
    start_time = time.time()

    hash_grid = GPUSpatialHashGrid(dimensions, cell_size=8)
    # Вставляем некоторые данные для тестирования
    sample_coords = torch.randint(0, 50, (100, 3), dtype=torch.long, device=device)
    sample_indices = torch.arange(100, device=device)
    hash_grid.insert_batch(sample_coords, sample_indices)

    neighbors = hash_grid.query_radius_batch(test_coordinates, radius)

    results["SpatialHashGrid"] = (time.time() - start_time) * 1000

    # 2. Adaptive GPU Spatial Hash
    print("  Тестирование Adaptive GPU Spatial Hash...")
    start_time = time.time()

    adaptive_hash = AdaptiveGPUSpatialHash(dimensions, target_memory_mb=256.0)
    adaptive_hash.insert_batch(sample_coords.float(), sample_indices)
    neighbors = adaptive_hash.query_radius_batch(test_coordinates, radius)

    results["AdaptiveSpatialHash"] = (time.time() - start_time) * 1000

    # 3. GPU Spatial Processor (интегрированное решение)
    print("  Тестирование GPU Spatial Processor...")
    start_time = time.time()

    processor = GPUSpatialProcessor(dimensions)
    result = processor.query_neighbors_sync(test_coordinates, radius, timeout=10.0)
    processor.shutdown()

    results["SpatialProcessor"] = (time.time() - start_time) * 1000

    # Выводим результаты
    print("Результаты производительности:")
    for method, time_ms in results.items():
        print(f"  {method:<20}: {time_ms:.2f}ms")

    fastest = min(results, key=results.get)
    print(f"Самый быстрый: {fastest} ({results[fastest]:.2f}ms)")
    print("✅ Сравнение производительности завершено\n")


def main():
    """Основная функция для демонстрации всех примеров"""
    print("=" * 60)
    print("GPU Spatial Optimization - Примеры использования")
    print("=" * 60)

    config = get_project_config()
    device_manager = get_device_manager()

    print(f"Устройство: {device_manager.get_device()}")
    print(f"CUDA доступна: {'✅' if device_manager.is_cuda() else '❌'}")
    print(f"Размер решетки: {config.lattice_dimensions}")
    print()

    try:
        # Запускаем примеры
        example_morton_encoding()
        example_spatial_hash_grid()
        example_adaptive_chunker()
        example_basic_usage()
        example_performance_comparison()

        # Async пример требует специального запуска
        print("🔄 Запуск async примера...")
        import asyncio

        asyncio.run(example_advanced_async())

        print("=" * 60)
        print("🎉 ВСЕ ПРИМЕРЫ УСПЕШНО ВЫПОЛНЕНЫ!")
        print(
            "GPU Spatial Optimization полностью интегрирован и готов к использованию."
        )
        print("=" * 60)

    except Exception as e:
        print(f"❌ Ошибка в примерах: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
