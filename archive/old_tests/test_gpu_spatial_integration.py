#!/usr/bin/env python3
"""
Тест интеграции GPU Spatial Optimization компонентов
==================================================

Проверяет правильную интеграцию всех компонентов:
- GPU Spatial Hashing (GPUMortonEncoder, GPUSpatialHashGrid, AdaptiveGPUSpatialHash)
- Adaptive Chunking (AdaptiveGPUChunker)
- Integrated Spatial Processor (GPUSpatialProcessor)

Основные проверки:
1. Инициализация всех компонентов
2. GPU Memory management
3. Spatial query performance
4. Component integration
5. Performance benchmarks
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from typing import List, Dict, Tuple
import traceback

from new_rebuild.config import get_project_config
from new_rebuild.utils.device_manager import get_device_manager
from new_rebuild.utils.logging import get_logger

# Импорты компонентов GPU Spatial Optimization
from new_rebuild.core.lattice.gpu_spatial_hashing import (
    GPUMortonEncoder,
    GPUSpatialHashGrid,
    AdaptiveGPUSpatialHash,
)
from new_rebuild.core.lattice.spatial_optimization.adaptive_chunker import (
    AdaptiveGPUChunker,
    AdaptiveChunkInfo,
)
from new_rebuild.core.lattice.spatial_optimization.gpu_spatial_processor import (
    GPUSpatialProcessor,
    SpatialQuery,
    SpatialQueryResult,
)

logger = get_logger(__name__)


class GPUSpatialIntegrationTester:
    """Тестер интеграции GPU Spatial Optimization компонентов"""

    def __init__(self):
        self.config = get_project_config()
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Тестовые размеры решеток
        self.test_dimensions = [
            (10, 10, 10),  # Маленькая
            (25, 25, 25),  # Средняя
            (50, 50, 50),  # Большая
        ]

        self.results = {}

    def run_all_tests(self) -> Dict[str, bool]:
        """Запускает все тесты интеграции"""
        logger.info("🚀 Запуск тестов GPU Spatial Optimization интеграции")

        tests = [
            ("GPU Morton Encoder", self.test_gpu_morton_encoder),
            ("GPU Spatial Hash Grid", self.test_gpu_spatial_hash_grid),
            ("Adaptive GPU Spatial Hash", self.test_adaptive_gpu_spatial_hash),
            ("Adaptive GPU Chunker", self.test_adaptive_gpu_chunker),
            ("GPU Spatial Processor", self.test_gpu_spatial_processor),
            ("Integration Performance", self.test_integration_performance),
            ("Memory Management", self.test_memory_management),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                logger.info(f"📊 Тест: {test_name}")
                start_time = time.time()
                success = test_func()
                elapsed = (time.time() - start_time) * 1000

                results[test_name] = success
                status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
                logger.info(f"   {status} за {elapsed:.2f}ms")

            except Exception as e:
                logger.error(f"   ❌ ОШИБКА: {e}")
                logger.error(f"   Traceback: {traceback.format_exc()}")
                results[test_name] = False

        return results

    def test_gpu_morton_encoder(self) -> bool:
        """Тест GPU Morton Encoder"""
        try:
            dimensions = (64, 64, 64)
            encoder = GPUMortonEncoder(dimensions)

            # Генерируем тестовые координаты
            test_coords = torch.tensor(
                [[10, 20, 30], [40, 50, 60], [5, 15, 25]],
                dtype=torch.long,
                device=self.device,
            )

            # Кодируем в Morton коды
            morton_codes = encoder.encode_batch(test_coords)
            assert morton_codes.shape == (3,), f"Неверная форма: {morton_codes.shape}"

            # Декодируем обратно
            decoded_coords = encoder.decode_batch(morton_codes)
            assert torch.allclose(
                test_coords.float(), decoded_coords.float()
            ), "Декодирование не совпадает"

            logger.debug(
                f"   Morton кодирование: {test_coords.tolist()} -> {morton_codes.tolist()}"
            )
            return True

        except Exception as e:
            logger.error(f"Morton Encoder ошибка: {e}")
            return False

    def test_gpu_spatial_hash_grid(self) -> bool:
        """Тест GPU Spatial Hash Grid"""
        try:
            dimensions = (32, 32, 32)
            hash_grid = GPUSpatialHashGrid(dimensions, cell_size=8)

            # Генерируем тестовые данные
            num_points = 100
            coordinates = torch.randint(
                0, 32, (num_points, 3), dtype=torch.long, device=self.device
            )
            indices = torch.arange(num_points, device=self.device)

            # Вставляем данные
            hash_grid.insert_batch(coordinates, indices)

            # Выполняем запрос
            query_points = torch.tensor(
                [[16, 16, 16]], dtype=torch.float32, device=self.device
            )
            neighbors = hash_grid.query_radius_batch(query_points, radius=5.0)

            assert (
                len(neighbors) == 1
            ), f"Неверное количество результатов: {len(neighbors)}"
            assert len(neighbors[0]) > 0, "Не найдено соседей"

            logger.debug(f"   Найдено соседей: {len(neighbors[0])}")
            return True

        except Exception as e:
            logger.error(f"Spatial Hash Grid ошибка: {e}")
            return False

    def test_adaptive_gpu_spatial_hash(self) -> bool:
        """Тест Adaptive GPU Spatial Hash"""
        try:
            dimensions = (50, 50, 50)
            adaptive_hash = AdaptiveGPUSpatialHash(dimensions, target_memory_mb=512.0)

            # Генерируем больше данных для проверки адаптации
            num_points = 500
            coordinates = torch.randint(
                0, 50, (num_points, 3), dtype=torch.float32, device=self.device
            )
            indices = torch.arange(num_points, device=self.device)

            # Вставляем данные
            adaptive_hash.insert_batch(coordinates, indices)

            # Выполняем несколько запросов для проверки адаптации
            for i in range(3):
                query_points = torch.rand(5, 3, device=self.device) * 50
                neighbors = adaptive_hash.query_radius_batch(query_points, radius=8.0)
                assert (
                    len(neighbors) == 5
                ), f"Неверное количество результатов: {len(neighbors)}"

            # Проверяем статистику
            stats = adaptive_hash.get_comprehensive_stats()
            assert "hash_grid" in stats, "Отсутствует статистика hash_grid"
            assert "adaptations" in stats, "Отсутствует статистика adaptations"

            logger.debug(f"   Адаптации: {stats['adaptations']}")
            return True

        except Exception as e:
            logger.error(f"Adaptive Spatial Hash ошибка: {e}")
            return False

    def test_adaptive_gpu_chunker(self) -> bool:
        """Тест Adaptive GPU Chunker"""
        try:
            dimensions = (40, 40, 40)
            chunker = AdaptiveGPUChunker(dimensions)

            # Получаем информацию о chunk'ах
            total_chunks = len(chunker.chunks)
            assert total_chunks > 0, "Не создано chunk'ов"

            # Тестируем получение chunk'а по координатам
            test_coords = (20, 20, 20)
            chunk_info = chunker.get_chunk_by_coords(test_coords)
            assert chunk_info is not None, "Не найден chunk для координат"

            # Тестируем schedule получения
            schedule = chunker.get_adaptive_processing_schedule()
            assert len(schedule) > 0, "Пустое расписание обработки"

            # Тестируем асинхронную обработку
            chunk_id = chunk_info.chunk_id
            future = chunker.process_chunk_async(chunk_id, "load")
            assert future is not None, "Не создана асинхронная задача"

            # Получаем статистику
            stats = chunker.get_comprehensive_stats()
            assert "chunks" in stats, "Отсутствует статистика chunks"
            assert (
                "total_chunks" in stats["chunks"]
            ), "Отсутствует статистика total_chunks"

            logger.debug(f"   Создано chunk'ов: {total_chunks}")
            logger.debug(f"   Статистика: {stats['chunks']['total_chunks']} chunk'ов")
            return True

        except Exception as e:
            logger.error(f"Adaptive Chunker ошибка: {e}")
            return False

    def test_gpu_spatial_processor(self) -> bool:
        """Тест GPU Spatial Processor (интегрированный компонент)"""
        try:
            dimensions = (30, 30, 30)
            processor = GPUSpatialProcessor(dimensions)

            # Подготавливаем тестовые координаты
            test_coordinates = torch.tensor(
                [[15, 15, 15], [10, 10, 10], [20, 20, 20]],
                dtype=torch.float32,
                device=self.device,
            )

            # Синхронный запрос
            result = processor.query_neighbors_sync(
                coordinates=test_coordinates, radius=5.0, timeout=10.0
            )

            assert result is not None, "Не получен результат синхронного запроса"
            assert (
                len(result.neighbor_lists) == 3
            ), f"Неверное количество результатов: {len(result.neighbor_lists)}"
            assert result.processing_time_ms > 0, "Время обработки не записано"

            # Асинхронный запрос
            query_id = processor.query_neighbors_async(
                coordinates=test_coordinates, radius=8.0, priority=10
            )

            assert query_id is not None, "Не получен ID асинхронного запроса"

            # Ждем завершения
            max_wait = 50  # 5 секунд
            completed = False
            for _ in range(max_wait):
                if processor.is_query_complete(query_id):
                    completed = True
                    break
                time.sleep(0.1)

            assert completed, "Асинхронный запрос не завершился"

            # Получаем результат
            async_result = processor.get_query_result(query_id)
            assert async_result is not None, "Не получен результат асинхронного запроса"

            # Проверяем статистику производительности
            stats = processor.get_performance_stats()
            assert "processor" in stats, "Отсутствует статистика processor"
            assert (
                stats["processor"]["total_queries"] >= 2
            ), "Неверное количество запросов в статистике"

            logger.debug(f"   Синхронный запрос: {result.processing_time_ms:.2f}ms")
            logger.debug(
                f"   Асинхронный запрос: {async_result.processing_time_ms:.2f}ms"
            )
            logger.debug(f"   Всего запросов: {stats['processor']['total_queries']}")

            # Завершаем работу
            processor.shutdown()
            return True

        except Exception as e:
            logger.error(f"Spatial Processor ошибка: {e}")
            return False

    def test_integration_performance(self) -> bool:
        """Тест производительности интеграции"""
        try:
            # Тестируем на разных размерах
            performance_results = {}

            for dimensions in self.test_dimensions:
                logger.debug(f"   Тестирование {dimensions}")

                processor = GPUSpatialProcessor(dimensions)

                # Генерируем тестовые данные
                num_queries = 10
                max_coord = max(dimensions)
                coordinates = torch.rand(num_queries, 3, device=self.device) * max_coord

                # Измеряем производительность
                start_time = time.time()

                result = processor.query_neighbors_sync(
                    coordinates=coordinates,
                    radius=max_coord * 0.1,  # 10% от максимального размера
                    timeout=30.0,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                performance_results[dimensions] = {
                    "time_ms": elapsed_ms,
                    "queries_per_second": num_queries / (elapsed_ms / 1000),
                    "processing_time_ms": result.processing_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                }

                processor.shutdown()

            # Логируем результаты производительности
            for dims, perf in performance_results.items():
                logger.debug(
                    f"   {dims}: {perf['time_ms']:.2f}ms, "
                    f"{perf['queries_per_second']:.1f} q/s, "
                    f"{perf['memory_usage_mb']:.2f}MB"
                )

            return True

        except Exception as e:
            logger.error(f"Performance тест ошибка: {e}")
            return False

    def test_memory_management(self) -> bool:
        """Тест управления памятью"""
        try:
            # Получаем начальные статистики памяти
            initial_stats = self.device_manager.get_memory_stats()

            dimensions = (60, 60, 60)  # Большая решетка для проверки памяти

            # Создаем и используем processor
            processor = GPUSpatialProcessor(dimensions)

            # Выполняем несколько операций для загрузки памяти
            for i in range(5):
                coordinates = torch.rand(20, 3, device=self.device) * 60
                result = processor.query_neighbors_sync(coordinates, radius=10.0)

                # Проверяем использование памяти
                current_stats = self.device_manager.get_memory_stats()
                logger.debug(
                    f"     Итерация {i+1}: {current_stats.get('used_mb', 0):.2f}MB использовано"
                )

            # Оптимизация производительности
            processor.optimize_performance()

            # Завершаем работу и проверяем очистку памяти
            processor.shutdown()

            # Принудительная очистка
            self.device_manager.cleanup()

            final_stats = self.device_manager.get_memory_stats()

            logger.debug(
                f"   Начальная память: {initial_stats.get('used_mb', 0):.2f}MB"
            )
            logger.debug(f"   Финальная память: {final_stats.get('used_mb', 0):.2f}MB")

            return True

        except Exception as e:
            logger.error(f"Memory management тест ошибка: {e}")
            return False


def main():
    """Основная функция для запуска тестов"""
    print("=" * 60)
    print("GPU Spatial Optimization Integration Test")
    print("=" * 60)

    tester = GPUSpatialIntegrationTester()
    results = tester.run_all_tests()

    # Подводим итоги
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1

    print("-" * 60)
    print(f"Пройдено: {passed}/{total} тестов ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! GPU Spatial Optimization успешно интегрирован!")
        return 0
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ. Требуется доработка.")
        return 1


if __name__ == "__main__":
    exit(main())
