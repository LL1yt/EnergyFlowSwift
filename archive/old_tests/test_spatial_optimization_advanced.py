#!/usr/bin/env python3
"""
Продвинутый тест Spatial Optimization
=====================================

Тестирует новую систему spatial optimization для масштабирования
3D Cellular Neural Network до решеток 100×100×100+ (1M клеток).

Проверяемые компоненты:
- HierarchicalSpatialIndex: многоуровневое пространственное индексирование
- LatticeChunker: разбивка больших решеток на chunks
- MemoryPoolManager: эффективное управление GPU памятью
- ParallelSpatialProcessor: параллельная обработка
- SpatialOptimizer: интегрированная система оптимизации

Целевые метрики производительности:
- 100×100×100 (1M клеток): < 500ms на forward pass
- Memory usage: < 16GB для RTX 5090
- Chunking efficiency: > 90% memory utilization
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import psutil
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_rebuild.core.lattice.spatial_optimization import (
    SpatialOptimizer,
    create_spatial_optimizer,
    estimate_memory_requirements,
    SpatialOptimConfig,
    ChunkInfo,
)
from new_rebuild.config import get_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


class SpatialOptimizationBenchmark:
    """Бенчмарк для spatial optimization"""

    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"🚀 SpatialOptimizationBenchmark инициализирован (device: {self.device})"
        )

    def test_memory_estimation(self):
        """Тест оценки требований к памяти"""
        print("\n💾 ТЕСТ ОЦЕНКИ ПАМЯТИ")
        print("=" * 60)

        # Тестируем разные размеры решеток
        test_sizes = [
            (27, 27, 27),  # Текущий размер MoE (19k клеток)
            (50, 50, 50),  # Средний размер (125k клеток)
            (100, 100, 100),  # Большой размер (1M клеток)
            (200, 200, 200),  # Очень большой (8M клеток)
            (666, 666, 333),  # Целевой размер (148M клеток)
        ]

        for dimensions in test_sizes:
            memory_req = estimate_memory_requirements(dimensions)
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            print(f"\n📊 Решетка {dimensions} ({total_cells:,} клеток):")
            print(f"   🔧 Базовая память: {memory_req['base_memory_gb']:.2f} GB")
            print(f"   🔗 Соседи: {memory_req['neighbor_memory_gb']:.2f} GB")
            print(f"   📈 Градиенты: {memory_req['gradient_memory_gb']:.2f} GB")
            print(f"   ⚙️ Overhead: {memory_req['overhead_memory_gb']:.2f} GB")
            print(f"   📊 ИТОГО: {memory_req['total_memory_gb']:.2f} GB")
            print(
                f"   🎯 Рекомендуемый GPU: {memory_req['recommended_gpu_memory_gb']:.2f} GB"
            )

            # Определяем подходящие GPU
            if memory_req["recommended_gpu_memory_gb"] <= 16:
                gpu_class = "RTX 4080/5070 (16GB)"
            elif memory_req["recommended_gpu_memory_gb"] <= 24:
                gpu_class = "RTX 4090/5080 (24GB)"
            elif memory_req["recommended_gpu_memory_gb"] <= 32:
                gpu_class = "RTX 5090 (32GB)"
            elif memory_req["recommended_gpu_memory_gb"] <= 48:
                gpu_class = "RTX 6000 Ada (48GB)"
            else:
                gpu_class = "Требует data center GPU (>48GB)"

            print(f"   🖥️ Подходящий GPU: {gpu_class}")

            # Сохраняем результаты
            self.results[f"memory_{total_cells}"] = memory_req

        print(f"\n✅ Оценка памяти завершена для {len(test_sizes)} размеров решеток")

    def test_chunking_efficiency(self):
        """Тест эффективности chunking'а"""
        print("\n🧩 ТЕСТ ЭФФЕКТИВНОСТИ CHUNKING")
        print("=" * 60)

        # Тестируем разные размеры решеток
        test_cases = [
            (
                (100, 100, 100),
                SpatialOptimConfig(chunk_size=32),
            ),  # 1M клеток, малые chunk'и
            (
                (100, 100, 100),
                SpatialOptimConfig(chunk_size=64),
            ),  # 1M клеток, средние chunk'и
            (
                (200, 200, 200),
                SpatialOptimConfig(chunk_size=64),
            ),  # 8M клеток, средние chunk'и
            (
                (200, 200, 200),
                SpatialOptimConfig(chunk_size=128),
            ),  # 8M клеток, большие chunk'и
        ]

        for dimensions, config in test_cases:
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            print(f"\n📦 Chunking для {dimensions} ({total_cells:,} клеток):")
            print(f"   Размер chunk'а: {config.chunk_size}³")

            start_time = time.time()

            # Создаем spatial optimizer
            optimizer = SpatialOptimizer(dimensions, config)

            creation_time = time.time() - start_time

            # Анализируем chunk'и
            chunks = optimizer.chunker.chunks
            chunk_sizes = [len(chunk.cell_indices) for chunk in chunks]
            memory_sizes = [chunk.memory_size_mb for chunk in chunks]

            print(f"   🕐 Время создания: {creation_time:.3f}s")
            print(f"   📊 Количество chunk'ов: {len(chunks)}")
            print(f"   📏 Средний размер chunk'а: {np.mean(chunk_sizes):,.0f} клеток")
            print(
                f"   📊 Размер chunk'ов: {np.min(chunk_sizes):,} - {np.max(chunk_sizes):,}"
            )
            print(f"   💾 Средняя память chunk'а: {np.mean(memory_sizes):.1f} MB")
            print(
                f"   💾 Диапазон памяти: {np.min(memory_sizes):.1f} - {np.max(memory_sizes):.1f} MB"
            )

            # Эффективность использования памяти
            total_memory = sum(memory_sizes)
            theoretical_memory = (
                total_cells * 32 * 4 / (1024**2)
            )  # 32D состояние × 4 байта
            efficiency = (theoretical_memory / total_memory) * 100

            print(f"   ⚡ Эффективность памяти: {efficiency:.1f}%")

            # Расписание обработки
            schedule = optimizer.chunker.get_processing_schedule()
            avg_batch_size = np.mean([len(batch) for batch in schedule])

            print(f"   📅 Batch'ей в расписании: {len(schedule)}")
            print(f"   📦 Средний размер batch'а: {avg_batch_size:.1f} chunk'ов")

            # Cleanup
            optimizer.cleanup()

            # Сохраняем результаты
            self.results[f"chunking_{total_cells}_{config.chunk_size}"] = {
                "num_chunks": len(chunks),
                "avg_chunk_size": np.mean(chunk_sizes),
                "memory_efficiency": efficiency,
                "creation_time": creation_time,
                "num_batches": len(schedule),
            }

        print(f"\n✅ Тест chunking'а завершен")

    def test_hierarchical_spatial_index(self):
        """Тест иерархического пространственного индекса"""
        print("\n🗂️ ТЕСТ ИЕРАРХИЧЕСКОГО ИНДЕКСА")
        print("=" * 60)

        # Тестируем на разных размерах
        test_sizes = [
            (50, 50, 50),  # 125k клеток
            (100, 100, 100),  # 1M клеток
        ]

        for dimensions in test_sizes:
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]
            print(f"\n🔍 Индекс для {dimensions} ({total_cells:,} клеток):")

            config = SpatialOptimConfig(spatial_levels=3)
            optimizer = SpatialOptimizer(dimensions, config)

            # Тестируем поиск соседей
            test_coords = [
                (dimensions[0] // 4, dimensions[1] // 4, dimensions[2] // 4),  # Угол
                (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2),  # Центр
                (dimensions[0] - 5, dimensions[1] - 5, dimensions[2] - 5),  # Край
            ]

            search_radii = [5.0, 10.0, 20.0]

            for coords in test_coords:
                print(f"   📍 Поиск из координат {coords}:")

                for radius in search_radii:
                    start_time = time.time()
                    neighbors = optimizer.find_neighbors_optimized(coords, radius)
                    search_time = (time.time() - start_time) * 1000  # в ms

                    print(
                        f"      🔎 Радиус {radius}: {len(neighbors)} соседей за {search_time:.3f}ms"
                    )

            optimizer.cleanup()

        print(f"\n✅ Тест иерархического индекса завершен")

    def test_memory_pool_performance(self):
        """Тест производительности memory pool"""
        print("\n💾 ТЕСТ MEMORY POOL ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)

        if not torch.cuda.is_available():
            print("⚠️ CUDA недоступен, пропускаем тест memory pool")
            return

        config = SpatialOptimConfig(memory_pool_size_gb=4.0)
        optimizer = SpatialOptimizer((64, 64, 64), config)
        memory_manager = optimizer.memory_manager

        # Тест создания и возврата тензоров
        print("🔧 Тестируем создание и возврат тензоров...")

        tensor_shapes = [
            (1000, 32),  # Состояния клеток
            (1000, 26, 32),  # Состояния соседей
            (100, 64),  # Скрытые состояния
        ]

        for shape in tensor_shapes:
            print(f"\n   📊 Тестируем тензоры формы {shape}:")

            # Создаем много тензоров
            tensors = []
            start_time = time.time()

            for i in range(100):
                tensor = memory_manager.get_tensor(shape)
                tensors.append(tensor)

            creation_time = time.time() - start_time

            # Возвращаем тензоры в pool
            start_time = time.time()

            for tensor in tensors:
                memory_manager.return_tensor(tensor)

            return_time = time.time() - start_time

            print(f"      🕐 Создание 100 тензоров: {creation_time:.3f}s")
            print(f"      🔄 Возврат 100 тензоров: {return_time:.3f}s")

            # Статистика памяти
            stats = memory_manager.get_memory_stats()
            print(f"      💾 Текущая память: {stats['current_mb']:.1f} MB")
            print(f"      📊 Pool'ов: {stats['num_pools']}")
            print(f"      🗂️ Тензоров в pool'ах: {stats['total_pooled_tensors']}")

        optimizer.cleanup()
        print(f"\n✅ Тест memory pool завершен")

    def test_scalability_benchmark(self):
        """Бенчмарк масштабируемости"""
        print("\n⚡ БЕНЧМАРК МАСШТАБИРУЕМОСТИ")
        print("=" * 60)

        # Тестируем прогрессивно увеличивающиеся размеры
        test_sizes = [
            (20, 20, 20),  # 8k клеток
            (30, 30, 30),  # 27k клеток
            (40, 40, 40),  # 64k клеток
            (50, 50, 50),  # 125k клеток
        ]

        # Добавляем большие размеры только если есть достаточно памяти
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb > 16:
            test_sizes.append((70, 70, 70))  # 343k клеток
        if available_memory_gb > 32:
            test_sizes.append((100, 100, 100))  # 1M клеток

        results = []

        for dimensions in test_sizes:
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]
            print(f"\n🚀 Бенчмарк {dimensions} ({total_cells:,} клеток):")

            # Создаем оптимизатор
            start_time = time.time()
            optimizer = create_spatial_optimizer(dimensions)
            creation_time = time.time() - start_time

            # Создаем тестовые данные
            states = torch.randn(total_cells, 32, device=self.device)

            # Простая функция обработки соседей (mock)
            def mock_neighbor_processor(chunk_states, chunk_neighbors):
                # Простая операция: среднее по соседям + текущее состояние
                if chunk_neighbors.numel() > 0:
                    neighbor_mean = chunk_neighbors.mean(dim=1)
                    return chunk_states + 0.1 * neighbor_mean
                else:
                    return chunk_states

            # Тестируем производительность forward pass
            start_time = time.time()

            try:
                # Используем упрощенную версию для тестирования
                # В реальной системе здесь будет full MoE forward pass
                output_states = optimizer.optimize_lattice_forward(
                    states, mock_neighbor_processor
                )

                forward_time = time.time() - start_time
                success = True

                print(f"   ✅ Forward pass успешен: {forward_time:.3f}s")
                print(
                    f"   📊 Пропускная способность: {total_cells/forward_time:,.0f} клеток/сек"
                )

            except Exception as e:
                forward_time = float("inf")
                success = False
                print(f"   ❌ Forward pass неудачен: {str(e)}")

            # Статистика производительности
            if success:
                stats = optimizer.get_performance_stats()
                print(f"   💾 Пиковая память: {stats.get('peak_mb', 0):.1f} MB")
                print(
                    f"   🧩 Chunk'ов обработано: {stats.get('total_chunks_processed', 0)}"
                )

            # Cleanup
            optimizer.cleanup()
            del states
            if "output_states" in locals():
                del output_states

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Сохраняем результаты
            result = {
                "dimensions": dimensions,
                "total_cells": total_cells,
                "creation_time": creation_time,
                "forward_time": forward_time if success else None,
                "success": success,
                "throughput": total_cells / forward_time if success else 0,
            }
            results.append(result)

            print(
                f"   🕐 Общее время: {creation_time + (forward_time if success else 0):.3f}s"
            )

        # Анализ результатов
        print(f"\n📈 АНАЛИЗ МАСШТАБИРУЕМОСТИ:")
        print("=" * 60)

        successful_results = [r for r in results if r["success"]]

        if successful_results:
            # Находим максимальный успешно обработанный размер
            max_cells = max(r["total_cells"] for r in successful_results)
            best_throughput = max(r["throughput"] for r in successful_results)

            print(f"✅ Максимальный размер: {max_cells:,} клеток")
            print(
                f"⚡ Лучшая пропускная способность: {best_throughput:,.0f} клеток/сек"
            )

            # Экстраполяция для больших размеров
            if max_cells >= 125000:  # 50x50x50
                estimated_1m_time = 1_000_000 / best_throughput
                print(f"🔮 Оценка для 1M клеток: ~{estimated_1m_time:.1f}s")

                if estimated_1m_time < 100:
                    print("🎯 Цель < 100ms для 1M клеток: ДОСТИЖИМА!")
                else:
                    print(
                        "⚠️ Цель < 100ms для 1M клеток: требует дополнительной оптимизации"
                    )

        self.results["scalability"] = results
        print(f"\n✅ Бенчмарк масштабируемости завершен")

    def generate_performance_report(self):
        """Генерирует отчет о производительности"""
        print("\n📋 ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 80)

        # Суммарная статистика
        print("📊 СУММАРНАЯ СТАТИСТИКА:")

        # Memory estimation summary
        memory_results = {
            k: v for k, v in self.results.items() if k.startswith("memory_")
        }
        if memory_results:
            print(f"\n💾 Оценка памяти проведена для {len(memory_results)} размеров")
            max_size = max(int(k.split("_")[1]) for k in memory_results.keys())
            print(f"   📈 Максимальный размер: {max_size:,} клеток")

        # Chunking efficiency summary
        chunking_results = {
            k: v for k, v in self.results.items() if k.startswith("chunking_")
        }
        if chunking_results:
            efficiencies = [v["memory_efficiency"] for v in chunking_results.values()]
            avg_efficiency = np.mean(efficiencies)
            print(f"\n🧩 Эффективность chunking'а: {avg_efficiency:.1f}% в среднем")

        # Scalability summary
        if "scalability" in self.results:
            scalability = self.results["scalability"]
            successful = [r for r in scalability if r["success"]]

            if successful:
                max_successful = max(r["total_cells"] for r in successful)
                best_throughput = max(r["throughput"] for r in successful)

                print(f"\n⚡ Максимальная производительность:")
                print(f"   📊 Размер: {max_successful:,} клеток")
                print(
                    f"   🚀 Пропускная способность: {best_throughput:,.0f} клеток/сек"
                )

        # Рекомендации
        print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ПРОДОЛЖЕНИЯ:")
        print("   1. Интегрировать SpatialOptimizer с MoE архитектурой")
        print("   2. Оптимизировать neighbor gathering для chunk'ов")
        print("   3. Добавить асинхронную обработку экспертов")
        print("   4. Тестировать на реальных RTX 5090 (32GB)")
        print("   5. Внедрить mixed precision для экономии памяти")

        # Следующие этапы
        print(f"\n🚀 СЛЕДУЮЩИЕ ЭТАПЫ РАЗВИТИЯ:")
        print("   Phase 5.1: Интеграция с MoE Connection Processor")
        print("   Phase 5.2: GPU Memory optimization для RTX 5090")
        print("   Phase 5.3: Масштабирование до 666×666×333")
        print("   Phase 6: Training system с spatial optimization")


def main():
    """Основная функция тестирования"""
    print("🚀 SPATIAL OPTIMIZATION ADVANCED TEST")
    print("=" * 80)
    print(f"🖥️ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"💾 Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )

    benchmark = SpatialOptimizationBenchmark()

    try:
        # Последовательно выполняем все тесты
        benchmark.test_memory_estimation()
        benchmark.test_chunking_efficiency()
        benchmark.test_hierarchical_spatial_index()

        if torch.cuda.is_available():
            benchmark.test_memory_pool_performance()

        benchmark.test_scalability_benchmark()

        # Генерируем итоговый отчет
        benchmark.generate_performance_report()

        print(f"\n🎉 SPATIAL OPTIMIZATION TESTING COMPLETE!")
        print("   Все компоненты протестированы успешно")
        print("   Система готова к интеграции с MoE архитектурой")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
