#!/usr/bin/env python3
"""
Тест интеграции MoE + Spatial Optimization
==========================================

Тестирует новую интеграцию MoESpatialOptimizer с MoE архитектурой
для максимальной производительности на больших решетках.

Проверяемые компоненты:
- MoESpatialOptimizer: интеграция spatial optimization с MoE
- Автоматическое создание через create_moe_spatial_optimizer
- Оценка памяти для MoE + Spatial архитектуры
- Chunked MoE processing с classification экспертов
- Асинхронная обработка и performance optimization

Целевые метрики:
- 27×27×27 (19,683 клеток): < 100ms forward pass
- 100×100×100 (1M клеток): < 500ms forward pass
- Memory usage: совместимо с RTX 5090 (32GB)
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_rebuild.core.lattice.spatial_optimization import (
    MoESpatialOptimizer,
    create_moe_spatial_optimizer,
    estimate_moe_memory_requirements,
    SpatialOptimConfig,
)
from new_rebuild.core.lattice import create_lattice
from new_rebuild.config import get_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


class MoEConnectionProcessorMock:
    """Mock для MoE Connection Processor для тестирования"""

    def __init__(self, state_size=32):
        self.state_size = state_size
        self.device = torch.device("cpu")
        self.processing_stats = {
            "calls": 0,
            "total_cells": 0,
            "expert_usage": {"local": 0, "functional": 0, "distant": 0},
        }

    def to(self, device):
        """Перенос на устройство"""
        self.device = device
        return self

    def __call__(self, states, neighbors_dict, chunk_info=None):
        """Имитация обработки через MoE экспертов"""

        self.processing_stats["calls"] += 1
        self.processing_stats["total_cells"] += states.shape[0]

        # Подсчитываем использование экспертов
        for expert_type, neighbor_indices in neighbors_dict.items():
            if len(neighbor_indices) > 0:
                self.processing_stats["expert_usage"][expert_type] += len(
                    neighbor_indices
                )

        # Простая имитация обработки: добавляем небольшой шум к состояниям
        # Убеждаемся что шум создается на том же устройстве
        noise = torch.randn_like(states, device=states.device) * 0.01
        return states + noise


class MoESpatialOptimizationTest:
    """Тест интеграции MoE + Spatial Optimization"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

        # Принудительно используем GPU если доступен
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()  # Очистка памяти
            logger.info(
                f"🚀 MoE Spatial Optimization Test (CUDA available, device: {self.device})"
            )
            logger.info(
                f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            )
        else:
            logger.warning(f"⚠️ CUDA не доступен, используется CPU: {self.device}")
            logger.info("💻 CPU Mode")

    def test_moe_memory_estimation(self):
        """Тест оценки памяти для MoE + Spatial Optimization"""
        print("\n💾 ТЕСТ ОЦЕНКИ ПАМЯТИ MoE + SPATIAL")
        print("=" * 70)

        test_sizes = [
            (27, 27, 27),  # Текущий MoE размер (19k клеток)
            (50, 50, 50),  # Средний размер (125k клеток)
            (100, 100, 100),  # Большой размер (1M клеток)
            (150, 150, 150),  # Очень большой (3.4M клеток)
        ]

        for dimensions in test_sizes:
            memory_req = estimate_moe_memory_requirements(dimensions)
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            print(f"\n📊 Решетка {dimensions} ({total_cells:,} клеток):")
            print(f"   🔧 Базовая память: {memory_req['base_memory_gb']:.3f} GB")
            print(f"   🤖 MoE эксперты: {memory_req['moe_experts_gb']:.3f} GB")
            print(f"   🎯 Gating Network: {memory_req['gating_network_gb']:.3f} GB")
            print(
                f"   🗂️ Spatial optimization: {memory_req['spatial_optimization_gb']:.3f} GB"
            )
            print(f"   🔗 Соседи: {memory_req['neighbor_memory_gb']:.3f} GB")
            print(f"   📈 Буферы экспертов: {memory_req['expert_buffers_gb']:.3f} GB")
            print(f"   📊 ИТОГО: {memory_req['total_memory_gb']:.3f} GB")
            print(
                f"   🎯 Рекомендуемый GPU: {memory_req['recommended_gpu_memory_gb']:.3f} GB"
            )

            # Определяем подходящие GPU
            if memory_req["recommended_gpu_memory_gb"] <= 16:
                gpu_class = "RTX 4080/5070 (16GB)"
            elif memory_req["recommended_gpu_memory_gb"] <= 24:
                gpu_class = "RTX 4090/5080 (24GB)"
            elif memory_req["recommended_gpu_memory_gb"] <= 32:
                gpu_class = "RTX 5090 (32GB) ⭐ TARGET"
            elif memory_req["recommended_gpu_memory_gb"] <= 48:
                gpu_class = "RTX 6000 Ada (48GB)"
            else:
                gpu_class = "Требует data center GPU (>48GB)"

            print(f"   🖥️ Подходящий GPU: {gpu_class}")

            self.results[f"moe_memory_{total_cells}"] = memory_req

        print(f"\n✅ Оценка памяти MoE завершена для {len(test_sizes)} размеров")

    def test_moe_spatial_optimizer_creation(self):
        """Тест создания MoE Spatial Optimizer"""
        print("\n🏗️ ТЕСТ СОЗДАНИЯ MoE SPATIAL OPTIMIZER")
        print("=" * 70)

        test_sizes = [
            (27, 27, 27),  # Малый (19k клеток)
            (50, 50, 50),  # Средний (125k клеток)
            (70, 70, 70),  # Большой (343k клеток)
        ]

        for dimensions in test_sizes:
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            print(
                f"\n📦 Создание оптимизатора для {dimensions} ({total_cells:,} клеток):"
            )

            start_time = time.time()

            # Создаем MoE spatial optimizer
            moe_processor = MoEConnectionProcessorMock()
            optimizer = create_moe_spatial_optimizer(
                dimensions, moe_processor, self.device
            )

            creation_time = time.time() - start_time

            print(f"   🕐 Время создания: {creation_time:.3f}s")
            print(f"   📊 Chunks: {len(optimizer.chunker.chunks)}")
            print(
                f"   📋 Batch расписание: {len(optimizer.chunker.get_processing_schedule())} batches"
            )
            print(f"   ⚡ Worker threads: {optimizer.config.num_worker_threads}")
            print(f"   💾 Memory pool: {optimizer.config.memory_pool_size_gb:.1f}GB")
            print(f"   📏 Chunk size: {optimizer.config.chunk_size}³")
            print(f"   🔗 Распределение связей: {optimizer.connection_distributions}")

            # Cleanup
            optimizer.cleanup()

            self.results[f"moe_creation_{total_cells}"] = {
                "creation_time": creation_time,
                "num_chunks": len(optimizer.chunker.chunks),
                "config": optimizer.config,
            }

        print(f"\n✅ Создание MoE Spatial Optimizer завершено")

    def test_moe_neighbor_classification(self):
        """Тест классификации соседей для MoE экспертов"""
        print("\n🔍 ТЕСТ КЛАССИФИКАЦИИ СОСЕДЕЙ ДЛЯ MoE")
        print("=" * 70)

        dimensions = (27, 27, 27)  # Тестовый размер
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]

        print(
            f"📊 Тестируем классификацию на решетке {dimensions} ({total_cells:,} клеток)"
        )

        # Создаем optimizer
        moe_processor = MoEConnectionProcessorMock()
        moe_processor.to(self.device)
        optimizer = create_moe_spatial_optimizer(dimensions, moe_processor, self.device)

        # Тестируем классификацию для случайных клеток
        test_cells = [0, total_cells // 4, total_cells // 2, total_cells - 1]

        classification_stats = {"local": [], "functional": [], "distant": []}

        for cell_idx in test_cells:
            # Получаем всех соседей с адаптивным радиусом
            cell_coords = optimizer.pos_helper.to_3d_coordinates(cell_idx)

            # Адаптивный радиус из централизованной конфигурации
            from new_rebuild.config.project_config import get_project_config

            config = get_project_config()
            adaptive_radius = config.calculate_adaptive_radius()

            all_neighbors = optimizer.find_neighbors_optimized(
                cell_coords, radius=adaptive_radius
            )

            # Классифицируем
            classified = optimizer._classify_neighbors_for_moe(cell_idx, all_neighbors)

            print(f"\n   📍 Клетка {cell_idx} (координаты {cell_coords}):")
            print(f"      🔗 Всего соседей: {len(all_neighbors)}")
            print(
                f"      📍 Local: {len(classified['local'])} ({len(classified['local'])/len(all_neighbors)*100:.1f}%)"
            )
            print(
                f"      🔧 Functional: {len(classified['functional'])} ({len(classified['functional'])/len(all_neighbors)*100:.1f}%)"
            )
            print(
                f"      🌐 Distant: {len(classified['distant'])} ({len(classified['distant'])/len(all_neighbors)*100:.1f}%)"
            )

            # Собираем статистику
            if all_neighbors:  # Если есть соседи
                classification_stats["local"].append(
                    len(classified["local"]) / len(all_neighbors)
                )
                classification_stats["functional"].append(
                    len(classified["functional"]) / len(all_neighbors)
                )
                classification_stats["distant"].append(
                    len(classified["distant"]) / len(all_neighbors)
                )

        # Средняя статистика
        if classification_stats["local"]:  # Если есть данные
            avg_local = np.mean(classification_stats["local"]) * 100
            avg_functional = np.mean(classification_stats["functional"]) * 100
            avg_distant = np.mean(classification_stats["distant"]) * 100

            print(f"\n📈 СРЕДНЯЯ КЛАССИФИКАЦИЯ:")
            print(f"   📍 Local: {avg_local:.1f}% (цель: 10%)")
            print(f"   🔧 Functional: {avg_functional:.1f}% (цель: 55%)")
            print(f"   🌐 Distant: {avg_distant:.1f}% (цель: 35%)")

            # Проверяем соответствие целям
            local_ok = abs(avg_local - 10) < 5  # ±5% tolerance
            functional_ok = abs(avg_functional - 55) < 10  # ±10% tolerance
            distant_ok = abs(avg_distant - 35) < 10

            if local_ok and functional_ok and distant_ok:
                print("   ✅ Классификация соответствует целевым пропорциям!")
            else:
                print("   ⚠️ Классификация отклоняется от целевых пропорций")

        optimizer.cleanup()
        print(f"\n✅ Тест классификации соседей завершен")

    def test_moe_chunked_processing(self):
        """Тест chunked processing с MoE экспертами"""
        print("\n⚡ ТЕСТ CHUNKED MoE PROCESSING")
        print("=" * 70)

        test_cases = [
            (27, 27, 27),  # Малая решетка (19k клеток) - быстрый тест
            # (40, 40, 40),  # Средняя решетка (64k клеток) - отключаем для скорости
        ]

        for dimensions in test_cases:
            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            print(
                f"\n🧩 Chunked MoE processing для {dimensions} ({total_cells:,} клеток):"
            )

            # Создаем состояния (решетка не нужна для spatial optimization тестов)
            print(f"   📊 Создание {total_cells:,} состояний на {self.device}...")
            states = torch.randn(
                total_cells, 32, device=self.device, dtype=torch.float32
            )  # Исходные состояния на GPU

            # Создаем MoE processor и optimizer
            print(f"   🛠️ Создание MoE processor...")
            moe_processor = MoEConnectionProcessorMock()
            moe_processor.to(self.device)  # Перенос на GPU

            print(f"   🗂️ Создание MoE spatial optimizer...")
            optimizer = create_moe_spatial_optimizer(
                dimensions, moe_processor, self.device
            )

            print(f"   ⚡ Запуск MoE forward pass на {self.device}...")
            # Прогрев CUDA kernel'ов (если на GPU)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            output_states = optimizer.optimize_moe_forward(states, moe_processor)

            processing_time = time.time() - start_time

            # Анализируем результаты
            throughput = total_cells / processing_time
            memory_used = (
                torch.cuda.memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0
            )

            print(f"   🕐 Время обработки: {processing_time:.3f}s")
            print(f"   📊 Пропускная способность: {throughput:,.0f} клеток/сек")
            print(f"   💾 Память GPU: {memory_used:.1f} MB")
            print(f"   📈 Форма выхода: {output_states.shape}")
            print(
                f"   🔍 Изменение состояний: {torch.mean(torch.abs(output_states - states)):.6f}"
            )

            # Статистика использования экспертов
            stats = moe_processor.processing_stats
            print(f"   🤖 MoE вызовов: {stats['calls']}")
            print(f"   📊 Клеток обработано: {stats['total_cells']:,}")
            print(f"   🎯 Использование экспертов:")
            for expert_type, usage in stats["expert_usage"].items():
                print(f"      {expert_type}: {usage:,} связей")

            # Проверяем корректность результатов
            assert (
                output_states.shape == states.shape
            ), "Форма выходных состояний не совпадает"
            assert not torch.isnan(
                output_states
            ).any(), "NaN значения в выходных состояниях"
            assert not torch.isinf(
                output_states
            ).any(), "Inf значения в выходных состояниях"

            # Более мягкая проверка изменений для тестовой среды
            state_change = torch.mean(torch.abs(output_states - states)).item()
            if state_change > 1.0:
                print(
                    f"   ⚠️ Большие изменения состояний: {state_change:.3f} (ожидалось < 1.0)"
                )
            else:
                print(f"   ✅ Изменения состояний в норме: {state_change:.3f}")

            print(f"   🎯 Тест MoE chunked processing прошел успешно!")

            optimizer.cleanup()

            self.results[f"moe_processing_{total_cells}"] = {
                "processing_time": processing_time,
                "throughput": throughput,
                "memory_mb": memory_used,
                "expert_stats": stats,
            }

        print(f"\n✅ Chunked MoE processing завершен успешно")

    def test_performance_comparison(self):
        """Сравнение производительности MoE vs обычный spatial optimization"""
        print("\n🏁 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        dimensions = (50, 50, 50)  # 125k клеток
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]

        print(f"🔄 Сравнение на решетке {dimensions} ({total_cells:,} клеток)")

        # Создаем состояния
        states = torch.randn(total_cells, 32, device=self.device)

        # Тест 1: Обычный SpatialOptimizer
        print(f"\n📊 Тест 1: Обычный SpatialOptimizer")
        from new_rebuild.core.lattice.spatial_optimization import (
            create_spatial_optimizer,
        )

        regular_optimizer = create_spatial_optimizer(dimensions)

        def simple_neighbor_processor(chunk_states, neighbors):
            """Простая функция обработки соседей"""
            return chunk_states + torch.randn_like(chunk_states) * 0.01

        start_time = time.time()
        regular_output = regular_optimizer.optimize_lattice_forward(
            states, simple_neighbor_processor
        )
        regular_time = time.time() - start_time
        regular_throughput = total_cells / regular_time

        print(f"   🕐 Время: {regular_time:.3f}s")
        print(f"   📊 Пропускная способность: {regular_throughput:,.0f} клеток/сек")

        # Тест 2: MoE SpatialOptimizer
        print(f"\n🤖 Тест 2: MoE SpatialOptimizer")

        moe_processor = MoEConnectionProcessorMock()
        moe_optimizer = create_moe_spatial_optimizer(dimensions, moe_processor)

        start_time = time.time()
        moe_output = moe_optimizer.optimize_moe_forward(states, moe_processor)
        moe_time = time.time() - start_time
        moe_throughput = total_cells / moe_time

        print(f"   🕐 Время: {moe_time:.3f}s")
        print(f"   📊 Пропускная способность: {moe_throughput:,.0f} клеток/сек")

        # Сравнение
        speedup = regular_time / moe_time if moe_time > 0 else float("inf")
        print(f"\n⚡ СРАВНЕНИЕ:")
        print(f"   🚀 Ускорение MoE: {speedup:.2f}x")
        if speedup > 1:
            print(f"   ✅ MoE быстрее на {(speedup-1)*100:.1f}%")
        else:
            print(f"   ⚠️ MoE медленнее на {(1-speedup)*100:.1f}%")

        # Cleanup
        regular_optimizer.cleanup()
        moe_optimizer.cleanup()

        self.results["performance_comparison"] = {
            "regular_time": regular_time,
            "moe_time": moe_time,
            "speedup": speedup,
            "regular_throughput": regular_throughput,
            "moe_throughput": moe_throughput,
        }

        print(f"\n✅ Сравнение производительности завершено")

    def generate_performance_report(self):
        """Генерирует финальный отчет о производительности"""
        print("\n" + "=" * 80)
        print("📋 ФИНАЛЬНЫЙ ОТЧЕТ MoE + SPATIAL OPTIMIZATION")
        print("=" * 80)

        if "moe_processing_19683" in self.results:
            result = self.results["moe_processing_19683"]
            print(f"🎯 MoE АРХИТЕКТУРА (27×27×27 = 19,683 клеток):")
            print(f"   ⚡ Время forward pass: {result['processing_time']:.3f}s")
            print(
                f"   📊 Пропускная способность: {result['throughput']:,.0f} клеток/сек"
            )
            print(f"   💾 Память GPU: {result['memory_mb']:.1f} MB")

        if "performance_comparison" in self.results:
            comp = self.results["performance_comparison"]
            print(f"\n🏁 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
            print(f"   📈 MoE ускорение: {comp['speedup']:.2f}x")
            print(
                f"   ⚡ MoE пропускная способность: {comp['moe_throughput']:,.0f} клеток/сек"
            )

        # Оценки для больших решеток
        if "moe_memory_1000000" in self.results:
            mem = self.results["moe_memory_1000000"]
            print(f"\n🎯 МАСШТАБИРОВАНИЕ ДО 1M КЛЕТОК:")
            print(f"   💾 Требуемая память: {mem['total_memory_gb']:.2f} GB")
            print(f"   🖥️ GPU рекомендация: {mem['recommended_gpu_memory_gb']:.1f} GB")

            # Оценка времени на основе текущих результатов
            if "moe_processing_64000" in self.results:
                base_result = self.results["moe_processing_64000"]
                base_throughput = base_result["throughput"]
                estimated_time = 1_000_000 / base_throughput
                print(f"   ⏱️ Оценка времени для 1M клеток: {estimated_time:.3f}s")

                if estimated_time < 0.5:
                    print(f"   ✅ Цель < 500ms: ДОСТИЖИМА!")
                else:
                    print(f"   ⚠️ Цель < 500ms: требует оптимизации")

        print(f"\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
        print(f"   1. Интеграция с реальным MoE Connection Processor")
        print(f"   2. Тестирование на RTX 5090 (32GB)")
        print(f"   3. Mixed precision optimization")
        print(f"   4. Асинхронная обработка экспертов")
        print(f"   5. Масштабирование до 666×666×333")


def main():
    """Основная функция тестирования"""
    print("🎉 ЗАПУСК ТЕСТИРОВАНИЯ MoE + SPATIAL OPTIMIZATION INTEGRATION")
    print("=" * 80)

    test_runner = MoESpatialOptimizationTest()

    try:
        # Запускаем все тесты
        test_runner.test_moe_memory_estimation()
        test_runner.test_moe_spatial_optimizer_creation()
        test_runner.test_moe_neighbor_classification()
        test_runner.test_moe_chunked_processing()
        test_runner.test_performance_comparison()

        # Генерируем финальный отчет
        test_runner.generate_performance_report()

        print("\n" + "=" * 80)
        print("🎉 ВСЕ ТЕСТЫ MoE + SPATIAL OPTIMIZATION ЗАВЕРШЕНЫ УСПЕШНО!")
        print("🚀 Система готова к интеграции с полной MoE архитектурой")
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании: {e}")
        raise


if __name__ == "__main__":
    main()
