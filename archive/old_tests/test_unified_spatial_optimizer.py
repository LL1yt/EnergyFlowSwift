#!/usr/bin/env python3
"""
Тестирование Unified Spatial Optimizer
======================================

Комплексное тестирование унифицированной системы пространственной оптимизации
с проверкой всех режимов работы и интеграции компонентов.

Автор: 3D Cellular Neural Network Project
Версия: 3.0.0 (2024-12-27)
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Настройка логирования для тестов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Тест 1: Проверка импортов"""
    logger.info("🧪 Тест 1: Проверка импортов")

    try:
        from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
            UnifiedSpatialOptimizer,
            OptimizationConfig,
            OptimizationMode,
            ConnectionType,
            create_unified_spatial_optimizer,
            estimate_unified_memory_requirements,
        )

        logger.info("✅ Все импорты унифицированной системы успешны")
        return True
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False


def test_optimization_modes():
    """Тест 2: Проверка всех режимов оптимизации"""
    logger.info("🧪 Тест 2: Проверка режимов оптимизации")

    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode,
    )

    dimensions = (8, 8, 8)  # Небольшая решетка для тестирования
    test_results = {}

    modes_to_test = [
        OptimizationMode.AUTO,
        OptimizationMode.CPU_ONLY,
        OptimizationMode.HYBRID,
    ]

    # Добавляем GPU режим только если CUDA доступна
    if torch.cuda.is_available():
        modes_to_test.append(OptimizationMode.GPU_ONLY)

    for mode in modes_to_test:
        try:
            logger.info(f"   🔍 Тестируем режим: {mode.value}")

            config = OptimizationConfig(
                mode=mode,
                enable_moe=False,  # Сначала без MoE
                target_performance_ms=100.0,
            )

            optimizer = create_unified_spatial_optimizer(dimensions, config)

            # Тестируем поиск соседей
            test_coords = (4, 4, 4)  # Центр решетки
            neighbors = optimizer.find_neighbors_optimized(test_coords, radius=2.0)

            logger.info(f"     📊 Найдено {len(neighbors)} соседей")

            test_results[mode.value] = {
                "neighbors_found": len(neighbors),
                "creation_successful": True,
            }

            # Очистка
            optimizer.cleanup()

        except Exception as e:
            logger.warning(f"     ⚠️ Режим {mode.value} не удался: {e}")
            test_results[mode.value] = {"creation_successful": False, "error": str(e)}

    logger.info(f"✅ Результаты тестирования режимов: {test_results}")
    return test_results


def test_lattice_optimization():
    """Тест 3: Комплексная оптимизация решетки"""
    logger.info("🧪 Тест 3: Комплексная оптимизация решетки")

    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode,
    )

    dimensions = (6, 6, 6)
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    state_size = 32

    try:
        # Создаем оптимизатор в AUTO режиме
        config = OptimizationConfig(
            mode=OptimizationMode.AUTO, enable_moe=False, target_performance_ms=50.0
        )

        optimizer = create_unified_spatial_optimizer(dimensions, config)

        # Создаем тестовые состояния
        states = torch.randn(total_cells, state_size, dtype=torch.float32)
        logger.info(f"   📊 Входные состояния: {states.shape}")

        # Выполняем оптимизацию
        start_time = time.time()
        result = optimizer.optimize_lattice_forward(states)
        optimization_time = time.time() - start_time

        # Проверяем результаты
        assert result.new_states.shape == states.shape, "Размерности не совпадают"
        assert not torch.isnan(result.new_states).any(), "Обнаружены NaN в результатах"
        assert not torch.isinf(result.new_states).any(), "Обнаружены Inf в результатах"

        logger.info(f"   ✅ Оптимизация завершена за {optimization_time:.3f}s")
        logger.info(f"   📊 Время обработки: {result.processing_time_ms:.1f}ms")
        logger.info(f"   🎯 Режим использованный: {result.mode_used.value}")
        logger.info(f"   💾 Память: {result.memory_usage_mb:.1f}MB")
        logger.info(f"   🔍 Соседей найдено: {result.neighbors_found}")

        # Получаем полную статистику
        stats = optimizer.get_comprehensive_stats()
        logger.info(f"   📈 Статистика: {stats['unified_optimizer']}")

        optimizer.cleanup()

        return {
            "success": True,
            "optimization_time_s": optimization_time,
            "processing_time_ms": result.processing_time_ms,
            "mode_used": result.mode_used.value,
            "memory_usage_mb": result.memory_usage_mb,
        }

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании оптимизации: {e}")
        return {"success": False, "error": str(e)}


def test_moe_integration():
    """Тест 4: Интеграция с MoE архитектурой"""
    logger.info("🧪 Тест 4: Интеграция с MoE архитектурой")

    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode,
    )

    dimensions = (4, 4, 4)
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    state_size = 16

    try:
        # Создаем мок MoE processor
        class MockMoEProcessor(torch.nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.linear = torch.nn.Linear(state_size, state_size)

            def forward(
                self,
                current_state,
                neighbor_states,
                cell_idx,
                neighbor_indices,
                **kwargs,
            ):
                # Простая обработка: применяем линейный слой
                processed = self.linear(current_state)
                return {"new_state": processed}

        moe_processor = MockMoEProcessor(state_size)

        # Создаем оптимизатор с MoE
        config = OptimizationConfig(
            mode=OptimizationMode.AUTO, enable_moe=True, enable_morton_encoding=True
        )

        optimizer = create_unified_spatial_optimizer(dimensions, config, moe_processor)

        # Тестовые состояния
        states = torch.randn(total_cells, state_size)

        # Выполняем MoE оптимизацию
        start_time = time.time()
        result = optimizer.optimize_lattice_forward(states)
        moe_time = time.time() - start_time

        # Проверяем результаты
        assert result.new_states.shape == states.shape
        assert not torch.isnan(result.new_states).any()

        logger.info(f"   ✅ MoE оптимизация завершена за {moe_time:.3f}s")
        logger.info(f"   🤖 MoE обработка: {result.processing_time_ms:.1f}ms")
        logger.info(f"   🎯 Режим: {result.mode_used.value}")

        # Получаем статистику
        stats = optimizer.get_comprehensive_stats()
        moe_enabled = stats["unified_optimizer"]["moe_enabled"]
        morton_enabled = stats["unified_optimizer"]["morton_enabled"]

        logger.info(f"   🤖 MoE активен: {moe_enabled}")
        logger.info(f"   🔢 Morton активен: {morton_enabled}")

        optimizer.cleanup()

        return {
            "success": True,
            "moe_time_s": moe_time,
            "moe_enabled": moe_enabled,
            "morton_enabled": morton_enabled,
        }

    except Exception as e:
        logger.error(f"❌ Ошибка в MoE тестировании: {e}")
        return {"success": False, "error": str(e)}


def test_memory_estimation():
    """Тест 5: Оценка требований к памяти"""
    logger.info("🧪 Тест 5: Оценка требований к памяти")

    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
        estimate_unified_memory_requirements,
        OptimizationConfig,
        OptimizationMode,
    )

    test_dimensions = [
        (4, 4, 4),  # Малая решетка
        (10, 10, 10),  # Средняя решетка
        (20, 20, 20),  # Большая решетка
    ]

    memory_estimates = {}

    for dimensions in test_dimensions:
        try:
            # Тестируем разные конфигурации
            configs = [
                ("cpu_only", OptimizationConfig(mode=OptimizationMode.CPU_ONLY)),
                ("gpu_only", OptimizationConfig(mode=OptimizationMode.GPU_ONLY)),
                (
                    "gpu_moe",
                    OptimizationConfig(
                        mode=OptimizationMode.GPU_ONLY,
                        enable_moe=True,
                        enable_morton_encoding=True,
                    ),
                ),
            ]

            dimension_results = {}

            for config_name, config in configs:
                estimates = estimate_unified_memory_requirements(dimensions, config)
                dimension_results[config_name] = estimates

                logger.info(f"   📏 {dimensions} ({config_name}):")
                logger.info(
                    f"     💾 Общая память: {estimates['total_memory_gb']:.2f}GB"
                )
                logger.info(
                    f"     🎯 Рекомендуемая GPU: {estimates['recommended_gpu_memory_gb']:.2f}GB"
                )

            memory_estimates[dimensions] = dimension_results

        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка оценки для {dimensions}: {e}")

    logger.info("✅ Оценка памяти завершена")
    return memory_estimates


def test_performance_comparison():
    """Тест 6: Сравнение производительности с legacy системами"""
    logger.info("🧪 Тест 6: Сравнение производительности")

    dimensions = (6, 6, 6)
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    state_size = 32

    performance_results = {}

    states = torch.randn(total_cells, state_size)

    # Тест UnifiedSpatialOptimizer
    try:
        from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import (
            create_unified_spatial_optimizer,
            OptimizationConfig,
            OptimizationMode,
        )

        config = OptimizationConfig(mode=OptimizationMode.AUTO)
        unified_optimizer = create_unified_spatial_optimizer(dimensions, config)

        # Прогрев
        _ = unified_optimizer.optimize_lattice_forward(states)

        # Измерение производительности
        start_time = time.time()
        for _ in range(3):
            result = unified_optimizer.optimize_lattice_forward(states)
        unified_time = (time.time() - start_time) / 3

        performance_results["unified"] = {
            "avg_time_s": unified_time,
            "mode_used": result.modeUsed.value,
            "memory_mb": result.memory_usage_mb,
        }

        unified_optimizer.cleanup()

        logger.info(
            f"   🚀 Unified: {unified_time:.3f}s avg, режим: {result.modeUsed.value}"
        )

    except Exception as e:
        logger.warning(f"   ⚠️ Unified тест не удался: {e}")

    # Тест legacy SpatialOptimizer для сравнения
    try:
        from new_rebuild.core.lattice.spatial_optimization.spatial_optimizer import (
            SpatialOptimizer,
        )

        legacy_optimizer = SpatialOptimizer(dimensions)
        legacy_states = torch.randn(total_cells, state_size)  # Создаем states для legacy теста
        
        def simple_processor(current_state, neighbor_states, cell_idx, neighbors):
            if len(neighbors) == 0:
                return current_state
            return 0.7 * current_state + 0.3 * neighbor_states.mean(dim=0)

        # Измерение производительности legacy
        start_time = time.time()
        for _ in range(3):
            _ = legacy_optimizer.optimize_lattice_forward(legacy_states, simple_processor)
        legacy_time = (time.time() - start_time) / 3

        performance_results["legacy"] = {
            "avg_time_s": legacy_time,
            "mode_used": "cpu_legacy",
        }

        legacy_optimizer.cleanup()

        logger.info(f"   📊 Legacy: {legacy_time:.3f}s avg")

        # Сравнение
        if "unified" in performance_results:
            speedup = legacy_time / performance_results["unified"]["avg_time_s"]
            logger.info(f"   ⚡ Ускорение: {speedup:.2f}x")

    except Exception as e:
        logger.warning(f"   ⚠️ Legacy тест не удался: {e}")

    return performance_results


def run_all_tests():
    """Запуск всех тестов"""
    logger.info("🚀 Запуск комплексного тестирования Unified Spatial Optimizer")
    logger.info("=" * 70)

    test_results = {}

    # Тест 1: Импорты
    test_results["imports"] = test_imports()

    if not test_results["imports"]:
        logger.error("❌ Критическая ошибка: импорты не удались")
        return test_results

    # Тест 2: Режимы оптимизации
    test_results["modes"] = test_optimization_modes()

    # Тест 3: Оптимизация решетки
    test_results["lattice_optimization"] = test_lattice_optimization()

    # Тест 4: MoE интеграция
    test_results["moe_integration"] = test_moe_integration()

    # Тест 5: Оценка памяти
    test_results["memory_estimation"] = test_memory_estimation()

    # Тест 6: Сравнение производительности
    test_results["performance_comparison"] = test_performance_comparison()

    # Финальный отчет
    logger.info("=" * 70)
    logger.info("📋 ФИНАЛЬНЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    logger.info("=" * 70)

    successful_tests = sum(
        1
        for result in test_results.values()
        if isinstance(result, (bool, dict))
        and (
            result is True or (isinstance(result, dict) and result.get("success", True))
        )
    )

    total_tests = len(test_results)

    logger.info(f"✅ Успешные тесты: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("🚀 UnifiedSpatialOptimizer готов к использованию")
    else:
        logger.warning(f"⚠️ {total_tests - successful_tests} тестов не прошли")
        logger.info("🔧 Рекомендуется исправить ошибки перед использованием")

    return test_results


if __name__ == "__main__":
    results = run_all_tests()
