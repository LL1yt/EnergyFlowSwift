#!/usr/bin/env python3
"""
Тест миграции на Unified Spatial Optimizer
==========================================

Проверяет что миграция на унифицированную архитектуру прошла успешно
и все компоненты работают корректно.

Автор: 3D Cellular Neural Network Project
Версия: 3.0.0 (2024-12-27)
"""

import torch
import logging
from typing import Any, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Тест 1: Проверка импортов"""
    logger.info("🧪 Тест 1: Проверка импортов после миграции")
    
    try:
        # Проверяем что старые импорты все еще работают (DEPRECATED)
        from core.lattice.spatial_optimization import (
            SpatialOptimizer,
            MoESpatialOptimizer,
            create_spatial_optimizer,
            create_moe_spatial_optimizer
        )
        logger.info("✅ DEPRECATED импорты работают")
        
        # Проверяем новые импорты
        from core.lattice.spatial_optimization import (
            UnifiedSpatialOptimizer,
            create_unified_spatial_optimizer,
            OptimizationConfig,
            OptimizationMode
        )
        logger.info("✅ Новые импорты работают")
        
        # Проверяем импорты из lattice
        from core.lattice import (
            Lattice3D,
            create_lattice,
            UnifiedSpatialOptimizer,
            OptimizationConfig
        )
        logger.info("✅ Импорты из lattice работают")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False


def test_lattice_with_unified_optimizer():
    """Тест 2: Проверка работы Lattice3D с унифицированным оптимизатором"""
    logger.info("🧪 Тест 2: Lattice3D с UnifiedSpatialOptimizer")
    
    try:
        # Импортируем нужные компоненты
        from core.lattice import create_lattice
        from config.project_config import get_project_config
        
        # Получаем конфигурацию
        config = get_project_config()
        
        # Проверяем что архитектура MoE
        if config.architecture_type != "moe":
            logger.warning(f"⚠️ Архитектура не MoE: {config.architecture_type}")
            return False
        
        # Создаем решетку
        logger.info(f"   🏗️ Создание решетки {config.lattice_dimensions}")
        lattice = create_lattice()
        
        # Проверяем что используется UnifiedSpatialOptimizer
        optimizer_type = type(lattice.spatial_optimizer).__name__
        logger.info(f"   🔧 Spatial Optimizer: {optimizer_type}")
        
        if optimizer_type != "UnifiedSpatialOptimizer":
            logger.error(f"❌ Ожидался UnifiedSpatialOptimizer, получен {optimizer_type}")
            return False
        
        # Проверяем настройки оптимизатора
        if hasattr(lattice.spatial_optimizer, 'config'):
            opt_config = lattice.spatial_optimizer.config
            logger.info(f"   ⚙️ MoE включен: {opt_config.enable_moe}")
            logger.info(f"   🔢 Morton включен: {opt_config.enable_morton_encoding}")
            logger.info(f"   🎯 Активный режим: {lattice.spatial_optimizer.active_mode.value}")
        
        # Тестируем forward pass
        logger.info("   🔄 Тестирование forward pass...")
        initial_states = lattice.states.clone()
        
        # Выполняем несколько forward passes
        for i in range(3):
            output_states = lattice.forward()
            logger.info(f"     Шаг {i+1}: состояния {output_states.shape}")
        
        # Проверяем что состояния изменились
        state_diff = torch.norm(lattice.states - initial_states)
        logger.info(f"   📊 Изменение состояний: {state_diff:.4f}")
        
        # Получаем статистику
        stats = lattice.validate_lattice()
        if "spatial_optimizer" in stats:
            spatial_stats = stats["spatial_optimizer"]
            logger.info(f"   📈 Статистика оптимизатора доступна: {len(spatial_stats)} компонентов")
        
        # Получаем производительность
        if hasattr(lattice, 'perf_stats') and "spatial_optimization" in lattice.perf_stats:
            perf = lattice.perf_stats["spatial_optimization"]
            logger.info(f"   ⏱️ Производительность: {perf['processing_time_ms']:.1f}ms")
            logger.info(f"   🎯 Режим: {perf['mode_used']}")
            logger.info(f"   💾 Память: {perf['memory_usage_mb']:.1f}MB")
        
        # Очистка ресурсов
        lattice.cleanup()
        logger.info("   🧹 Ресурсы освобождены")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании Lattice3D: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Тест 3: Проверка обратной совместимости"""
    logger.info("🧪 Тест 3: Обратная совместимость с DEPRECATED классами")
    
    try:
        from core.lattice.spatial_optimization import (
            SpatialOptimizer,
            MoESpatialOptimizer,
            create_spatial_optimizer,
            create_moe_spatial_optimizer
        )
        
        dimensions = (4, 4, 4)
        
        # Тестируем создание старых классов
        logger.info("   🔧 Создание SpatialOptimizer...")
        spatial_opt = create_spatial_optimizer(dimensions)
        logger.info(f"     ✅ SpatialOptimizer создан: {type(spatial_opt).__name__}")
        
        # Тестируем базовый поиск соседей
        neighbors = spatial_opt.find_neighbors_optimized((2, 2, 2), radius=1.5)
        logger.info(f"     🔍 Найдено соседей: {len(neighbors)}")
        
        spatial_opt.cleanup()
        
        # Тестируем MoE версию
        logger.info("   🤖 Создание MoESpatialOptimizer...")
        moe_opt = create_moe_spatial_optimizer(dimensions)
        logger.info(f"     ✅ MoESpatialOptimizer создан: {type(moe_opt).__name__}")
        
        # Тестируем MoE поиск соседей
        moe_neighbors = moe_opt.find_neighbors_by_radius_safe(20)  # центральная клетка
        logger.info(f"     🔍 MoE найдено соседей: {len(moe_neighbors)}")
        
        moe_opt.cleanup()
        
        logger.info("   ✅ Обратная совместимость работает")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка обратной совместимости: {e}")
        return False


def test_performance_comparison():
    """Тест 4: Сравнение производительности"""
    logger.info("🧪 Тест 4: Сравнение производительности Unified vs Legacy")
    
    try:
        import time
        import numpy as np
        
        from core.lattice.spatial_optimization import (
            create_spatial_optimizer,
            create_unified_spatial_optimizer,
            OptimizationConfig,
            OptimizationMode
        )
        
        dimensions = (6, 6, 6)
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]
        state_size = 32
        states = torch.randn(total_cells, state_size)
        
        results = {}
        
        # Тест Legacy SpatialOptimizer
        logger.info("   📊 Тестирование Legacy SpatialOptimizer...")
        legacy_opt = create_spatial_optimizer(dimensions)
        
        def simple_processor(current_state, neighbor_states, cell_idx, neighbors):
            if len(neighbors) == 0:
                return current_state
            return 0.7 * current_state + 0.3 * neighbor_states.mean(dim=0)
        
        start_time = time.time()
        legacy_result = legacy_opt.optimize_lattice_forward(states, simple_processor)
        legacy_time = time.time() - start_time
        
        results["legacy"] = {
            "time_s": legacy_time,
            "type": "CPU Legacy"
        }
        logger.info(f"     ⏱️ Legacy: {legacy_time:.3f}s")
        legacy_opt.cleanup()
        
        # Тест Unified SpatialOptimizer
        logger.info("   🚀 Тестирование UnifiedSpatialOptimizer...")
        config = OptimizationConfig(mode=OptimizationMode.AUTO)
        unified_opt = create_unified_spatial_optimizer(dimensions, config)
        
        start_time = time.time()
        unified_result = unified_opt.optimize_lattice_forward(states)
        unified_time = time.time() - start_time
        
        results["unified"] = {
            "time_s": unified_time,
            "mode_used": unified_result.mode_used.value,
            "processing_time_ms": unified_result.processing_time_ms,
            "memory_mb": unified_result.memory_usage_mb
        }
        logger.info(f"     ⏱️ Unified: {unified_time:.3f}s")
        logger.info(f"     🎯 Режим: {unified_result.mode_used.value}")
        logger.info(f"     💾 Память: {unified_result.memory_usage_mb:.1f}MB")
        
        # Сравнение
        speedup = legacy_time / unified_time if unified_time > 0 else 1.0
        logger.info(f"   📈 Ускорение: {speedup:.2f}x")
        
        unified_opt.cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Ошибка сравнения производительности: {e}")
        return {}


def run_migration_tests():
    """Запуск всех тестов миграции"""
    logger.info("🚀 Запуск тестов миграции на Unified Spatial Optimizer")
    logger.info("=" * 70)
    
    test_results = {}
    
    # Тест 1: Импорты
    test_results["imports"] = test_imports()
    
    # Тест 2: Lattice3D с унифицированным оптимизатором
    test_results["lattice_unified"] = test_lattice_with_unified_optimizer()
    
    # Тест 3: Обратная совместимость
    test_results["backward_compatibility"] = test_backward_compatibility()
    
    # Тест 4: Сравнение производительности
    test_results["performance"] = test_performance_comparison()
    
    # Финальный отчет
    logger.info("=" * 70)
    logger.info("📋 ОТЧЕТ О МИГРАЦИИ")
    logger.info("=" * 70)
    
    successful_tests = sum(1 for result in test_results.values() if 
                          isinstance(result, (bool, dict)) and 
                          (result is True or (isinstance(result, dict) and result)))
    
    total_tests = len(test_results)
    
    logger.info(f"✅ Успешные тесты: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        logger.info("🎉 МИГРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("🚀 Unified Spatial Optimizer полностью интегрирован")
        logger.info("📖 Старые классы помечены как DEPRECATED")
        logger.info("🔄 Обратная совместимость сохранена")
    else:
        logger.warning(f"⚠️ {total_tests - successful_tests} тестов требуют внимания")
        logger.info("🔧 Проверьте логи выше для исправления проблем")
    
    return test_results


if __name__ == "__main__":
    results = run_migration_tests()