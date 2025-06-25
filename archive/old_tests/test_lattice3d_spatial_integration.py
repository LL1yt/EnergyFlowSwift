#!/usr/bin/env python3
"""
Тест MoE Lattice3D с Spatial Optimization
=========================================

Проверяет работоспособность упрощенного Lattice3D
с MoE архитектурой и spatial optimization.

ЦЕЛЬ ТЕСТА:
- Убедиться что MoE forward pass работает корректно
- Проверить что spatial optimization интегрирован правильно
- Валидировать MoE архитектуру
"""

import pytest
import torch
import sys
import os
from typing import Dict, Any
import logging

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath("."))

# Настройка логирования для тестов
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_moe_lattice3d_initialization():
    """
    Тест инициализации MoE Lattice3D
    """
    try:
        from new_rebuild.core.lattice.lattice import Lattice3D, create_lattice
        from new_rebuild.config import get_project_config

        logger.info("🚀 Тестирование MoE Lattice3D инициализации...")

        # Убеждаемся что архитектура MoE
        config = get_project_config()
        config.architecture_type = "moe"

        # Создаем решетку
        lattice = create_lattice()

        # Проверяем что spatial optimizer создан
        assert hasattr(lattice, "spatial_optimizer"), "Spatial optimizer не создан"
        assert lattice.spatial_optimizer is not None, "Spatial optimizer равен None"

        logger.info(f"✅ Spatial optimizer: {type(lattice.spatial_optimizer).__name__}")

        # Проверяем основные компоненты
        assert hasattr(lattice, "states"), "States не инициализированы"
        assert hasattr(lattice, "cells"), "Cells не созданы"
        assert lattice.states.shape[0] > 0, "States пустые"

        logger.info(f"✅ States shape: {lattice.states.shape}")
        logger.info(f"✅ Device: {lattice.device}")
        logger.info(f"✅ Cells type: {type(lattice.cells).__name__}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании инициализации: {e}")
        raise


def test_moe_forward_pass():
    """
    Критический тест MoE forward pass
    """
    try:
        from new_rebuild.core.lattice.lattice import create_lattice
        from new_rebuild.config import get_project_config

        logger.info("🚀 Тестирование MoE forward pass...")

        # Убеждаемся что архитектура MoE
        config = get_project_config()
        config.architecture_type = "moe"

        lattice = create_lattice()
        initial_states = lattice.states.clone()

        # Проверяем что нет NaN или inf в начальных состояниях
        assert not torch.isnan(initial_states).any(), "NaN в начальных состояниях"
        assert not torch.isinf(initial_states).any(), "Inf в начальных состояниях"

        # Выполняем MoE forward pass - КРИТИЧЕСКАЯ ПРОВЕРКА
        new_states = lattice.forward()

        # Проверяем результаты
        assert new_states is not None, "Forward pass вернул None"
        assert new_states.shape == initial_states.shape, "Размеры состояний изменились"
        assert not torch.isnan(new_states).any(), "NaN в новых состояниях"
        assert not torch.isinf(new_states).any(), "Inf в новых состояниях"

        logger.info(f"✅ MoE Forward pass успешен!")
        logger.info(f"   📊 States shape: {new_states.shape}")
        logger.info(f"   📊 Device: {new_states.device}")
        logger.info(
            f"   📊 States changed: {not torch.equal(initial_states, new_states)}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в MoE forward pass: {e}")
        raise


def test_moe_spatial_optimizer_performance():
    """
    Тест производительности MoE spatial optimizer
    """
    try:
        from new_rebuild.core.lattice.lattice import create_lattice
        from new_rebuild.config import get_project_config
        import time

        logger.info("🚀 Тестирование производительности MoE spatial optimizer...")

        # Убеждаемся что архитектура MoE
        config = get_project_config()
        config.architecture_type = "moe"

        lattice = create_lattice()

        # Измеряем время forward pass
        start_time = time.time()
        for i in range(3):  # Несколько итераций
            states = lattice.forward()
        end_time = time.time()

        avg_time = (end_time - start_time) / 3
        logger.info(f"✅ Средняя скорость MoE forward pass: {avg_time:.4f}s")

        # Проверяем статистику spatial optimizer
        if hasattr(lattice.spatial_optimizer, "get_performance_stats"):
            stats = lattice.spatial_optimizer.get_performance_stats()
            logger.info(f"   📊 Spatial optimizer stats: {stats}")

        # Проверяем статистику решетки
        perf_stats = lattice.get_performance_stats()
        logger.info(f"   📊 Lattice performance: {perf_stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании производительности MoE: {e}")
        raise


def test_wrong_architecture_rejection():
    """
    Тест отклонения неправильной архитектуры
    """
    try:
        from new_rebuild.core.lattice.lattice import Lattice3D
        from new_rebuild.config import get_project_config

        logger.info("🚀 Тестирование отклонения неправильной архитектуры...")

        # Пробуем не-MoE архитектуру
        config = get_project_config()
        config.architecture_type = "gnn"  # Не MoE

        # Должно вызвать ошибку
        try:
            lattice = Lattice3D()
            assert False, "Должна была быть ошибка для не-MoE архитектуры"
        except ValueError as e:
            logger.info(f"✅ Правильно отклонена не-MoE архитектура: {e}")
            return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании отклонения архитектуры: {e}")
        raise


def test_moe_architecture_validation():
    """
    Тест валидации MoE архитектуры
    """
    try:
        from new_rebuild.core.lattice.lattice import create_lattice
        from new_rebuild.config import get_project_config

        logger.info("🚀 Тестирование валидации MoE архитектуры...")

        # Устанавливаем MoE архитектуру
        config = get_project_config()
        config.architecture_type = "moe"

        lattice = create_lattice()

        # Валидируем решетку
        validation_stats = lattice.validate_lattice()

        assert (
            validation_stats["architecture_type"] == "moe"
        ), "Неправильный тип архитектуры"
        assert (
            "spatial_optimizer" in validation_stats
        ), "Отсутствует статистика spatial optimizer"

        logger.info(f"✅ MoE архитектура валидирована успешно!")
        logger.info(f"   📊 Validation stats: {validation_stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в валидации MoE архитектуры: {e}")
        raise


def run_all_tests():
    """
    Запускает все тесты MoE Lattice3D
    """
    tests = [
        ("MoE инициализация", test_moe_lattice3d_initialization),
        ("MoE forward pass", test_moe_forward_pass),
        ("MoE производительность", test_moe_spatial_optimizer_performance),
        ("Отклонение неправильной архитектуры", test_wrong_architecture_rejection),
        ("MoE валидация", test_moe_architecture_validation),
    ]

    results = {}
    logger.info("🔥 НАЧИНАЕМ ТЕСТИРОВАНИЕ MOE LATTICE3D")
    logger.info("=" * 60)

    for test_name, test_func in tests:
        try:
            logger.info(f"\n📋 Тест: {test_name}")
            logger.info("-" * 40)

            result = test_func()
            results[test_name] = "✅ PASSED"
            logger.info(f"✅ {test_name}: PASSED")

        except Exception as e:
            results[test_name] = f"❌ FAILED: {str(e)}"
            logger.error(f"❌ {test_name}: FAILED - {e}")

    # Итоговый отчет
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ MOE LATTICE3D")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        logger.info(f"{result}")
        if "PASSED" in result:
            passed += 1

    logger.info(f"\n🎯 РЕЗУЛЬТАТ: {passed}/{total} тестов прошли успешно")

    if passed == total:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! MoE Lattice3D работает корректно!")
        return True
    else:
        logger.error(
            f"⚠️ {total - passed} тестов не прошли. Требуется дополнительная работа."
        )
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
