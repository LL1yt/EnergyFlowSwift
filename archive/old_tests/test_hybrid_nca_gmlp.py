#!/usr/bin/env python3
"""
Тест гибридной архитектуры NCA + GatedMLP
========================================

Проверяет:
1. MinimalNCACell с фиксированными параметрами (69 params)
2. Отключение автомасштабирования NCA
3. Корректную работу с новой конфигурацией
4. Валидацию параметров архитектуры
"""

import sys
import os
import torch
import logging

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath("."))

from core.cell_prototype.architectures.minimal_nca_cell import (
    MinimalNCACell,
    create_nca_cell_from_config,
)
from core.lattice_3d import create_lattice_from_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_nca_fixed_scaling():
    """Тест MinimalNCACell с отключенным масштабированием"""
    logger.info("🧪 Тестирование MinimalNCACell с фиксированными параметрами...")

    # Конфигурация для фиксированной NCA клетки
    nca_config = {
        "state_size": 36,
        "neighbor_count": 26,
        "hidden_dim": 6,
        "external_input_size": 2,
        "activation": "tanh",
        "target_params": 69,
        "enable_lattice_scaling": False,  # КРИТИЧНО: отключение масштабирования
    }

    try:
        # Создание клетки
        cell_fixed = MinimalNCACell(**nca_config)

        # Проверка параметров
        total_params = sum(p.numel() for p in cell_fixed.parameters())
        info = cell_fixed.get_info()

        logger.info(f"[NCA-FIXED] Параметры: {total_params}")
        logger.info(f"[NCA-FIXED] Режим масштабирования: {info['scaling_mode']}")
        logger.info(
            f"[NCA-FIXED] Размеры: state={info['state_size']}, hidden={info['hidden_dim']}"
        )

        # Проверки
        assert (
            info["scaling_mode"] == "fixed"
        ), f"Expected fixed scaling, got {info['scaling_mode']}"
        assert (
            info["state_size"] == 36
        ), f"Expected state_size=36, got {info['state_size']}"
        assert (
            info["hidden_dim"] == 6
        ), f"Expected hidden_dim=6, got {info['hidden_dim']}"
        assert total_params <= 100, f"Expected ≤100 params, got {total_params}"

        logger.info("✅ MinimalNCACell фиксированный режим: ОК")
        return True

    except Exception as e:
        logger.error(f"❌ MinimalNCACell тест failed: {e}")
        return False


def test_nca_dynamic_scaling():
    """Тест MinimalNCACell с включенным масштабированием (для сравнения)"""
    logger.info("🧪 Тестирование MinimalNCACell с динамическим масштабированием...")

    # Та же конфигурация, но с включенным масштабированием
    nca_config = {
        "state_size": 36,
        "neighbor_count": 26,
        "hidden_dim": 6,
        "external_input_size": 2,
        "activation": "tanh",
        "target_params": 1000,  # Больший target для демонстрации масштабирования
        "enable_lattice_scaling": True,  # Включено масштабирование
    }

    try:
        # Создание клетки
        cell_dynamic = MinimalNCACell(**nca_config)

        # Проверка параметров
        total_params = sum(p.numel() for p in cell_dynamic.parameters())
        info = cell_dynamic.get_info()

        logger.info(f"[NCA-DYNAMIC] Параметры: {total_params}")
        logger.info(f"[NCA-DYNAMIC] Режим масштабирования: {info['scaling_mode']}")
        logger.info(
            f"[NCA-DYNAMIC] Размеры: state={info['state_size']}, hidden={info['hidden_dim']}"
        )

        # Проверки
        assert (
            info["scaling_mode"] == "dynamic"
        ), f"Expected dynamic scaling, got {info['scaling_mode']}"
        assert (
            info["state_size"] > 36
        ), f"Expected scaled state_size>36, got {info['state_size']}"

        logger.info("✅ MinimalNCACell динамический режим: ОК")
        return True

    except Exception as e:
        logger.error(f"❌ MinimalNCACell dynamic test failed: {e}")
        return False


def test_hybrid_config_creation():
    """Тест создания NCA клетки из гибридной конфигурации"""
    logger.info("🧪 Тестирование создания NCA из гибридной конфигурации...")

    # Симулируем структуру из hybrid_nca_gmlp.yaml
    hybrid_config = {
        "minimal_nca_cell": {
            "state_size": 36,
            "neighbor_count": 26,
            "hidden_dim": 6,
            "external_input_size": 2,
            "activation": "tanh",
            "target_params": 69,
            "enable_lattice_scaling": False,  # КРИТИЧНО
        }
    }

    try:
        # Создание клетки из конфигурации
        cell = create_nca_cell_from_config(hybrid_config)

        # Проверка параметров
        total_params = sum(p.numel() for p in cell.parameters())
        info = cell.get_info()

        logger.info(f"[CONFIG] Параметры: {total_params}")
        logger.info(f"[CONFIG] Режим: {info['scaling_mode']}")

        # Проверки
        assert (
            info["scaling_mode"] == "fixed"
        ), "Config should create fixed scaling cell"
        assert total_params <= 100, f"Expected ≤100 params, got {total_params}"

        logger.info("✅ Создание из гибридной конфигурации: ОК")
        return True

    except Exception as e:
        logger.error(f"❌ Config creation test failed: {e}")
        return False


def test_forward_pass():
    """Тест forward pass с фиксированной NCA клеткой"""
    logger.info("🧪 Тестирование forward pass...")

    # Создание фиксированной NCA клетки
    cell = MinimalNCACell(
        state_size=36,
        neighbor_count=26,
        hidden_dim=6,
        external_input_size=2,
        enable_lattice_scaling=False,
        target_params=69,
    )

    try:
        # Тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 36)
        own_state = torch.randn(batch_size, 36)
        external_input = torch.randn(batch_size, 2)

        # Forward pass
        new_state = cell(neighbor_states, own_state, external_input)

        # Проверки
        assert new_state.shape == (
            batch_size,
            36,
        ), f"Wrong output shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values in output"
        assert not torch.isinf(new_state).any(), "Inf values in output"

        logger.info(f"[FORWARD] Input shape: {own_state.shape}")
        logger.info(f"[FORWARD] Output shape: {new_state.shape}")
        logger.info(f"[FORWARD] Output norm: {new_state.norm().item():.4f}")

        logger.info("✅ Forward pass: ОК")
        return True

    except Exception as e:
        logger.error(f"❌ Forward pass test failed: {e}")
        return False


def test_parameter_efficiency():
    """Тест эффективности параметров"""
    logger.info("🧪 Тестирование эффективности параметров...")

    configs = [
        {"target_params": 69, "enable_lattice_scaling": False, "expected_max": 100},
        {"target_params": 150, "enable_lattice_scaling": False, "expected_max": 200},
        {"target_params": 300, "enable_lattice_scaling": False, "expected_max": 400},
    ]

    results = []

    for config in configs:
        try:
            cell = MinimalNCACell(
                state_size=36,
                neighbor_count=26,
                hidden_dim=6,
                target_params=config["target_params"],
                enable_lattice_scaling=config["enable_lattice_scaling"],
            )

            total_params = sum(p.numel() for p in cell.parameters())
            efficiency = total_params / config["target_params"]

            logger.info(
                f"[EFFICIENCY] Target: {config['target_params']}, Actual: {total_params}, Ratio: {efficiency:.2f}"
            )

            # Проверка, что параметры в разумных пределах для фиксированного режима
            assert (
                total_params <= config["expected_max"]
            ), f"Too many params: {total_params} > {config['expected_max']}"

            results.append(
                {
                    "target": config["target_params"],
                    "actual": total_params,
                    "efficiency": efficiency,
                }
            )

        except Exception as e:
            logger.error(
                f"❌ Efficiency test failed for {config['target_params']}: {e}"
            )
            return False

    logger.info("📊 Результаты эффективности:")
    for result in results:
        logger.info(
            f"   Target: {result['target']}, Actual: {result['actual']}, Efficiency: {result['efficiency']:.2f}x"
        )

    logger.info("✅ Parameter efficiency: ОК")
    return True


def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестов гибридной архитектуры NCA+GatedMLP")
    logger.info("=" * 60)

    tests = [
        ("NCA Fixed Scaling", test_nca_fixed_scaling),
        ("NCA Dynamic Scaling", test_nca_dynamic_scaling),
        ("Hybrid Config Creation", test_hybrid_config_creation),
        ("Forward Pass", test_forward_pass),
        ("Parameter Efficiency", test_parameter_efficiency),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n🔍 Тест: {test_name}")
        logger.info("-" * 40)

        success = test_func()
        results.append((test_name, success))

        if success:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")

    # Итоговый отчет
    logger.info("\n" + "=" * 60)
    logger.info("📋 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")

    logger.info(f"\n📊 Результат: {passed}/{total} тестов прошли")

    if passed == total:
        logger.info("🎉 Все тесты гибридной архитектуры прошли успешно!")
        logger.info("🔥 Готово к переходу к Фазе 2.1 (Spatial Hashing)")
        return True
    else:
        logger.error("💥 Некоторые тесты не прошли. Требуется исправление.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
