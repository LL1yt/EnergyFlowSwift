#!/usr/bin/env python3
"""
Тест интеграции гибридной архитектуры с динамической конфигурацией
================================================================

Проверяет:
1. Создание конфигурации в режиме testing
2. MinimalNCACell с <100 параметрами
3. Интеграцию с dynamic_config.py
4. Корректную работу режима тестирования
"""

import sys
import os
import torch
import logging

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath("."))

from utils.config_manager.dynamic_config import DynamicConfigManager
from core.cell_prototype.architectures.minimal_nca_cell import (
    create_nca_cell_from_config,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_dynamic_config_testing_mode():
    """Тест динамической конфигурации в режиме testing"""
    logger.info("🧪 Тестирование динамической конфигурации (testing mode)...")

    try:
        # Создание менеджера конфигурации
        config_manager = DynamicConfigManager()

        # Генерация конфигурации для режима testing
        config = config_manager.create_config_for_mode("testing")

        logger.info(
            f"[CONFIG] Режим: {config.get('_metadata', {}).get('mode', 'unknown')}"
        )
        logger.info(
            f"[CONFIG] Scale factor: {config.get('_metadata', {}).get('scale_factor', 'unknown')}"
        )

        # Проверка архитектурных настроек
        arch_config = config.get("architecture", {})
        logger.info(f"[ARCH] Hybrid mode: {arch_config.get('hybrid_mode', False)}")
        logger.info(
            f"[ARCH] NCA scaling disabled: {arch_config.get('disable_nca_scaling', False)}"
        )

        # Проверка NCA конфигурации
        nca_config = config.get("minimal_nca_cell", {})
        logger.info(f"[NCA] State size: {nca_config.get('state_size', 'unknown')}")
        logger.info(f"[NCA] Hidden dim: {nca_config.get('hidden_dim', 'unknown')}")
        logger.info(
            f"[NCA] Target params: {nca_config.get('target_params', 'unknown')}"
        )
        logger.info(
            f"[NCA] Scaling enabled: {nca_config.get('enable_lattice_scaling', 'unknown')}"
        )

        # Проверки
        assert (
            arch_config.get("hybrid_mode") == True
        ), "Hybrid mode should be enabled in testing"
        assert (
            arch_config.get("disable_nca_scaling") == True
        ), "NCA scaling should be disabled"
        assert (
            nca_config.get("enable_lattice_scaling") == False
        ), "NCA lattice scaling should be disabled"

        logger.info("✅ Динамическая конфигурация (testing mode): ОК")
        return config

    except Exception as e:
        logger.error(f"❌ Dynamic config test failed: {e}")
        return None


def test_nca_from_dynamic_config(config):
    """Тест создания NCA клетки из динамической конфигурации"""
    logger.info("🧪 Тестирование создания NCA из динамической конфигурации...")

    try:
        # Создание NCA клетки из динамической конфигурации
        cell = create_nca_cell_from_config(config)

        # Проверка параметров
        total_params = sum(p.numel() for p in cell.parameters())
        info = cell.get_info()

        logger.info(f"[CELL] Architecture: {info['architecture']}")
        logger.info(f"[CELL] Total params: {total_params}")
        logger.info(f"[CELL] Target params: {info['target_parameters']}")
        logger.info(f"[CELL] Scaling mode: {info['scaling_mode']}")
        logger.info(f"[CELL] State size: {info['state_size']}")
        logger.info(f"[CELL] Hidden dim: {info['hidden_dim']}")

        # Проверки для режима testing
        assert (
            info["scaling_mode"] == "fixed"
        ), f"Expected fixed scaling, got {info['scaling_mode']}"
        assert total_params < 100, f"Expected <100 params, got {total_params}"
        assert (
            info["lattice_scaling_enabled"] == False
        ), "Lattice scaling should be disabled"

        logger.info("✅ Создание NCA из динамической конфигурации: ОК")
        return cell

    except Exception as e:
        logger.error(f"❌ NCA creation from dynamic config failed: {e}")
        return None


def test_parameter_count_optimization():
    """Тест оптимизации количества параметров"""
    logger.info("🧪 Тестирование оптимизации количества параметров...")

    test_configs = [
        # Тест 1: Очень маленькая конфигурация
        {
            "minimal_nca_cell": {
                "state_size": 6,
                "neighbor_count": 26,
                "hidden_dim": 2,
                "external_input_size": 1,
                "enable_lattice_scaling": False,
                "target_params": 50,
            }
        },
        # Тест 2: Средняя конфигурация
        {
            "minimal_nca_cell": {
                "state_size": 8,
                "neighbor_count": 26,
                "hidden_dim": 3,
                "external_input_size": 1,
                "enable_lattice_scaling": False,
                "target_params": 69,
            }
        },
        # Тест 3: Максимальная конфигурация для <100 params
        {
            "minimal_nca_cell": {
                "state_size": 10,
                "neighbor_count": 26,
                "hidden_dim": 4,
                "external_input_size": 1,
                "enable_lattice_scaling": False,
                "target_params": 99,
            }
        },
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        try:
            logger.info(f"[TEST {i}] Config: {config['minimal_nca_cell']}")

            cell = create_nca_cell_from_config(config)
            total_params = sum(p.numel() for p in cell.parameters())
            info = cell.get_info()

            logger.info(f"[TEST {i}] Actual params: {total_params}")
            logger.info(
                f"[TEST {i}] Dimensions: state={info['state_size']}, hidden={info['hidden_dim']}"
            )

            # Проверка что параметры < 100
            assert total_params < 100, f"Test {i}: Too many params: {total_params}"

            results.append(
                {
                    "test": i,
                    "config": config["minimal_nca_cell"],
                    "params": total_params,
                    "efficiency": total_params
                    / config["minimal_nca_cell"]["target_params"],
                }
            )

            logger.info(f"✅ Test {i}: PASSED ({total_params} params)")

        except Exception as e:
            logger.error(f"❌ Test {i} failed: {e}")
            return False

    # Итоговый отчет
    logger.info("📊 Результаты оптимизации параметров:")
    for result in results:
        logger.info(
            f"   Test {result['test']}: {result['params']} params (efficiency: {result['efficiency']:.2f}x)"
        )

    logger.info("✅ Parameter optimization: ОК")
    return True


def test_forward_pass_performance():
    """Тест производительности forward pass"""
    logger.info("🧪 Тестирование производительности forward pass...")

    try:
        # Создание конфигурации с очень маленькими параметрами
        config = {
            "minimal_nca_cell": {
                "state_size": 8,
                "neighbor_count": 26,
                "hidden_dim": 3,
                "external_input_size": 1,
                "enable_lattice_scaling": False,
                "target_params": 69,
            }
        }

        cell = create_nca_cell_from_config(config)
        total_params = sum(p.numel() for p in cell.parameters())

        logger.info(f"[PERF] Cell params: {total_params}")

        # Тестовые данные разных размеров
        test_sizes = [4, 16, 64]  # Batch sizes

        for batch_size in test_sizes:
            # Создание тестовых данных
            neighbor_states = torch.randn(batch_size, 26, 8)
            own_state = torch.randn(batch_size, 8)
            external_input = torch.randn(batch_size, 1)

            # Измерение времени
            import time

            start_time = time.time()

            with torch.no_grad():
                for _ in range(10):  # 10 итераций для среднего времени
                    output = cell(neighbor_states, own_state, external_input)

            end_time = time.time()
            avg_time = (end_time - start_time) / 10 * 1000  # ms

            logger.info(f"[PERF] Batch {batch_size}: {avg_time:.2f}ms per forward pass")

            # Проверка корректности вывода
            assert output.shape == (
                batch_size,
                8,
            ), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "NaN in output"

        logger.info("✅ Forward pass performance: ОК")
        return True

    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def main():
    """Основная функция тестирования"""
    logger.info(
        "🚀 Запуск тестов интеграции гибридной архитектуры с динамической конфигурацией"
    )
    logger.info("=" * 80)

    # Тест 1: Динамическая конфигурация
    config = test_dynamic_config_testing_mode()
    if not config:
        logger.error("💥 Не удалось создать динамическую конфигурацию")
        return False

    # Тест 2: Создание NCA из конфигурации
    cell = test_nca_from_dynamic_config(config)
    if not cell:
        logger.error("💥 Не удалось создать NCA клетку")
        return False

    # Тест 3: Оптимизация параметров
    if not test_parameter_count_optimization():
        logger.error("💥 Оптимизация параметров не прошла")
        return False

    # Тест 4: Производительность
    if not test_forward_pass_performance():
        logger.error("💥 Тест производительности не прошел")
        return False

    # Итоговый отчет
    logger.info("\n" + "=" * 80)
    logger.info("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
    logger.info("=" * 80)
    logger.info("📋 Результаты:")
    logger.info("   ✅ Динамическая конфигурация работает")
    logger.info("   ✅ Режим testing настроен правильно")
    logger.info("   ✅ MinimalNCACell создается с <100 параметрами")
    logger.info("   ✅ Отключение масштабирования работает")
    logger.info("   ✅ Forward pass стабилен и быстр")
    logger.info("")
    logger.info("🔥 ГОТОВО К ИНТЕГРАЦИИ В ОСНОВНОЙ ПРОЕКТ!")
    logger.info("🔥 Можно переходить к оптимизации GatedMLPCell")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
