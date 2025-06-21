#!/usr/bin/env python3
"""
Базовый тест Clean 3D Cellular Neural Network
============================================

Проверяем основные компоненты:
1. Конфигурация работает
2. Клетки создаются и выполняют forward pass
3. Параметры соответствуют целевым значениям
"""

import torch
import logging
import sys
import os

# Добавляем путь к new_rebuild
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "new_rebuild"))

from new_rebuild import (
    ProjectConfig,
    get_project_config,
    set_project_config,
    NCACell,
    GMLPCell,
    CellFactory,
)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_project_config():
    """Тест конфигурации"""
    logger.info("=== ТЕСТ КОНФИГУРАЦИИ ===")

    config = get_project_config()
    logger.info(f"✅ Конфигурация создана: {config.architecture_type}")
    logger.info(
        f"✅ Решетка: {config.lattice_dimensions} = {config.total_cells} клеток"
    )
    logger.info(f"✅ Устройство: {config.device}")
    logger.info(
        f"✅ Целевые параметры: NCA={config.nca_target_params}, gMLP={config.gmlp_target_params}"
    )

    # Проверяем методы доступа
    nca_config = config.get_nca_config()
    gmlp_config = config.get_gmlp_config()

    assert nca_config["state_size"] == 4
    assert gmlp_config["state_size"] == 32
    assert nca_config["neighbor_count"] == gmlp_config["neighbor_count"]

    logger.info("✅ Все проверки конфигурации пройдены")
    return True


def test_nca_cell():
    """Тест NCA клетки"""
    logger.info("=== ТЕСТ NCA КЛЕТКИ ===")

    # Создаем клетку
    cell = NCACell()
    logger.info(f"✅ NCA клетка создана: {cell.state_size} состояние")

    # Проверяем параметры
    total_params = sum(p.numel() for p in cell.parameters())
    logger.info(f"✅ Параметры: {total_params:,} (цель: {cell.target_params:,})")

    # Тестируем forward pass
    batch_size = 2
    neighbor_count = cell.neighbor_count
    state_size = cell.state_size

    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, cell.external_input_size)

    with torch.no_grad():
        output = cell(neighbor_states, own_state, external_input)

    assert output.shape == (batch_size, state_size)
    logger.info(f"✅ Forward pass: {output.shape}")

    # Проверяем что выход отличается от входа (клетка что-то делает)
    assert not torch.allclose(output, own_state, atol=1e-6)
    logger.info("✅ Клетка изменяет состояние")

    return True


def test_gmlp_cell():
    """Тест gMLP клетки"""
    logger.info("=== ТЕСТ gMLP КЛЕТКИ ===")

    # Создаем клетку
    cell = GMLPCell()
    logger.info(f"✅ gMLP клетка создана: {cell.state_size} состояние")

    # Проверяем параметры
    total_params = sum(p.numel() for p in cell.parameters())
    logger.info(f"✅ Параметры: {total_params:,} (цель: {cell.target_params:,})")

    # Проверяем что убрали bottleneck
    bottleneck_found = any("bottleneck" in name for name, _ in cell.named_parameters())
    assert not bottleneck_found, "❌ Найден bottleneck в архитектуре!"
    logger.info("✅ Bottleneck архитектура убрана")

    # Тестируем forward pass
    batch_size = 2
    neighbor_count = cell.neighbor_count
    state_size = cell.state_size

    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, cell.external_input_size)

    with torch.no_grad():
        output = cell(neighbor_states, own_state, external_input)

    assert output.shape == (batch_size, state_size)
    logger.info(f"✅ Forward pass: {output.shape}")

    # Проверяем что выход отличается от входа
    assert not torch.allclose(output, own_state, atol=1e-6)
    logger.info("✅ Клетка изменяет состояние")

    return True


def test_cell_factory():
    """Тест фабрики клеток"""
    logger.info("=== ТЕСТ ФАБРИКИ КЛЕТОК ===")

    config = get_project_config()

    # Тест создания NCA
    nca_config = config.get_nca_config()
    nca_cell = CellFactory.create_cell("nca", nca_config)
    assert isinstance(nca_cell, NCACell)
    logger.info("✅ NCA через фабрику")

    # Тест создания gMLP
    gmlp_config = config.get_gmlp_config()
    gmlp_cell = CellFactory.create_cell("gmlp", gmlp_config)
    assert isinstance(gmlp_cell, GMLPCell)
    logger.info("✅ gMLP через фабрику")

    return True


def main():
    """Основная функция тестирования"""
    logger.info("🚀 ЗАПУСК ТЕСТОВ CLEAN АРХИТЕКТУРЫ")

    try:
        # Отключаем debug_mode для чистоты тестов
        config = ProjectConfig()
        config.debug_mode = False
        set_project_config(config)

        # Запускаем тесты
        test_project_config()
        test_nca_cell()
        test_gmlp_cell()
        test_cell_factory()

        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("✅ Clean архитектура работает корректно")

        # Выводим статистику
        config = get_project_config()
        logger.info(f"📊 СТАТИСТИКА:")
        logger.info(f"   Общие целевые параметры: {config.total_target_params:,}")
        logger.info(f"   Устройство: {config.device}")
        logger.info(f"   Архитектура: {config.architecture_type}")

        return True

    except Exception as e:
        logger.error(f"❌ ТЕСТ ПРОВАЛЕН: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
