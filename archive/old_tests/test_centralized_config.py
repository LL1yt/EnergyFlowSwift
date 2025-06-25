#!/usr/bin/env python3
"""
Тест централизованной конфигурации
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))


def test_centralized_config():
    """Тест базовой функциональности централизованной конфигурации"""
    print("🧪 TESTING CENTRALIZED CONFIG")
    print("=" * 50)

    try:
        from utils.centralized_config import get_centralized_config

        # Получаем конфигурацию
        config = get_centralized_config()

        # Тест NCA конфигурации
        nca_config = config.get_nca_config()
        print(f"✅ NCA config: {nca_config}")

        # Тест minimal_nca_cell конфигурации
        minimal_nca_config = config.get_minimal_nca_cell_config()
        print(f"✅ minimal_nca_cell config: {minimal_nca_config}")

        # Тест полной конфигурации
        full_config = config.get_full_config_dict()
        print(f"✅ Full config keys: {list(full_config.keys())}")

        # Проверяем что все ключевые значения присутствуют
        assert (
            nca_config["state_size"] == 4
        ), f"Expected state_size=4, got {nca_config['state_size']}"
        assert (
            nca_config["neighbor_count"] == 26
        ), f"Expected neighbor_count=26, got {nca_config['neighbor_count']}"
        assert (
            "minimal_nca_cell" in full_config
        ), "missing minimal_nca_cell in full config"

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_create_nca_cell():
    """Тест создания NCA клетки с централизованной конфигурацией"""
    print("\n🧪 TESTING NCA CELL CREATION")
    print("=" * 50)

    try:
        from utils.centralized_config import get_centralized_config
        from core.cell_prototype.architectures.minimal_nca_cell import (
            create_nca_cell_from_config,
        )

        # Получаем централизованную конфигурацию
        central_config = get_centralized_config()
        full_config = central_config.get_full_config_dict()

        # Создаем NCA клетку
        cell = create_nca_cell_from_config(full_config)

        # Проверяем параметры
        info = cell.get_info()
        print(f"✅ Created NCA cell: {info['total_parameters']} parameters")
        print(f"   state_size: {info['state_size']}")
        print(f"   neighbor_count: {info['neighbor_count']}")
        print(f"   architecture: {info['architecture']}")

        # Тест forward pass
        import torch

        batch_size = 2
        neighbor_states = torch.randn(
            batch_size, info["neighbor_count"], info["state_size"]
        )
        own_state = torch.randn(batch_size, info["state_size"])
        external_input = torch.randn(batch_size, info["external_input_size"])

        with torch.no_grad():
            output = cell(neighbor_states, own_state, external_input)

        print(f"✅ Forward pass successful: {own_state.shape} → {output.shape}")

        return True

    except Exception as e:
        print(f"❌ NCA cell test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_create_cell_from_config():
    """Тест create_cell_from_config с централизованной конфигурацией"""
    print("\n🧪 TESTING create_cell_from_config")
    print("=" * 50)

    try:
        from utils.centralized_config import get_centralized_config
        from core.cell_prototype.main import create_cell_from_config

        # Получаем централизованную конфигурацию
        central_config = get_centralized_config()
        full_config = central_config.get_full_config_dict()

        # Создаем конфигурацию в формате ожидаемом create_cell_from_config
        cell_config = {
            "prototype_name": "minimal_nca_cell",
            "minimal_nca_cell": full_config["minimal_nca_cell"],
        }

        print(f"📋 Cell config: {cell_config}")

        # Создаем клетку
        cell = create_cell_from_config(cell_config)

        print(f"✅ Created cell from config: {type(cell).__name__}")

        return True

    except Exception as e:
        print(f"❌ create_cell_from_config test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 CENTRALIZED CONFIG TEST SUITE")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    if test_centralized_config():
        tests_passed += 1

    if test_create_nca_cell():
        tests_passed += 1

    if test_create_cell_from_config():
        tests_passed += 1

    print(f"\n📊 RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Centralized config is working!")
        exit(0)
    else:
        print("❌ Some tests failed. Check the errors above.")
        exit(1)
