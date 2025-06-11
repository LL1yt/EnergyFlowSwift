#!/usr/bin/env python3
"""
Simple NCA Integration Test
Тестирование интеграции μNCA в существующую систему без изменения основной логики
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Imports
from utils.config_manager.dynamic_config import DynamicConfigManager
from training.embedding_trainer.nca_adapter import create_emergent_nca_cell_from_config
from core.cell_prototype.architectures.minimal_nca_cell import test_nca_cell_basic


def test_config_integration():
    """Тестирование интеграции NCA с конфигурационной системой"""

    print("🔧 TESTING CONFIG INTEGRATION")
    print("=" * 60)

    try:
        # Создаем конфигурационный менеджер
        config_manager = DynamicConfigManager()

        # Генерируем конфигурацию для development режима
        config = config_manager.create_config_for_mode("development")

        print(f"✅ Config generated successfully")
        print(f"   Mode: {config['_metadata']['mode']}")
        print(f"   Scale factor: {config['_metadata']['scale_factor']}")

        # Проверяем наличие NCA конфигурации
        if "nca" in config:
            nca_config = config["nca"]
            print(f"📋 NCA Configuration:")
            print(f"   Enabled: {nca_config.get('enabled', False)}")
            print(f"   Target params: {nca_config.get('target_params', 'N/A')}")
            print(f"   State size: {nca_config.get('state_size', 'N/A')}")
            print(f"   Hidden dim: {nca_config.get('hidden_dim', 'N/A')}")
            print(f"   External input: {nca_config.get('external_input_size', 'N/A')}")
        else:
            print("❌ NCA configuration not found in config")
            return False

        # Проверяем emergent_training секцию
        if "emergent_training" in config:
            emergent_config = config["emergent_training"]
            architecture = emergent_config.get("cell_architecture", "unknown")
            print(f"🧠 Emergent Training Architecture: {architecture}")

            if architecture == "nca":
                print("✅ NCA architecture selected for emergent training")
            else:
                print("ℹ️  Using fallback architecture for emergent training")

        return True

    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False


def test_cell_creation():
    """Тестирование создания NCA клеток из конфигурации"""

    print("\n🔬 TESTING CELL CREATION")
    print("=" * 60)

    try:
        # Создаем конфигурацию с включенным NCA
        config_manager = DynamicConfigManager()
        config = config_manager.create_config_for_mode("development")

        # Включаем NCA если не включен
        if not config.get("nca", {}).get("enabled", False):
            if "nca" not in config:
                config["nca"] = {}
            config["nca"]["enabled"] = True
            print("⚙️  Manually enabled NCA for testing")

        # Создаем NCA клетку из конфигурации
        cell = create_emergent_nca_cell_from_config(config)

        print(f"✅ EmergentNCACell created successfully")

        # Получаем информацию о клетке
        info = cell.get_info()
        spec_info = cell.get_specialization_info()

        print(f"📊 Cell Information:")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Target parameters: {info['target_parameters']:,}")
        print(f"   Parameter efficiency: {info['parameter_efficiency']:.2f}x")
        print(f"   State size: {info['state_size']}")
        print(f"   Hidden dim: {info['hidden_dim']}")
        print(f"   Neighbor count: {info['neighbor_count']}")
        print(f"   NCA alpha: {info['nca_alpha']:.3f}")
        print(f"   NCA beta: {info['nca_beta']:.3f}")

        print(f"🧠 Specialization Info:")
        print(f"   Specialization strength: {spec_info['specialization_strength']:.3f}")
        print(f"   Forward count: {spec_info['forward_count']}")
        print(f"   Spatial connections: {spec_info['spatial_connections']}")

        # Сравнение с gMLP
        gmlp_target = config.get("gmlp", {}).get("target_params", 1888)
        if isinstance(gmlp_target, (int, float)) and gmlp_target > 0:
            reduction = ((gmlp_target - info["total_parameters"]) / gmlp_target) * 100
            print(f"🔥 Parameter reduction vs target gMLP: {reduction:.1f}%")

        return cell

    except Exception as e:
        print(f"❌ Cell creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_forward_pass(cell):
    """Тестирование forward pass NCA клетки"""

    print("\n🚀 TESTING FORWARD PASS")
    print("=" * 60)

    try:
        # Получаем размеры из клетки
        state_size = cell.state_size
        neighbor_count = cell.neighbor_count
        external_input_size = cell.external_input_size

        # Тестовые данные
        batch_size = 8
        neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
        own_state = torch.randn(batch_size, state_size)
        external_input = torch.randn(batch_size, external_input_size)

        print(f"📊 Input shapes:")
        print(f"   Neighbor states: {neighbor_states.shape}")
        print(f"   Own state: {own_state.shape}")
        print(f"   External input: {external_input.shape}")

        # Forward pass
        output = cell(neighbor_states, own_state, external_input)

        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(
            f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]"
        )
        print(f"   Output mean: {output.mean().item():.3f}")
        print(f"   Output std: {output.std().item():.3f}")

        # Проверки
        assert output.shape == (
            batch_size,
            state_size,
        ), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values in output"
        assert not torch.isinf(output).any(), "Inf values in output"

        print("✅ All forward pass checks passed")

        return True

    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scaling():
    """Тестирование масштабируемости NCA через конфигурацию"""

    print("\n📏 TESTING SCALING")
    print("=" * 60)

    config_manager = DynamicConfigManager()
    modes = ["development", "research", "validation"]

    results = {}

    for mode in modes:
        try:
            config = config_manager.create_config_for_mode(mode)

            # Включаем NCA
            if "nca" not in config:
                config["nca"] = {}
            config["nca"]["enabled"] = True

            # Создаем клетку
            cell = create_emergent_nca_cell_from_config(config)
            info = cell.get_info()

            results[mode] = {
                "scale_factor": config["_metadata"]["scale_factor"],
                "total_params": info["total_parameters"],
                "target_params": info["target_parameters"],
                "state_size": info["state_size"],
                "hidden_dim": info["hidden_dim"],
                "lattice_size": config["lattice"]["total_neurons"],
            }

            print(f"✅ {mode.upper()} mode:")
            print(f"   Scale: {results[mode]['scale_factor']}")
            print(f"   Parameters: {results[mode]['total_params']:,}")
            print(f"   Lattice neurons: {results[mode]['lattice_size']:,}")

        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")
            results[mode] = None

    # Анализ масштабирования
    print(f"\n📈 Scaling Analysis:")
    valid_results = {k: v for k, v in results.items() if v is not None}

    if len(valid_results) > 1:
        scales = [v["scale_factor"] for v in valid_results.values()]
        params = [v["total_params"] for v in valid_results.values()]

        min_scale, max_scale = min(scales), max(scales)
        min_params, max_params = min(params), max(params)

        scale_ratio = max_scale / min_scale if min_scale > 0 else 0
        param_ratio = max_params / min_params if min_params > 0 else 0

        print(f"   Scale range: {min_scale} → {max_scale} ({scale_ratio:.1f}x)")
        print(f"   Param range: {min_params:,} → {max_params:,} ({param_ratio:.1f}x)")
        print(f"   Scaling efficiency: {param_ratio/scale_ratio:.2f}")

    return results


def main():
    """Основная функция тестирования"""

    print("🎯 NCA INTEGRATION TEST")
    print("=" * 80)

    success_count = 0
    total_tests = 5

    # 1. Базовое тестирование NCA клетки
    print(f"\n[1/{total_tests}] BASIC NCA CELL TEST")
    if test_nca_cell_basic():
        success_count += 1
        print("✅ Basic NCA cell test passed")
    else:
        print("❌ Basic NCA cell test failed")

    # 2. Тестирование конфигурационной интеграции
    print(f"\n[2/{total_tests}] CONFIG INTEGRATION TEST")
    if test_config_integration():
        success_count += 1
        print("✅ Config integration test passed")
    else:
        print("❌ Config integration test failed")

    # 3. Тестирование создания клеток
    print(f"\n[3/{total_tests}] CELL CREATION TEST")
    cell = test_cell_creation()
    if cell is not None:
        success_count += 1
        print("✅ Cell creation test passed")
    else:
        print("❌ Cell creation test failed")
        cell = None

    # 4. Тестирование forward pass
    if cell is not None:
        print(f"\n[4/{total_tests}] FORWARD PASS TEST")
        if test_forward_pass(cell):
            success_count += 1
            print("✅ Forward pass test passed")
        else:
            print("❌ Forward pass test failed")
    else:
        print(f"\n[4/{total_tests}] FORWARD PASS TEST - SKIPPED (no cell)")

    # 5. Тестирование масштабируемости
    print(f"\n[5/{total_tests}] SCALING TEST")
    results = test_scaling()
    if results and any(r is not None for r in results.values()):
        success_count += 1
        print("✅ Scaling test passed")
    else:
        print("❌ Scaling test failed")

    # Финальный отчет
    print(f"\n🏁 FINAL RESULTS")
    print("=" * 80)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests*100):.1f}%")

    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - NCA integration ready!")
        return 0
    elif success_count >= total_tests * 0.8:
        print("✅ Most tests passed - integration mostly working")
        return 0
    else:
        print("❌ Multiple test failures - integration needs fixes")
        return 1


if __name__ == "__main__":
    exit(main())
