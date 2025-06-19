#!/usr/bin/env python3
"""
Quick test для проверки масштабирования параметров NCA
"""

from utils.config_manager.dynamic_config import DynamicConfigManager
from training.embedding_trainer.nca_adapter import create_emergent_nca_cell_from_config


def test_nca_scaling():
    print("🧪 TESTING NCA PARAMETER SCALING")
    print("=" * 60)

    manager = DynamicConfigManager()
    modes = ["development", "research", "validation"]

    for mode in modes:
        config = manager.create_config_for_mode(mode)

        # Включаем NCA
        config["nca"]["enabled"] = True

        nca_config = config["nca"]
        scale = config["_metadata"]["scale_factor"]

        print(f"\n📊 {mode.upper()} mode (scale={scale}):")
        print(f'   Config target params: {nca_config["target_params"]:,}')
        print(f'   Config state size: {nca_config["state_size"]}')
        print(f'   Config hidden dim: {nca_config["hidden_dim"]}')
        print(f'   Config external input: {nca_config["external_input_size"]}')

        # Создаем клетку и проверяем реальные параметры
        try:
            cell = create_emergent_nca_cell_from_config(config)
            info = cell.get_info()
            actual_params = info["total_parameters"]
            target_params = info["target_parameters"]

            print(f"   ✅ Cell created successfully:")

            # Показываем все размерности
            original = info.get("original_dimensions", {})
            print(
                f'   State size: {info["state_size"]} (original: {original.get("state_size", "N/A")})'
            )
            print(
                f'   Hidden dim: {info["hidden_dim"]} (original: {original.get("hidden_dim", "N/A")})'
            )
            print(
                f'   External input: {info["external_input_size"]} (original: {original.get("external_input_size", "N/A")})'
            )

            # Параметры
            print(f"   Actual params: {actual_params:,}")
            print(f"   Target params: {target_params:,}")

            if target_params > 0:
                efficiency = actual_params / target_params
                print(f"   Parameter efficiency: {efficiency:.3f}x")

            optimization_used = info.get("architecture_optimized", False)
            print(
                f'   Architecture optimization: {"✅ YES" if optimization_used else "❌ NO"}'
            )

        except Exception as e:
            print(f"   ❌ ERROR creating cell: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_nca_scaling()
