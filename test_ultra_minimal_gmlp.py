#!/usr/bin/env python3
"""
Ultra-minimal GatedMLPCell optimization для точного достижения 10K параметров
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from core.cell_prototype.architectures.gmlp_cell_minimal import MinimalGatedMLPCell
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_bottleneck_configurations():
    """Тестирование разных размеров bottleneck для достижения 10K"""

    print("🎯 ULTRA-MINIMAL OPTIMIZATION для 10K параметров")
    print("=" * 60)

    target = 10000
    best_config = None
    best_diff = float("inf")

    # Тестируем разные bottleneck размеры
    test_configs = [
        {"bottleneck_dim": 8, "hidden_dim": 24, "external_input_size": 2},
        {"bottleneck_dim": 6, "hidden_dim": 20, "external_input_size": 2},
        {"bottleneck_dim": 4, "hidden_dim": 16, "external_input_size": 2},
        {"bottleneck_dim": 3, "hidden_dim": 12, "external_input_size": 1},
        {"bottleneck_dim": 2, "hidden_dim": 12, "external_input_size": 1},
        {
            "bottleneck_dim": 1,
            "hidden_dim": 8,
            "external_input_size": 1,
        },  # Экстремальный случай
    ]

    print("Тестируем конфигурации bottleneck:")
    print("bottleneck | hidden | ext_input | params | diff | ratio")
    print("-" * 55)

    for config in test_configs:
        base_config = {
            "state_size": 36,
            "neighbor_count": 26,
            "hidden_dim": 32,
            "bottleneck_dim": 16,
            "external_input_size": 4,
            "target_params": target,
        }

        # Обновляем тестовой конфигурацией
        base_config.update(config)

        try:
            cell = MinimalGatedMLPCell(**base_config)
            actual_params = sum(p.numel() for p in cell.parameters())
            diff = abs(actual_params - target)
            ratio = actual_params / target

            print(
                f"    {config['bottleneck_dim']:2d}    |   {config['hidden_dim']:2d}   |     {config['external_input_size']:1d}     | {actual_params:6,d} | {diff:5,d} | {ratio:.2f}x"
            )

            if diff < best_diff:
                best_diff = diff
                best_config = base_config.copy()

        except Exception as e:
            print(f"    {config['bottleneck_dim']:2d}    |   ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ (diff: {best_diff:,}):")
    for key, value in best_config.items():
        print(f"   {key}: {value}")

    return best_config


def calculate_parameter_breakdown(config):
    """Детальный breakdown параметров для лучшей конфигурации"""

    print(f"\n🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕЙ КОНФИГУРАЦИИ")
    print("=" * 50)

    # Создаем клетку с лучшей конфигурацией
    cell = MinimalGatedMLPCell(**config)

    # Рассчитываем каждый компонент
    neighbor_input_size = 26 * 36  # 936
    total_input_size = (
        neighbor_input_size + 36 + config["external_input_size"]
    )  # 936 + 36 + ext

    print(f"📊 ВХОДНЫЕ РАЗМЕРЫ:")
    print(f"   neighbor_input: {neighbor_input_size}")
    print(f"   own_state: 36")
    print(f"   external_input: {config['external_input_size']}")
    print(f"   total_input: {total_input_size}")

    print(f"\n🧮 РАСЧЕТ ПАРАМЕТРОВ:")

    components = {}

    # 1. Input Norm
    components["input_norm"] = total_input_size * 2

    # 2. Input Bottleneck
    components["input_bottleneck"] = total_input_size * config["bottleneck_dim"]

    # 3. Bottleneck to Hidden
    components["bottleneck_to_hidden"] = (
        config["bottleneck_dim"] * config["hidden_dim"] + config["hidden_dim"]
    )

    # 4. Pre-gating
    components["pre_gating"] = config["hidden_dim"] * (config["hidden_dim"] * 2)

    # 5. Spatial Gating
    seq_len = 27  # neighbor_count + 1
    components["spatial_proj"] = seq_len * seq_len  # No bias
    components["spatial_norm"] = config["hidden_dim"] * 2

    # 6. FFN
    components["ffn"] = config["hidden_dim"] * config["hidden_dim"]  # No bias

    # 7. Output Norm
    components["output_norm"] = config["hidden_dim"] * 2

    # 8. Output Projection
    components["output_projection"] = config["hidden_dim"] * 36 + 36

    # 9. Compressed Residual
    components["compressed_residual"] = config["bottleneck_dim"] * 36  # No bias

    total_calculated = sum(components.values())
    actual_params = sum(p.numel() for p in cell.parameters())

    print(f"   1. Input Norm: {components['input_norm']:,}")
    print(f"   2. Input Bottleneck: {components['input_bottleneck']:,}")
    print(f"   3. Bottleneck→Hidden: {components['bottleneck_to_hidden']:,}")
    print(f"   4. Pre-gating: {components['pre_gating']:,}")
    print(f"   5. Spatial Proj: {components['spatial_proj']:,}")
    print(f"   6. Spatial Norm: {components['spatial_norm']:,}")
    print(f"   7. FFN: {components['ffn']:,}")
    print(f"   8. Output Norm: {components['output_norm']:,}")
    print(f"   9. Output Projection: {components['output_projection']:,}")
    print(f"  10. Compressed Residual: {components['compressed_residual']:,}")

    print(f"\n📈 ИТОГО:")
    print(f"   Расчетное: {total_calculated:,}")
    print(f"   Фактическое: {actual_params:,}")
    print(f"   Target: {config['target_params']:,}")
    print(f"   Отклонение от target: {actual_params - config['target_params']:,}")

    return components


def test_forward_pass_optimal(config):
    """Тестирование forward pass лучшей конфигурации"""

    print(f"\n🧪 ТЕСТИРОВАНИЕ FORWARD PASS")
    print("=" * 40)

    try:
        cell = MinimalGatedMLPCell(**config)

        # Test data
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 36)
        own_state = torch.randn(batch_size, 36)
        connection_weights = torch.randn(batch_size, 26)
        external_input = torch.randn(batch_size, config["external_input_size"])

        # Forward pass timing
        import time

        start_time = time.time()
        output = cell(neighbor_states, own_state, connection_weights, external_input)
        forward_time = (time.time() - start_time) * 1000

        print(f"✅ Forward pass successful!")
        print(f"   Input: {own_state.shape}")
        print(f"   Output: {output.shape}")
        print(f"   Time: {forward_time:.2f}ms")

        # Gradient test
        loss = output.sum()
        loss.backward()

        grad_count = sum(1 for p in cell.parameters() if p.grad is not None)
        total_params_count = len(list(cell.parameters()))

        print(f"✅ Gradient flow successful!")
        print(f"   Gradients: {grad_count}/{total_params_count}")

        # Output statistics
        print(f"📊 Output statistics:")
        print(f"   Mean: {output.mean():.4f}")
        print(f"   Std: {output.std():.4f}")
        print(f"   Min: {output.min():.4f}")
        print(f"   Max: {output.max():.4f}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Находим оптимальную конфигурацию
    best_config = test_bottleneck_configurations()

    if best_config:
        # Детальный анализ
        components = calculate_parameter_breakdown(best_config)

        # Тестирование
        success = test_forward_pass_optimal(best_config)

        print(f"\n🎉 РЕЗУЛЬТАТ: {'✅ УСПЕХ' if success else '❌ НЕУДАЧА'}")

        if success:
            actual_params = sum(
                p.numel() for p in MinimalGatedMLPCell(**best_config).parameters()
            )
            target = best_config["target_params"]
            print(f"📊 Финальные параметры: {actual_params:,} (target: {target:,})")
            print(
                f"📊 Отклонение: {abs(actual_params - target):,} ({actual_params/target:.3f}x)"
            )
