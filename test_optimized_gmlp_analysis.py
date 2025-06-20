#!/usr/bin/env python3
"""
Детальный анализ оптимизированной GatedMLPCell
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from core.cell_prototype.architectures.gmlp_cell_optimized import OptimizedGatedMLPCell
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_optimized_gmlp():
    """Детальный анализ параметров оптимизированной GatedMLPCell"""

    print("🔬 АНАЛИЗ ОПТИМИЗИРОВАННОЙ GatedMLPCell")
    print("=" * 60)

    # Конфигурация для тестирования
    config = {
        "state_size": 36,  # Синхронизировано с NCA
        "neighbor_count": 26,  # 3D Moore neighborhood
        "hidden_dim": 48,  # Оптимизировано для ~10K
        "external_input_size": 8,  # Уменьшено vs 12
        "memory_dim": 24,  # Оптимизировано
        "target_params": 10000,
        "use_memory": True,
        "activation": "gelu",
        "dropout": 0.05,
    }

    print("\n📊 КОНФИГУРАЦИЯ:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Создаем клетку
    cell = OptimizedGatedMLPCell(**config)

    # Рассчитываем входные размеры
    neighbor_input_size = 26 * 36  # 936
    total_input_size = neighbor_input_size + 36 + 8  # 980

    print(f"\n📏 ВХОДНЫЕ РАЗМЕРЫ:")
    print(f"   neighbor_input_size: {neighbor_input_size}")
    print(f"   own_state: {config['state_size']}")
    print(f"   external_input: {config['external_input_size']}")
    print(f"   total_input_size: {total_input_size}")

    # Детальный анализ каждого компонента
    print(f"\n🧮 АНАЛИЗ СЛОЕВ:")

    total_calculated = 0

    # 1. Input Norm
    input_norm_params = total_input_size * 2  # weight + bias
    total_calculated += input_norm_params
    print(f"1. Input Norm: {input_norm_params:,} params")

    # 2. Input Projection
    input_proj_params = total_input_size * config["hidden_dim"] + config["hidden_dim"]
    total_calculated += input_proj_params
    print(f"2. Input Projection: {input_proj_params:,} params")

    # 3. Pre-gating
    pre_gating_params = config["hidden_dim"] * (config["hidden_dim"] * 2) + (
        config["hidden_dim"] * 2
    )
    total_calculated += pre_gating_params
    print(f"3. Pre-gating: {pre_gating_params:,} params")

    # 4. Spatial Gating Unit
    seq_len = config["neighbor_count"] + 1  # 27
    spatial_proj_params = seq_len * seq_len + seq_len  # Linear layer
    spatial_norm_params = config["hidden_dim"] * 2  # LayerNorm
    spatial_total = spatial_proj_params + spatial_norm_params
    total_calculated += spatial_total
    print(
        f"4. Spatial Gating: {spatial_total:,} params ({spatial_proj_params} proj + {spatial_norm_params} norm)"
    )

    # 5. FFN
    ffn_dim = int(config["hidden_dim"] * 1.5)  # 72
    ffn1_params = config["hidden_dim"] * ffn_dim + ffn_dim
    ffn2_params = ffn_dim * config["hidden_dim"] + config["hidden_dim"]
    ffn_total = ffn1_params + ffn2_params
    total_calculated += ffn_total
    print(f"5. FFN: {ffn_total:,} params ({ffn1_params} + {ffn2_params})")

    # 6. Memory (GRU)
    if config["use_memory"]:
        # GRU parameters: input_size * hidden_size * 3 (gates) + hidden_size * hidden_size * 3 + bias
        memory_ih = config["hidden_dim"] * config["memory_dim"] * 3  # input to hidden
        memory_hh = config["memory_dim"] * config["memory_dim"] * 3  # hidden to hidden
        memory_bias = config["memory_dim"] * 6  # bias for ih and hh
        memory_to_output_params = (
            config["memory_dim"] * config["hidden_dim"] + config["hidden_dim"]
        )
        memory_total = memory_ih + memory_hh + memory_bias + memory_to_output_params
        total_calculated += memory_total
        print(f"6. Memory (GRU + Linear): {memory_total:,} params")
        print(f"   - GRU ih: {memory_ih:,}, hh: {memory_hh:,}, bias: {memory_bias:,}")
        print(f"   - Memory to output: {memory_to_output_params:,}")

    # 7. Output Norm
    output_norm_params = config["hidden_dim"] * 2
    total_calculated += output_norm_params
    print(f"7. Output Norm: {output_norm_params:,} params")

    # 8. Output Projection
    output_proj_params = (
        config["hidden_dim"] * config["state_size"] + config["state_size"]
    )
    total_calculated += output_proj_params
    print(f"8. Output Projection: {output_proj_params:,} params")

    # 9. Input Residual (ПРОБЛЕМА!)
    if total_input_size != config["state_size"]:
        input_residual_params = total_input_size * config["state_size"]  # NO BIAS!
        total_calculated += input_residual_params
        print(
            f"9. Input Residual: {input_residual_params:,} params ⚠️ БОЛЬШОЙ ПОТРЕБИТЕЛЬ!"
        )
    else:
        print(f"9. Input Residual: 0 params (Identity)")

    # Фактический подсчет
    actual_params = sum(p.numel() for p in cell.parameters())

    print(f"\n📈 ИТОГО:")
    print(f"   Расчетное: {total_calculated:,} параметров")
    print(f"   Фактическое: {actual_params:,} параметров")
    print(f"   Target: {config['target_params']:,} параметров")
    print(
        f"   Превышение: {actual_params - config['target_params']:,} ({actual_params/config['target_params']:.1f}x)"
    )

    # Проблемные места
    print(f"\n⚠️ ПРОБЛЕМНЫЕ КОМПОНЕНТЫ:")
    print(
        f"   1. Input Residual: {input_residual_params:,} params ({input_residual_params/actual_params*100:.1f}%)"
    )
    print(
        f"   2. Input Projection: {input_proj_params:,} params ({input_proj_params/actual_params*100:.1f}%)"
    )

    return {
        "actual_params": actual_params,
        "target_params": config["target_params"],
        "config": config,
        "major_consumers": {
            "input_residual": input_residual_params,
            "input_projection": input_proj_params,
            "memory": memory_total if config["use_memory"] else 0,
        },
    }


def test_parameter_optimization():
    """Тестирование разных конфигураций для достижения 10K параметров"""

    print(f"\n🎯 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ 10K TARGET")
    print("=" * 60)

    target = 10000
    best_config = None
    best_diff = float("inf")

    # Тестируем разные конфигурации
    test_configs = [
        # Уменьшаем hidden_dim для экономии в input_projection
        {"hidden_dim": 24, "memory_dim": 16, "external_input_size": 4},
        {"hidden_dim": 32, "memory_dim": 20, "external_input_size": 6},
        {"hidden_dim": 20, "memory_dim": 12, "external_input_size": 4},
        {"hidden_dim": 16, "memory_dim": 10, "external_input_size": 2},
        # Без memory для экономии
        {
            "hidden_dim": 40,
            "memory_dim": 0,
            "external_input_size": 8,
            "use_memory": False,
        },
        {
            "hidden_dim": 48,
            "memory_dim": 0,
            "external_input_size": 6,
            "use_memory": False,
        },
    ]

    for i, test_config in enumerate(test_configs):
        base_config = {
            "state_size": 36,
            "neighbor_count": 26,
            "hidden_dim": 48,
            "external_input_size": 8,
            "memory_dim": 24,
            "target_params": target,
            "use_memory": True,
            "activation": "gelu",
            "dropout": 0.05,
        }

        # Обновляем тестовой конфигурацией
        base_config.update(test_config)

        try:
            cell = OptimizedGatedMLPCell(**base_config)
            actual_params = sum(p.numel() for p in cell.parameters())
            diff = abs(actual_params - target)
            ratio = actual_params / target

            print(
                f"Config {i+1}: hidden={base_config['hidden_dim']:2d}, "
                f"memory={base_config['memory_dim']:2d}, "
                f"ext_input={base_config['external_input_size']:2d}, "
                f"use_mem={base_config['use_memory']}, "
                f"params={actual_params:6,d} ({ratio:.2f}x, diff={diff:6,d})"
            )

            if diff < best_diff:
                best_diff = diff
                best_config = base_config.copy()

        except Exception as e:
            print(f"Config {i+1}: FAILED - {e}")

    print(f"\n🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ:")
    if best_config:
        for key, value in best_config.items():
            print(f"   {key}: {value}")

        cell = OptimizedGatedMLPCell(**best_config)
        actual = sum(p.numel() for p in cell.parameters())
        print(
            f"\n📊 РЕЗУЛЬТАТ: {actual:,} параметров (target: {target:,}, diff: {best_diff:,})"
        )

    return best_config


def test_forward_pass(config):
    """Тестирование forward pass с оптимальной конфигурацией"""

    print(f"\n🧪 ТЕСТИРОВАНИЕ FORWARD PASS")
    print("=" * 40)

    try:
        cell = OptimizedGatedMLPCell(**config)

        # Тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 36)
        own_state = torch.randn(batch_size, 36)
        connection_weights = torch.randn(batch_size, 26)
        external_input = torch.randn(batch_size, config["external_input_size"])

        # Forward pass
        import time

        start_time = time.time()
        output = cell(neighbor_states, own_state, connection_weights, external_input)
        forward_time = (time.time() - start_time) * 1000

        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {own_state.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Forward time: {forward_time:.2f}ms")

        # Тестирование градиентов
        loss = output.sum()
        loss.backward()

        grad_count = sum(1 for p in cell.parameters() if p.grad is not None)
        total_params = len(list(cell.parameters()))

        print(f"✅ Gradient flow successful!")
        print(f"   Parameters with gradients: {grad_count}/{total_params}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Анализ текущей версии
    analysis = analyze_optimized_gmlp()

    # Оптимизация параметров
    best_config = test_parameter_optimization()

    # Тестирование лучшей конфигурации
    if best_config:
        test_forward_pass(best_config)
