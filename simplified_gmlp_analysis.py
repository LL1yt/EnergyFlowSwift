#!/usr/bin/env python3
"""
Анализ упрощенной gMLP архитектуры для target=300 параметров
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn


def analyze_simplified_configs():
    """Анализ различных упрощенных конфигураций"""

    print("=== АНАЛИЗ УПРОЩЕННЫХ КОНФИГУРАЦИЙ gMLP ===")
    print()

    target_params = 300

    # Базовые константы
    state_size = 8
    neighbor_count = 6

    configs = [
        {
            "name": "MINIMAL",
            "state_size": state_size,
            "neighbor_count": neighbor_count,
            "hidden_dim": 2,
            "external_input_size": 1,
            "use_memory": False,
            "use_spatial_gating": False,
            "use_ffn": False,
        },
        {
            "name": "BASIC",
            "state_size": state_size,
            "neighbor_count": neighbor_count,
            "hidden_dim": 4,
            "external_input_size": 2,
            "use_memory": False,
            "use_spatial_gating": False,
            "use_ffn": True,
        },
        {
            "name": "MEDIUM",
            "state_size": state_size,
            "neighbor_count": neighbor_count,
            "hidden_dim": 6,
            "external_input_size": 2,
            "use_memory": False,
            "use_spatial_gating": True,
            "use_ffn": True,
        },
        {
            "name": "CURRENT",
            "state_size": state_size,
            "neighbor_count": neighbor_count,
            "hidden_dim": 8,
            "external_input_size": 4,
            "use_memory": True,
            "use_spatial_gating": True,
            "use_ffn": True,
        },
    ]

    for config in configs:
        print(f"=== {config['name']} КОНФИГУРАЦИЯ ===")
        params = calculate_simplified_params(config)

        print(f"  Target: {target_params} параметров")
        print(f"  Actual: {params} параметров")
        print(f"  Ratio: {params/target_params:.2f}x")
        print(f"  Difference: {params - target_params:+d}")

        if params <= target_params * 1.1:  # 10% допуск
            print("  ✅ ПОДХОДИТ ПОД TARGET!")
        elif params <= target_params * 1.5:  # 50% допуск
            print("  ⚠️  Близко к target")
        else:
            print("  ❌ Слишком много параметров")

        print(
            f"  Components: hidden_dim={config['hidden_dim']}, "
            + f"external={config['external_input_size']}, "
            + f"memory={config['use_memory']}"
        )
        print()


def calculate_simplified_params(config):
    """Расчет параметров для упрощенной архитектуры"""

    state_size = config["state_size"]
    neighbor_count = config["neighbor_count"]
    hidden_dim = config["hidden_dim"]
    external_input_size = config["external_input_size"]

    # Входной размер
    neighbor_input_size = neighbor_count * state_size  # 6 * 8 = 48
    total_input_size = neighbor_input_size + state_size + external_input_size

    params = 0

    # 1. Input processing (обязательно)
    params += total_input_size * 2  # LayerNorm
    params += total_input_size * hidden_dim + hidden_dim  # Linear projection

    # 2. Spatial Gating (опционально)
    if config.get("use_spatial_gating", False):
        params += hidden_dim * (hidden_dim * 2) + (hidden_dim * 2)  # pre_gating
        params += hidden_dim * (neighbor_count + 1)  # spatial gating

    # 3. Feed Forward Network (опционально)
    if config.get("use_ffn", False):
        ffn_hidden = hidden_dim * 2
        params += hidden_dim * ffn_hidden + ffn_hidden  # FFN layer 1
        params += ffn_hidden * hidden_dim + hidden_dim  # FFN layer 2

    # 4. Memory (опционально)
    if config.get("use_memory", False):
        memory_dim = config.get("memory_dim", hidden_dim // 2)
        # GRU параметры (3 гейта)
        params += 3 * (hidden_dim * memory_dim + memory_dim * memory_dim + memory_dim)
        params += memory_dim * hidden_dim + hidden_dim  # memory to output

    # 5. Output processing (обязательно)
    params += hidden_dim * 2  # Output LayerNorm
    params += hidden_dim * state_size + state_size  # Output projection

    # 6. Residual connection (если размеры не совпадают)
    if total_input_size != state_size:
        params += total_input_size * state_size + state_size

    return params


def create_optimal_config():
    """Находим оптимальную конфигурацию близкую к 300 параметрам"""

    print("=== ПОИСК ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ ===")
    print()

    target = 300
    best_config = None
    best_diff = float("inf")

    state_size = 8
    neighbor_count = 6

    # Перебираем варианты
    for hidden_dim in [2, 3, 4, 5, 6]:
        for external_input_size in [1, 2, 3]:
            for use_memory in [False]:  # Память отключаем
                for use_spatial_gating in [False, True]:
                    for use_ffn in [False, True]:

                        config = {
                            "state_size": state_size,
                            "neighbor_count": neighbor_count,
                            "hidden_dim": hidden_dim,
                            "external_input_size": external_input_size,
                            "use_memory": use_memory,
                            "use_spatial_gating": use_spatial_gating,
                            "use_ffn": use_ffn,
                        }

                        params = calculate_simplified_params(config)
                        diff = abs(params - target)

                        if diff < best_diff:
                            best_diff = diff
                            best_config = config.copy()
                            best_config["calculated_params"] = params

    print("ЛУЧШАЯ НАЙДЕННАЯ КОНФИГУРАЦИЯ:")
    print(f"  Параметры: {best_config['calculated_params']} (target: {target})")
    print(f"  Отклонение: {best_diff} параметров")
    print(f"  hidden_dim: {best_config['hidden_dim']}")
    print(f"  external_input_size: {best_config['external_input_size']}")
    print(f"  use_memory: {best_config['use_memory']}")
    print(f"  use_spatial_gating: {best_config['use_spatial_gating']}")
    print(f"  use_ffn: {best_config['use_ffn']}")

    return best_config


def explain_architecture_philosophy():
    """Объясняет философию упрощенной архитектуры"""

    print()
    print("=== ФИЛОСОФИЯ УПРОЩЕННОЙ АРХИТЕКТУРЫ ===")
    print()

    print("🧠 БИОЛОГИЧЕСКИЕ ПРИНЦИПЫ:")
    print("  1. Нейрон = простой процессор (минимум внутренних состояний)")
    print("  2. Сложность = из взаимодействий, не из сложности клетки")
    print("  3. Память = не индивидуальная, а сетевая (паттерны активации)")
    print("  4. Эмерджентность = коллективное поведение простых элементов")
    print()

    print("⚙️ ТЕХНИЧЕСКИЕ ПРЕИМУЩЕСТВА:")
    print("  1. Меньше параметров = быстрее обучение")
    print("  2. Проще архитектура = стабильнее градиенты")
    print("  3. Меньше overfitting на индивидуальных клетках")
    print("  4. Масштабируемость на большие решетки")
    print()

    print("🎯 КОМПРОМИССЫ:")
    print("  ✅ Отключаем memory - эмерджентность важнее индивидуальной памяти")
    print("  ✅ Уменьшаем hidden_dim - простота важнее богатства состояний")
    print("  ⚠️  Spatial gating - может быть полезен для соседских взаимодействий")
    print("  ⚠️  FFN - базовая нелинейность может быть нужна")


if __name__ == "__main__":
    analyze_simplified_configs()
    optimal_config = create_optimal_config()
    explain_architecture_philosophy()
