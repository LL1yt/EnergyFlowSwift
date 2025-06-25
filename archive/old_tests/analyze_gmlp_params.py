#!/usr/bin/env python3
"""
Анализ параметров gMLP ячейки
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from core.cell_prototype.architectures.gmlp_cell import GatedMLPCell
import torch

# Конфигурация из логов
config = {
    "state_size": 8,
    "neighbor_count": 6,
    "hidden_dim": 8,
    "external_input_size": 4,
    "memory_dim": 4,
    "target_params": 300,
    "use_memory": True,
    "activation": "gelu",
    "dropout": 0.1,
}

print("=== АНАЛИЗ ПАРАМЕТРОВ gMLP КЛЕТКИ ===")
print()

# Создаем клетку
cell = GatedMLPCell(**config)

# Рассчитываем входной размер
neighbor_input_size = 6 * 8  # neighbor_count * state_size = 48
total_input_size = neighbor_input_size + 8 + 4  # neighbors + own_state + external = 60

print("ВХОДНЫЕ РАЗМЕРЫ:")
print(f"  neighbor_input_size: {neighbor_input_size}")
print(f'  own_state: {config["state_size"]}')
print(f'  external_input: {config["external_input_size"]}')
print(f"  total_input_size: {total_input_size}")
print()

# Анализируем каждый компонент
print("АНАЛИЗ СЛОЕВ:")

# 1. Input processing
input_norm_params = total_input_size * 2  # LayerNorm bias + weight
input_projection_params = (
    total_input_size * config["hidden_dim"] + config["hidden_dim"]
)  # Linear weight + bias
print(f"1. Input Norm: {input_norm_params} params")
print(f"2. Input Projection: {input_projection_params} params")

# 2. Pre-gating
pre_gating_params = config["hidden_dim"] * (config["hidden_dim"] * 2) + (
    config["hidden_dim"] * 2
)
print(f"3. Pre-gating: {pre_gating_params} params")

# 3. Spatial gating unit
spatial_gating_params = config["hidden_dim"] * 7  # seq_len = neighbor_count + 1 = 7
print(f"4. Spatial Gating: {spatial_gating_params} params")

# 4. FFN
ffn_1_params = config["hidden_dim"] * (config["hidden_dim"] * 2) + (
    config["hidden_dim"] * 2
)
ffn_2_params = (config["hidden_dim"] * 2) * config["hidden_dim"] + config["hidden_dim"]
ffn_total = ffn_1_params + ffn_2_params
print(f"5. FFN Layer 1: {ffn_1_params} params")
print(f"6. FFN Layer 2: {ffn_2_params} params")

# 5. Memory (GRU)
if config["use_memory"]:
    # GRU имеет 3 гейта (reset, update, new) и hidden state
    gru_input_size = config["hidden_dim"]
    gru_hidden_size = config["memory_dim"]
    # Параметры для input-to-hidden и hidden-to-hidden связей для каждого из 3 гейтов
    gru_params = 3 * (
        gru_input_size * gru_hidden_size
        + gru_hidden_size * gru_hidden_size
        + gru_hidden_size
    )
    memory_to_output_params = (
        gru_hidden_size * config["hidden_dim"] + config["hidden_dim"]
    )
    memory_total = gru_params + memory_to_output_params
    print(f"7. GRU Memory: {gru_params} params")
    print(f"8. Memory to Output: {memory_to_output_params} params")
else:
    memory_total = 0
    print("7-8. Memory: DISABLED")

# 6. Output
output_norm_params = config["hidden_dim"] * 2  # LayerNorm
output_projection_params = (
    config["hidden_dim"] * config["state_size"] + config["state_size"]
)
print(f"9. Output Norm: {output_norm_params} params")
print(f"10. Output Projection: {output_projection_params} params")

# 7. Residual connection
if total_input_size != config["state_size"]:
    residual_params = total_input_size * config["state_size"] + config["state_size"]
else:
    residual_params = 0
print(f"11. Input Residual: {residual_params} params")

# Общий подсчет
estimated_total = (
    input_norm_params
    + input_projection_params
    + pre_gating_params
    + spatial_gating_params
    + ffn_total
    + memory_total
    + output_norm_params
    + output_projection_params
    + residual_params
)

actual_total = sum(p.numel() for p in cell.parameters())

print()
print("ИТОГО:")
print(f"  Расчетное: {estimated_total} параметров")
print(f"  Фактическое: {actual_total} параметров")
print(f'  Target: {config["target_params"]} параметров')
print(
    f'  Превышение: {actual_total - config["target_params"]} ({actual_total/config["target_params"]:.1f}x)'
)
print()

# Детальный анализ каждого слоя
print("ДЕТАЛЬНЫЙ АНАЛИЗ:")
for name, param in cell.named_parameters():
    print(f"  {name}: {param.numel()} params, shape: {list(param.shape)}")

print()
print("=== РЕКОМЕНДАЦИИ ДЛЯ УМЕНЬШЕНИЯ ПАРАМЕТРОВ ===")
print(f'Для достижения target_params={config["target_params"]}:')
print("1. Уменьшить hidden_dim с 8 до 4-6")
print("2. Уменьшить external_input_size с 4 до 2-3")
print("3. Рассмотреть отключение memory (use_memory=False)")
print("4. Упростить архитектуру (убрать некоторые слои)")
