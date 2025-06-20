#!/usr/bin/env python3
"""
Расчет параметров для minimal NCA cell
"""


def calculate_nca_params(state_size, neighbor_count, hidden_dim, external_input_size=1):
    """Рассчитать количество параметров для NCA cell"""

    print(f"🧮 Расчет параметров NCA:")
    print(f"   state_size: {state_size}")
    print(f"   neighbor_count: {neighbor_count}")
    print(f"   hidden_dim: {hidden_dim}")
    print(f"   external_input_size: {external_input_size}")
    print()

    # Input размеры
    neighbor_input_size = neighbor_count * state_size
    total_input_size = neighbor_input_size + state_size + external_input_size

    print(f"📊 Размеры входов:")
    print(
        f"   neighbor_input_size: {neighbor_count} × {state_size} = {neighbor_input_size}"
    )
    print(f"   own_state_size: {state_size}")
    print(f"   external_input_size: {external_input_size}")
    print(f"   total_input_size: {total_input_size}")
    print()

    # Слои NCA
    print(f"🔧 Слои NCA:")

    # Input projection: total_input -> hidden_dim
    input_proj_params = total_input_size * hidden_dim + hidden_dim  # weight + bias
    print(
        f"   input_projection: {total_input_size} × {hidden_dim} + {hidden_dim} = {input_proj_params}"
    )

    # Update gate: hidden_dim -> state_size
    update_gate_params = hidden_dim * state_size + state_size  # weight + bias
    print(
        f"   update_gate: {hidden_dim} × {state_size} + {state_size} = {update_gate_params}"
    )

    # Output projection: hidden_dim -> state_size
    output_proj_params = hidden_dim * state_size + state_size  # weight + bias
    print(
        f"   output_projection: {hidden_dim} × {state_size} + {state_size} = {output_proj_params}"
    )

    # Общее количество параметров
    total_params = input_proj_params + update_gate_params + output_proj_params

    print()
    print(f"🎯 ИТОГО параметров: {total_params}")

    return total_params


if __name__ == "__main__":
    print("=" * 50)
    print("РАСЧЕТ ПАРАМЕТРОВ MINIMAL NCA CELL")
    print("=" * 50)

    # Текущие параметры из конфигурации
    current_params = calculate_nca_params(
        state_size=4, neighbor_count=26, hidden_dim=3, external_input_size=1
    )

    print()
    print("=" * 50)
    print("СРАВНЕНИЕ С ДРУГИМИ КОНФИГУРАЦИЯМИ")
    print("=" * 50)

    # Старые параметры (state_size=8)
    print("\n📋 Старая конфигурация (state_size=8):")
    old_params = calculate_nca_params(
        state_size=8, neighbor_count=26, hidden_dim=3, external_input_size=1
    )

    print(
        f"\n🔄 Изменение параметров: {old_params} → {current_params} ({current_params - old_params:+d})"
    )
    print(
        f"   Процентное изменение: {(current_params - old_params) / old_params * 100:.1f}%"
    )
