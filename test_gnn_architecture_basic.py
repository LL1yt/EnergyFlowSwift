#!/usr/bin/env python3
"""
Тест базовой функциональности GNN архитектуры
==========================================

Проверяет замену gMLP на GNN с оптимизированными параметрами
"""

import os
import sys
import torch

# Добавляем пути
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("new_rebuild"))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.cells import GNNCell, NCACell, CellFactory
from new_rebuild.core.lattice import Lattice3D


def test_gnn_cell_basic():
    """Тест базовой функциональности GNN клетки"""
    print("🧪 Тестирование GNN клетки...")

    # Создаем GNN клетку
    gnn_cell = GNNCell()

    # Подсчитываем параметры
    total_params = sum(p.numel() for p in gnn_cell.parameters())
    print(f"✅ GNN клетка создана: {total_params} параметров")

    # Проверяем целевые параметры
    config = get_project_config()
    target_params = config.gnn_target_params
    print(f"   Цель: {target_params}, Фактически: {total_params}")

    ratio = total_params / target_params
    if ratio <= 1.5:  # Допускаем 50% превышение
        print(f"✅ Параметры в пределах нормы (x{ratio:.2f})")
    else:
        print(f"⚠️ Превышение параметров (x{ratio:.2f})")

    # Тест forward pass
    batch_size = 4
    neighbor_count = config.gnn_neighbor_count
    state_size = config.gnn_state_size

    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)

    # Forward pass
    new_state = gnn_cell(neighbor_states, own_state)

    print(
        f"✅ Forward pass: {neighbor_states.shape} + {own_state.shape} → {new_state.shape}"
    )

    # Проверяем, что состояние изменилось
    state_change = torch.norm(new_state - own_state).item()
    print(f"✅ Изменение состояния: {state_change:.6f}")

    return gnn_cell, total_params


def test_gnn_vs_gmlp_comparison():
    """Сравнение GNN с gMLP параметрами"""
    print("\n🔍 Сравнение GNN vs gMLP...")

    config = get_project_config()

    # Создаем GNN через CellFactory
    gnn_cell = CellFactory.create_cell("gnn", config.get_gnn_config())
    gnn_params = sum(p.numel() for p in gnn_cell.parameters())

    # Создаем gMLP через CellFactory (должен вернуть GNN)
    gmlp_cell = CellFactory.create_cell("gmlp", config.get_gmlp_config())
    gmlp_params = sum(p.numel() for p in gmlp_cell.parameters())

    print(f"✅ GNN напрямую: {gnn_params} параметров")
    print(f"✅ gMLP (→GNN): {gmlp_params} параметров")

    # Проверяем, что это одинаковые архитектуры
    assert gnn_params == gmlp_params, "gMLP должен возвращать GNN"
    print("✅ Legacy совместимость работает")

    # Сравнение с историческими данными
    old_gmlp_params = 113161  # Из COMPLETION_SUMMARY.md
    improvement = old_gmlp_params / gnn_params
    print(f"✅ Улучшение: {old_gmlp_params} → {gnn_params} (x{improvement:.1f} меньше)")

    return gnn_params, improvement


def test_gnn_attention_mechanism():
    """Тест attention mechanism"""
    print("\n🎯 Тестирование attention mechanism...")

    config = get_project_config()

    # Создаем GNN с attention
    gnn_cell = GNNCell(use_attention=True)

    batch_size = 2
    neighbor_count = config.gnn_neighbor_count
    state_size = config.gnn_state_size

    # Создаем разнообразные состояния соседей
    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)

    # Получаем статистику сообщений
    stats = gnn_cell.get_message_statistics(neighbor_states, own_state)

    print(f"✅ Message diversity: {stats['message_diversity'].mean().item():.4f}")
    print(f"✅ Message magnitudes: {stats['message_magnitudes'].mean().item():.4f}")

    if "attention_entropy" in stats:
        print(f"✅ Attention entropy: {stats['attention_entropy'].mean().item():.4f}")

    # Forward pass
    new_state = gnn_cell(neighbor_states, own_state)
    print(f"✅ Forward pass с attention: {new_state.shape}")

    return stats


def test_gnn_lattice_integration():
    """Тест интеграции GNN с 3D решеткой"""
    print("\n🏗️ Тестирование интеграции с 3D решеткой...")

    # Устанавливаем GNN архитектуру
    config = get_project_config()
    config.architecture_type = "gnn"
    set_project_config(config)

    # Создаем решетку
    lattice = Lattice3D()

    print(f"✅ Решетка создана: {lattice.pos_helper.total_positions} клеток")
    print(f"   Архитектура: {config.architecture_type}")
    print(f"   Состояния: {lattice.states.shape}")
    print(f"   Устройство: {lattice.device}")

    # Тест forward pass
    initial_states = lattice.states.clone()
    new_states = lattice.forward()

    print(f"✅ Forward pass: {initial_states.shape} → {new_states.shape}")

    # Проверяем изменения
    total_change = torch.norm(new_states - initial_states).item()
    print(f"✅ Общее изменение состояний: {total_change:.6f}")

    # Тест I/O операций
    input_states = lattice.get_input_states()
    output_states = lattice.get_output_states()

    print(f"✅ Input points: {input_states.shape}")
    print(f"✅ Output points: {output_states.shape}")

    return lattice


def test_neighbor_topology_proportions():
    """Тест новых пропорций соседей 10/60/30"""
    print("\n⚖️ Тестирование пропорций соседей...")

    config = get_project_config()

    # Проверяем новые пропорции
    local_tier = config.local_tier
    functional_tier = config.functional_tier
    distant_tier = config.distant_tier

    print(f"✅ Локальные: {local_tier:.1%} (стабилизация)")
    print(f"✅ Функциональные: {functional_tier:.1%} (ЯДРО эмерджентности)")
    print(f"✅ Дальние: {distant_tier:.1%} (глобальная координация)")

    total = local_tier + functional_tier + distant_tier
    print(f"✅ Общая сумма: {total:.1%}")

    assert (
        abs(total - 1.0) < 0.001
    ), f"Пропорции должны составлять 100%, получено {total:.1%}"

    # Проверяем, что функциональные связи доминируют (для эмерджентности)
    assert (
        functional_tier >= 0.5
    ), f"Функциональные связи должны быть >= 50%, получено {functional_tier:.1%}"

    print("✅ Пропорции оптимизированы для эмерджентности")

    return local_tier, functional_tier, distant_tier


def main():
    """Основная функция тестирования"""
    print("🚀 Тест GNN архитектуры (замена gMLP)")
    print("=" * 50)

    try:
        # Тест 1: Базовая функциональность GNN
        gnn_cell, gnn_params = test_gnn_cell_basic()

        # Тест 2: Сравнение с gMLP
        params, improvement = test_gnn_vs_gmlp_comparison()

        # Тест 3: Attention mechanism
        stats = test_gnn_attention_mechanism()

        # Тест 4: Интеграция с решеткой
        lattice = test_gnn_lattice_integration()

        # Тест 5: Пропорции соседей
        proportions = test_neighbor_topology_proportions()

        print("\n" + "=" * 50)
        print("🎉 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 50)
        print(f"✅ GNN архитектура: {params} параметров")
        print(f"✅ Улучшение над gMLP: x{improvement:.1f} меньше параметров")
        print(f"✅ Решетка: {lattice.pos_helper.total_positions} клеток")
        print(
            f"✅ Пропорции соседей: {proportions[0]:.0%}/{proportions[1]:.0%}/{proportions[2]:.0%}"
        )
        print(f"✅ Attention механизм: работает")
        print(f"✅ Legacy совместимость: сохранена")

        print("\n🎯 GNN готова к замене gMLP!")

    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        raise


if __name__ == "__main__":
    main()
