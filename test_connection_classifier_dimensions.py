#!/usr/bin/env python3
"""
Тест для проверки корректности обработки размерностей в connection_classifier
"""

import torch
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.core.moe.connection_types import ConnectionCategory


def test_dimension_handling():
    """Тестируем различные варианты размерностей"""
    print("🔍 Тестируем обработку размерностей в connection_classifier\n")
    
    # Создаем классификатор
    lattice_dims = (5, 5, 5)
    classifier = UnifiedConnectionClassifier(lattice_dims)
    state_size = 32
    
    print("✅ Тест 1: Стандартный случай")
    # cell_state: [32], neighbor_states: [5, 32]
    cell_idx = 12
    neighbor_indices = [7, 11, 13, 17, 37]
    cell_state = torch.randn(state_size)
    neighbor_states = torch.randn(5, state_size)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print(f"   cell_state: {cell_state.shape}")
        print(f"   neighbor_states: {neighbor_states.shape}")
        print(f"   Результат: {sum(len(conns) for conns in result.values())} связей классифицировано")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    print("✅ Тест 2: cell_state с batch dimension")
    # cell_state: [1, 32], neighbor_states: [5, 32]
    cell_state = torch.randn(1, state_size)
    neighbor_states = torch.randn(5, state_size)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print(f"   cell_state: {cell_state.shape}")
        print(f"   neighbor_states: {neighbor_states.shape}")
        print(f"   Результат: {sum(len(conns) for conns in result.values())} связей классифицировано")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    print("✅ Тест 3: Один сосед")
    # cell_state: [32], neighbor_states: [32]
    neighbor_indices = [13]
    cell_state = torch.randn(state_size)
    neighbor_states = torch.randn(state_size)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print(f"   cell_state: {cell_state.shape}")
        print(f"   neighbor_states: {neighbor_states.shape}")
        print(f"   Результат: {sum(len(conns) for conns in result.values())} связей классифицировано")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    print("✅ Тест 4: neighbor_states с лишним batch dimension")
    # cell_state: [32], neighbor_states: [1, 5, 32]
    neighbor_indices = [7, 11, 13, 17, 37]
    cell_state = torch.randn(state_size)
    neighbor_states = torch.randn(1, 5, state_size)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print(f"   cell_state: {cell_state.shape}")
        print(f"   neighbor_states: {neighbor_states.shape}")
        print(f"   Результат: {sum(len(conns) for conns in result.values())} связей классифицировано")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    print("✅ Тест 5: Пустые соседи")
    # cell_state: [32], neighbor_states: [0, 32]
    neighbor_indices = []
    cell_state = torch.randn(state_size)
    neighbor_states = torch.empty(0, state_size)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print(f"   cell_state: {cell_state.shape}")
        print(f"   neighbor_states: {neighbor_states.shape}")
        print(f"   Результат: {sum(len(conns) for conns in result.values())} связей классифицировано")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")
    
    print("✅ Тест 6: Batch классификация")
    # Проверяем batch версию напрямую
    batch_size = 3
    max_neighbors = 4
    total_cells = 125  # 5*5*5
    
    cell_indices = torch.tensor([10, 20, 30])
    neighbor_indices = torch.tensor([
        [5, 15, 25, 35],
        [15, 25, 35, 45],
        [25, 35, 45, -1],  # -1 означает padding
    ])
    
    # Состояния всех клеток
    all_states = torch.randn(total_cells, state_size)
    
    try:
        batch_result = classifier.classify_connections_batch(
            cell_indices, neighbor_indices, all_states
        )
        print(f"   cell_indices: {cell_indices.shape}")
        print(f"   neighbor_indices: {neighbor_indices.shape}")
        print(f"   all_states: {all_states.shape}")
        print(f"   Результат масок:")
        for key, mask in batch_result.items():
            print(f"     {key}: {mask.shape}, активных: {mask.sum().item()}")
        print("   ✓ Успешно!\n")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}\n")


def test_error_cases():
    """Тестируем обработку ошибок"""
    print("\n🔍 Тестируем обработку ошибочных случаев\n")
    
    lattice_dims = (5, 5, 5)
    classifier = UnifiedConnectionClassifier(lattice_dims)
    state_size = 32
    
    print("❌ Тест ошибки 1: Несовместимые размеры state_size")
    cell_idx = 12
    neighbor_indices = [7, 11, 13]
    cell_state = torch.randn(32)  # state_size = 32
    neighbor_states = torch.randn(3, 64)  # state_size = 64 (неправильно!)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print("   ⚠️ Ошибка не обнаружена!\n")
    except ValueError as e:
        print(f"   ✓ Ошибка корректно обработана: {e}\n")
    except Exception as e:
        print(f"   ⚠️ Неожиданная ошибка: {e}\n")
    
    print("❌ Тест ошибки 2: Скалярный cell_state")
    cell_state = torch.tensor(1.0)  # скаляр
    neighbor_states = torch.randn(3, 32)
    
    try:
        result = classifier.classify_connections(
            cell_idx, neighbor_indices, cell_state, neighbor_states
        )
        print("   ⚠️ Ошибка не обнаружена!\n")
    except ValueError as e:
        print(f"   ✓ Ошибка корректно обработана: {e}\n")
    except Exception as e:
        print(f"   ⚠️ Неожиданная ошибка: {e}\n")


if __name__ == "__main__":
    print("="*60)
    print("Тестирование обработки размерностей в UnifiedConnectionClassifier")
    print("="*60)
    
    test_dimension_handling()
    test_error_cases()
    
    print("\n✅ Тестирование завершено!")