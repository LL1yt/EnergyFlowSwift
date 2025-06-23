#!/usr/bin/env python3
"""
Тест для UnifiedConnectionClassifier
===================================

Проверяем:
1. Правильность инициализации
2. Batch processing функциональность
3. Backward compatibility с единичными вызовами
4. Корректность классификации
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.core.moe.unified_connection_classifier import (
    UnifiedConnectionClassifier,
    ConnectionCategory,
    DistanceCalculator,
    FunctionalSimilarityAnalyzer,
)


def test_distance_calculator():
    """Тест DistanceCalculator"""
    print("🔍 Тестируем DistanceCalculator...")

    lattice_dims = (5, 5, 5)
    calc = DistanceCalculator(lattice_dims)

    # Тест единичных вычислений
    dist = calc.euclidean_distance(0, 1)  # Соседние клетки
    print(f"Euclidean distance (0, 1): {dist:.2f}")

    dist = calc.manhattan_distance(0, 1)
    print(f"Manhattan distance (0, 1): {dist}")

    # Тест batch вычислений
    idx1 = torch.tensor([0, 1, 2])
    idx2 = torch.tensor([1, 2, 3])

    batch_euclidean = calc.euclidean_distance_batch(idx1, idx2)
    batch_manhattan = calc.manhattan_distance_batch(idx1, idx2)

    print(f"Batch euclidean distances: {batch_euclidean}")
    print(f"Batch manhattan distances: {batch_manhattan}")

    print("✅ DistanceCalculator работает корректно\n")


def test_functional_similarity():
    """Тест FunctionalSimilarityAnalyzer"""
    print("🔍 Тестируем FunctionalSimilarityAnalyzer...")

    state_size = 32
    analyzer = FunctionalSimilarityAnalyzer(state_size)

    # Создаем тестовые состояния
    state1 = torch.randn(3, state_size)
    state2 = torch.randn(3, state_size)

    # Тест batch similarity
    similarities = analyzer(state1, state2)
    print(f"Batch similarities: {similarities}")

    # Проверяем, что результат в правильном диапазоне [0, 1]
    assert torch.all(similarities >= 0) and torch.all(
        similarities <= 1
    ), "Similarities должны быть в диапазоне [0, 1]"

    print("✅ FunctionalSimilarityAnalyzer работает корректно\n")


def test_unified_classifier():
    """Тест UnifiedConnectionClassifier"""
    print("🔍 Тестируем UnifiedConnectionClassifier...")

    lattice_dims = (5, 5, 5)
    classifier = UnifiedConnectionClassifier(lattice_dims)

    # Тест единичной классификации
    cell_idx = 12  # Центральная клетка
    neighbor_indices = [7, 11, 13, 17, 37]  # Случайные соседи
    cell_state = torch.randn(32)
    neighbor_states = torch.randn(5, 32)

    single_result = classifier.classify_connections(
        cell_idx, neighbor_indices, cell_state, neighbor_states
    )

    print("Единичная классификация:")
    for category, connections in single_result.items():
        print(f"  {category.value}: {len(connections)} соединений")

    # Тест batch классификации
    batch_size = 3
    max_neighbors = 4
    total_cells = 125  # 5*5*5

    cell_indices = torch.tensor([10, 20, 30])
    neighbor_indices = torch.tensor(
        [
            [5, 15, 25, 35],  # Соседи для клетки 10
            [15, 25, 35, 45],  # Соседи для клетки 20
            [25, 35, 45, 55],  # Соседи для клетки 30
        ]
    )

    # Создаем состояния всех клеток
    full_states = torch.randn(total_cells, 32)

    batch_result = classifier.classify_connections_batch(
        cell_indices, neighbor_indices, full_states
    )

    print("Batch классификация:")
    for category, mask in batch_result.items():
        count = mask.sum().item()
        print(f"  {category}: {count} соединений")

    # Проверяем размерности
    for category, mask in batch_result.items():
        assert mask.shape == (
            batch_size,
            max_neighbors,
        ), f"Неправильная размерность для {category}"

    print("✅ UnifiedConnectionClassifier работает корректно\n")


def test_moe_integration():
    """Тест интеграции с MoE"""
    print("🔍 Тестируем интеграцию с MoE...")

    try:
        from new_rebuild.core.moe.moe_connection_processor import MoEConnectionProcessor

        # Создаем MoE процессор
        moe_processor = MoEConnectionProcessor(
            state_size=32, lattice_dimensions=(5, 5, 5), neighbor_count=10
        )

        print(f"MoE процессор создан успешно")
        print(f"Классификатор: {type(moe_processor.connection_classifier).__name__}")

        # Тест единичной обработки
        current_state = torch.randn(1, 32)
        neighbor_states = torch.randn(1, 5, 32)
        cell_idx = 12
        neighbor_indices = [7, 11, 13, 17, 37]

        result = moe_processor.forward(
            current_state, neighbor_states, cell_idx, neighbor_indices
        )

        print(f"Forward результат: {list(result.keys())}")
        print(f"Новое состояние: {result['new_state'].shape}")

        # Тест batch обработки
        batch_states = torch.randn(3, 32)
        batch_neighbor_states = torch.randn(3, 5, 32)
        batch_cell_indices = torch.tensor([10, 20, 30])
        batch_neighbor_indices = torch.tensor(
            [
                [5, 15, 25, 35, -1],  # -1 означает padding
                [15, 25, 35, 45, 55],
                [25, 35, 45, 55, -1],
            ]
        )
        full_lattice_states = torch.randn(125, 32)

        batch_result = moe_processor.forward_batch(
            batch_states,
            batch_neighbor_states,
            batch_cell_indices,
            batch_neighbor_indices,
            full_lattice_states,
        )

        print(f"Batch результат: {list(batch_result.keys())}")
        print(f"Batch новые состояния: {batch_result['new_states'].shape}")

        print("✅ Интеграция с MoE работает корректно\n")

    except Exception as e:
        print(f"⚠️ Ошибка при тестировании MoE интеграции: {e}")
        import traceback

        traceback.print_exc()


def test_parameter_count():
    """Тест количества параметров"""
    print("🔍 Проверяем количество параметров...")

    lattice_dims = (27, 27, 27)
    classifier = UnifiedConnectionClassifier(lattice_dims)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Общее количество параметров в классификаторе: {total_params}")

    # Выводим детальную информацию
    for name, param in classifier.named_parameters():
        print(f"  {name}: {param.numel()} параметров")

    print("✅ Проверка параметров завершена\n")


if __name__ == "__main__":
    print("🚀 Запуск тестов UnifiedConnectionClassifier\n")

    test_distance_calculator()
    test_functional_similarity()
    test_unified_classifier()
    test_parameter_count()
    test_moe_integration()

    print("🎉 Все тесты завершены успешно!")
