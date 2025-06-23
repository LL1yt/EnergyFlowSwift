#!/usr/bin/env python3
"""
Тест оптимизированного SimpleLinearExpert
=========================================

Проверяем:
1. Фиксированное количество параметров независимо от соседей
2. Работу с переменным количеством соседей
3. Attention-based агрегацию
4. Настройку через централизованный конфиг
"""

import torch
import sys
import os

# Добавляем путь к new_rebuild
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "new_rebuild"))

from new_rebuild.core.moe import OptimizedSimpleLinearExpert, SimpleLinearExpert
from new_rebuild.config import get_project_config


def test_fixed_parameters():
    """Тест фиксированного количества параметров"""
    print("🧪 Тестируем фиксированную архитектуру...")

    state_size = 32
    batch_size = 4

    # Создаем эксперты
    expert = OptimizedSimpleLinearExpert(state_size)

    # Считаем параметры
    total_params = sum(p.numel() for p in expert.parameters())
    print(f"   Общее количество параметров: {total_params}")

    # Тестируем с разным количеством соседей
    neighbor_counts = [0, 1, 5, 10, 26, 50, 100]

    for num_neighbors in neighbor_counts:
        current_state = torch.randn(batch_size, state_size)
        neighbor_states = torch.randn(batch_size, num_neighbors, state_size)

        # Forward pass
        output = expert(current_state, neighbor_states)

        print(f"   ✅ {num_neighbors} соседей: output shape {output.shape}")

        # Проверяем что output имеет правильную форму
        assert output.shape == (
            batch_size,
            state_size,
        ), f"Wrong output shape for {num_neighbors} neighbors"

    print(f"   ✅ Параметры фиксированы: {total_params}")


def test_attention_mechanism():
    """Тест attention-based агрегации"""
    print("🧪 Тестируем attention механизм...")

    state_size = 32
    batch_size = 2
    num_neighbors = 5

    expert = OptimizedSimpleLinearExpert(state_size)

    # Создаем текущее состояние
    current_state = torch.randn(batch_size, state_size)

    # Создаем соседей с разной похожестью на текущее состояние
    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)

    # Делаем одного соседа очень похожим на текущее состояние
    neighbor_states[:, 0] = current_state + 0.1 * torch.randn_like(current_state)

    output = expert(current_state, neighbor_states)

    print(f"   ✅ Attention агрегация работает: output shape {output.shape}")
    assert output.shape == (batch_size, state_size)


def test_config_integration():
    """Тест интеграции с централизованным конфигом"""
    print("🧪 Тестируем интеграцию с конфигом...")

    config = get_project_config()
    local_config = config.get_local_expert_config()

    print(f"   Конфиг из centralized_config:")
    print(f"   - neighbor_agg_hidden1: {local_config['neighbor_agg_hidden1']}")
    print(f"   - neighbor_agg_hidden2: {local_config['neighbor_agg_hidden2']}")
    print(f"   - processor_hidden: {local_config['processor_hidden']}")
    print(f"   - use_attention: {local_config['use_attention']}")
    print(f"   - alpha: {local_config['alpha']}, beta: {local_config['beta']}")

    # Создаем эксперт и проверяем что параметры из конфига используются
    expert = OptimizedSimpleLinearExpert(state_size=32)
    assert expert.use_attention == local_config["use_attention"]

    print("   ✅ Конфигурация успешно применена")


def test_backward_compatibility():
    """Тест обратной совместимости"""
    print("🧪 Тестируем обратную совместимость...")

    # Проверяем что SimpleLinearExpert это alias для OptimizedSimpleLinearExpert
    assert SimpleLinearExpert is OptimizedSimpleLinearExpert

    # Создаем через старый alias
    expert = SimpleLinearExpert(state_size=32)
    assert isinstance(expert, OptimizedSimpleLinearExpert)

    print("   ✅ Обратная совместимость работает")


def test_parameter_info():
    """Тест получения информации о параметрах"""
    print("🧪 Тестируем информацию о параметрах...")

    expert = OptimizedSimpleLinearExpert(state_size=32)
    info = expert.get_parameter_info()

    print(f"   Информация о параметрах:")
    print(f"   - Всего: {info['total_params']}")
    print(f"   - Цель: {info['target_params']}")
    print(f"   - Эффективность: {info['efficiency']}")
    print(f"   - Архитектура: {info['architecture']}")
    print(f"   - Детализация: {info['breakdown']}")

    assert info["architecture"] == "fixed"
    assert info["adaptive_neighbors"] == True

    print("   ✅ Информация о параметрах корректна")


if __name__ == "__main__":
    print("🚀 Запускаем тесты OptimizedSimpleLinearExpert...")
    print()

    try:
        test_fixed_parameters()
        print()

        test_attention_mechanism()
        print()

        test_config_integration()
        print()

        test_backward_compatibility()
        print()

        test_parameter_info()
        print()

        print("🎉 Все тесты прошли успешно!")
        print("✅ OptimizedSimpleLinearExpert готов к использованию")

    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
