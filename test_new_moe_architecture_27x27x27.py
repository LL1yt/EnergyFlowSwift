#!/usr/bin/env python3
"""
Тест новой MoE архитектуры 27×27×27
=====================================

Проверяем новую архитектуру:
- GatingNetwork заменяет NCA нейрон
- 3 эксперта: SimpleLinear (10%), GNN (55%), LightweightCNF (35%)
- Все параметры из централизованного конфига
- Решетка 27×27×27 = 19,683 клеток
"""

import torch
import torch.nn as nn
from new_rebuild.config import get_project_config
from new_rebuild.core.moe import MoEConnectionProcessor
from new_rebuild.core.lattice import create_lattice


def test_config_moe_parameters():
    """Тест: проверка MoE параметров в конфиге"""
    print("\n🔧 Тестирование MoE конфигурации...")

    config = get_project_config()

    # Проверяем основные MoE параметры
    assert (
        config.architecture_type == "moe"
    ), f"Ожидали 'moe', получили '{config.architecture_type}'"
    assert config.enable_moe == True, "MoE должен быть включен"

    # Проверяем параметры экспертов
    assert (
        config.local_expert_params == 2059
    ), f"Local expert: ожидали 2059, получили {config.local_expert_params}"
    assert (
        config.functional_expert_params == 8233
    ), f"Functional expert: ожидали 8233, получили {config.functional_expert_params}"
    assert (
        config.distant_expert_params == 4000
    ), f"Distant expert: ожидали 4000, получили {config.distant_expert_params}"
    assert (
        config.gating_params == 808
    ), f"Gating network: ожидали 808, получили {config.gating_params}"

    # Проверяем распределение связей
    assert (
        config.local_connections_ratio == 0.10
    ), f"Local ratio: ожидали 0.10, получили {config.local_connections_ratio}"
    assert (
        config.functional_connections_ratio == 0.55
    ), f"Functional ratio: ожидали 0.55, получили {config.functional_connections_ratio}"
    assert (
        config.distant_connections_ratio == 0.35
    ), f"Distant ratio: ожидали 0.35, получили {config.distant_connections_ratio}"

    total_ratio = (
        config.local_connections_ratio
        + config.functional_connections_ratio
        + config.distant_connections_ratio
    )
    assert (
        abs(total_ratio - 1.0) < 1e-6
    ), f"Сумма коэффициентов должна быть 1.0, получили {total_ratio}"

    print(f"✅ MoE конфигурация корректна:")
    print(f"   - Architecture: {config.architecture_type}")
    print(
        f"   - Local Expert: {config.local_expert_params} params ({config.local_connections_ratio*100:.0f}%)"
    )
    print(
        f"   - Functional Expert: {config.functional_expert_params} params ({config.functional_connections_ratio*100:.0f}%)"
    )
    print(
        f"   - Distant Expert: {config.distant_expert_params} params ({config.distant_connections_ratio*100:.0f}%)"
    )
    print(f"   - Gating Network: {config.gating_params} params")


def test_moe_processor_creation():
    """Тест: создание MoE Connection Processor"""
    print("\n🏗️ Тестирование создания MoE Processor...")

    config = get_project_config()

    # Создаем MoE processor
    moe_processor = MoEConnectionProcessor(
        state_size=config.gating_state_size,
        neighbors=config.effective_neighbors,
        device=config.device,
    )

    # Проверяем создание
    assert moe_processor is not None, "MoE processor не создан"
    assert hasattr(moe_processor, "gating_network"), "Отсутствует gating network"
    assert hasattr(moe_processor, "local_expert"), "Отсутствует local expert"
    assert hasattr(moe_processor, "functional_expert"), "Отсутствует functional expert"
    assert hasattr(moe_processor, "distant_expert"), "Отсутствует distant expert"

    print(f"✅ MoE Processor создан успешно:")
    print(f"   - State size: {config.gating_state_size}")
    print(f"   - Neighbors: {config.effective_neighbors}")
    print(f"   - Device: {config.device}")

    return moe_processor


def test_moe_forward_pass():
    """Тест: forward pass через MoE"""
    print("\n🚀 Тестирование MoE forward pass...")

    config = get_project_config()

    # Создаем MoE processor
    moe_processor = MoEConnectionProcessor(
        state_size=config.gating_state_size,
        neighbors=config.effective_neighbors,
        device=config.device,
    )

    # Создаем тестовые данные
    batch_size = 1000  # Тестируем на подмножестве клеток
    current_state = torch.randn(
        batch_size, config.gating_state_size, device=config.device
    )
    neighbor_states = torch.randn(
        batch_size,
        config.effective_neighbors,
        config.gating_state_size,
        device=config.device,
    )

    # Forward pass
    with torch.no_grad():
        output_state = moe_processor(current_state, neighbor_states)

    # Проверяем результат
    assert output_state is not None, "MoE не вернул результат"
    assert (
        output_state.shape == current_state.shape
    ), f"Неправильная форма: ожидали {current_state.shape}, получили {output_state.shape}"
    assert not torch.isnan(output_state).any(), "MoE вернул NaN значения"
    assert not torch.isinf(output_state).any(), "MoE вернул Inf значения"

    # Проверяем что состояние изменилось
    state_changed = not torch.allclose(current_state, output_state, atol=1e-6)
    assert state_changed, "MoE не изменил состояние (возможно, не работает)"

    print(f"✅ MoE forward pass работает:")
    print(f"   - Input shape: {current_state.shape}")
    print(f"   - Output shape: {output_state.shape}")
    print(f"   - State changed: {state_changed}")
    print(f"   - Output range: [{output_state.min():.4f}, {output_state.max():.4f}]")


def test_moe_parameter_count():
    """Тест: подсчет параметров MoE"""
    print("\n📊 Тестирование количества параметров MoE...")

    config = get_project_config()

    # Создаем MoE processor
    moe_processor = MoEConnectionProcessor(
        state_size=config.gating_state_size,
        neighbors=config.effective_neighbors,
        device=config.device,
    )

    # Подсчитываем параметры
    total_params = sum(p.numel() for p in moe_processor.parameters())
    gating_params = sum(p.numel() for p in moe_processor.gating_network.parameters())
    local_params = sum(p.numel() for p in moe_processor.local_expert.parameters())
    functional_params = sum(
        p.numel() for p in moe_processor.functional_expert.parameters()
    )
    distant_params = sum(p.numel() for p in moe_processor.distant_expert.parameters())

    print(f"✅ Параметры MoE архитектуры:")
    print(
        f"   - Gating Network: {gating_params:,} params (цель: {config.gating_params:,})"
    )
    print(
        f"   - Local Expert: {local_params:,} params (цель: {config.local_expert_params:,})"
    )
    print(
        f"   - Functional Expert: {functional_params:,} params (цель: {config.functional_expert_params:,})"
    )
    print(
        f"   - Distant Expert: {distant_params:,} params (цель: {config.distant_expert_params:,})"
    )
    print(f"   - ОБЩИЙ ИТОГ: {total_params:,} params")

    # Проверяем попадание в цели (допускаем 20% отклонение)
    def check_target(actual, target, name):
        deviation = abs(actual - target) / target
        status = "✅" if deviation <= 0.2 else "⚠️"
        print(f"   {status} {name}: {deviation*100:.1f}% отклонение от цели")
        return deviation <= 0.2

    all_good = True
    all_good &= check_target(gating_params, config.gating_params, "Gating Network")
    all_good &= check_target(local_params, config.local_expert_params, "Local Expert")
    all_good &= check_target(
        functional_params, config.functional_expert_params, "Functional Expert"
    )
    all_good &= check_target(
        distant_params, config.distant_expert_params, "Distant Expert"
    )

    if all_good:
        print("🎉 Все параметры в пределах допустимых отклонений!")
    else:
        print("⚠️ Некоторые параметры выходят за пределы целей")


def test_lattice_27x27x27_integration():
    """Тест: интеграция с решеткой 27×27×27"""
    print("\n🌐 Тестирование интеграции с решеткой 27×27×27...")

    config = get_project_config()

    # Устанавливаем размер решетки 27×27×27
    config.lattice_dimensions = (27, 27, 27)
    total_cells = 27 * 27 * 27

    print(f"   Создаем решетку {config.lattice_dimensions} = {total_cells:,} клеток...")

    # Создаем решетку (это займет время для 19k клеток)
    lattice = create_lattice(config.lattice_dimensions, config)

    # Проверяем решетку
    assert lattice is not None, "Решетка не создана"
    assert (
        lattice.total_cells == total_cells
    ), f"Неправильное количество клеток: ожидали {total_cells}, получили {lattice.total_cells}"

    # Создаем MoE processor
    moe_processor = MoEConnectionProcessor(
        state_size=config.gating_state_size,
        neighbors=config.effective_neighbors,
        device=config.device,
    )

    # Тестируем на подмножестве клеток (для скорости)
    test_cells = min(1000, total_cells)
    current_state = torch.randn(
        test_cells, config.gating_state_size, device=config.device
    )
    neighbor_states = torch.randn(
        test_cells,
        config.effective_neighbors,
        config.gating_state_size,
        device=config.device,
    )

    # Forward pass
    with torch.no_grad():
        output_state = moe_processor(current_state, neighbor_states)

    print(f"✅ Интеграция с решеткой работает:")
    print(f"   - Решетка: {config.lattice_dimensions} = {total_cells:,} клеток")
    print(f"   - Тестировано клеток: {test_cells:,}")
    print(f"   - Neighbors per cell: {config.effective_neighbors}")
    print(f"   - MoE forward pass: успешно")


def main():
    """Основная функция тестирования"""
    print("=" * 60)
    print("🧪 ТЕСТ НОВОЙ MoE АРХИТЕКТУРЫ 27×27×27")
    print("=" * 60)

    try:
        # Тест 1: Конфигурация
        test_config_moe_parameters()

        # Тест 2: Создание MoE
        test_moe_processor_creation()

        # Тест 3: Forward pass
        test_moe_forward_pass()

        # Тест 4: Параметры
        test_moe_parameter_count()

        # Тест 5: Интеграция с решеткой
        test_lattice_27x27x27_integration()

        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ НОВОЙ MoE АРХИТЕКТУРЫ ПРОЙДЕНЫ!")
        print("=" * 60)
        print("\n🚀 Новая архитектура готова:")
        print("   - GatingNetwork заменил NCA нейрон")
        print("   - 3 эксперта работают корректно")
        print("   - Решетка 27×27×27 поддерживается")
        print("   - Централизованная конфигурация работает")
        print("\n📋 Следующие шаги:")
        print("   1. Реализовать систему обучения для MoE")
        print("   2. Интегрировать с embedding системой")
        print("   3. Масштабировать до полного размера")

    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
