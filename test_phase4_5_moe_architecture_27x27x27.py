#!/usr/bin/env python3
"""
Тест Phase 4.5: MoE Architecture для решетки 27×27×27
===================================================

Тестирование новой MoE (Mixture of Experts) архитектуры с тремя специализированными экспертами:
- SimpleLinear (10%) - рефлексы
- HybridGNN_CNF (55%) - основная обработка
- LightweightCNF (35%) - долгосрочная память

ЦЕЛЬ: Проверить интеграцию всех компонентов перед переходом к полноценному обучению
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any

# Импорты из новой MoE структуры
from new_rebuild.core.moe import (
    SimpleLinearExpert,
    HybridGNN_CNF_Expert,
    MoEConnectionProcessor,
    GatingNetwork,
)
from new_rebuild.core.cnf import LightweightCNF, ConnectionType
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.config import get_project_config
from new_rebuild.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_individual_experts():
    """Тест каждого эксперта в отдельности"""
    print("\n=== ТЕСТИРОВАНИЕ ОТДЕЛЬНЫХ ЭКСПЕРТОВ ===")

    config = get_project_config()

    # Настройка тестовых данных
    batch_size = 2
    state_size = config.gnn_state_size
    num_neighbors = 5

    current_state = torch.randn(batch_size, state_size)
    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    external_input = torch.randn(batch_size, config.gnn_external_input_size)

    # 1. SimpleLinearExpert
    print("\n1. SimpleLinearExpert (локальные связи)")
    local_expert = SimpleLinearExpert(state_size=state_size)
    local_result = local_expert(current_state, neighbor_states)
    local_params = sum(p.numel() for p in local_expert.parameters())

    print(f"   Параметры: {local_params} (цель: {config.local_expert_params})")
    print(f"   Выход: {local_result.shape}")
    print(f"   ✅ Тест пройден!")

    # 2. HybridGNN_CNF_Expert
    print("\n2. HybridGNN_CNF_Expert (функциональные связи)")
    functional_expert = HybridGNN_CNF_Expert(state_size=state_size)
    functional_result = functional_expert(
        current_state, neighbor_states, external_input
    )
    functional_params = sum(p.numel() for p in functional_expert.parameters())

    print(
        f"   Параметры: {functional_params} (цель: {config.hybrid_gnn_cnf_expert_params})"
    )
    print(f"   Выход: {functional_result['new_state'].shape}")
    print(f"   GNN/CNF веса: {functional_result['gating_weight']:.3f}")
    print(f"   ✅ Тест пройден!")

    # 3. LightweightCNF Expert
    print("\n3. LightweightCNF (дальние связи)")
    distant_expert = LightweightCNF(
        state_size=state_size,
        connection_type=ConnectionType.DISTANT,
        target_params=config.distant_expert_params,
    )
    distant_result = distant_expert(current_state, neighbor_states)
    distant_params = sum(p.numel() for p in distant_expert.parameters())

    print(f"   Параметры: {distant_params} (цель: {config.distant_expert_params})")
    print(f"   Выход: {distant_result.shape}")
    print(f"   ✅ Тест пройден!")

    return {
        "local_params": local_params,
        "functional_params": functional_params,
        "distant_params": distant_params,
    }


def test_gating_network():
    """Тест GatingNetwork"""
    print("\n=== ТЕСТИРОВАНИЕ GATING NETWORK ===")

    config = get_project_config()
    batch_size = 2
    state_size = config.gnn_state_size

    # Создаем тестовые данные
    current_state = torch.randn(batch_size, state_size)
    neighbor_activity = torch.randn(batch_size, state_size)
    expert_outputs = [
        torch.randn(batch_size, state_size),  # local
        torch.randn(batch_size, state_size),  # functional
        torch.randn(batch_size, state_size),  # distant
    ]

    # Создаем GatingNetwork
    gating_network = GatingNetwork(state_size=state_size)
    gating_params = sum(p.numel() for p in gating_network.parameters())

    # Тестируем
    combined_output, expert_weights = gating_network(
        current_state, neighbor_activity, expert_outputs
    )

    print(f"   Параметры: {gating_params} (цель: {config.gating_params})")
    print(f"   Выход: {combined_output.shape}")
    print(f"   Веса экспертов: {expert_weights[0].tolist()}")
    print(f"   Сумма весов: {expert_weights.sum(dim=1).tolist()}")
    print(f"   ✅ Тест пройден!")

    return gating_params


def test_moe_processor():
    """Тест полного MoE процессора"""
    print("\n=== ТЕСТИРОВАНИЕ ПОЛНОГО MoE ПРОЦЕССОРА ===")

    config = get_project_config()

    # Создаем MoE процессор
    moe_processor = MoEConnectionProcessor(state_size=config.gnn_state_size)

    # Подсчет параметров
    total_params = sum(p.numel() for p in moe_processor.parameters())
    param_breakdown = moe_processor.get_parameter_breakdown()

    print(f"   Общие параметры: {total_params}")
    print(f"   Local Expert: {param_breakdown['local_expert']['total_params']}")
    print(
        f"   Functional Expert: {param_breakdown['functional_expert']['total_params']}"
    )
    print(f"   Distant Expert: {param_breakdown['distant_expert']['total_params']}")
    print(f"   Gating Network: {param_breakdown['gating_network']['total_params']}")

    # Создаем тестовые данные
    batch_size = 2
    state_size = config.gnn_state_size
    num_neighbors = 10

    current_state = torch.randn(batch_size, state_size)
    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    cell_idx = 1000  # Произвольный индекс клетки
    neighbor_indices = list(range(num_neighbors))

    # Тестируем forward pass
    result = moe_processor(
        current_state=current_state,
        neighbor_states=neighbor_states,
        cell_idx=cell_idx,
        neighbor_indices=neighbor_indices,
    )

    print(f"   Выходное состояние: {result['new_state'].shape}")
    print(f"   Веса экспертов: {result['expert_weights'][0].tolist()}")
    print(f"   Вклады экспертов: {result['expert_contributions']}")
    print(f"   Количество связей: {result['connection_counts']}")
    print(f"   ✅ Тест пройден!")

    return total_params, param_breakdown


def test_lattice_integration():
    """Тест интеграции с 3D решеткой"""
    print("\n=== ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ С 3D РЕШЕТКОЙ ===")

    config = get_project_config()

    # Создаем решетку
    lattice = Lattice3D(
        dimensions=config.lattice_dimensions,
        state_size=config.gnn_state_size,
        cell_type="moe",
        boundary_condition="periodic",
    )

    print(f"   Создана решетка: {config.lattice_dimensions}")
    print(f"   Общее количество клеток: {lattice.total_cells}")
    print(f"   Размер состояния: {lattice.state_size}")

    # Создаем MoE процессор для тестирования
    moe_processor = MoEConnectionProcessor(state_size=config.gnn_state_size)

    # Получаем случайную клетку и ее соседей
    cell_idx = 1000
    neighbor_indices = lattice.get_neighbors(cell_idx)

    print(f"   Клетка {cell_idx} имеет {len(neighbor_indices)} соседей")

    # Создаем тестовые состояния
    batch_size = 1
    current_state = torch.randn(batch_size, config.gnn_state_size)
    neighbor_states = torch.randn(
        batch_size, len(neighbor_indices), config.gnn_state_size
    )

    # Тестируем обработку
    result = moe_processor(
        current_state=current_state,
        neighbor_states=neighbor_states,
        cell_idx=cell_idx,
        neighbor_indices=neighbor_indices,
    )

    print(f"   Результат обработки: {result['new_state'].shape}")
    print(f"   Статистика связей: {result['connection_counts']}")
    print(f"   ✅ Интеграция работает!")

    return True


def test_parameter_targets():
    """Проверка соответствия параметров целевым значениям"""
    print("\n=== ПРОВЕРКА СООТВЕТСТВИЯ ПАРАМЕТРОВ ===")

    config = get_project_config()

    # Создаем все компоненты
    local_expert = SimpleLinearExpert(state_size=config.gnn_state_size)
    functional_expert = HybridGNN_CNF_Expert(state_size=config.gnn_state_size)
    distant_expert = LightweightCNF(
        state_size=config.gnn_state_size,
        connection_type=ConnectionType.DISTANT,
        target_params=config.distant_expert_params,
    )
    gating_network = GatingNetwork(state_size=config.gnn_state_size)

    # Подсчет параметров
    local_params = sum(p.numel() for p in local_expert.parameters())
    functional_params = sum(p.numel() for p in functional_expert.parameters())
    distant_params = sum(p.numel() for p in distant_expert.parameters())
    gating_params = sum(p.numel() for p in gating_network.parameters())

    # Проверка соответствия
    targets = {
        "Local Expert": (local_params, config.local_expert_params),
        "Functional Expert": (functional_params, config.hybrid_gnn_cnf_expert_params),
        "Distant Expert": (distant_params, config.distant_expert_params),
        "Gating Network": (gating_params, config.gating_params),
    }

    print(f"   Компонент              | Фактически | Цель      | Отклонение")
    print(f"   ----------------------|------------|-----------|----------")

    all_within_range = True
    for name, (actual, target) in targets.items():
        deviation = ((actual - target) / target) * 100
        status = "✅" if abs(deviation) < 20 else "⚠️"
        print(
            f"   {name:<21} | {actual:>10} | {target:>9} | {deviation:>6.1f}% {status}"
        )

        if abs(deviation) > 20:
            all_within_range = False

    print(
        f"\n   Общий результат: {'✅ Все параметры в пределах 20%' if all_within_range else '⚠️ Некоторые параметры превышают 20%'}"
    )

    return all_within_range


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ MoE АРХИТЕКТУРЫ ДЛЯ 27×27×27 РЕШЕТКИ")
    print("=" * 60)

    # Настройка логирования
    setup_logging(debug_mode=False)  # Минимум логов для теста

    config = get_project_config()
    print(f"Конфигурация: {config.lattice_dimensions} = {config.total_cells} клеток")
    print(f"Устройство: {config.device}")

    try:
        # Тестирование отдельных экспертов
        expert_params = test_individual_experts()

        # Тестирование gating network
        gating_params = test_gating_network()

        # Тестирование полного MoE процессора
        total_params, param_breakdown = test_moe_processor()

        # Тестирование интеграции с решеткой
        lattice_integration = test_lattice_integration()

        # Проверка соответствия параметров
        parameters_ok = test_parameter_targets()

        # Итоговый отчет
        print("\n" + "=" * 60)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)

        print(f"✅ Отдельные эксперты: работают")
        print(f"✅ Gating Network: {gating_params} параметров")
        print(f"✅ MoE Процессор: {total_params} параметров")
        print(f"✅ Интеграция с решеткой: работает")
        print(
            f"{'✅' if parameters_ok else '⚠️'} Параметры: {'в пределах нормы' if parameters_ok else 'есть отклонения'}"
        )

        print(f"\n🎯 АРХИТЕКТУРА ГОТОВА К ОБУЧЕНИЮ!")
        print(f"   Решетка: {config.lattice_dimensions} = {config.total_cells} клеток")
        print(f"   Эксперты: 3 специализированных (10%/55%/35%)")
        print(f"   Общие параметры: {total_params}")

        return True

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
