#!/usr/bin/env python3
"""
Тест HybridCell (NCA + GNN) - Phase 3.3
=======================================

Тестирование гибридной архитектуры, объединяющей:
- NCA (10% влияние) - локальная стабильная динамика
- GNN (90% влияние) - глобальная коммуникация
"""

import torch
import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "new_rebuild"))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.cells import HybridCell, CellFactory
from new_rebuild.utils.logging import setup_logging, get_logger

# Настройка логирования
setup_logging(debug_mode=True)
logger = get_logger(__name__)


def test_hybrid_cell_creation():
    """Тест создания HybridCell"""
    logger.info("🧪 Тест создания HybridCell")

    # Устанавливаем конфигурацию для hybrid архитектуры
    config = ProjectConfig(
        architecture_type="hybrid",
        debug_mode=True,
        lattice_dimensions=(6, 6, 6),
        # NCA параметры (локальная динамика)
        nca_state_size=4,
        nca_hidden_dim=3,
        nca_external_input_size=1,
        nca_neighbor_count=26,
        nca_target_params=69,
        # GNN параметры (глобальная коммуникация)
        gnn_state_size=32,
        gnn_message_dim=16,
        gnn_hidden_dim=32,
        gnn_external_input_size=8,
        gnn_neighbor_count=26,
        gnn_target_params=8000,
        # Hybrid веса
        hybrid_nca_weight=0.1,
        hybrid_gnn_weight=0.9,
    )
    set_project_config(config)

    # Создаем HybridCell
    hybrid_cell = HybridCell()

    # Проверяем параметры
    total_params = sum(p.numel() for p in hybrid_cell.parameters())
    logger.info(f"✅ HybridCell создана:")
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   Target params: {hybrid_cell.target_params:,}")
    logger.info(f"   State size: {hybrid_cell.state_size}")
    logger.info(f"   NCA weight: {hybrid_cell.nca_weight:.1f}")
    logger.info(f"   GNN weight: {hybrid_cell.gnn_weight:.1f}")

    return hybrid_cell, total_params


def test_hybrid_cell_forward():
    """Тест forward pass HybridCell"""
    logger.info("🧪 Тест forward pass HybridCell")

    hybrid_cell, total_params = test_hybrid_cell_creation()

    # Подготавливаем тестовые данные
    batch_size = 2
    num_neighbors = 26
    state_size = hybrid_cell.state_size  # 32 (GNN размер)
    external_input_size = hybrid_cell.external_input_size  # 8

    # Создаем тестовые тензоры
    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    logger.info(f"📊 Входные данные:")
    logger.info(f"   neighbor_states: {neighbor_states.shape}")
    logger.info(f"   own_state: {own_state.shape}")
    logger.info(f"   external_input: {external_input.shape}")

    # Forward pass
    with torch.no_grad():
        new_state = hybrid_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
        )

    logger.info(f"📊 Выходные данные:")
    logger.info(f"   new_state: {new_state.shape}")
    logger.info(
        f"   Изменение состояния: {torch.mean(torch.abs(new_state - own_state)):.6f}"
    )

    # Проверки
    assert (
        new_state.shape == own_state.shape
    ), f"Размер выхода {new_state.shape} != размер входа {own_state.shape}"
    assert not torch.allclose(
        new_state, own_state
    ), "Состояние не изменилось (проблема с обновлением)"

    logger.info("✅ Forward pass работает корректно")

    return hybrid_cell, new_state


def test_hybrid_component_analysis():
    """Тест анализа компонентов HybridCell"""
    logger.info("🧪 Тест анализа компонентов HybridCell")

    hybrid_cell, _ = test_hybrid_cell_creation()

    # Подготавливаем тестовые данные
    batch_size = 2
    num_neighbors = 26
    state_size = hybrid_cell.state_size
    external_input_size = hybrid_cell.external_input_size

    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    # Получаем состояния отдельных компонентов
    with torch.no_grad():
        component_states = hybrid_cell.get_component_states(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
        )

    logger.info("📊 Анализ компонентов:")
    logger.info(f"   GNN state shape: {component_states['gnn_state'].shape}")
    logger.info(f"   NCA state (expanded) shape: {component_states['nca_state'].shape}")
    logger.info(
        f"   NCA state (small) shape: {component_states['nca_state_small'].shape}"
    )
    logger.info(f"   Hybrid state shape: {component_states['hybrid_state'].shape}")
    logger.info(f"   GNN weight: {component_states['gnn_weight']:.1f}")
    logger.info(f"   NCA weight: {component_states['nca_weight']:.1f}")

    # Проверяем влияние компонентов
    gnn_influence = torch.mean(torch.abs(component_states["gnn_state"] - own_state))
    nca_influence = torch.mean(torch.abs(component_states["nca_state"] - own_state))
    hybrid_influence = torch.mean(
        torch.abs(component_states["hybrid_state"] - own_state)
    )

    logger.info(f"📊 Влияние компонентов:")
    logger.info(f"   GNN influence: {gnn_influence:.6f}")
    logger.info(f"   NCA influence: {nca_influence:.6f}")
    logger.info(f"   Hybrid influence: {hybrid_influence:.6f}")

    # Проверки
    assert component_states["gnn_state"].shape == own_state.shape
    assert component_states["nca_state"].shape == own_state.shape
    assert (
        component_states["nca_state_small"].shape[1] == hybrid_cell.nca_state_size
    )  # 4
    assert component_states["hybrid_state"].shape == own_state.shape

    logger.info("✅ Анализ компонентов прошел успешно")

    return component_states


def test_cell_factory_hybrid():
    """Тест создания HybridCell через CellFactory"""
    logger.info("🧪 Тест CellFactory для HybridCell")

    config = get_project_config()
    gnn_config = config.get_gnn_config()

    # Создаем через фабрику
    hybrid_cell_factory = CellFactory.create_cell("hybrid", gnn_config)

    # Проверяем тип
    assert isinstance(
        hybrid_cell_factory, HybridCell
    ), f"Ожидался HybridCell, получен {type(hybrid_cell_factory)}"

    # Проверяем параметры
    total_params = sum(p.numel() for p in hybrid_cell_factory.parameters())
    logger.info(f"✅ CellFactory создал HybridCell:")
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   Type: {type(hybrid_cell_factory).__name__}")

    return hybrid_cell_factory


def test_parameter_analysis():
    """Детальный анализ параметров HybridCell"""
    logger.info("🧪 Детальный анализ параметров")

    hybrid_cell, total_params = test_hybrid_cell_creation()

    # Детализация параметров
    nca_params = sum(p.numel() for p in hybrid_cell.nca_cell.parameters())
    gnn_params = sum(p.numel() for p in hybrid_cell.gnn_cell.parameters())
    aligner_params = sum(p.numel() for p in hybrid_cell.state_aligner.parameters())

    logger.info("📊 Параметры по компонентам:")
    logger.info(f"   NCA component: {nca_params:,} params")
    logger.info(f"   GNN component: {gnn_params:,} params")
    logger.info(f"   State Aligner: {aligner_params:,} params")
    logger.info(f"   Total: {total_params:,} params")
    logger.info(f"   Target: {hybrid_cell.target_params:,} params")

    # Расчет эффективности
    efficiency = (
        total_params / hybrid_cell.target_params if hybrid_cell.target_params else None
    )
    logger.info(f"   Efficiency: {efficiency:.2f}x от цели")

    # Проверка целевых значений
    config = get_project_config()
    expected_nca = config.nca_target_params  # 69
    expected_gnn = config.gnn_target_params  # 8000
    expected_total = expected_nca + expected_gnn  # 8069

    logger.info("🎯 Соответствие целевым значениям:")
    logger.info(
        f"   NCA: {nca_params} / {expected_nca} = {nca_params/expected_nca:.2f}x"
    )
    logger.info(
        f"   GNN: {gnn_params} / {expected_gnn} = {gnn_params/expected_gnn:.2f}x"
    )
    logger.info(
        f"   Total: {total_params} / {expected_total} = {total_params/expected_total:.2f}x"
    )

    return {
        "total": total_params,
        "nca": nca_params,
        "gnn": gnn_params,
        "aligner": aligner_params,
        "efficiency": efficiency,
    }


def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестирования HybridCell (Phase 3.3)")

    try:
        # Тест 1: Создание клетки
        logger.info("\n" + "=" * 50)
        hybrid_cell, total_params = test_hybrid_cell_creation()

        # Тест 2: Forward pass
        logger.info("\n" + "=" * 50)
        hybrid_cell, new_state = test_hybrid_cell_forward()

        # Тест 3: Анализ компонентов
        logger.info("\n" + "=" * 50)
        component_states = test_hybrid_component_analysis()

        # Тест 4: CellFactory
        logger.info("\n" + "=" * 50)
        factory_cell = test_cell_factory_hybrid()

        # Тест 5: Детальный анализ параметров
        logger.info("\n" + "=" * 50)
        param_analysis = test_parameter_analysis()

        # Итоговая сводка
        logger.info("\n" + "🎉" + "=" * 48)
        logger.info("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("✅ HybridCell (NCA + GNN) работает корректно!")
        logger.info(
            f"✅ Архитектура: NCA ({param_analysis['nca']:,}) + GNN ({param_analysis['gnn']:,}) + Aligner ({param_analysis['aligner']:,})"
        )
        logger.info(f"✅ Общие параметры: {param_analysis['total']:,}")
        logger.info(f"✅ Эффективность: {param_analysis['efficiency']:.2f}x от цели")
        logger.info("✅ Phase 3.3 завершена успешно!")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестах: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
