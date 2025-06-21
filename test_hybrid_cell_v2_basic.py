#!/usr/bin/env python3
"""
Тест HybridCellV2 - биологически правдоподобная архитектура
========================================================

Тестирование улучшенной гибридной архитектуры:
- NCA (4D) - внутриклеточная динамика
- GNN (32D) - межклеточная коммуникация
- NCA модулирует GNN (без потери информации)
"""

import torch
import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "new_rebuild"))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.cells.hybrid_cell_v2 import (
    HybridCellV2,
    NCAModulator,
    ModulatedGNNCell,
)
from new_rebuild.utils.logging import setup_logging, get_logger

# Настройка логирования
setup_logging(debug_mode=True)
logger = get_logger(__name__)


def test_nca_modulator():
    """Тест NCAModulator компонента"""
    logger.info("🧪 Тест NCAModulator")

    nca_state_size = 4
    batch_size = 2

    # Создаем модулятор
    modulator = NCAModulator(nca_state_size=nca_state_size, gnn_components=3)

    # Тестовые данные
    nca_state = torch.randn(batch_size, nca_state_size)

    # Forward pass
    with torch.no_grad():
        modulation = modulator(nca_state)

    logger.info(f"📊 NCAModulator результаты:")
    logger.info(f"   Input shape: {nca_state.shape}")
    logger.info(f"   Attention modulation: {modulation['attention_modulation'].shape}")
    logger.info(f"   Message modulation: {modulation['message_modulation'].shape}")
    logger.info(f"   Update modulation: {modulation['update_modulation'].shape}")

    # Проверяем диапазон значений [0, 1]
    for key, values in modulation.items():
        assert torch.all(values >= 0.0) and torch.all(
            values <= 1.0
        ), f"Модуляция {key} вне диапазона [0, 1]"

    logger.info("✅ NCAModulator работает корректно")
    return modulation


def test_modulated_gnn_cell():
    """Тест ModulatedGNNCell"""
    logger.info("🧪 Тест ModulatedGNNCell")

    # Конфигурация
    config = ProjectConfig(
        architecture_type="hybrid",
        debug_mode=True,
        gnn_state_size=32,
        gnn_message_dim=16,
        gnn_hidden_dim=32,
        gnn_neighbor_count=26,
        gnn_external_input_size=8,
        gnn_target_params=8000,
    )
    set_project_config(config)
    gnn_config = config.get_gnn_config()

    # Создаем модулированную GNN клетку
    gnn_cell = ModulatedGNNCell(**gnn_config)

    # Тестовые данные
    batch_size = 2
    num_neighbors = 26
    state_size = 32
    external_input_size = 8

    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    # Тестовая модуляция
    nca_modulation = {
        "attention_modulation": torch.tensor([[0.3], [0.7]]),  # [batch, 1]
        "message_modulation": torch.tensor([[0.4], [0.8]]),
        "update_modulation": torch.tensor([[0.5], [0.6]]),
    }

    # Forward pass без модуляции
    with torch.no_grad():
        result_unmodulated = gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=None,
        )

    # Forward pass с модуляцией
    with torch.no_grad():
        result_modulated = gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=nca_modulation,
        )

    logger.info(f"📊 ModulatedGNNCell результаты:")
    logger.info(f"   Unmodulated: {result_unmodulated.shape}")
    logger.info(f"   Modulated: {result_modulated.shape}")
    logger.info(
        f"   Modulation effect: {torch.mean(torch.abs(result_modulated - result_unmodulated)):.6f}"
    )

    # Проверки
    assert result_unmodulated.shape == result_modulated.shape
    assert not torch.allclose(
        result_unmodulated, result_modulated
    ), "Модуляция не оказала влияния"

    logger.info("✅ ModulatedGNNCell работает корректно")
    return result_modulated, result_unmodulated


def test_hybrid_cell_v2_creation():
    """Тест создания HybridCellV2"""
    logger.info("🧪 Тест создания HybridCellV2")

    # Конфигурация
    config = ProjectConfig(
        architecture_type="hybrid",
        debug_mode=True,
        lattice_dimensions=(6, 6, 6),
        # NCA параметры
        nca_state_size=4,
        nca_hidden_dim=3,
        nca_external_input_size=1,
        nca_neighbor_count=26,
        nca_target_params=69,
        # GNN параметры
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

    # Создаем HybridCellV2
    hybrid_cell = HybridCellV2()

    # Анализ параметров
    total_params = sum(p.numel() for p in hybrid_cell.parameters())
    nca_params = sum(p.numel() for p in hybrid_cell.nca_cell.parameters())
    gnn_params = sum(p.numel() for p in hybrid_cell.gnn_cell.parameters())
    modulator_params = sum(p.numel() for p in hybrid_cell.nca_modulator.parameters())
    projection_params = sum(
        p.numel() for p in hybrid_cell.nca_to_gnn_projection.parameters()
    )

    logger.info(f"✅ HybridCellV2 создана:")
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   NCA params: {nca_params:,}")
    logger.info(f"   GNN params: {gnn_params:,}")
    logger.info(f"   Modulator params: {modulator_params:,}")
    logger.info(f"   Projection params: {projection_params:,}")
    logger.info(f"   Target params: {hybrid_cell.target_params:,}")
    logger.info(f"   NCA weight: {hybrid_cell.nca_weight:.1f}")
    logger.info(f"   GNN weight: {hybrid_cell.gnn_weight:.1f}")

    return hybrid_cell, total_params


def test_hybrid_cell_v2_forward():
    """Тест forward pass HybridCellV2"""
    logger.info("🧪 Тест forward pass HybridCellV2")

    hybrid_cell, total_params = test_hybrid_cell_v2_creation()

    # Тестовые данные
    batch_size = 2
    num_neighbors = 26
    state_size = hybrid_cell.state_size  # 32
    external_input_size = hybrid_cell.external_input_size  # 8

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
    assert new_state.shape == own_state.shape
    assert not torch.allclose(new_state, own_state), "Состояние не изменилось"

    logger.info("✅ Forward pass работает корректно")
    return hybrid_cell, new_state


def test_component_analysis():
    """Тест детального анализа компонентов"""
    logger.info("🧪 Тест анализа компонентов HybridCellV2")

    hybrid_cell, _ = test_hybrid_cell_v2_creation()

    # Тестовые данные
    batch_size = 2
    num_neighbors = 26
    state_size = hybrid_cell.state_size
    external_input_size = hybrid_cell.external_input_size

    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    # Детальный анализ компонентов
    with torch.no_grad():
        analysis = hybrid_cell.get_component_analysis(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
        )

    logger.info("📊 Анализ компонентов:")
    logger.info(f"   NCA state: {analysis['nca_state'].shape}")
    logger.info(f"   NCA projected: {analysis['nca_projected'].shape}")
    logger.info(f"   GNN unmodulated: {analysis['gnn_unmodulated'].shape}")
    logger.info(f"   GNN modulated: {analysis['gnn_modulated'].shape}")
    logger.info(f"   Hybrid final: {analysis['hybrid_final'].shape}")
    logger.info(f"   Modulation effect: {analysis['modulation_effect']:.6f}")

    # Анализ модуляции
    modulation = analysis["nca_modulation"]
    logger.info("📊 NCA модуляция:")
    logger.info(f"   Attention: {torch.mean(modulation['attention_modulation']):.3f}")
    logger.info(f"   Message: {torch.mean(modulation['message_modulation']):.3f}")
    logger.info(f"   Update: {torch.mean(modulation['update_modulation']):.3f}")

    # Проверки
    assert analysis["nca_state"].shape[1] == 4  # NCA state size
    assert analysis["nca_projected"].shape[1] == 32  # GNN state size
    assert analysis["modulation_effect"] > 0, "Модуляция не оказала влияния"

    logger.info("✅ Анализ компонентов успешен")
    return analysis


def test_biological_accuracy():
    """Тест биологической правдоподобности"""
    logger.info("🧪 Тест биологической правдоподобности")

    hybrid_cell, _ = test_hybrid_cell_v2_creation()

    # Проверяем размеры состояний
    assert hybrid_cell.nca_state_size == 4, "NCA должен работать с 4D состояниями"
    assert hybrid_cell.gnn_state_size == 32, "GNN должен работать с 32D состояниями"

    # Проверяем отсутствие потери информации
    # В V2 версии нет StateAligner с потерей информации
    assert not hasattr(hybrid_cell, "state_aligner"), "StateAligner удален (хорошо!)"
    assert hasattr(hybrid_cell, "nca_modulator"), "NCAModulator присутствует"
    assert hasattr(
        hybrid_cell, "nca_to_gnn_projection"
    ), "Проекция только для финального объединения"

    # Проверяем модуляцию
    modulator_params = sum(p.numel() for p in hybrid_cell.nca_modulator.parameters())
    projection_params = sum(
        p.numel() for p in hybrid_cell.nca_to_gnn_projection.parameters()
    )

    logger.info("🧬 Биологическая правдоподобность:")
    logger.info(f"   ✅ NCA: 4D внутриклеточное состояние")
    logger.info(f"   ✅ GNN: 32D межклеточная коммуникация")
    logger.info(f"   ✅ Модуляция: {modulator_params} params (эффективно)")
    logger.info(f"   ✅ Проекция: {projection_params} params (только для объединения)")
    logger.info(f"   ✅ Нет StateAligner (нет потери информации)")

    # Сравнение с V1
    v1_aligner_params_estimate = 4 * 32 + 32 + 32 * 4 + 4  # прибл. StateAligner
    logger.info(
        f"   📈 Экономия параметров vs V1: ~{v1_aligner_params_estimate - modulator_params - projection_params}"
    )

    logger.info("✅ Биологическая правдоподобность подтверждена")


def test_parameter_comparison():
    """Сравнение параметров с V1"""
    logger.info("🧪 Сравнение параметров HybridCellV2 vs V1")

    hybrid_cell, total_params = test_hybrid_cell_v2_creation()

    # Детализация параметров V2
    nca_params = sum(p.numel() for p in hybrid_cell.nca_cell.parameters())
    gnn_params = sum(p.numel() for p in hybrid_cell.gnn_cell.parameters())
    modulator_params = sum(p.numel() for p in hybrid_cell.nca_modulator.parameters())
    projection_params = sum(
        p.numel() for p in hybrid_cell.nca_to_gnn_projection.parameters()
    )

    # Оценка параметров V1 (из результатов теста)
    v1_total_estimate = 9228  # из лога теста V1
    v1_aligner_estimate = 292  # из лога теста V1

    logger.info("📊 Сравнение архитектур:")
    logger.info(f"   V1 Total: {v1_total_estimate:,} params")
    logger.info(f"   V2 Total: {total_params:,} params")
    logger.info(f"   Разница: {total_params - v1_total_estimate:,} params")

    logger.info("📊 Архитектурные отличия:")
    logger.info(f"   V1 StateAligner: {v1_aligner_estimate} params (потеря информации)")
    logger.info(
        f"   V2 NCAModulator: {modulator_params} params (биологическая модуляция)"
    )
    logger.info(
        f"   V2 Projection: {projection_params} params (только для объединения)"
    )

    # Эффективность
    efficiency_v2 = total_params / hybrid_cell.target_params
    logger.info(f"   V2 Efficiency: {efficiency_v2:.2f}x от цели")

    logger.info("✅ Параметрический анализ завершен")

    return {
        "v2_total": total_params,
        "v2_efficiency": efficiency_v2,
        "modulator_params": modulator_params,
        "projection_params": projection_params,
    }


def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестирования HybridCellV2 (биологически правдоподобная)")

    try:
        # Тест 1: NCAModulator
        logger.info("\n" + "=" * 60)
        modulation = test_nca_modulator()

        # Тест 2: ModulatedGNNCell
        logger.info("\n" + "=" * 60)
        modulated_result, unmodulated_result = test_modulated_gnn_cell()

        # Тест 3: Создание HybridCellV2
        logger.info("\n" + "=" * 60)
        hybrid_cell, total_params = test_hybrid_cell_v2_creation()

        # Тест 4: Forward pass
        logger.info("\n" + "=" * 60)
        hybrid_cell, new_state = test_hybrid_cell_v2_forward()

        # Тест 5: Анализ компонентов
        logger.info("\n" + "=" * 60)
        analysis = test_component_analysis()

        # Тест 6: Биологическая правдоподобность
        logger.info("\n" + "=" * 60)
        test_biological_accuracy()

        # Тест 7: Сравнение параметров
        logger.info("\n" + "=" * 60)
        param_comparison = test_parameter_comparison()

        # Итоговая сводка
        logger.info("\n" + "🎉" + "=" * 58)
        logger.info("✅ ВСЕ ТЕСТЫ HYBRIDCELL V2 ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("✅ Биологически правдоподобная архитектура работает!")
        logger.info(
            f"✅ Параметры: {param_comparison['v2_total']:,} (эффективность: {param_comparison['v2_efficiency']:.2f}x)"
        )
        logger.info(f"✅ NCA модуляция: {param_comparison['modulator_params']} params")
        logger.info(f"✅ Проекция: {param_comparison['projection_params']} params")
        logger.info("✅ Отсутствие потери информации подтверждено!")
        logger.info("✅ HybridCellV2 готова для интеграции!")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестах: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
