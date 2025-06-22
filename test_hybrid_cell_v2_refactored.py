#!/usr/bin/env python3
"""
Тест рефакторинга HybridCellV2 - Phase 3.4
=========================================

Проверяем что рефакторинг прошел успешно:
1. Импорты новых модулей работают
2. HybridCellV2 создается корректно
3. Forward pass работает как до рефакторинга
4. Все компоненты доступны
"""

import torch
import torch.nn as nn


def test_refactored_imports():
    """Тест импортов после рефакторинга"""
    print("=== ТЕСТ ИМПОРТОВ ПОСЛЕ РЕФАКТОРИНГА ===")

    try:
        # Импорты отдельных модулей
        from new_rebuild.core.cells.nca_modulator import NCAModulator
        from new_rebuild.core.cells.modulated_gnn import ModulatedGNNCell
        from new_rebuild.core.cells.hybrid_cell_v2 import HybridCellV2

        print("✅ Отдельные импорты работают")

        # Импорт через __init__.py
        from new_rebuild.core.cells import (
            NCAModulator as NCAMod,
            ModulatedGNNCell as ModGNN,
            HybridCellV2 as HybridV2,
        )

        print("✅ Импорты через __init__.py работают")

        # Проверяем что это те же классы
        assert NCAModulator is NCAMod, "NCAModulator импорт не совпадает"
        assert ModulatedGNNCell is ModGNN, "ModulatedGNNCell импорт не совпадает"
        assert HybridCellV2 is HybridV2, "HybridCellV2 импорт не совпадает"

        print("✅ Импорты корректны")
        return True

    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


def test_nca_modulator():
    """Тест NCAModulator как отдельного модуля"""
    print("\n=== ТЕСТ NCA MODULATOR ===")

    try:
        from new_rebuild.core.cells.nca_modulator import NCAModulator

        # Создаем модулятор
        modulator = NCAModulator(nca_state_size=4, gnn_components=3)

        # Тестовый вход
        nca_state = torch.randn(2, 4)  # batch=2, nca_state_size=4

        # Forward pass
        modulation = modulator(nca_state)

        # Проверки
        assert "attention_modulation" in modulation
        assert "message_modulation" in modulation
        assert "update_modulation" in modulation

        assert modulation["attention_modulation"].shape == (2, 1)
        assert modulation["message_modulation"].shape == (2, 1)
        assert modulation["update_modulation"].shape == (2, 1)

        # Проверяем диапазон [0, 1]
        for key, value in modulation.items():
            assert torch.all(value >= 0) and torch.all(
                value <= 1
            ), f"{key} вне диапазона [0,1]"

        print(
            f"✅ NCAModulator работает: {sum(p.numel() for p in modulator.parameters())} параметров"
        )
        return True

    except Exception as e:
        print(f"❌ Ошибка NCAModulator: {e}")
        return False


def test_modulated_gnn():
    """Тест ModulatedGNNCell как отдельного модуля"""
    print("\n=== ТЕСТ MODULATED GNN ===")

    try:
        from new_rebuild.core.cells.modulated_gnn import ModulatedGNNCell

        # Создаем модулированную GNN клетку
        gnn_cell = ModulatedGNNCell(
            state_size=32,
            neighbor_count=26,
            message_dim=16,
            hidden_dim=64,
            external_input_size=8,
            activation="gelu",
            target_params=8000,
            use_attention=True,
        )

        # Тестовые входы
        batch_size = 2
        neighbor_states = torch.randn(batch_size, 26, 32)
        own_state = torch.randn(batch_size, 32)
        external_input = torch.randn(batch_size, 8)

        # Модуляция
        nca_modulation = {
            "attention_modulation": torch.sigmoid(torch.randn(batch_size, 1)),
            "message_modulation": torch.sigmoid(torch.randn(batch_size, 1)),
            "update_modulation": torch.sigmoid(torch.randn(batch_size, 1)),
        }

        # Forward pass с модуляцией
        new_state_modulated = gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=nca_modulation,
        )

        # Forward pass без модуляции
        new_state_normal = gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            nca_modulation=None,
        )

        # Проверки
        assert new_state_modulated.shape == (batch_size, 32)
        assert new_state_normal.shape == (batch_size, 32)

        # Результаты должны отличаться из-за модуляции
        assert not torch.allclose(new_state_modulated, new_state_normal, atol=1e-6)

        print(
            f"✅ ModulatedGNNCell работает: {sum(p.numel() for p in gnn_cell.parameters())} параметров"
        )
        return True

    except Exception as e:
        print(f"❌ Ошибка ModulatedGNNCell: {e}")
        return False


def test_hybrid_cell_v2():
    """Тест HybridCellV2 после рефакторинга"""
    print("\n=== ТЕСТ HYBRID CELL V2 ===")

    try:
        from new_rebuild.core.cells import HybridCellV2

        # Создаем гибридную клетку
        hybrid_cell = HybridCellV2()

        # Тестовые входы
        batch_size = 2
        neighbor_count = 26
        state_size = 32

        neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
        own_state = torch.randn(batch_size, state_size)
        external_input = torch.randn(batch_size, 8)

        # Forward pass
        new_state = hybrid_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
        )

        # Проверки
        assert new_state.shape == (batch_size, state_size)
        assert not torch.allclose(new_state, own_state)  # состояние должно измениться

        # Анализ компонентов
        analysis = hybrid_cell.get_component_analysis(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
        )

        # Проверяем что анализ содержит ожидаемые ключи
        expected_keys = [
            "nca_state",
            "nca_modulation",
            "nca_projected",
            "gnn_unmodulated",
            "gnn_modulated",
            "hybrid_final",
            "modulation_effect",
            "nca_weight",
            "gnn_weight",
        ]

        for key in expected_keys:
            assert key in analysis, f"Отсутствует ключ: {key}"

        # Информация о клетке
        info = hybrid_cell.get_info()
        assert info["architecture"] == "hybrid_v2"
        assert "nca_params" in info
        assert "gnn_params" in info
        assert "modulator_params" in info
        assert "projection_params" in info

        total_params = sum(p.numel() for p in hybrid_cell.parameters())
        print(f"✅ HybridCellV2 работает: {total_params} параметров")
        print(f"   - NCA: {info['nca_params']} параметров")
        print(f"   - GNN: {info['gnn_params']} параметров")
        print(f"   - Modulator: {info['modulator_params']} параметров")
        print(f"   - Projection: {info['projection_params']} параметров")

        return True

    except Exception as e:
        print(f"❌ Ошибка HybridCellV2: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ РЕФАКТОРИНГА HYBRID CELL V2 - PHASE 3.4")
    print("=" * 60)

    results = []

    # Тесты
    results.append(test_refactored_imports())
    results.append(test_nca_modulator())
    results.append(test_modulated_gnn())
    results.append(test_hybrid_cell_v2())

    # Результаты
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ: {passed}/{total}")
        print("✅ Рефакторинг Phase 3.4 завершен успешно!")
        print("\nСТРУКТУРА МОДУЛЕЙ:")
        print("├── nca_modulator.py     - NCAModulator класс")
        print("├── modulated_gnn.py     - ModulatedGNNCell класс")
        print("└── hybrid_cell_v2.py    - HybridCellV2 (упрощенная)")
    else:
        print(f"❌ ЕСТЬ ОШИБКИ: {passed}/{total} тестов пройдено")
        print("🔧 Требуется доработка рефакторинга")


if __name__ == "__main__":
    main()
