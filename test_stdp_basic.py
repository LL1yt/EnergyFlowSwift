#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Базовый тест для STDP механизма в Lattice3D
==================================================

Проверяет:
1. Корректную инициализацию STDP параметров
2. Отслеживание активности клеток
3. Применение STDP правил для обновления весов
4. Биологическую правдоподобность обновлений

Использует тестовую конфигурацию config/adaptive_connectivity.yaml
"""

import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent))

from core.lattice_3d import create_lattice_from_config


def test_stdp_initialization():
    """Тест 1: Проверка инициализации STDP механизма"""
    print("🧠 Тест 1: Инициализация STDP механизма")

    # Загружаем конфигурацию
    config_path = "config/adaptive_connectivity.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Конфигурация не найдена: {config_path}")
        return False

    try:
        # === ОТЛАДКА: Проверяем что загружается ===
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        print(f"🔍 Отладка конфигурации:")
        print(f"   cell_prototype есть: {'cell_prototype' in raw_config}")
        print(f"   cell есть: {'cell' in raw_config}")
        if "cell_prototype" in raw_config:
            print(f"   cell_prototype содержимое: {raw_config['cell_prototype']}")
        if "cell" in raw_config:
            print(f"   cell содержимое: {raw_config['cell']}")

        lattice = create_lattice_from_config(config_path=config_path)

        # === ОТЛАДКА: Проверяем что попало в lattice ===
        print(f"🔍 Lattice config отладка:")
        print(f"   cell_config: {lattice.config.cell_config}")
        print(f"   neighbors в lattice: {lattice.config.neighbors}")
        print(f"   cell_prototype state_size: {lattice.cell_prototype.state_size}")
        print(f"   cell_prototype тип: {type(lattice.cell_prototype)}")

        # Проверяем, что STDP включен
        assert lattice.enable_stdp == True, "STDP должен быть включен"

        # Проверяем наличие STDP параметров
        assert hasattr(lattice, "activity_threshold"), "Отсутствует activity_threshold"
        assert hasattr(lattice, "learning_rate"), "Отсутствует learning_rate"
        assert hasattr(lattice, "A_plus"), "Отсутствует A_plus"
        assert hasattr(lattice, "A_minus"), "Отсутствует A_minus"
        assert hasattr(lattice, "weight_bounds"), "Отсутствует weight_bounds"

        # Проверяем previous_states
        assert lattice.previous_states is not None, "previous_states не инициализирован"
        assert (
            lattice.previous_states.shape == lattice.states.shape
        ), "Размерность previous_states некорректна"

        # Проверяем activity_history
        assert (
            lattice.activity_history is not None
        ), "activity_history не инициализирован"
        assert (
            lattice.activity_history.maxlen == 10
        ), "Некорректный размер activity_history"

        print(f"✅ STDP инициализация успешна")
        print(f"   - Активный порог: {lattice.activity_threshold}")
        print(f"   - Скорость обучения: {lattice.learning_rate}")
        print(f"   - Диапазон весов: {lattice.weight_bounds}")
        print(f"   - Размер решетки: {lattice.config.dimensions}")
        print(f"   - Общее количество клеток: {lattice.config.total_cells}")

        return True

    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False


def test_activity_tracking():
    """Тест 2: Проверка отслеживания активности"""
    print("\n📊 Тест 2: Отслеживание активности")

    try:
        lattice = create_lattice_from_config(
            config_path="config/adaptive_connectivity.yaml"
        )

        # Начальное состояние - без активности
        initial_history_len = len(lattice.activity_history)

        # Выполняем несколько forward passes
        for step in range(5):
            # Добавляем небольшое внешнее возмущение для активации
            external_input = (
                torch.randn(
                    len(lattice.input_indices),
                    lattice.cell_prototype.external_input_size,
                    device=lattice.device,
                )
                * 0.1
            )

            lattice.forward(external_input)

            # Проверяем, что история растет
            assert (
                len(lattice.activity_history) == step + 1
            ), f"Неправильный размер истории на шаге {step}"

            # Проверяем структуру записи активности
            last_activity = lattice.activity_history[-1]
            assert "step" in last_activity, "Отсутствует 'step' в записи активности"
            assert (
                "active_cells" in last_activity
            ), "Отсутствует 'active_cells' в записи активности"
            assert (
                "state_change" in last_activity
            ), "Отсутствует 'state_change' в записи активности"

            # Проверяем размерности
            assert (
                len(last_activity["active_cells"]) == lattice.config.total_cells
            ), "Неправильный размер active_cells"
            assert (
                len(last_activity["state_change"]) == lattice.config.total_cells
            ), "Неправильный размер state_change"

        print(f"✅ Отслеживание активности работает корректно")
        print(f"   - Записей в истории: {len(lattice.activity_history)}")

        # Статистика активности
        last_activity = lattice.activity_history[-1]
        active_count = np.sum(last_activity["active_cells"])
        avg_state_change = np.mean(last_activity["state_change"])

        print(f"   - Активных клеток: {active_count}/{lattice.config.total_cells}")
        print(f"   - Среднее изменение состояния: {avg_state_change:.4f}")

        return True

    except Exception as e:
        print(f"❌ Ошибка отслеживания активности: {e}")
        return False


def test_stdp_update():
    """Тест 3: Проверка STDP обновлений"""
    print("\n⚡ Тест 3: STDP обновление весов")

    try:
        lattice = create_lattice_from_config(
            config_path="config/adaptive_connectivity.yaml"
        )

        # Сохраняем исходные веса для сравнения
        initial_weights = lattice.connection_weights.clone()
        initial_mean = float(initial_weights.mean())
        initial_std = float(initial_weights.std())

        print(f"   Исходные веса - mean: {initial_mean:.4f}, std: {initial_std:.4f}")

        # Выполняем forward passes для создания истории активности
        for step in range(3):
            # Добавляем сильное внешнее возмущение для гарантии активности
            external_input = (
                torch.randn(
                    len(lattice.input_indices),
                    lattice.cell_prototype.external_input_size,
                    device=lattice.device,
                )
                * 0.5
            )
            lattice.forward(external_input)

        # Проверяем, что есть достаточно истории для STDP
        assert len(lattice.activity_history) >= 2, "Недостаточно истории для STDP"

        # Применяем STDP обновление
        stdp_stats = lattice.apply_stdp_update()

        # Проверяем статистику STDP
        assert isinstance(stdp_stats, dict), "STDP должно возвращать словарь статистики"
        required_keys = [
            "active_cells",
            "ltp_updates",
            "ltd_updates",
            "total_weight_change",
        ]
        for key in required_keys:
            assert key in stdp_stats, f"Отсутствует ключ {key} в статистике STDP"

        print(f"✅ STDP обновление выполнено")
        print(f"   - Активных клеток: {stdp_stats['active_cells']}")
        print(f"   - LTP обновлений: {stdp_stats['ltp_updates']}")
        print(f"   - LTD обновлений: {stdp_stats['ltd_updates']}")
        print(f"   - Общее изменение весов: {stdp_stats['total_weight_change']:.6f}")

        # Проверяем, что веса изменились (если была активность)
        final_weights = lattice.connection_weights
        if stdp_stats["active_cells"] > 0:
            weight_diff = torch.norm(final_weights - initial_weights).item()
            assert weight_diff > 0, "Веса должны измениться при наличии активности"
            print(f"   - Норма изменения весов: {weight_diff:.6f}")

        # Проверяем bounds checking
        assert (
            float(final_weights.min()) >= lattice.weight_bounds[0]
        ), "Веса нарушают нижнюю границу"
        assert (
            float(final_weights.max()) <= lattice.weight_bounds[1]
        ), "Веса нарушают верхнюю границу"

        # Статистика финальных весов
        if "connection_weights_stats" in stdp_stats:
            final_stats = stdp_stats["connection_weights_stats"]
            print(
                f"   Финальные веса - mean: {final_stats['mean']:.4f}, std: {final_stats['std']:.4f}"
            )
            print(
                f"                  - min: {final_stats['min']:.4f}, max: {final_stats['max']:.4f}"
            )

        return True

    except Exception as e:
        print(f"❌ Ошибка STDP обновления: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_biological_plausibility():
    """Тест 4: Проверка биологической правдоподобности"""
    print("\n🧬 Тест 4: Биологическая правдоподобность")

    try:
        lattice = create_lattice_from_config(
            config_path="config/adaptive_connectivity.yaml"
        )

        # Проверяем диапазон параметров
        assert (
            0.001 <= lattice.learning_rate <= 0.1
        ), "Скорость обучения не в биологическом диапазоне"
        assert 0.001 <= lattice.A_plus <= 0.1, "A_plus не в биологическом диапазоне"
        assert 0.001 <= lattice.A_minus <= 0.1, "A_minus не в биологическом диапазоне"
        assert 1 <= lattice.tau_plus <= 100, "tau_plus не в биологическом диапазоне"
        assert 1 <= lattice.tau_minus <= 100, "tau_minus не в биологическом диапазоне"

        # Проверяем размер решетки (должен быть управляемым для тестирования)
        total_cells = lattice.config.total_cells
        assert (
            1000 <= total_cells <= 10000
        ), f"Размер решетки {total_cells} не подходит для тестирования"

        # Проверяем соотношение соседей (биологически правдоподобно)
        neighbors_ratio = lattice.config.neighbors / 26  # Максимум 26 в 3D решетке
        assert (
            0.5 <= neighbors_ratio <= 1.0
        ), "Количество соседей не биологически правдоподобно"

        print(f"✅ Биологическая правдоподобность подтверждена")
        print(
            f"   - Размер решетки: {lattice.config.dimensions} ({total_cells} клеток)"
        )
        print(f"   - Соседей на клетку: {lattice.config.neighbors}")
        print(f"   - Соотношение соседей: {neighbors_ratio:.2f}")
        print(
            f"   - Временные константы: τ+ = {lattice.tau_plus}, τ- = {lattice.tau_minus}"
        )

        return True

    except Exception as e:
        print(f"❌ Ошибка проверки биологической правдоподобности: {e}")
        return False


def main():
    """Запуск всех тестов STDP механизма"""
    print("🚀 Запуск тестов STDP механизма (Шаг 2.3)")
    print("=" * 60)

    tests = [
        test_stdp_initialization,
        test_activity_tracking,
        test_stdp_update,
        test_biological_plausibility,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Тест {test.__name__} провалился!")

    print("\n" + "=" * 60)
    print(f"📊 Результаты: {passed}/{total} тестов прошли успешно")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! STDP механизм готов к использованию")
        return True
    else:
        print("⚠️  Некоторые тесты провалились. Требуются исправления.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
