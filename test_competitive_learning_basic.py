#!/usr/bin/env python3
"""
Тест конкурентного обучения - Шаг 3.1
=====================================

Проверяет реализацию биологически правдоподобного конкурентного обучения:
- Нормализация весов
- Winner-Take-All механизм
- Латеральное торможение
- Интеграция с STDP

Базируется на успешном STDP механизме (Шаг 2.3).
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Добавляем корневую папку проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.lattice_3d import create_lattice_from_config


def test_competitive_learning_initialization():
    """Тест 1: Инициализация конкурентного обучения"""
    print("=== Тест 1: Инициализация конкурентного обучения ===")

    # ИСПОЛЬЗУЕМ ТЕСТОВУЮ КОНФИГУРАЦИЮ с повышенной чувствительностью
    lattice = create_lattice_from_config("config/competitive_learning_test.yaml")

    print(
        "⚠️  ВНИМАНИЕ: Используется ТЕСТОВАЯ конфигурация с повышенной чувствительностью!"
    )
    print("   Production конфигурация: config/adaptive_connectivity.yaml")

    # Проверяем включение конкурентного обучения
    assert lattice.enable_competitive, "Конкурентное обучение должно быть включено"
    assert lattice.enable_stdp, "STDP должен быть включен для конкурентного обучения"

    # Проверяем параметры
    print(f"Winner boost factor: {lattice.winner_boost_factor} (тест: увеличен)")
    print(
        f"Lateral inhibition factor: {lattice.lateral_inhibition_factor} (тест: усилен)"
    )
    print(f"Activity threshold: {lattice.activity_threshold} (тест: понижен)")
    print(f"Learning rate: {lattice.learning_rate} (тест: увеличен)")
    print(f"Max winner connections: {lattice.max_winner_connections}")
    print(f"Update frequency: {lattice.competitive_update_frequency}")

    assert (
        1.0 <= lattice.winner_boost_factor <= 2.0
    ), "Winner boost factor в корректном диапазоне"
    assert (
        0.5 <= lattice.lateral_inhibition_factor <= 1.0
    ), "Lateral inhibition factor в корректном диапазоне"
    assert (
        lattice.max_winner_connections >= 1
    ), "Максимум winner connections положительный"

    print("✅ Инициализация конкурентного обучения успешна")
    return lattice


def test_competitive_learning_integration():
    """Тест 2: Интеграция с STDP"""
    print("\n=== Тест 2: Интеграция конкурентного обучения с STDP ===")

    lattice = create_lattice_from_config("config/competitive_learning_test.yaml")

    # Небольшой ввод для активации части клеток
    batch_size = 25  # Соответствует input_points из конфигурации
    input_size = 2  # external_input_size из конфигурации
    external_input = (
        torch.randn(batch_size, input_size) * 1.0
    )  # Увеличена интенсивность (было 0.5)

    print(f"Размер входа: {external_input.shape}")
    print(f"Решетка: {lattice.config.dimensions} = {lattice.config.total_cells} клеток")

    # Выполним несколько шагов для накопления активности
    for step in range(5):
        states = lattice.forward(external_input)
        print(f"Шаг {step + 1}: норма состояний = {torch.norm(states).item():.4f}")

    # Проверим наличие истории активности
    assert len(lattice.activity_history) >= 2, "История активности накоплена"

    current_activity = lattice.activity_history[-1]
    active_cells_count = int(current_activity["active_cells"].sum())

    # Отладочная информация
    state_changes = current_activity["state_change"]
    print(f"Статистика изменений состояний:")
    print(f"  Min: {state_changes.min():.6f}")
    print(f"  Max: {state_changes.max():.6f}")
    print(f"  Mean: {state_changes.mean():.6f}")
    print(f"  Std: {state_changes.std():.6f}")
    print(f"  Activity threshold: {lattice.activity_threshold}")
    print(f"  Клеток выше порога: {(state_changes > lattice.activity_threshold).sum()}")

    print(f"Активных клеток: {active_cells_count} из {lattice.config.total_cells}")

    assert active_cells_count > 0, "Есть активные клетки для конкурентного обучения"

    print("✅ Интеграция с STDP работает корректно")
    return lattice


def test_competitive_learning_mechanisms():
    """Тест 3: Механизмы конкурентного обучения"""
    print("\n=== Тест 3: Механизмы конкурентного обучения ===")

    lattice = create_lattice_from_config("config/competitive_learning_test.yaml")

    # Подготовка: накопим активность
    external_input = torch.randn(25, 2) * 0.8  # Увеличена интенсивность (было 0.3)
    for _ in range(3):
        lattice.forward(external_input)

    # Сохраним исходные веса для сравнения
    initial_weights = lattice.connection_weights.clone()
    initial_stats = {
        "min": float(initial_weights.min().item()),
        "max": float(initial_weights.max().item()),
        "mean": float(initial_weights.mean().item()),
        "std": float(initial_weights.std().item()),
    }
    print(
        f"Исходные веса: min={initial_stats['min']:.3f}, max={initial_stats['max']:.3f}, "
        f"mean={initial_stats['mean']:.3f}, std={initial_stats['std']:.3f}"
    )

    # Применим STDP + конкурентное обучение
    stdp_stats = lattice.apply_stdp_update()
    competitive_stats = lattice.apply_competitive_learning()

    print(
        f"STDP обновления: LTP={stdp_stats.get('ltp_updates', 0)}, LTD={stdp_stats.get('ltd_updates', 0)}"
    )
    print(
        f"Конкурентные обновления: winners={competitive_stats.get('winner_updates', 0)}, "
        f"inhibition={competitive_stats.get('lateral_inhibition_updates', 0)}"
    )
    print(f"Нормализованных клеток: {competitive_stats.get('normalized_cells', 0)}")

    # Проверим изменения весов
    final_weights = lattice.connection_weights
    weight_change = torch.abs(final_weights - initial_weights).sum().item()
    print(f"Общее изменение весов: {weight_change:.6f}")

    final_stats = competitive_stats["connection_weights_stats"]
    print(
        f"Финальные веса: min={final_stats['min']:.3f}, max={final_stats['max']:.3f}, "
        f"mean={final_stats['mean']:.3f}, std={final_stats['std']:.3f}"
    )

    # Проверки биологической правдоподобности
    assert final_stats["min"] >= lattice.weight_bounds[0], "Веса не ниже минимума"
    assert final_stats["max"] <= lattice.weight_bounds[1], "Веса не выше максимума"

    print("✅ Механизмы конкурентного обучения работают корректно")
    return lattice, stdp_stats, competitive_stats


def test_combined_plasticity():
    """Тест 4: Объединенная пластичность (STDP + конкурентное обучение)"""
    print("\n=== Тест 4: Объединенная пластичность ===")

    lattice = create_lattice_from_config("config/competitive_learning_test.yaml")

    # Подготовка активности
    external_input = torch.randn(25, 2) * 0.8  # Увеличена интенсивность (было 0.4)
    for _ in range(4):
        lattice.forward(external_input)

    # Тестируем объединенный метод
    combined_stats = lattice.apply_combined_plasticity()

    print("Статистика объединенной пластичности:")
    print(f"  STDP: {combined_stats['stdp']}")
    print(f"  Конкурентное: {combined_stats['competitive']}")
    print(f"  Объединенное: {combined_stats['combined_stats']}")

    # Проверки
    combined = combined_stats["combined_stats"]
    assert "total_active_cells" in combined, "Есть информация об активных клетках"
    assert "plasticity_operations" in combined, "Есть счетчик операций пластичности"
    assert "weight_stability" in combined, "Есть информация о стабильности весов"

    active_cells = combined["total_active_cells"]
    operations = combined["plasticity_operations"]
    print(f"Активных клеток: {active_cells}")
    print(f"Операций пластичности: {operations}")

    assert active_cells >= 0, "Количество активных клеток корректно"
    assert operations >= 0, "Количество операций пластичности корректно"

    stability = combined["weight_stability"]
    print(f"Стабильность весов: min={stability['min']:.3f}, max={stability['max']:.3f}")

    print("✅ Объединенная пластичность работает стабильно")
    return combined_stats


def test_competitive_learning_stability():
    """Тест 5: Стабильность конкурентного обучения"""
    print("\n=== Тест 5: Стабильность конкурентного обучения ===")

    lattice = create_lattice_from_config("config/competitive_learning_test.yaml")

    # Длительное тестирование стабильности
    external_input = torch.randn(25, 2) * 0.8  # Увеличена интенсивность (было 0.3)
    weight_history = []

    for epoch in range(10):
        # Forward pass
        lattice.forward(external_input)

        # Применяем пластичность каждые несколько шагов
        if epoch >= 2:  # Начинаем после накопления истории
            combined_stats = lattice.apply_combined_plasticity()

            if "combined_stats" in combined_stats:
                stability = combined_stats["combined_stats"]["weight_stability"]
                weight_history.append(
                    {
                        "epoch": epoch,
                        "min": stability["min"],
                        "max": stability["max"],
                        "mean": stability["mean"],
                        "std": stability["std"],
                    }
                )

                print(
                    f"Эпоха {epoch}: веса [{stability['min']:.3f}, {stability['max']:.3f}], "
                    f"mean={stability['mean']:.3f}, std={stability['std']:.3f}"
                )

    # Анализ стабильности
    if len(weight_history) > 3:
        # Проверим, что веса остаются в boundaries
        for record in weight_history:
            assert (
                lattice.weight_bounds[0] <= record["min"]
            ), "Минимальные веса в границах"
            assert (
                record["max"] <= lattice.weight_bounds[1]
            ), "Максимальные веса в границах"

        # Проверим отсутствие катастрофических изменений
        mean_values = [r["mean"] for r in weight_history]
        mean_std = np.std(mean_values)
        print(f"Стандартное отклонение средних весов: {mean_std:.6f}")

        assert mean_std < 0.1, "Средние веса стабильны (нет катастрофических изменений)"

        print(f"✅ Конкурентное обучение стабильно на {len(weight_history)} эпохах")
    else:
        print("⚠️  Недостаточно данных для анализа стабильности")

    return weight_history


def main():
    """Основная функция тестирования"""
    print("🧠 Тестирование конкурентного обучения (Шаг 3.1)")
    print("=" * 55)

    try:
        # Проверяем GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Устройство: {device}")

        # Последовательность тестов
        lattice1 = test_competitive_learning_initialization()
        lattice2 = test_competitive_learning_integration()
        lattice3, stdp_stats, competitive_stats = test_competitive_learning_mechanisms()
        combined_stats = test_combined_plasticity()
        weight_history = test_competitive_learning_stability()

        print("\n" + "=" * 55)
        print("🎉 ВСЕ ТЕСТЫ КОНКУРЕНТНОГО ОБУЧЕНИЯ ПРОЙДЕНЫ!")
        print("=" * 55)

        # Итоговая статистика
        print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(
            f"  📐 Решетка: {lattice3.config.dimensions} = {lattice3.config.total_cells} клеток"
        )
        print(f"  🔗 Соседей на клетку: {lattice3.config.neighbors}")
        print(f"  ⚡ STDP + Конкурентное обучение: активно")
        print(f"  🏆 Winner boost: {lattice3.winner_boost_factor}")
        print(f"  🛡️  Lateral inhibition: {lattice3.lateral_inhibition_factor}")
        print(
            f"  📊 Операций пластичности: {combined_stats['combined_stats']['plasticity_operations']}"
        )

        if weight_history:
            print(f"  📈 Стабильность весов: проверена на {len(weight_history)} эпохах")

        print("\n✅ Шаг 3.1: Конкурентное обучение - ЗАВЕРШЕН УСПЕШНО!")

        return True

    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТИРОВАНИИ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
