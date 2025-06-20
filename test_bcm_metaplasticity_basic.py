#!/usr/bin/env python3
"""
Комплексный тест BCM метапластичности (Шаг 3.2)
=================================================

Тестирует реализацию Bienenstock-Cooper-Munro (BCM) правила обучения:
- Адаптивные пороги активности
- Интеграция с STDP и конкурентным обучением
- Биологическая правдоподобность механизмов
- Стабилизация обучения через метапластичность

Результат: проверка что BCM правило работает корректно и стабилизирует сеть.
"""

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from core.lattice_3d.config import load_lattice_config
from core.lattice_3d.lattice import create_lattice_from_config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_bcm_initialization():
    """Тест 1: Инициализация BCM механизмов"""
    print("🧠 Тест 1: Инициализация BCM метапластичности...")

    # Загружаем конфигурацию
    config = load_lattice_config("config/bcm_metaplasticity_test.yaml")

    # Проверяем что конфигурация корректна
    assert config.enable_metaplasticity == True
    assert config.enable_plasticity == True
    assert config.enable_competitive_learning == True

    print(f"  ✓ Конфигурация загружена: {config.dimensions} решетка")
    print(f"  ✓ BCM параметры: tau_theta={config.bcm_config['tau_theta']}")
    print(
        f"  ✓ Диапазон порогов: [{config.bcm_config['min_threshold']}, {config.bcm_config['max_threshold']}]"
    )

    # Создаем решетку напрямую из config объекта
    from core.lattice_3d.lattice import Lattice3D

    lattice = Lattice3D(config)

    # Проверяем что BCM механизм инициализирован
    assert hasattr(lattice, "adaptive_threshold")
    assert lattice.adaptive_threshold is not None
    assert lattice.enable_metaplasticity == True

    # Проверяем начальные значения порогов
    thresholds = lattice.adaptive_threshold.thresholds
    assert thresholds.shape == (config.total_cells,)
    assert torch.all(thresholds >= config.bcm_config["min_threshold"])
    assert torch.all(thresholds <= config.bcm_config["max_threshold"])

    print(f"  ✓ Адаптивные пороги инициализированы: {thresholds.shape}")
    print(
        f"  ✓ Начальные пороги: min={thresholds.min():.4f}, max={thresholds.max():.4f}"
    )
    print(
        f"  ✓ Решетка готова: {config.total_cells} клеток, {config.neighbors} соседей"
    )

    return lattice, config


def test_bcm_threshold_adaptation():
    """Тест 2: Адаптация порогов активности"""
    print("\n🔄 Тест 2: Адаптация порогов активности...")

    lattice, config = test_bcm_initialization()

    # Запоминаем начальные пороги
    initial_thresholds = lattice.adaptive_threshold.thresholds.clone()

    # Генерируем внешний ввод для стимуляции активности
    total_cells = config.total_cells
    input_size = 2  # Из конфигурации
    num_input_cells = len(lattice.input_indices)

    # Сильный внешний сигнал для активации клеток
    external_input = (
        torch.randn(num_input_cells, input_size) * 2.0
    )  # Увеличенная амплитуда

    print(
        f"  ✓ Внешний ввод: {external_input.shape}, амплитуда={external_input.abs().mean():.3f}"
    )

    # Несколько шагов для адаптации порогов
    adaptation_steps = 10
    threshold_history = []

    for step in range(adaptation_steps):
        # Forward pass
        states = lattice.forward(external_input)

        # Записываем состояние порогов
        current_thresholds = lattice.adaptive_threshold.thresholds.clone()
        threshold_history.append(
            {
                "step": step,
                "thresholds": current_thresholds,
                "mean_threshold": current_thresholds.mean().item(),
                "std_threshold": current_thresholds.std().item(),
            }
        )

        # Изменяем внешний ввод для разнообразия активности
        external_input = torch.randn(num_input_cells, input_size) * (2.0 + step * 0.1)

    # Проверяем что пороги адаптировались
    final_thresholds = lattice.adaptive_threshold.thresholds
    threshold_change = torch.abs(final_thresholds - initial_thresholds).mean()

    print(f"  ✓ Адаптация за {adaptation_steps} шагов:")
    print(
        f"    Начальные пороги: {initial_thresholds.mean():.4f} ± {initial_thresholds.std():.4f}"
    )
    print(
        f"    Финальные пороги: {final_thresholds.mean():.4f} ± {final_thresholds.std():.4f}"
    )
    print(f"    Среднее изменение: {threshold_change:.4f}")

    # Проверяем, что пороги действительно адаптировались
    assert (
        threshold_change > 1e-4
    ), f"Пороги не адаптировались: изменение={threshold_change}"

    # Проверяем, что пороги остались в допустимых границах
    assert torch.all(final_thresholds >= config.bcm_config["min_threshold"])
    assert torch.all(final_thresholds <= config.bcm_config["max_threshold"])

    print(f"  ✓ Адаптация успешна, пороги остались в границах")

    return lattice, config, threshold_history


def test_bcm_stdp_integration():
    """Тест 3: Интеграция BCM с STDP"""
    print("\n🔗 Тест 3: Интеграция BCM с STDP...")

    lattice, config, _ = test_bcm_threshold_adaptation()

    # Накапливаем историю активности для STDP
    num_input_cells = len(lattice.input_indices)
    input_size = 2

    # Несколько шагов для накопления истории
    for step in range(5):
        external_input = torch.randn(num_input_cells, input_size) * 1.5
        lattice.forward(external_input)

    # Проверяем что история активности накоплена
    assert len(lattice.activity_history) >= 2
    print(f"  ✓ История активности: {len(lattice.activity_history)} записей")

    # Применяем BCM-enhanced STDP
    stdp_stats = lattice.apply_stdp_update()

    # Проверяем статистику STDP
    assert "stdp_updates" in stdp_stats
    assert "active_cells" in stdp_stats

    # Проверяем BCM-специфичную статистику
    if lattice.enable_metaplasticity:
        assert "bcm_updates" in stdp_stats
        assert "adaptive_thresholds" in stdp_stats

        bcm_updates = stdp_stats["bcm_updates"]
        print(f"  ✓ BCM LTP обновления: {bcm_updates.get('bcm_ltp_updates', 0)}")
        print(f"  ✓ BCM LTD обновления: {bcm_updates.get('bcm_ltd_updates', 0)}")

    # Проверяем классическую STDP статистику
    stdp_updates = stdp_stats["stdp_updates"]
    print(f"  ✓ Классический STDP LTP: {stdp_updates['ltp_updates']}")
    print(f"  ✓ Классический STDP LTD: {stdp_updates['ltd_updates']}")
    print(f"  ✓ Активные клетки: {stdp_stats['active_cells']}")
    print(f"  ✓ Общее изменение весов: {stdp_stats['total_weight_change']:.6f}")

    # Проверяем состояние весов связей
    weights_stats = stdp_stats["connection_weights_stats"]
    print(f"  ✓ Веса связей: {weights_stats['mean']:.4f} ± {weights_stats['std']:.4f}")

    return lattice, config, stdp_stats


def test_combined_plasticity():
    """Тест 4: Объединенная пластичность (STDP + Конкурентное + BCM)"""
    print("\n🎯 Тест 4: Объединенная пластичность...")

    lattice, config, _ = test_bcm_stdp_integration()

    # Применяем объединенную пластичность
    combined_stats = lattice.apply_combined_plasticity()

    # Проверяем структуру статистики
    assert "stdp" in combined_stats
    assert "competitive" in combined_stats
    assert "combined_stats" in combined_stats

    stdp_stats = combined_stats["stdp"]
    competitive_stats = combined_stats["competitive"]
    overall_stats = combined_stats["combined_stats"]

    print(f"  ✓ STDP операции:")
    if "stdp_updates" in stdp_stats:
        print(f"    LTP: {stdp_stats['stdp_updates']['ltp_updates']}")
        print(f"    LTD: {stdp_stats['stdp_updates']['ltd_updates']}")
    if "bcm_updates" in stdp_stats:
        print(f"    BCM LTP: {stdp_stats['bcm_updates']['bcm_ltp_updates']}")
        print(f"    BCM LTD: {stdp_stats['bcm_updates']['bcm_ltd_updates']}")

    print(f"  ✓ Конкурентное обучение:")
    if isinstance(competitive_stats, dict) and "winner_updates" in competitive_stats:
        print(f"    Победители: {competitive_stats['winner_updates']}")
        print(f"    Торможение: {competitive_stats['lateral_inhibition_updates']}")
        print(f"    Нормализация: {competitive_stats.get('normalized_cells', 0)}")

    print(f"  ✓ Общая статистика:")
    print(f"    Активные клетки: {overall_stats['total_active_cells']}")
    print(f"    Всего операций пластичности: {overall_stats['plasticity_operations']}")

    # Проверяем стабильность весов
    weights_stability = overall_stats["weight_stability"]
    print(
        f"    Стабильность весов: {weights_stability['mean']:.4f} ± {weights_stability['std']:.4f}"
    )

    return lattice, config, combined_stats


def test_long_term_stability():
    """Тест 5: Долгосрочная стабильность BCM"""
    print("\n⚖️ Тест 5: Долгосрочная стабильность...")

    lattice, config, _ = test_combined_plasticity()

    # Долгосрочное обучение
    num_epochs = 20
    stability_metrics = []

    num_input_cells = len(lattice.input_indices)
    input_size = 2

    print(f"  Тестирование стабильности на {num_epochs} эпохах...")

    for epoch in range(num_epochs):
        # Генерируем разнообразные внешние сигналы
        for micro_step in range(3):
            external_input = torch.randn(num_input_cells, input_size) * (
                1.0 + epoch * 0.05
            )
            lattice.forward(external_input)

        # Применяем объединенную пластичность
        combined_stats = lattice.apply_combined_plasticity()

        # Записываем метрики стабильности
        if "combined_stats" in combined_stats:
            stability = combined_stats["combined_stats"]["weight_stability"]
            thresholds = lattice.adaptive_threshold.thresholds

            stability_metrics.append(
                {
                    "epoch": epoch,
                    "weight_mean": stability["mean"],
                    "weight_std": stability["std"],
                    "threshold_mean": thresholds.mean().item(),
                    "threshold_std": thresholds.std().item(),
                    "active_cells": combined_stats["combined_stats"][
                        "total_active_cells"
                    ],
                }
            )

    # Анализ стабильности
    print(f"  ✓ Анализ стабильности:")

    final_metrics = stability_metrics[-1]
    initial_metrics = stability_metrics[0]

    weight_drift = abs(final_metrics["weight_mean"] - initial_metrics["weight_mean"])
    threshold_drift = abs(
        final_metrics["threshold_mean"] - initial_metrics["threshold_mean"]
    )

    print(f"    Дрейф весов: {weight_drift:.6f}")
    print(f"    Дрейф порогов: {threshold_drift:.6f}")
    print(
        f"    Финальные веса: {final_metrics['weight_mean']:.4f} ± {final_metrics['weight_std']:.4f}"
    )
    print(
        f"    Финальные пороги: {final_metrics['threshold_mean']:.4f} ± {final_metrics['threshold_std']:.4f}"
    )

    # Проверяем что система стабильна
    assert weight_drift < 0.5, f"Веса слишком нестабильны: дрейф={weight_drift}"
    assert threshold_drift < 0.1, f"Пороги слишком нестабильны: дрейф={threshold_drift}"

    # Проверяем что пороги остались в границах
    final_thresholds = lattice.adaptive_threshold.thresholds
    assert torch.all(final_thresholds >= config.bcm_config["min_threshold"])
    assert torch.all(final_thresholds <= config.bcm_config["max_threshold"])

    print(f"  ✓ Долгосрочная стабильность подтверждена")

    return lattice, config, stability_metrics


def main():
    """Основная функция тестирования BCM метапластичности"""
    print("🧠 Комплексный тест BCM метапластичности (Шаг 3.2)")
    print("=" * 60)

    start_time = time.time()

    try:
        # Выполняем все тесты последовательно
        test_bcm_initialization()
        test_bcm_threshold_adaptation()
        test_bcm_stdp_integration()
        test_combined_plasticity()
        lattice, config, stability_metrics = test_long_term_stability()

        # Финальная сводка
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        print(f"📊 Финальная статистика:")
        print(f"  Решетка: {config.dimensions} = {config.total_cells} клеток")
        print(f"  Соседей на клетку: {config.neighbors}")
        print(f"  BCM tau_theta: {config.bcm_config['tau_theta']}")
        print(f"  Финальная стабильность: достигнута")

        # Итоговое состояние адаптивных порогов
        final_thresholds = lattice.adaptive_threshold.thresholds
        print(
            f"  Адаптивные пороги: {final_thresholds.mean():.4f} ± {final_thresholds.std():.4f}"
        )

        # Производительность
        total_time = time.time() - start_time
        print(f"  Время выполнения: {total_time:.2f}s")

        print("\n✅ BCM МЕТАПЛАСТИЧНОСТЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
