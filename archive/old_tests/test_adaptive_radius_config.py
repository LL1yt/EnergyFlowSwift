#!/usr/bin/env python3
"""
Тест для проверки настраиваемого adaptive_radius в ProjectConfig

Проверяем:
1. Настройка adaptive_radius_ratio работает корректно
2. Ограничения min/max соблюдаются
3. Вычисления по размеру решетки корректны
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "new_rebuild"))

from new_rebuild.config import (
    get_project_config,
    ProjectConfig,
    ModelSettings,
    LatticeSettings,
    EulerSettings,
)


def test_adaptive_radius_basic():
    """Тест базового функционала adaptive radius"""
    print("\n🧪 ТЕСТ ADAPTIVE RADIUS КОНФИГУРАЦИИ")
    print("=" * 60)

    # Создаем тестовую конфигурацию
    config = ProjectConfig()
    config.lattice_dimensions = (27, 27, 27)  # max_dim = 27
    config.adaptive_radius_ratio = 0.3  # 30%
    config.adaptive_radius_min = 1.5
    config.adaptive_radius_max = 50.0
    config.adaptive_radius_enabled = True

    print(f"📊 Размер решетки: {config.lattice_dimensions}")
    print(
        f"🔧 Adaptive radius ratio: {config.adaptive_radius_ratio} ({config.adaptive_radius_ratio*100}%)"
    )
    print(f"📏 Минимальный радиус: {config.adaptive_radius_min}")
    print(f"📏 Максимальный радиус: {config.adaptive_radius_max}")

    # Вычисляем адаптивный радиус
    calculated_radius = config.calculate_adaptive_radius()
    expected_radius = max(config.lattice_dimensions) * config.adaptive_radius_ratio

    print(f"\n📐 Ожидаемый радиус: {expected_radius} (27 * 0.3)")
    print(f"📐 Вычисленный радиус: {calculated_radius}")

    assert (
        calculated_radius == expected_radius
    ), f"Радиус {calculated_radius} != {expected_radius}"
    print("✅ Базовое вычисление корректно")


def test_adaptive_radius_limits():
    """Тест ограничений min/max для adaptive radius"""
    print("\n🔒 ТЕСТ ОГРАНИЧЕНИЙ ADAPTIVE RADIUS")
    print("=" * 60)

    # Тест минимального ограничения
    config = ProjectConfig()
    config.lattice_dimensions = (3, 3, 3)  # очень маленькая решетка
    config.adaptive_radius_ratio = 0.3  # 30% от 3 = 0.9
    config.adaptive_radius_min = 2.0  # больше чем 0.9
    config.adaptive_radius_max = 50.0

    calculated_radius = config.calculate_adaptive_radius()
    print(f"🔻 Маленькая решетка {config.lattice_dimensions}:")
    print(
        f"   Расчетный радиус: {3 * 0.3} -> ограничен до {config.adaptive_radius_min}"
    )
    print(f"   Итоговый радиус: {calculated_radius}")

    assert (
        calculated_radius == config.adaptive_radius_min
    ), f"Минимальное ограничение не работает: {calculated_radius}"
    print("✅ Минимальное ограничение работает")

    # Тест максимального ограничения
    config.lattice_dimensions = (200, 200, 200)  # очень большая решетка
    config.adaptive_radius_ratio = 0.5  # 50% от 200 = 100
    config.adaptive_radius_min = 1.0
    config.adaptive_radius_max = 25.0  # меньше чем 100

    calculated_radius = config.calculate_adaptive_radius()
    print(f"\n🔺 Большая решетка {config.lattice_dimensions}:")
    print(
        f"   Расчетный радиус: {200 * 0.5} -> ограничен до {config.adaptive_radius_max}"
    )
    print(f"   Итоговый радиус: {calculated_radius}")

    assert (
        calculated_radius == config.adaptive_radius_max
    ), f"Максимальное ограничение не работает: {calculated_radius}"
    print("✅ Максимальное ограничение работает")


def test_adaptive_radius_disabled():
    """Тест отключения adaptive radius"""
    print("\n❌ ТЕСТ ОТКЛЮЧЕНИЯ ADAPTIVE RADIUS")
    print("=" * 60)

    config = ProjectConfig()
    config.lattice_dimensions = (27, 27, 27)
    config.adaptive_radius_ratio = 0.3
    config.adaptive_radius_enabled = False  # Отключаем
    config.adaptive_radius_max = 10.0

    calculated_radius = config.calculate_adaptive_radius()
    print(f"🔧 Adaptive radius отключен")
    print(f"📐 Возвращается max радиус: {calculated_radius}")

    assert (
        calculated_radius == config.adaptive_radius_max
    ), f"При отключении должен возвращаться max: {calculated_radius}"
    print("✅ Отключение adaptive radius работает")


def test_different_ratios():
    """Тест разных соотношений adaptive_radius_ratio"""
    print("\n📊 ТЕСТ РАЗНЫХ СООТНОШЕНИЙ")
    print("=" * 60)

    config = ProjectConfig()
    config.lattice_dimensions = (50, 50, 50)  # max_dim = 50
    config.adaptive_radius_min = 0.1
    config.adaptive_radius_max = 100.0

    test_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"🏗️ Размер решетки: {config.lattice_dimensions} (max_dim = 50)")
    print(f"📏 Соотношения и результаты:")

    for ratio in test_ratios:
        config.adaptive_radius_ratio = ratio
        calculated_radius = config.calculate_adaptive_radius()
        expected = 50 * ratio

        print(
            f"   {ratio*100:3.0f}% -> {calculated_radius:5.1f} (ожидалось {expected:5.1f})"
        )
        assert calculated_radius == expected, f"Неверный расчет для ratio {ratio}"

    print("✅ Все соотношения работают корректно")


def test_neighbor_strategy_config_integration():
    """Тест интеграции с get_neighbor_strategy_config()"""
    print("\n🔗 ТЕСТ ИНТЕГРАЦИИ С NEIGHBOR STRATEGY CONFIG")
    print("=" * 60)

    config = ProjectConfig()
    config.lattice_dimensions = (30, 20, 40)  # max_dim = 40
    config.adaptive_radius_ratio = 0.25  # 25%
    config.adaptive_radius_min = 2.0
    config.adaptive_radius_max = 30.0

    neighbor_config = config.get_neighbor_strategy_config()

    print(f"📊 Neighbor strategy config содержит:")
    for key, value in neighbor_config.items():
        if "adaptive" in key:
            print(f"   {key}: {value}")

    expected_radius = 40 * 0.25  # 10.0
    assert (
        neighbor_config["adaptive_radius"] == expected_radius
    ), f"Неверный радиус в neighbor config"
    assert (
        neighbor_config["adaptive_radius_ratio"] == 0.25
    ), "Неверный ratio в neighbor config"

    print("✅ Интеграция с neighbor strategy config работает")


def test_global_config_integration():
    """Тест интеграции с глобальным экземпляром конфигурации"""
    print("\n🌐 ТЕСТ ГЛОБАЛЬНОГО ЭКЗЕМПЛЯРА CONFIG")
    print("=" * 60)

    # Создаем и устанавливаем кастомную конфигурацию
    custom_config = ProjectConfig()
    custom_config.lattice_dimensions = (15, 15, 15)
    custom_config.adaptive_radius_ratio = 0.4  # 40%
    custom_config.adaptive_radius_min = 1.0
    custom_config.adaptive_radius_max = 20.0

    set_project_config(custom_config)

    # Получаем глобальную конфигурацию
    global_config = get_project_config()
    calculated_radius = global_config.calculate_adaptive_radius()
    expected_radius = 15 * 0.4  # 6.0

    print(f"🔧 Глобальная конфигурация:")
    print(f"   Размер решетки: {global_config.lattice_dimensions}")
    print(f"   Adaptive radius ratio: {global_config.adaptive_radius_ratio}")
    print(f"   Вычисленный радиус: {calculated_radius}")

    assert (
        calculated_radius == expected_radius
    ), f"Глобальная конфигурация работает неверно"
    print("✅ Глобальная конфигурация работает корректно")


def main():
    """Запуск всех тестов adaptive radius конфигурации"""
    try:
        test_adaptive_radius_basic()
        test_adaptive_radius_limits()
        test_adaptive_radius_disabled()
        test_different_ratios()
        test_neighbor_strategy_config_integration()
        test_global_config_integration()

        print("\n🎉 ВСЕ ТЕСТЫ ADAPTIVE RADIUS ПРОШЛИ УСПЕШНО!")
        print("=" * 60)
        print("✅ Адаптивный радиус теперь полностью настраиваем")
        print("✅ Все ограничения и проверки работают")
        print("✅ Интеграция с существующим кодом функциональна")
        print("📝 Теперь можно легко менять процент радиуса в ProjectConfig")

    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
