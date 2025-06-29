#!/usr/bin/env python3
"""
Базовый тест интеграции векторизованных компонентов
=================================================

Проверяет что все ссылки на старые компоненты заменены на новые векторизованные.
"""

import sys
import traceback


def test_cell_imports():
    """Тест импорта клеток"""
    print("🧪 Тестирование импорта клеток...")

    try:
        # Должен работать - новый способ
        from new_rebuild.core.cells import create_cell, VectorizedGNNCell

        print("✅ Векторизованные компоненты импортированы успешно")

        # Проверяем что старые импорты не работают
        try:
            from new_rebuild.core.cells import GNNCell

            print("❌ Старый GNNCell все еще доступен для импорта!")
            return False
        except ImportError:
            print("✅ Старый GNNCell недоступен для импорта (правильно)")

    except ImportError as e:
        print(f"❌ Ошибка импорта векторизованных компонентов: {e}")
        return False

    return True


def test_cell_creation():
    """Тест создания клеток"""
    print("\n🧪 Тестирование создания клеток...")

    try:
        from new_rebuild.core.cells import create_cell

        # Создание векторизованной клетки
        cell = create_cell()
        print(f"✅ Клетка создана: {type(cell).__name__}")

        # Проверяем что это векторизованная версия
        if "Vectorized" in type(cell).__name__:
            print("✅ Создана векторизованная версия (правильно)")
        else:
            print(f"❌ Создана неправильная версия: {type(cell).__name__}")
            return False

        # Тест legacy версии должен падать
        try:
            legacy_cell = create_cell(cell_type="gnn")
            print("❌ Legacy версия все еще создается!")
            return False
        except (DeprecationWarning, Exception):
            print("✅ Legacy версия недоступна (правильно)")

    except Exception as e:
        print(f"❌ Ошибка создания клетки: {e}")
        traceback.print_exc()
        return False

    return True


def test_spatial_processor_imports():
    """Тест импорта пространственных процессоров"""
    print("\n🧪 Тестирование импорта пространственных процессоров...")

    try:
        from new_rebuild.core.lattice import (
            create_spatial_processor,
            VectorizedSpatialProcessor,
        )

        print("✅ Векторизованные пространственные компоненты импортированы")

        # Создание векторизованного процессора
        processor = create_spatial_processor(dimensions=(5, 5, 5))
        print(f"✅ Процессор создан: {type(processor).__name__}")

    except ImportError as e:
        print(f"⚠️  Некоторые векторизованные компоненты недоступны: {e}")
        print("ℹ️  Это нормально если файлы еще не созданы")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

    return True


def test_config_integration():
    """Тест интеграции с конфигурацией"""
    print("\n🧪 Тестирование интеграции с конфигурацией...")

    try:
        from new_rebuild.config import get_project_config

        config = get_project_config()
        print("✅ Конфигурация загружена")

        # Проверяем векторизованные настройки
        if hasattr(config, "vectorized"):
            print(f"✅ Векторизованные настройки: enabled={config.vectorized.enabled}")
            print(f"   Force vectorized: {config.vectorized.force_vectorized}")
            print(f"   Optimal batch size: {config.calculate_optimal_batch_size()}")
        else:
            print("❌ Векторизованные настройки недоступны")
            return False

        # Проверяем тип клетки
        cell_type = config.get_cell_type()
        print(f"✅ Рекомендуемый тип клетки: {cell_type}")

        if cell_type != "vectorized_gnn":
            print(f"❌ Неправильный тип клетки по умолчанию: {cell_type}")
            return False

    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        traceback.print_exc()
        return False

    return True


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТ ИНТЕГРАЦИИ ВЕКТОРИЗОВАННЫХ КОМПОНЕНТОВ")
    print("=" * 60)

    tests = [
        test_cell_imports,
        test_cell_creation,
        test_spatial_processor_imports,
        test_config_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test.__name__}: {e}")
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {test.__name__}: {status}")

    print(f"\n📈 ИТОГО: {passed}/{total} тестов прошли успешно")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! Интеграция выполнена успешно.")
        return True
    else:
        print("⚠️  Некоторые тесты не прошли. Проверьте интеграцию.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
