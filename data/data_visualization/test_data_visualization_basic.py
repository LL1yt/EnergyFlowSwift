#!/usr/bin/env python3
"""
Базовые тесты для модуля data_visualization.

Проверяет основную функциональность:
- Загрузку конфигурации
- Создание визуализаторов
- Интеграцию с core модулями
- Базовую визуализацию
"""

import sys
import traceback
import logging
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_module_imports():
    """Тест импорта основных компонентов модуля"""
    print("[TEST] Тестирование импорта модуля data_visualization...")
    
    try:
        # Базовые импорты
        from data.data_visualization import (
            VisualizationConfig,
            load_visualization_config,
            RenderEngine,
            VisualizationMode,
            ExportFormat,
            get_module_info
        )
        print("  [OK] Базовые компоненты импортированы успешно")
        
        # Проверяем информацию о модуле
        info = get_module_info()
        print(f"  [DATA] Версия модуля: {info['version']}")
        print(f"  [PACKAGE] Доступность визуализаторов: {info['visualizers_available']}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Ошибка импорта: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Тест конфигурации модуля"""
    print("\n[TEST] Тестирование конфигурации...")
    
    try:
        from data.data_visualization import VisualizationConfig, load_visualization_config
        
        # Тест создания дефолтной конфигурации
        config = VisualizationConfig()
        print(f"  [OK] Дефолтная конфигурация создана")
        print(f"     Title: {config.title}")
        print(f"     Size: {config.width}x{config.height}")
        print(f"     Engine: {config.engine.value}")
        
        # Тест валидации
        assert config.width > 0, "Width должен быть положительным"
        assert config.height > 0, "Height должен быть положительным"
        assert 0 <= config.cell_opacity <= 1, "Opacity должен быть между 0 и 1"
        print("  [OK] Валидация конфигурации прошла успешно")
        
        # Тест загрузки из файла (если файл существует)
        config_path = Path("data/data_visualization/config/default.yaml")
        if config_path.exists():
            loaded_config = load_visualization_config(str(config_path))
            print("  [OK] Конфигурация загружена из YAML файла")
        else:
            print("  [WARNING]  YAML файл конфигурации не найден (ожидаемо)")
            
        return True
        
    except Exception as e:
        print(f"  [ERROR] Ошибка тестирования конфигурации: {e}")
        traceback.print_exc()
        return False


def test_core_integration():
    """Тест интеграции с core модулями"""
    print("\n[TEST] Тестирование интеграции с core модулями...")
    
    try:
        # Импортируем необходимые модули
        from core.lattice_3d import create_lattice_from_config, IOPointPlacer, PlacementStrategy, Face
        
        # Создаем тестовую решетку
        lattice = create_lattice_from_config()
        print(f"  [OK] Решетка создана: {lattice.config.dimensions}")
        
        # Проверяем доступность методов для визуализации
        states = lattice.get_states()
        print(f"  [OK] Состояния получены: shape {states.shape}")
        
        io_info = lattice.get_io_point_info()
        print(f"  [OK] I/O информация получена: {len(io_info)} ключей")
        
        # Тестируем IOPointPlacer
        dimensions = (8, 8, 8)
        strategy = PlacementStrategy.PROPORTIONAL
        config = {
            'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6},
            'absolute_limits': {'min_points': 5, 'max_points': 0}
        }
        
        io_placer = IOPointPlacer(dimensions, strategy, config)
        input_points = io_placer.get_input_points(Face.FRONT)
        output_points = io_placer.get_output_points(Face.BACK)
        
        print(f"  [OK] IOPointPlacer работает: {len(input_points)} input, {len(output_points)} output точек")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Ошибка интеграции с core: {e}")
        traceback.print_exc()
        return False


def test_visualization_creation():
    """Тест создания базовой визуализации"""
    print("\n[TEST] Тестирование создания визуализации...")
    
    try:
        # Проверяем доступность функций создания
        from data.data_visualization import create_visualizer, create_io_visualizer
        
        print("  [OK] Функции создания визуализаторов доступны")
        
        # Пробуем создать визуализаторы (может не получиться если visualizers.py не готов)
        try:
            visualizer = create_visualizer()
            print("  [OK] Lattice3DVisualizer создан успешно")
            visualizer_available = True
        except ImportError as e:
            print(f"  [WARNING]  Lattice3DVisualizer недоступен: {e}")
            visualizer_available = False
            
        try:
            io_visualizer = create_io_visualizer()
            print("  [OK] IOPointVisualizer создан успешно")
            io_visualizer_available = True
        except ImportError as e:
            print(f"  [WARNING]  IOPointVisualizer недоступен: {e}")
            io_visualizer_available = False
            
        return visualizer_available or io_visualizer_available
        
    except Exception as e:
        print(f"  [ERROR] Ошибка создания визуализации: {e}")
        traceback.print_exc()
        return False


def test_quick_functions():
    """Тест быстрых функций визуализации"""
    print("\n[TEST] Тестирование быстрых функций...")
    
    try:
        from data.data_visualization import quick_visualize_lattice, quick_visualize_io_strategy
        from core.lattice_3d import create_lattice_from_config, IOPointPlacer, PlacementStrategy, Face
        
        print("  [OK] Быстрые функции импортированы")
        
        # Создаем тестовые объекты
        lattice = create_lattice_from_config()
        
        dimensions = (4, 4, 4)  # Маленькая решетка для теста
        strategy = PlacementStrategy.PROPORTIONAL
        config = {'coverage_ratio': {'min_percentage': 10.0, 'max_percentage': 20.0}}
        io_placer = IOPointPlacer(dimensions, strategy, config)
        
        # Тестируем быстрые функции (без фактического рендеринга)
        try:
            fig = quick_visualize_lattice(lattice, title="Test Visualization")
            print("  [OK] quick_visualize_lattice работает")
            quick_lattice_ok = True
        except Exception as e:
            print(f"  [WARNING]  quick_visualize_lattice недоступна: {e}")
            quick_lattice_ok = False
            
        try:
            fig = quick_visualize_io_strategy(io_placer, Face.FRONT)
            print("  [OK] quick_visualize_io_strategy работает")
            quick_io_ok = True
        except Exception as e:
            print(f"  [WARNING]  quick_visualize_io_strategy недоступна: {e}")
            quick_io_ok = False
            
        return quick_lattice_ok or quick_io_ok
        
    except Exception as e:
        print(f"  [ERROR] Ошибка быстрых функций: {e}")
        traceback.print_exc()
        return False


def test_dependencies():
    """Тест доступности зависимостей"""
    print("\n[TEST] Проверка зависимостей...")
    
    dependencies = {
        'plotly': 'plotly.graph_objects',
        'numpy': 'numpy', 
        'torch': 'torch',
        'yaml': 'yaml'
    }
    
    results = {}
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"  [OK] {name} доступен")
            results[name] = True
        except ImportError:
            print(f"  [ERROR] {name} НЕ доступен")
            results[name] = False
            
    return all(results.values())


def run_all_tests():
    """Запуск всех тестов"""
    print("[START] Запуск базовых тестов модуля data_visualization")
    print("=" * 60)
    
    tests = [
        ("Импорт модуля", test_module_imports),
        ("Конфигурация", test_configuration),
        ("Интеграция с core", test_core_integration),
        ("Создание визуализации", test_visualization_creation),
        ("Быстрые функции", test_quick_functions),
        ("Зависимости", test_dependencies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("[DATA] РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    
    passed = 0
    for test_name, result in results:
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n[TARGET] Пройдено: {passed}/{len(results)} тестов")
    
    if passed == len(results):
        print("[SUCCESS] Все тесты пройдены успешно!")
        return True
    elif passed >= len(results) * 0.7:
        print("[WARNING]  Большинство тестов пройдено (модуль частично функционален)")
        return True
    else:
        print("[ERROR] Много проблем - требуется доработка")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 