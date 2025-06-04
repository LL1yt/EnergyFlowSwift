#!/usr/bin/env python3
"""
Тестирование базовой функциональности модуля Lattice 3D

Этот скрипт проверяет:
1. Импорт всех компонентов
2. Создание и валидация LatticeConfig
3. Работу системы координат Position3D  
4. Функциональность NeighborTopology
5. Загрузку конфигурации из YAML
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Тест 1: Проверка импорта всех компонентов"""
    print("🧪 Тест 1: Импорт компонентов lattice_3d")
    
    try:
        from core.lattice_3d import (
            LatticeConfig, Position3D, NeighborTopology, 
            BoundaryCondition, Face, load_lattice_config,
            create_lattice_from_config, validate_lattice_config,
            Coordinates3D, Dimensions3D, Lattice3D
        )
        print("  ✅ Все компоненты импортированы успешно")
        return True
    except Exception as e:
        print(f"  ❌ Ошибка импорта: {e}")
        return False


def test_lattice_config():
    """Тест 2: Создание и валидация LatticeConfig"""
    print("\n🧪 Тест 2: LatticeConfig функциональность")
    
    try:
        from core.lattice_3d import LatticeConfig, BoundaryCondition, Face
        
        # Тест создания базовой конфигурации
        config = LatticeConfig(
            dimensions=(3, 3, 3),
            boundary_conditions=BoundaryCondition.WALLS
        )
        
        print(f"  ✅ Базовая конфигурация создана: {config.dimensions}")
        print(f"    - Общее количество клеток: {config.total_cells}")
        print(f"    - Граничные условия: {config.boundary_conditions.value}")
        print(f"    - Устройство: {config.device}")
        
        # Тест валидации размеров
        assert config.total_cells == 27, f"Неверное количество клеток: {config.total_cells}"
        assert config.dimensions == (3, 3, 3), f"Неверные размеры: {config.dimensions}"
        
        # Тест различных граничных условий
        for bc in BoundaryCondition:
            test_config = LatticeConfig(dimensions=(5, 5, 5), boundary_conditions=bc)
            assert test_config.boundary_conditions == bc
            print(f"    - Граничные условия {bc.value}: ✅")
            
        print("  ✅ LatticeConfig все тесты пройдены")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в LatticeConfig: {e}")
        traceback.print_exc()
        return False


def test_position_3d():
    """Тест 3: Система координат Position3D"""
    print("\n🧪 Тест 3: Position3D - система координат")
    
    try:
        from core.lattice_3d import Position3D
        
        # Создаем систему координат
        pos_system = Position3D((4, 4, 4))
        print(f"  ✅ Position3D создан для решетки {pos_system.dimensions}")
        
        # Тест преобразования координат
        test_coords = [
            (0, 0, 0),    # Угол
            (1, 1, 1),    # Центр
            (3, 3, 3),    # Противоположный угол
            (2, 1, 3),    # Произвольная точка
        ]
        
        for coords in test_coords:
            # 3D -> линейный -> 3D
            linear_idx = pos_system.to_linear_index(coords)
            back_coords = pos_system.to_3d_coordinates(linear_idx)
            
            assert coords == back_coords, f"Преобразование не сохранилось: {coords} -> {linear_idx} -> {back_coords}"
            print(f"    - {coords} ↔ {linear_idx}: ✅")
            
        # Тест валидации координат
        valid_coords = (1, 1, 1)
        invalid_coords = (5, 5, 5)  # Вне границ
        
        assert pos_system.is_valid_coordinates(valid_coords), "Валидные координаты не прошли проверку"
        assert not pos_system.is_valid_coordinates(invalid_coords), "Невалидные координаты прошли проверку"
        
        # Тест расстояний
        coord1, coord2 = (0, 0, 0), (1, 1, 1)
        manhattan_dist = pos_system.manhattan_distance(coord1, coord2)
        euclidean_dist = pos_system.euclidean_distance(coord1, coord2)
        
        assert manhattan_dist == 3, f"Неверное манхэттенское расстояние: {manhattan_dist}"
        assert abs(euclidean_dist - 1.732) < 0.01, f"Неверное евклидово расстояние: {euclidean_dist}"
        
        print(f"    - Манхэттенское расстояние {coord1}-{coord2}: {manhattan_dist}")
        print(f"    - Евклидово расстояние {coord1}-{coord2}: {euclidean_dist:.3f}")
        
        print("  ✅ Position3D все тесты пройдены")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в Position3D: {e}")
        traceback.print_exc()
        return False


def test_neighbor_topology():
    """Тест 4: Топология соседства NeighborTopology"""
    print("\n🧪 Тест 4: NeighborTopology - система соседства")
    
    try:
        from core.lattice_3d import NeighborTopology, LatticeConfig, BoundaryCondition
        
        # Тест с граничными условиями "стенки"
        config = LatticeConfig(dimensions=(3, 3, 3), boundary_conditions=BoundaryCondition.WALLS)
        topology = NeighborTopology(config)
        
        print(f"  ✅ NeighborTopology создан для {config.dimensions} с {config.boundary_conditions.value}")
        
        # Тест соседства для центральной клетки (должно быть 6 соседей)
        center_coords = (1, 1, 1)
        neighbors = topology.get_neighbors(center_coords)
        
        print(f"    - Центральная клетка {center_coords} имеет {len(neighbors)} соседей")
        assert len(neighbors) == 6, f"У центральной клетки должно быть 6 соседей, а не {len(neighbors)}"
        
        expected_neighbors = [
            (0, 1, 1), (2, 1, 1),  # ±X
            (1, 0, 1), (1, 2, 1),  # ±Y  
            (1, 1, 0), (1, 1, 2),  # ±Z
        ]
        
        for expected in expected_neighbors:
            assert expected in neighbors, f"Ожидаемый сосед {expected} не найден"
            
        # Тест соседства для угловой клетки (должно быть 3 соседа)
        corner_coords = (0, 0, 0)
        corner_neighbors = topology.get_neighbors(corner_coords)
        
        print(f"    - Угловая клетка {corner_coords} имеет {len(corner_neighbors)} соседей")
        assert len(corner_neighbors) == 3, f"У угловой клетки должно быть 3 соседа, а не {len(corner_neighbors)}"
        
        # Тест периодических граничных условий
        periodic_config = LatticeConfig(dimensions=(3, 3, 3), boundary_conditions=BoundaryCondition.PERIODIC)
        periodic_topology = NeighborTopology(periodic_config)
        
        periodic_corner_neighbors = periodic_topology.get_neighbors(corner_coords)
        print(f"    - Угловая клетка (периодические): {len(periodic_corner_neighbors)} соседей")
        assert len(periodic_corner_neighbors) == 6, "В периодических условиях у всех клеток 6 соседей"
        
        # Тест валидации топологии
        stats = topology.validate_topology()
        print(f"    - Топология валидна: симметрия={stats['symmetry_check']}")
        print(f"    - Статистика соседей: {stats['neighbor_counts']}")
        
        assert stats['symmetry_check'], "Топология должна быть симметричной"
        
        print("  ✅ NeighborTopology все тесты пройдены")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в NeighborTopology: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Тест 5: Загрузка конфигурации из YAML"""
    print("\n🧪 Тест 5: Загрузка конфигурации из YAML")
    
    try:
        from core.lattice_3d import load_lattice_config, create_lattice_from_config
        
        # Тест загрузки конфигурации по умолчанию
        config = load_lattice_config()
        print(f"  ✅ Конфигурация загружена из YAML: {config.dimensions}")
        print(f"    - Граничные условия: {config.boundary_conditions.value}")
        print(f"    - GPU включен: {config.gpu_enabled}")
        print(f"    - Автосинхронизация cell_prototype: {config.auto_sync_cell_config}")
        
        # Проверяем основные параметры
        assert isinstance(config.dimensions, tuple), "Размеры должны быть tuple"
        assert len(config.dimensions) == 3, "Размеры должны быть 3D"
        assert all(dim > 0 for dim in config.dimensions), "Все размеры должны быть положительными"
        
        # Тест создания решетки из конфигурации
        lattice = create_lattice_from_config()
        print(f"  ✅ Lattice3D создан из конфигурации")
        print(f"    - Размеры решетки: {lattice.config.dimensions}")
        print(f"    - Общее количество клеток: {lattice.config.total_cells}")
        
        print("  ✅ Загрузка конфигурации все тесты пройдены")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка загрузки конфигурации: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Тест 6: Интеграция компонентов"""
    print("\n🧪 Тест 6: Интеграция всех компонентов")
    
    try:
        from core.lattice_3d import (
            LatticeConfig, Position3D, NeighborTopology, 
            BoundaryCondition, validate_lattice_config
        )
        
        # Создаем комплексную конфигурацию
        config = LatticeConfig(
            dimensions=(5, 5, 5),
            boundary_conditions=BoundaryCondition.PERIODIC,
            cache_neighbors=True,
            validate_connections=True
        )
        
        # Валидация конфигурации
        validation_result = validate_lattice_config(config)
        print(f"  ✅ Валидация конфигурации: {validation_result['valid']}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                print(f"    ⚠️  {warning}")
                
        # Создаем все компоненты
        pos_system = Position3D(config.dimensions)
        topology = NeighborTopology(config)
        
        # Проверяем совместимость
        total_positions = pos_system.total_positions
        assert total_positions == config.total_cells, "Несоответствие количества позиций"
        
        # Проверяем, что топология работает для всех позиций
        all_coords = pos_system.get_all_coordinates()
        neighbor_counts = []
        
        for coords in all_coords:
            neighbors = topology.get_neighbors(coords)
            neighbor_counts.append(len(neighbors))
            
        # В периодических условиях у всех должно быть 6 соседей
        assert all(count == 6 for count in neighbor_counts), "В периодических условиях у всех клеток должно быть 6 соседей"
        
        print(f"    - Всего позиций: {total_positions}")
        print(f"    - Проверено соседство для всех позиций: ✅")
        print(f"    - Все клетки имеют {neighbor_counts[0]} соседей")
        
        print("  ✅ Интеграция компонентов успешна")
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка интеграции: {e}")
        traceback.print_exc()
        return False


def main():
    """Основная функция запуска всех тестов"""
    print("🚀 ТЕСТИРОВАНИЕ МОДУЛЯ LATTICE 3D")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_lattice_config,
        test_position_3d,
        test_neighbor_topology,
        test_config_loading,
        test_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Тест {test_func.__name__} провален")
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test_func.__name__}: {e}")
            
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n✅ Модуль lattice_3d готов к следующему этапу разработки")
        return True
    else:
        print(f"⚠️  {total - passed} тестов провалено. Требуется исправление.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 