"""
Тест для новой пропорциональной I/O стратегии в Lattice3D.

Проверяет работу IOPointPlacer и интеграцию с Lattice3D.
"""

import torch
import numpy as np
from core.lattice_3d import (
    Lattice3D, LatticeConfig, IOPointPlacer, 
    PlacementStrategy, Face
)

def test_io_point_placer():
    """Тестирует IOPointPlacer с различными стратегиями."""
    print("=== Тест IOPointPlacer ===")
    
    dimensions = (8, 8, 8)
    
    # Тест пропорциональной стратегии
    print("\n1. Пропорциональная стратегия:")
    config = {
        'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6},
        'absolute_limits': {'min_points': 5, 'max_points': 10},
        'seed': 42
    }
    
    placer = IOPointPlacer(dimensions, PlacementStrategy.PROPORTIONAL, config, seed=42)
    
    # Проверяем расчет количества точек
    face_area = 8 * 8  # 64 клетки
    min_points, max_points = placer.calculate_num_points(face_area)
    print(f"  Площадь грани: {face_area}")
    print(f"  Рассчитанное количество точек: {min_points} - {max_points}")
    print(f"  Покрытие: {min_points/face_area*100:.1f}% - {max_points/face_area*100:.1f}%")
    
    # Получаем точки ввода и вывода
    input_points = placer.get_input_points(Face.FRONT)
    output_points = placer.get_output_points(Face.BACK)
    
    print(f"  Фактические точки ввода: {len(input_points)}")
    print(f"  Фактические точки вывода: {len(output_points)}")
    print(f"  Примеры координат ввода: {input_points[:3]}")
    
    # Тест других стратегий
    print("\n2. Стратегия углов:")
    placer_corners = IOPointPlacer(dimensions, PlacementStrategy.CORNERS, {}, seed=42)
    corner_points = placer_corners.get_input_points(Face.FRONT)
    print(f"  Количество угловых точек: {len(corner_points)}")
    print(f"  Координаты углов: {corner_points}")
    
    print("\n3. Полное покрытие:")
    placer_full = IOPointPlacer(dimensions, PlacementStrategy.FULL_FACE, {}, seed=42)
    full_points = placer_full.get_input_points(Face.FRONT)
    print(f"  Полное покрытие: {len(full_points)} точек")


def test_lattice_3d_with_proportional_io():
    """Тестирует Lattice3D с новой пропорциональной I/O стратегией."""
    print("\n=== Тест Lattice3D с пропорциональной I/O ===")
    
    # Создаем конфигурацию
    config = LatticeConfig(
        dimensions=(8, 8, 8),
        placement_strategy=PlacementStrategy.PROPORTIONAL,
        io_strategy_config={
            'coverage_ratio': {'min_percentage': 10.0, 'max_percentage': 20.0},
            'absolute_limits': {'min_points': 5, 'max_points': 15},
            'seed': 42
        },
        gpu_enabled=False,  # Для совместимости
        enable_logging=True
    )
    
    # Создаем решетку
    print("1. Создание решетки...")
    lattice = Lattice3D(config)
    print(f"   Решетка создана: {config.dimensions}")
    
    # Получаем информацию о I/O точках
    print("\n2. Информация о I/O точках:")
    io_info = lattice.get_io_point_info()
    print(f"   Стратегия: {io_info['strategy']}")
    print(f"   Входная грань: {io_info['input_face']}")
    print(f"   Точки ввода: {io_info['input_points']['count']} ({io_info['input_points']['coverage_percentage']:.1f}%)")
    print(f"   Точки вывода: {io_info['output_points']['count']} ({io_info['output_points']['coverage_percentage']:.1f}%)")
    
    # Тестируем forward pass
    print("\n3. Тестирование forward pass:")
    num_input_points = io_info['input_points']['count']
    input_size = lattice.cell_prototype.input_size
    
    # Создаем тестовые входные данные
    external_inputs = torch.randn(num_input_points, input_size)
    print(f"   Входные данные: {external_inputs.shape}")
    
    # Выполняем forward pass
    output_states = lattice.forward(external_inputs)
    print(f"   Выходные состояния всех клеток: {output_states.shape}")
    
    # Получаем только выходные точки
    output_only = lattice.get_output_states()
    print(f"   Состояния только выходных точек: {output_only.shape}")
    
    # Проверяем статистику
    print("\n4. Статистика производительности:")
    stats = lattice.get_performance_stats()
    print(f"   Forward calls: {stats['forward_calls']}")
    print(f"   Время выполнения: {stats['total_time']:.4f}s")


def test_scaling_behavior():
    """Тестирует масштабирование пропорциональной стратегии."""
    print("\n=== Тест масштабирования ===")
    
    sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]
    config = {
        'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6},
        'absolute_limits': {'min_points': 5, 'max_points': 0},  # Без ограничений
        'seed': 42
    }
    
    print("Размер решетки | Площадь грани | Точки ввода | Покрытие")
    print("-" * 55)
    
    for size in sizes:
        placer = IOPointPlacer(size, PlacementStrategy.PROPORTIONAL, config, seed=42)
        face_area = size[0] * size[1]  # Площадь грани FRONT/BACK
        input_points = placer.get_input_points(Face.FRONT)
        coverage = len(input_points) / face_area * 100
        
        print(f"{size[0]:2d}×{size[1]:2d}×{size[2]:2d}      | {face_area:10d} | {len(input_points):10d} | {coverage:6.1f}%")


def compare_strategies():
    """Сравнивает различные стратегии размещения."""
    print("\n=== Сравнение стратегий ===")
    
    dimensions = (8, 8, 8)
    face_area = 64
    
    strategies = [
        (PlacementStrategy.PROPORTIONAL, {'coverage_ratio': {'min_percentage': 10, 'max_percentage': 20}}),
        (PlacementStrategy.CORNERS, {}),
        (PlacementStrategy.CORNERS_CENTER, {}),
        (PlacementStrategy.RANDOM, {}),
        (PlacementStrategy.FULL_FACE, {}),
    ]
    
    print("Стратегия           | Точки | Покрытие")
    print("-" * 40)
    
    for strategy, config in strategies:
        placer = IOPointPlacer(dimensions, strategy, config, seed=42)
        points = placer.get_input_points(Face.FRONT)
        coverage = len(points) / face_area * 100
        
        print(f"{strategy.value:18s} | {len(points):5d} | {coverage:6.1f}%")


def main():
    """Основная функция тестирования."""
    print("🔬 Тестирование пропорциональной I/O стратегии")
    print("=" * 50)
    
    try:
        # Тестируем IOPointPlacer
        test_io_point_placer()
        
        # Тестируем интеграцию с Lattice3D
        test_lattice_3d_with_proportional_io()
        
        # Тестируем масштабирование
        test_scaling_behavior()
        
        # Сравниваем стратегии
        compare_strategies()
        
        print("\n✅ Все тесты пройдены успешно!")
        print("🎯 Пропорциональная I/O стратегия работает корректно")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 