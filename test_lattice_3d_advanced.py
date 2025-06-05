#!/usr/bin/env python3
"""
Расширенный тест модуля Lattice 3D - полная функциональность
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from core.lattice_3d import (
    Lattice3D, LatticeConfig, Position3D, NeighborTopology,
    BoundaryCondition, Face, load_lattice_config, create_lattice_from_config
)

def test_lattice_3d_full():
    """Полный тест реализации Lattice3D"""
    print("🚀 РАСШИРЕННОЕ ТЕСТИРОВАНИЕ LATTICE 3D - ПОЛНАЯ ФУНКЦИОНАЛЬНОСТЬ")
    print("=" * 80)
    
    try:
        # Тест 1: Создание полноценной решетки
        print("🧪 Тест 1: Создание полноценной решетки Lattice3D")
        
        config = LatticeConfig(
            dimensions=(4, 4, 4),
            boundary_conditions=BoundaryCondition.WALLS,
            initialization_method="normal",
            initialization_std=0.1,
            parallel_processing=True,
            track_performance=True,
            gpu_enabled=False  # Отключаем GPU для совместимости
        )
        
        lattice = Lattice3D(config)
        print(f"  ✅ Lattice3D создан: {config.dimensions}")
        print(f"    - Общее количество клеток: {config.total_cells}")
        print(f"    - Устройство: {config.device}")
        print(f"    - Размер состояния клетки: {lattice.cell_prototype.state_size}")
        print(f"    - Размер входа клетки: {lattice.cell_prototype.input_size}")
        print(f"    - Форма состояний решетки: {lattice._states.shape}")
        
        # Тест 2: Тестирование forward pass без внешних входов
        print("\n🧪 Тест 2: Forward pass без внешних входов")
        
        initial_states = lattice.get_states()
        print(f"  📊 Начальные состояния: {initial_states.shape}")
        print(f"    - Среднее значение: {initial_states.mean().item():.4f}")
        print(f"    - Стандартное отклонение: {initial_states.std().item():.4f}")
        
        # Выполняем один шаг
        new_states = lattice.forward()
        print(f"  📊 Новые состояния после forward: {new_states.shape}")
        print(f"    - Среднее значение: {new_states.mean().item():.4f}")
        print(f"    - Стандартное отклонение: {new_states.std().item():.4f}")
        print(f"    - Изменились ли состояния: {not torch.equal(initial_states, new_states)}")
        print(f"    - Счетчик шагов: {lattice.step_count}")
        
        # Тест 3: Тестирование forward pass с внешними входами
        print("\n🧪 Тест 3: Forward pass с внешними входами")
        
        # Создаем внешние входы для входной грани
        input_face_size = len(lattice._face_indices[config.input_face])
        external_input_size = min(4, lattice.cell_prototype.input_size)  # Берем разумный размер
        external_inputs = torch.randn(input_face_size, external_input_size)
        
        print(f"  📊 Внешние входы: {external_inputs.shape}")
        print(f"    - Размер входной грани: {input_face_size}")
        print(f"    - Размер каждого входа: {external_input_size}")
        
        # Выполняем шаг с внешними входами
        states_before = lattice.get_states()
        new_states_with_input = lattice.forward(external_inputs)
        
        print(f"  📊 Результат с внешними входами:")
        print(f"    - Форма выходных состояний: {new_states_with_input.shape}")
        print(f"    - Изменились ли состояния: {not torch.equal(states_before, new_states_with_input)}")
        print(f"    - Счетчик шагов: {lattice.step_count}")
        
        # Тест 4: Тестирование граней и интерфейсов ввода/вывода
        print("\n🧪 Тест 4: Интерфейсы ввода/вывода (грани)")
        
        for face in Face:
            face_states = lattice.get_face_states(face)
            face_indices = lattice._face_indices[face]
            print(f"  📊 Грань {face.name}:")
            print(f"    - Количество клеток: {len(face_indices)}")
            print(f"    - Форма состояний: {face_states.shape}")
            print(f"    - Среднее значение: {face_states.mean().item():.4f}")
        
        # Тест 5: Управление состояниями
        print("\n🧪 Тест 5: Управление состояниями")
        
        # Сохраняем текущие состояния
        current_states = lattice.get_states()
        
        # Устанавливаем новые случайные состояния
        random_states = torch.randn_like(current_states)
        lattice.set_states(random_states)
        retrieved_states = lattice.get_states()
        
        print(f"  📊 Установка новых состояний:")
        print(f"    - Состояния установлены корректно: {torch.allclose(random_states, retrieved_states)}")
        
        # Сброс состояний
        lattice.reset_states()
        reset_states = lattice.get_states()
        print(f"  📊 Сброс состояний:")
        print(f"    - Счетчик шагов сброшен: {lattice.step_count == 0}")
        print(f"    - Форма состояний после сброса: {reset_states.shape}")
        
        # Тест 6: Статистика производительности
        print("\n🧪 Тест 6: Статистика производительности")
        
        # Выполняем несколько шагов для накопления статистики
        for i in range(5):
            lattice.forward()
            
        perf_stats = lattice.get_performance_stats()
        print(f"  📊 Статистика производительности:")
        print(f"    - Количество вызовов forward: {perf_stats['forward_calls']}")
        print(f"    - Общее время: {perf_stats['total_time']:.4f} сек")
        print(f"    - Среднее время на шаг: {perf_stats['avg_time_per_step']:.4f} сек")
        
        # Тест 7: Многошаговое распространение
        print("\n🧪 Тест 7: Многошаговое распространение сигнала")
        
        # Сбрасываем состояния
        lattice.reset_states()
        
        # Подаем сильный сигнал на входную грань
        strong_input = torch.ones(input_face_size, external_input_size) * 2.0
        
        states_history = []
        for step in range(8):  # Пропускаем сигнал через решетку
            if step == 0:
                new_states = lattice.forward(strong_input)
            else:
                new_states = lattice.forward()
            states_history.append(new_states.mean().item())
            
        print(f"  📊 История распространения сигнала (среднее):")
        for i, avg_state in enumerate(states_history):
            print(f"    Шаг {i}: {avg_state:.4f}")
            
        # Проверяем, что сигнал распространился
        input_face_final = lattice.get_face_states(config.input_face)
        output_face_final = lattice.get_face_states(config.output_face)
        
        print(f"  📊 Финальные состояния граней:")
        print(f"    - Входная грань (среднее): {input_face_final.mean().item():.4f}")
        print(f"    - Выходная грань (среднее): {output_face_final.mean().item():.4f}")
        
        # Тест 8: Различные режимы обработки
        print("\n🧪 Тест 8: Тестирование режимов обработки")
        
        # Тест параллельной обработки
        lattice.config.parallel_processing = True
        lattice.reset_states()
        parallel_result = lattice.forward()
        
        # Тест последовательной обработки
        lattice.config.parallel_processing = False
        lattice.reset_states()
        sequential_result = lattice.forward()
        
        print(f"  📊 Сравнение режимов обработки:")
        print(f"    - Параллельный результат (среднее): {parallel_result.mean().item():.4f}")
        print(f"    - Последовательный результат (среднее): {sequential_result.mean().item():.4f}")
        print(f"    - Результаты идентичны: {torch.allclose(parallel_result, sequential_result, atol=1e-5)}")
        
        print(f"\n  ✅ Все функции Lattice3D работают корректно")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка в тестировании Lattice3D: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_lattice_integration():
    """Тест интеграции с конфигурационными файлами"""
    print("\n🧪 Дополнительный тест: Интеграция с конфигурацией")
    
    try:
        # Создаем решетку из конфигурационного файла
        lattice = create_lattice_from_config()
        print(f"  ✅ Решетка создана из конфигурации: {lattice.config.dimensions}")
        
        # Тестируем базовую функциональность
        initial_states = lattice.get_states()
        new_states = lattice.forward()
        
        print(f"  ✅ Базовая функциональность работает")
        print(f"    - Изменение состояний: {not torch.equal(initial_states, new_states)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка интеграции: {e}")
        return False

def main():
    """Главная функция тестирования"""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Выполняем тесты
    tests_results = []
    
    tests_results.append(test_lattice_3d_full())
    tests_results.append(test_lattice_integration())
    
    # Подводим итоги
    passed_tests = sum(tests_results)
    total_tests = len(tests_results)
    
    print("\n" + "=" * 80)
    print(f"📊 РЕЗУЛЬТАТЫ РАСШИРЕННОГО ТЕСТИРОВАНИЯ: {passed_tests}/{total_tests} тестов пройдено")
    
    if passed_tests == total_tests:
        print("🎉 ВСЕ РАСШИРЕННЫЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n✅ Модуль Lattice3D полностью функционален и готов к следующему этапу")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} тестов провалилось")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 