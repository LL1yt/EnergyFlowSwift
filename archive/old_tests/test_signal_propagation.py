"""
Тест модуля Signal Propagation

Проверяет основную функциональность:
- Инициализация компонентов
- Интеграция с другими модулями
- Базовая симуляция
- Анализ паттернов
"""

import torch
import sys
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

# Импорты наших модулей
from core.signal_propagation import (
    TimeManager, TimeConfig, TimeMode,
    SignalPropagator, PropagationConfig, PropagationMode,
    PropagationPatterns, PatternAnalyzer, 
    ConvergenceDetector, ConvergenceConfig, ConvergenceMode
)
from core.lattice_3d import Lattice3D, LatticeConfig
from core.cell_prototype import CellPrototype

def setup_logging():
    """Настройка логирования для тестов"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_time_manager():
    """Тест TimeManager"""
    print("\n=== Тест TimeManager ===")
    
    # Создаем конфигурацию
    config = TimeConfig(
        max_time_steps=10,
        time_mode=TimeMode.SYNCHRONOUS,
        save_history=True,
        history_length=5
    )
    
    # Создаем менеджер
    time_manager = TimeManager(config)
    
    # Тестируем основные операции
    print(f"TimeManager создан: {time_manager}")
    
    # Запускаем симуляцию
    time_manager.start_simulation()
    assert time_manager.is_running
    
    # Тестовые состояния
    test_state = torch.randn(3, 3, 3, 4)
    
    # Выполняем несколько шагов
    for i in range(5):
        new_state = time_manager.step(test_state)
        print(f"Шаг {i+1}: shape={new_state.shape}")
    
    # Проверяем статистику
    stats = time_manager.get_stats()
    print(f"Статистика: {stats}")
    
    # Останавливаем
    time_manager.stop_simulation()
    assert not time_manager.is_running
    
    print("[OK] TimeManager тест пройден")

def test_convergence_detector():
    """Тест ConvergenceDetector"""
    print("\n=== Тест ConvergenceDetector ===")
    
    # Создаем конфигурацию
    config = ConvergenceConfig(
        mode=ConvergenceMode.COMBINED,
        tolerance=1e-3,  # Низкий порог для быстрого тестирования
        patience=3,
        min_steps=2
    )
    
    # Создаем детектор
    detector = ConvergenceDetector(config)
    print(f"ConvergenceDetector создан: {detector}")
    
    # Создаем последовательность сходящихся состояний
    base_state = torch.ones(2, 2, 2, 3)
    
    states = [
        base_state * 1.0,
        base_state * 0.99,   # Небольшое изменение
        base_state * 0.98,   # Еще меньшее изменение
        base_state * 0.979,  # Очень малое изменение
        base_state * 0.9789, # Сходимость
        base_state * 0.9789, # Стабильность
    ]
    
    # Проверяем сходимость
    converged = False
    for i, state in enumerate(states):
        converged = detector.add_state(state)
        info = detector.get_convergence_info()
        print(f"Шаг {i+1}: converged={converged}, "
              f"metrics={info['current_metrics']}")
        
        if converged:
            print(f"Сходимость достигнута на шаге {i+1}")
            break
    
    # Финальная информация
    final_info = detector.get_convergence_info()
    print(f"Финальная информация: {final_info}")
    
    print("[OK] ConvergenceDetector тест пройден")

def test_pattern_analyzer():
    """Тест PatternAnalyzer"""
    print("\n=== Тест PatternAnalyzer ===")
    
    analyzer = PatternAnalyzer()
    patterns = PropagationPatterns()
    
    # Создаем искусственную историю сигналов
    history = []
    
    # Симулируем волновое распространение
    for t in range(5):
        # Создаем состояние с волной
        state = torch.zeros(4, 4, 4, 2)
        
        # Волна распространяется по оси X
        wave_pos = min(t, 3)
        state[wave_pos, :, :, :] = torch.sin(torch.arange(4).float()).unsqueeze(-1).unsqueeze(-1).repeat(1, 4, 2)
        
        history.append(state)
    
    # Анализируем паттерн
    result = patterns.analyze_propagation(history)
    
    print(f"Обнаруженный паттерн: {result.pattern_type.value}")
    print(f"Уверенность: {result.confidence:.3f}")
    print(f"Характеристики: {result.characteristics}")
    
    # Сводка
    summary = patterns.get_pattern_summary()
    print(f"Сводка по паттернам: {summary}")
    
    print("[OK] PatternAnalyzer тест пройден")

def test_signal_propagator_basic():
    """Базовый тест SignalPropagator"""
    print("\n=== Базовый тест SignalPropagator ===")
    
    # Создаем компоненты
    lattice_config = LatticeConfig(
        dimensions=(3, 3, 3),
        boundary_conditions='periodic',
        neighbors=6,
        parallel_processing=True,
        gpu_enabled=False  # Отключаем GPU для совместимости
    )
    
    time_config = TimeConfig(
        max_time_steps=5,
        time_mode=TimeMode.SYNCHRONOUS,
        save_history=True
    )
    
    propagation_config = PropagationConfig(
        mode=PropagationMode.WAVE,
        signal_strength=1.0,
        decay_rate=0.1
    )
    
    # Создаем решетку и менеджер времени
    lattice = Lattice3D(lattice_config)
    time_manager = TimeManager(time_config)
    
    # Создаем пропагатор
    propagator = SignalPropagator(lattice, time_manager, propagation_config)
    print(f"SignalPropagator создан: {propagator}")
    
    # Инициализируем сигналы (размер состояния клетки получаем из решетки)
    cell_state_size = lattice.cell_prototype.state_size
    input_signals = torch.randn(3, 3, cell_state_size) * 0.5  # Небольшие входные сигналы
    propagator.initialize_signals(input_signals, input_face="front")
    
    print(f"Сигналы инициализированы на грани 'front'")
    print(f"Форма сигналов: {propagator.current_signals.shape}")
    
    # Выполняем несколько шагов
    for step in range(3):
        new_state = propagator.propagate_step()
        stats = propagator.get_stats()
        print(f"Шаг {step+1}: средняя сила сигнала = {stats['average_signal_strength']:.6f}")
    
    # Получаем выходные сигналы
    output_signals = propagator.get_output_signals(output_face="back")
    print(f"Выходные сигналы: форма={output_signals.shape}, "
          f"среднее={output_signals.mean().item():.6f}")
    
    # Финальная статистика
    final_stats = propagator.get_stats()
    print(f"Финальная статистика: {final_stats}")
    
    print("[OK] SignalPropagator базовый тест пройден")

def test_full_integration():
    """Полный интеграционный тест"""
    print("\n=== Полный интеграционный тест ===")
    
    try:
        # Создаем все компоненты
        lattice_config = LatticeConfig(
            dimensions=(4, 4, 4),
            boundary_conditions='reflecting',
            neighbors=6,
            parallel_processing=True,
            gpu_enabled=False  # Отключаем GPU для совместимости
        )
        
        time_config = TimeConfig(
            max_time_steps=10,
            time_mode=TimeMode.SYNCHRONOUS,
            save_history=True,
            history_length=20
        )
        
        propagation_config = PropagationConfig(
            mode=PropagationMode.DIFFUSION,
            signal_strength=1.0,
            decay_rate=0.05,
            diffusion_coefficient=0.3
        )
        
        convergence_config = ConvergenceConfig(
            mode=ConvergenceMode.COMBINED,
            tolerance=1e-4,
            patience=3,
            min_steps=3
        )
        
        # Создаем систему
        lattice = Lattice3D(lattice_config)
        time_manager = TimeManager(time_config)
        propagator = SignalPropagator(lattice, time_manager, propagation_config)
        convergence_detector = ConvergenceDetector(convergence_config)
        pattern_analyzer = PropagationPatterns()
        
        print("Все компоненты созданы успешно")
        
        # Инициализируем сигналы
        cell_state_size = lattice.cell_prototype.state_size
        input_signals = torch.randn(4, 4, cell_state_size) * 0.3
        propagator.initialize_signals(input_signals, input_face="left")
        
        # Запускаем симуляцию с мониторингом сходимости
        simulation_history = []
        
        for step in range(15):  # Максимум 15 шагов
            # Шаг распространения
            new_state = propagator.propagate_step()
            simulation_history.append(new_state.clone())
            
            # Проверка сходимости
            converged = convergence_detector.add_state(new_state)
            
            # Статистика
            if step % 3 == 0:
                stats = propagator.get_stats()
                conv_info = convergence_detector.get_convergence_info()
                print(f"Шаг {step+1}: сила={stats['average_signal_strength']:.6f}, "
                      f"сходимость={conv_info['convergence_count']}/{conv_info['required_patience']}")
            
            # Остановка при сходимости
            if converged:
                print(f"Симуляция остановлена - сходимость достигнута на шаге {step+1}")
                break
        
        # Анализ паттернов
        pattern_result = pattern_analyzer.analyze_propagation(simulation_history)
        print(f"Обнаруженный паттерн: {pattern_result.pattern_type.value} "
              f"(уверенность: {pattern_result.confidence:.3f})")
        
        # Финальная статистика
        final_stats = propagator.get_stats()
        conv_info = convergence_detector.get_convergence_info()
        pattern_summary = pattern_analyzer.get_pattern_summary()
        
        print("\n--- Финальные результаты ---")
        print(f"Всего шагов симуляции: {len(simulation_history)}")
        print(f"Система сошлась: {conv_info['is_converged']}")
        print(f"Средняя сила сигнала: {final_stats['average_signal_strength']:.6f}")
        print(f"Обнаруженные паттерны: {pattern_summary.get('pattern_distribution', {})}")
        
        print("[OK] Полный интеграционный тест пройден")
        
    except Exception as e:
        print(f"[ERROR] Ошибка в интеграционном тесте: {e}")
        raise

def main():
    """Основная функция тестирования"""
    print("[START] Запуск тестов Signal Propagation модуля")
    setup_logging()
    
    try:
        # Базовые тесты компонентов
        test_time_manager()
        test_convergence_detector()
        test_pattern_analyzer()
        test_signal_propagator_basic()
        
        # Интеграционный тест
        test_full_integration()
        
        print("\n[SUCCESS] Все тесты Signal Propagation модуля пройдены успешно!")
        
    except Exception as e:
        print(f"\n[ERROR] Тесты завершились с ошибкой: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 