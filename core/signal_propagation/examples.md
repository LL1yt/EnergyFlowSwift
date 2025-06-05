# Signal Propagation Module - Examples

## Базовые Примеры Использования

### Пример 1: Простое Волновое Распространение

```python
import torch
from core.cell_prototype import CellConfig, create_cell_from_config
from core.lattice_3d import LatticeConfig, Lattice3D
from core.signal_propagation import (
    TimeManager, TimeConfig, TimeMode,
    SignalPropagator, PropagationConfig, PropagationMode
)

# Конфигурация клетки
cell_config = CellConfig(
    state_size=8,
    neighbor_input_size=6 * 8,  # 6 соседей × 8 состояний
    external_input_size=4,
    hidden_size=16,
    output_size=8
)

# Конфигурация решетки
lattice_config = LatticeConfig(
    size=(5, 5, 5),
    cell_config=cell_config,
    boundary_condition="reflective",
    gpu_enabled=False  # Для совместимости
)

# Конфигурация времени
time_config = TimeConfig(
    dt=0.01,
    max_time_steps=20,
    mode=TimeMode.FIXED
)

# Конфигурация распространения
propagation_config = PropagationConfig(
    mode=PropagationMode.WAVE,
    signal_strength=1.0,
    wave_speed=0.5,
    decay_rate=0.05
)

# Создание компонентов
cell_prototype = create_cell_from_config(cell_config)
lattice = Lattice3D(lattice_config, cell_prototype)
time_manager = TimeManager(time_config)
propagator = SignalPropagator(lattice, time_manager, propagation_config)

# Создание входных сигналов
input_signals = torch.randn(5, 5, 8) * 0.5  # Случайные сигналы на передней грани

# Инициализация и запуск
propagator.initialize_signals(input_signals, input_face="front")

print("Запуск волнового распространения...")
for step in range(10):
    current_state = propagator.propagate_step()
    avg_activity = current_state.mean().item()
    max_activity = current_state.max().item()
    print(f"Шаг {step+1}: средняя активность = {avg_activity:.4f}, макс = {max_activity:.4f}")

# Получение результатов
output_signals = propagator.get_output_signals("back")
print(f"\nВыходные сигналы: {output_signals.shape}")
print(f"Средний выходной сигнал: {output_signals.mean().item():.4f}")
```

### Пример 2: Диффузионное Распространение с Анализом Паттернов

```python
from core.signal_propagation import PatternAnalyzer, PatternConfig

# Конфигурация для диффузии
propagation_config = PropagationConfig(
    mode=PropagationMode.DIFFUSION,
    signal_strength=0.8,
    diffusion_coefficient=0.3,
    decay_rate=0.02,
    min_signal_threshold=1e-5
)

# Конфигурация анализа паттернов
pattern_config = PatternConfig(
    enable_analysis=True,
    confidence_threshold=0.4,
    spatial_window_size=3,
    temporal_window_size=5
)

# Создание компонентов
time_manager = TimeManager(time_config)
pattern_analyzer = PatternAnalyzer(pattern_config)
propagator = SignalPropagator(lattice, time_manager, propagation_config)
propagator.pattern_analyzer = pattern_analyzer

# Создание центрального импульса
input_signals = torch.zeros(5, 5, 8)
input_signals[2, 2, :] = 2.0  # Центральный импульс

propagator.initialize_signals(input_signals, input_face="front")

print("Запуск диффузионного распространения с анализом паттернов...")
history = []

for step in range(15):
    current_state = propagator.propagate_step()
    history.append(current_state.clone())

    # Анализ паттернов каждые 3 шага
    if step % 3 == 0:
        patterns = propagator.pattern_analyzer.analyze_patterns(
            current_state.unsqueeze(0),  # Добавляем batch dimension
            history[-3:] if len(history) >= 3 else history
        )

        print(f"Шаг {step+1}:")
        for pattern_type, confidence in patterns.items():
            if confidence > 0.3:
                print(f"  - {pattern_type.value}: {confidence:.3f}")

# Финальная статистика
stats = propagator.get_stats()
print(f"\nФинальная статистика:")
print(f"Всего шагов: {stats['total_propagations']}")
print(f"Средняя сила сигнала: {stats['average_signal_strength']:.4f}")
print(f"Активных клеток: {stats['active_cells_count']}")
```

### Пример 3: Направленное Распространение с Детекцией Сходимости

```python
from core.signal_propagation import ConvergenceDetector, ConvergenceConfig, ConvergenceMode

# Направленное распространение (слева направо)
propagation_config = PropagationConfig(
    mode=PropagationMode.DIRECTIONAL,
    signal_strength=1.2,
    direction_vector=(1.0, 0.0, 0.0),  # Направление по X
    decay_rate=0.03
)

# Конфигурация сходимости
convergence_config = ConvergenceConfig(
    mode=ConvergenceMode.COMBINED,
    tolerance=1e-4,
    patience=5,
    check_frequency=2
)

# Создание компонентов
time_manager = TimeManager(time_config)
convergence_detector = ConvergenceDetector(convergence_config)
propagator = SignalPropagator(lattice, time_manager, propagation_config)
propagator.convergence_detector = convergence_detector

# Создание полосы сигналов на левой грани
input_signals = torch.zeros(5, 5, 8)
input_signals[:, 1:4, :] = 1.5  # Полоса в центре

propagator.initialize_signals(input_signals, input_face="left")

print("Запуск направленного распространения с детекцией сходимости...")
step = 0
converged = False

while step < 25 and not converged:
    current_state = propagator.propagate_step()
    step += 1

    # Проверка сходимости каждые 2 шага
    if step % 2 == 0:
        convergence_detector.update_history(current_state)
        converged = convergence_detector.check_convergence()

        print(f"Шаг {step}: средняя активность = {current_state.mean().item():.4f}")
        if converged:
            convergence_time = convergence_detector.get_convergence_time()
            print(f"🎯 Система сошлась на шаге {step} (время: {convergence_time:.2f})")
            break

# Анализ результатов направленного движения
output_signals = propagator.get_output_signals("right")
print(f"\nНаправленное распространение завершено:")
print(f"Выходные сигналы справа: {output_signals.mean().item():.4f}")
print(f"Сходимость достигнута: {converged}")
```

### Пример 4: Полная Симуляция с Автоматической Остановкой

```python
from core.signal_propagation import create_signal_propagator

# Конфигурации в виде словарей (как из YAML)
lattice_config_dict = {
    'size': [4, 4, 4],
    'boundary_condition': 'reflective',
    'gpu_enabled': False
}

time_config_dict = {
    'dt': 0.02,
    'max_time_steps': 50,
    'mode': 'fixed',
    'history_length': 20
}

propagation_config_dict = {
    'mode': 'wave',
    'signal_strength': 1.0,
    'wave_speed': 0.8,
    'decay_rate': 0.1,
    'min_signal_threshold': 1e-4
}

# Создание с помощью фабричной функции
propagator = create_signal_propagator(
    lattice_config_dict,
    time_config_dict,
    propagation_config_dict
)

# Создание сложного паттерна входных сигналов
input_signals = torch.zeros(4, 4, 8)
# Угловые импульсы
input_signals[0, 0, :] = 1.0
input_signals[0, 3, :] = 1.0
input_signals[3, 0, :] = 1.0
input_signals[3, 3, :] = 1.0

propagator.initialize_signals(input_signals, input_face="front")

print("Запуск полной симуляции с автоматической остановкой...")

# Полная симуляция с автоматической остановкой
history = propagator.run_simulation(max_steps=30)

print(f"\nСимуляция завершена:")
print(f"Выполнено шагов: {len(history)}")

# Детальная статистика
final_stats = propagator.get_stats()
print(f"\nДетальная статистика:")
print(f"Время симуляции: {final_stats['time_manager_stats']['current_step']}")
print(f"Максимальный сигнал: {final_stats['max_signal_reached']:.4f}")
print(f"Финальная активность: {final_stats['average_signal_strength']:.4f}")

# Анализ обнаруженных паттернов
if hasattr(propagator, 'pattern_analyzer'):
    detected_patterns = propagator.pattern_analyzer.get_detected_patterns()
    print(f"\nОбнаруженные паттерны:")
    for pattern_type, count in detected_patterns.items():
        print(f"  - {pattern_type}: {count} раз(а)")
```

### Пример 5: Сравнение Режимов Распространения

```python
import matplotlib.pyplot as plt

def compare_propagation_modes():
    """Сравнение разных режимов распространения"""

    modes = [
        (PropagationMode.WAVE, "Волновое"),
        (PropagationMode.DIFFUSION, "Диффузионное"),
        (PropagationMode.DIRECTIONAL, "Направленное")
    ]

    results = {}

    for mode, name in modes:
        print(f"\nТестирование режима: {name}")

        # Базовая конфигурация
        config = PropagationConfig(
            mode=mode,
            signal_strength=1.0,
            wave_speed=0.5,
            diffusion_coefficient=0.4,
            direction_vector=(1.0, 0.0, 0.0),
            decay_rate=0.05
        )

        # Создание симуляции
        time_manager = TimeManager(TimeConfig(max_time_steps=15))
        propagator = SignalPropagator(lattice, time_manager, config)

        # Одинаковые входные сигналы
        input_signals = torch.ones(5, 5, 8) * 0.5
        propagator.initialize_signals(input_signals, input_face="front")

        # Запуск симуляции
        mode_history = []
        for step in range(10):
            state = propagator.propagate_step()
            mode_history.append(state.mean().item())

        results[name] = mode_history
        print(f"Финальная активность: {mode_history[-1]:.4f}")

    # Визуализация (если доступна matplotlib)
    try:
        plt.figure(figsize=(10, 6))
        for name, history in results.items():
            plt.plot(history, label=name, marker='o')
        plt.xlabel('Шаг симуляции')
        plt.ylabel('Средняя активность')
        plt.title('Сравнение режимов распространения сигналов')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib недоступен для визуализации")

    return results

# Запуск сравнения
comparison_results = compare_propagation_modes()
```

### Пример 6: Обработка Ошибок и Граничных Случаев

```python
def robust_signal_propagation_example():
    """Пример с обработкой ошибок и проверками"""

    try:
        # Создание конфигурации с проверками
        propagation_config = PropagationConfig(
            mode=PropagationMode.WAVE,
            signal_strength=2.0,  # Высокая сила сигнала
            decay_rate=0.1
        )

        # Валидация конфигурации
        if propagation_config.signal_strength > propagation_config.max_signal_amplitude:
            print("⚠️ Сила сигнала превышает максимум, корректирую...")
            propagation_config.signal_strength = propagation_config.max_signal_amplitude

        time_manager = TimeManager(TimeConfig(max_time_steps=20))
        propagator = SignalPropagator(lattice, time_manager, propagation_config)

        # Проверка корректности входных сигналов
        input_signals = torch.randn(5, 5, 8)

        # Ограничение амплитуды входных сигналов
        input_signals = torch.clamp(input_signals, -2.0, 2.0)

        print(f"Входные сигналы: min={input_signals.min():.3f}, max={input_signals.max():.3f}")

        propagator.initialize_signals(input_signals, input_face="front")

        # Безопасное выполнение симуляции
        step = 0
        max_attempts = 25

        while step < max_attempts:
            try:
                current_state = propagator.propagate_step()
                step += 1

                # Проверка на взрывной рост
                if current_state.max() > 10.0:
                    print(f"⚠️ Обнаружен взрывной рост на шаге {step}, остановка")
                    break

                # Проверка на NaN или Inf
                if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                    print(f"⚠️ Обнаружены NaN/Inf значения на шаге {step}, остановка")
                    break

                print(f"Шаг {step}: OK, активность = {current_state.mean().item():.4f}")

            except RuntimeError as e:
                print(f"❌ Ошибка на шаге {step}: {e}")
                break

        # Безопасное получение результатов
        try:
            output_signals = propagator.get_output_signals("back")
            print(f"✅ Симуляция завершена успешно. Выходные сигналы: {output_signals.shape}")
        except Exception as e:
            print(f"❌ Ошибка при получении выходных сигналов: {e}")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print("Проверьте конфигурацию и зависимости")

# Запуск безопасного примера
robust_signal_propagation_example()
```

## Готовые Конфигурации

### Быстрая Волна

```python
quick_wave_config = PropagationConfig(
    mode=PropagationMode.WAVE,
    signal_strength=1.5,
    wave_speed=1.0,
    decay_rate=0.05
)
```

### Медленная Диффузия

```python
slow_diffusion_config = PropagationConfig(
    mode=PropagationMode.DIFFUSION,
    signal_strength=0.8,
    diffusion_coefficient=0.2,
    decay_rate=0.02
)
```

### Направленный Поток

```python
directional_flow_config = PropagationConfig(
    mode=PropagationMode.DIRECTIONAL,
    signal_strength=1.2,
    direction_vector=(0.7, 0.7, 0.0),  # Диагональное направление
    decay_rate=0.03
)
```

## Советы по Использованию

1. **Размер решетки**: Начинайте с малых размеров (3×3×3, 5×5×5) для тестирования
2. **Параметры сигналов**: Используйте `signal_strength` 0.5-2.0 для стабильности
3. **Скорость затухания**: `decay_rate` 0.01-0.1 предотвращает взрывной рост
4. **Детекция сходимости**: Включайте для автоматической остановки
5. **Анализ паттернов**: Полезен для понимания поведения системы
6. **Обработка ошибок**: Всегда проверяйте входные данные и результаты

**Эти примеры показывают основные сценарии использования модуля signal_propagation в реальных применениях.**
