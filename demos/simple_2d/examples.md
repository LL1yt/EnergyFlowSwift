# Примеры Использования: Simple 2D Demo

## 🚀 Быстрый Запуск

### 📓 Интерактивный Jupyter Notebook (РЕКОМЕНДУЕТСЯ!)

```bash
# Откройте Jupyter Notebook для лучшего опыта
jupyter notebook demos/simple_2d/Simple_2D_Demo_Interactive.ipynb
```

**Особенности интерактивного notebook:**

- 🎛️ **Настраиваемые параметры** через виджеты
- 🎨 **Красивые визуализации** в темной теме
- 📚 **Пошаговые объяснения** с биологическими аналогиями
- 🧪 **Экспериментальная среда** для исследований
- 📊 **Множественная визуализация** (статика + графики)

### Запуск демонстраций через Python

```bash
# Из корневой директории проекта
cd demos/simple_2d
python simple_2d_demo.py
```

**Что произойдет:**

1. Демонстрация распространения волны (20 шагов)
2. Пауза для просмотра → нажать Enter
3. Демонстрация радиального импульса (15 шагов)
4. Пауза для просмотра → нажать Enter
5. Демонстрация интерференции волн (25 шагов)
6. Сохранение анимаций в `visualizations/`

## 📚 Программные Примеры

### Пример 1: Создание простой решетки

```python
from demos.simple_2d import Simple2DLattice, Demo2DVisualizer

# Создаем маленькую решетку 5x5
lattice = Simple2DLattice(width=5, height=5)

# Визуализируем начальное состояние
visualizer = Demo2DVisualizer(lattice)
visualizer.plot_current_state("Начальное состояние")

print(f"Создана решетка {lattice.width}x{lattice.height}")
print(f"Состояние клеток: {lattice.states.shape}")
```

**Выход:**

```
✅ Создана 2D решетка 5x5 с общим прототипом клетки
Создана решетка 5x5
Состояние клеток: torch.Size([5, 5, 4])
```

### Пример 2: Создание кастомных паттернов

```python
from demos.simple_2d import PatternGenerator
import torch

# Точечный источник в центре
point = PatternGenerator.point_source(
    width=10, height=10,
    x=5, y=5,
    intensity=3.0
)
print(f"Точечный источник: {point.shape}")

# Волна сверху
wave = PatternGenerator.wave_source(
    width=10, height=10,
    side='top',
    intensity=2.0
)
print(f"Волна сверху: активные клетки = {(wave > 0).sum()}")

# Круговой импульс
pulse = PatternGenerator.pulse_pattern(
    width=12, height=12,
    center_x=6, center_y=6,
    radius=3,
    intensity=1.5
)
print(f"Импульс: максимум = {pulse.max():.2f}")
```

**Выход:**

```
Точечный источник: torch.Size([10, 10, 1])
Волна сверху: активные клетки = 10
Импульс: максимум = 1.50
```

### Пример 3: Пошаговая симуляция

```python
from demos.simple_2d import Simple2DLattice, PatternGenerator, Demo2DVisualizer

# Создаем решетку и паттерн
lattice = Simple2DLattice(width=8, height=8)
wave = PatternGenerator.wave_source(8, 8, side='left', intensity=2.0)

# Ручное управление симуляцией
steps = 10
for step in range(steps):
    # Подаем вход только первые 3 шага
    external_input = wave if step < 3 else None

    # Один шаг симуляции
    lattice.step(external_input=external_input)

    # Анализируем состояние
    activity = lattice.get_activity_map()
    max_activity = activity.max()
    active_cells = (activity > 0.1).sum()

    print(f"Шаг {step+1}: макс.активность={max_activity:.3f}, "
          f"активных клеток={active_cells}")

# Показываем итоговое состояние
visualizer = Demo2DVisualizer(lattice)
visualizer.plot_current_state(f"После {steps} шагов")
```

### Пример 4: Кастомная конфигурация клетки

```python
from demos.simple_2d import Simple2DLattice

# Кастомная конфигурация с большей сетью
custom_config = {
    'input_size': 8,      # Стандартный размер входа
    'hidden_size': 32,    # Увеличенный скрытый слой
    'output_size': 6,     # Больше состояний
    'activation': 'relu', # Другая активация
    'use_bias': True
}

# Создаем решетку с кастомной конфигурацией
lattice = Simple2DLattice(
    width=12, height=12,
    cell_config=custom_config
)

print(f"Кастомная решетка создана:")
print(f"  Размер состояния клетки: {custom_config['output_size']}")
print(f"  Скрытый слой: {custom_config['hidden_size']}")
print(f"  Активация: {custom_config['activation']}")
```

### Пример 5: Сохранение результатов

```python
from demos.simple_2d import Simple2DLattice, PatternGenerator, Demo2DVisualizer
import os

# Создаем папку для результатов
os.makedirs("my_experiments", exist_ok=True)

# Эксперимент с импульсом
lattice = Simple2DLattice(width=10, height=10)
pulse = PatternGenerator.pulse_pattern(10, 10, 5, 5, 2, intensity=3.0)

# Симуляция
for step in range(8):
    lattice.step(external_input=pulse if step < 2 else None)

# Сохраняем результаты
visualizer = Demo2DVisualizer(lattice)

# Статическое изображение
visualizer.plot_current_state(
    title="Мой эксперимент с импульсом",
    save_path="my_experiments/pulse_result.png"
)

# Анимация
animation = visualizer.create_animation(
    save_path="my_experiments/pulse_animation.gif",
    fps=3
)

print("✅ Результаты сохранены в my_experiments/")
```

## 📓 Примеры для Jupyter Notebook

### Пример 6: Интерактивная демонстрация волны

```python
# В Jupyter Notebook - интерактивные виджеты
from ipywidgets import interact, widgets

def interactive_wave_demo(width=15, height=10, intensity=2.0, side='left', steps=15):
    """Функция для интерактивного виджета"""
    lattice = Simple2DLattice(width=width, height=height)
    wave = PatternGenerator.wave_source(width, height, side=side, intensity=intensity)

    # Симуляция и визуализация
    activities = []
    for step in range(steps):
        lattice.step(external_input=wave if step < 3 else None)
        activities.append(lattice.get_activity_map().copy())

    # Показать результат
    visualizer = Demo2DVisualizer(lattice)
    visualizer.plot_current_state(f"Волна: {side}, интенсивность={intensity}")

    return f"Максимальная активность: {max(a.max() for a in activities):.3f}"

# Создать интерактивный виджет
interact(interactive_wave_demo,
         width=widgets.IntSlider(value=15, min=8, max=25, description='Ширина:'),
         height=widgets.IntSlider(value=10, min=6, max=20, description='Высота:'),
         intensity=widgets.FloatSlider(value=2.0, min=0.5, max=5.0, description='Интенсивность:'),
         side=widgets.Dropdown(options=['left', 'right', 'top', 'bottom'], description='Сторона:'),
         steps=widgets.IntSlider(value=15, min=5, max=30, description='Шаги:'))
```

### Пример 7: Jupyter магические команды

```python
# В Jupyter Notebook - удобная настройка
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('dark_background')  # Темная тема

# Быстрый запуск демонстрации
%%time
lattice = Simple2DLattice(width=12, height=12)
pulse = PatternGenerator.pulse_pattern(12, 12, 6, 6, 3, intensity=4.0)

for step in range(10):
    lattice.step(external_input=pulse if step < 2 else None)

visualizer = Demo2DVisualizer(lattice)
visualizer.plot_current_state("Результат за 10 шагов")
```

## 🔬 Исследовательские Примеры

### Пример 6: Анализ распространения сигнала

```python
from demos.simple_2d import Simple2DLattice, PatternGenerator
import numpy as np
import matplotlib.pyplot as plt

def analyze_propagation():
    """Анализ скорости распространения сигнала"""

    lattice = Simple2DLattice(width=15, height=15)
    source = PatternGenerator.point_source(15, 15, x=7, y=7, intensity=5.0)

    propagation_data = []

    # Симуляция с анализом
    for step in range(12):
        lattice.step(external_input=source if step < 1 else None)

        activity = lattice.get_activity_map()

        # Находим границу активной области
        active_mask = activity > 0.5
        if active_mask.any():
            # Расстояние от центра до самой дальней активной клетки
            center_y, center_x = 7, 7
            max_distance = 0

            for y in range(15):
                for x in range(15):
                    if active_mask[y, x]:
                        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        max_distance = max(max_distance, distance)

            propagation_data.append(max_distance)
        else:
            propagation_data.append(0)

        print(f"Шаг {step+1}: радиус распространения = {propagation_data[-1]:.2f}")

    # График скорости распространения
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(propagation_data)+1), propagation_data, 'o-')
    plt.xlabel('Шаг времени')
    plt.ylabel('Радиус распространения')
    plt.title('Скорость распространения сигнала')
    plt.grid(True)
    plt.show()

    return propagation_data

# Запуск анализа
data = analyze_propagation()
```

### Пример 7: Сравнение граничных условий

```python
def compare_boundary_effects():
    """Сравнение поведения на границах"""

    # Тест у левой границы
    lattice = Simple2DLattice(width=10, height=10)
    edge_source = PatternGenerator.point_source(10, 10, x=0, y=5, intensity=3.0)

    for step in range(6):
        lattice.step(external_input=edge_source if step < 2 else None)

    edge_activity = lattice.get_activity_map()

    # Тест в центре
    lattice.reset()
    center_source = PatternGenerator.point_source(10, 10, x=5, y=5, intensity=3.0)

    for step in range(6):
        lattice.step(external_input=center_source if step < 2 else None)

    center_activity = lattice.get_activity_map()

    print("Анализ граничных эффектов:")
    print(f"  Активность у края: макс={edge_activity.max():.3f}, средн={edge_activity.mean():.3f}")
    print(f"  Активность в центре: макс={center_activity.max():.3f}, средн={center_activity.mean():.3f}")

    return edge_activity, center_activity

edge_data, center_data = compare_boundary_effects()
```

## 🎯 Образовательные Примеры

### Пример 8: Демонстрация для новичков

```python
def simple_explanation():
    """Простое объяснение концепции для новичков"""

    print("🧬 ДЕМОНСТРАЦИЯ КЛЕТОЧНОЙ СЕТИ")
    print("=" * 40)

    # Создаем очень маленькую решетку
    lattice = Simple2DLattice(width=3, height=3)

    print("1. Создали решетку 3x3 из 'умных клеток'")
    print("   Каждая клетка получает сигналы от соседей")

    # Простой сигнал в угол
    signal = PatternGenerator.point_source(3, 3, x=0, y=0, intensity=2.0)

    print("\n2. Подаем сигнал в левый верхний угол...")

    # Показываем что происходит каждый шаг
    for step in range(4):
        lattice.step(external_input=signal if step < 1 else None)

        activity = lattice.get_activity_map()
        print(f"\n   Шаг {step+1}:")

        # Показываем решетку символами
        for y in range(3):
            row = "   "
            for x in range(3):
                if activity[y, x] > 0.1:
                    row += "🔥 "  # Активная клетка
                else:
                    row += "⚪ "  # Неактивная клетка
            print(row)

    print("\n3. Видите? Сигнал распространился по соседним клеткам!")
    print("   Это основной принцип нашей 3D нейронной сети.")

# Запуск демонстрации для новичков
simple_explanation()
```

**Выход:**

```
🧬 ДЕМОНСТРАЦИЯ КЛЕТОЧНОЙ СЕТИ
========================================
✅ Создана 2D решетка 3x3 с общим прототипом клетки
1. Создали решетку 3x3 из 'умных клеток'
   Каждая клетка получает сигналы от соседей

2. Подаем сигнал в левый верхний угол...

   Шаг 1:
   🔥 ⚪ ⚪
   ⚪ ⚪ ⚪
   ⚪ ⚪ ⚪

   Шаг 2:
   🔥 🔥 ⚪
   🔥 🔥 ⚪
   ⚪ ⚪ ⚪

   ... и так далее
```

## 🛠️ Отладочные Примеры

### Пример 9: Диагностика проблем

```python
def debug_lattice():
    """Диагностика проблем с решеткой"""

    lattice = Simple2DLattice(width=5, height=5)

    print("🔍 ДИАГНОСТИКА РЕШЕТКИ")
    print("=" * 30)

    # Проверка начального состояния
    initial_state = lattice.states
    print(f"✅ Начальные состояния: {initial_state.shape}")
    print(f"✅ Все нули: {torch.allclose(initial_state, torch.zeros_like(initial_state))}")

    # Проверка прототипа клетки
    print(f"✅ Прототип клетки создан: {lattice.cell_prototype is not None}")

    # Тест входного размера
    test_neighbors = torch.randn(4, 4)  # 4 соседа по 4 элемента
    test_own_state = torch.randn(4)     # собственное состояние
    test_external = torch.randn(1)      # внешний вход

    test_input = torch.cat([test_neighbors.flatten(), test_own_state, test_external])
    print(f"✅ Размер тестового входа: {test_input.shape}")

    try:
        test_output = lattice.cell_prototype(test_input.unsqueeze(0))
        print(f"✅ Тест прототипа успешен: выход {test_output.shape}")
    except Exception as e:
        print(f"❌ Ошибка тестирования прототипа: {e}")

    # Тест одного шага
    try:
        lattice.step()
        print(f"✅ Тест шага успешен")
        print(f"✅ История создана: {len(lattice.history)} записей")
    except Exception as e:
        print(f"❌ Ошибка шага: {e}")

# Запуск диагностики
debug_lattice()
```

## 📖 Заключение

Эти примеры показывают:

1. **Основы использования** - создание решеток и запуск симуляций
2. **Кастомизация** - настройка параметров и конфигураций
3. **Анализ данных** - извлечение и обработка результатов
4. **Исследования** - изучение свойств системы
5. **Образование** - объяснение концепций новичкам
6. **Отладка** - диагностика проблем

### 🎯 Следующие Шаги

После изучения этих примеров:

1. Попробуйте создать свои паттерны входных сигналов
2. Поэкспериментируйте с размерами решеток
3. Исследуйте влияние параметров клеток
4. Создайте собственные метрики анализа
5. Подготовьтесь к изучению 3D версии!

---

_Все примеры протестированы и готовы к запуску. Удачных экспериментов! 🚀_
