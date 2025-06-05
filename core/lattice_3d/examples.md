# Примеры Использования: Lattice 3D

## Обзор

Модуль `lattice_3d` реализует трехмерную решетку "умных клеток" для клеточной нейронной сети. Каждая клетка взаимодействует с соседями и может обрабатывать внешние входы.

## Базовое Использование

### Пример 1: Создание Простой Решетки

```python
from core.lattice_3d import Lattice3D, LatticeConfig, BoundaryCondition

# Создаем конфигурацию
config = LatticeConfig(
    dimensions=(4, 4, 4),                    # Решетка 4x4x4
    boundary_conditions=BoundaryCondition.WALLS,  # Стенки на границах
    initialization_method="normal",          # Нормальная инициализация
    gpu_enabled=False                        # Используем CPU
)

# Создаем решетку
lattice = Lattice3D(config)

print(f"Создана решетка: {config.dimensions}")
print(f"Всего клеток: {config.total_cells}")
print(f"Размер состояния: {lattice.cell_prototype.state_size}")
```

### Пример 2: Один Шаг Обновления

```python
# Получаем начальные состояния
initial_states = lattice.get_states()
print(f"Начальные состояния: {initial_states.shape}")

# Выполняем один шаг обновления
new_states = lattice.forward()
print(f"Новые состояния: {new_states.shape}")

# Проверяем изменения
changed = not torch.equal(initial_states, new_states)
print(f"Состояния изменились: {changed}")
```

### Пример 3: Подача Внешнего Сигнала

```python
import torch
from core.lattice_3d import Face

# Получаем размер входной грани
input_face_size = len(lattice._face_indices[Face.FRONT])
input_size = 4  # Размер каждого внешнего входа

# Создаем внешний сигнал
external_inputs = torch.randn(input_face_size, input_size)

# Подаем сигнал на входную грань
result_states = lattice.forward(external_inputs)

print(f"Подан внешний сигнал: {external_inputs.shape}")
print(f"Результат: {result_states.shape}")
```

## Продвинутые Примеры

### Пример 4: Многошаговое Распространение Сигнала

```python
# Сбрасываем состояния
lattice.reset_states()

# Создаем сильный входной сигнал
strong_signal = torch.ones(input_face_size, input_size) * 2.0

# Отслеживаем распространение сигнала
states_history = []

for step in range(8):
    if step == 0:
        # Подаем сигнал только на первом шаге
        new_states = lattice.forward(strong_signal)
    else:
        # Последующие шаги без внешнего входа
        new_states = lattice.forward()

    # Сохраняем среднее значение для анализа
    avg_state = new_states.mean().item()
    states_history.append(avg_state)

    print(f"Шаг {step}: средне значение = {avg_state:.4f}")

# Анализируем результат
print(f"\nИстория распространения: {states_history}")
```

### Пример 5: Работа с Гранями

```python
from core.lattice_3d import Face

# Получаем состояния всех граней
for face in Face:
    face_states = lattice.get_face_states(face)
    avg_value = face_states.mean().item()
    print(f"Грань {face.name}: среднее = {avg_value:.4f}")

# Сравниваем входную и выходную грани
input_states = lattice.get_face_states(Face.FRONT)
output_states = lattice.get_face_states(Face.BACK)

print(f"\nВходная грань (FRONT): {input_states.mean().item():.4f}")
print(f"Выходная грань (BACK): {output_states.mean().item():.4f}")
```

### Пример 6: Управление Состояниями

```python
# Сохраняем текущие состояния
saved_states = lattice.get_states()

# Устанавливаем специальные состояния
special_states = torch.zeros_like(saved_states)
special_states[0] = torch.ones(lattice.cell_prototype.state_size)  # Активируем первую клетку

lattice.set_states(special_states)

# Смотрим, как распространяется активация
for step in range(5):
    lattice.forward()
    active_cells = (lattice.get_states().abs() > 0.1).any(dim=1).sum()
    print(f"Шаг {step}: активных клеток = {active_cells}")

# Восстанавливаем исходные состояния
lattice.set_states(saved_states)
```

## Конфигурационные Примеры

### Пример 7: Загрузка из YAML

```python
from core.lattice_3d import create_lattice_from_config

# Создаем решетку из конфигурационного файла
lattice_from_config = create_lattice_from_config()

print(f"Размеры из конфигурации: {lattice_from_config.config.dimensions}")
print(f"Граничные условия: {lattice_from_config.config.boundary_conditions}")

# Тестируем функциональность
test_states = lattice_from_config.forward()
print(f"Тест прошел успешно: {test_states.shape}")
```

### Пример 8: Различные Граничные Условия

```python
# Тестируем разные граничные условия
boundary_types = [
    BoundaryCondition.WALLS,
    BoundaryCondition.PERIODIC,
    BoundaryCondition.ABSORBING,
    BoundaryCondition.REFLECTING
]

for boundary in boundary_types:
    config = LatticeConfig(
        dimensions=(3, 3, 3),
        boundary_conditions=boundary,
        gpu_enabled=False
    )

    test_lattice = Lattice3D(config)

    # Тестируем одну итерацию
    result = test_lattice.forward()

    print(f"Граничное условие {boundary.value}: OK, форма = {result.shape}")
```

## Анализ Производительности

### Пример 9: Статистика Производительности

```python
# Включаем отслеживание производительности
config.track_performance = True
lattice = Lattice3D(config)

# Выполняем несколько итераций
for i in range(10):
    lattice.forward()

# Получаем статистику
stats = lattice.get_performance_stats()
print(f"Вызовов forward: {stats['forward_calls']}")
print(f"Общее время: {stats['total_time']:.4f} сек")
print(f"Среднее время на шаг: {stats['avg_time_per_step']:.4f} сек")
print(f"Производительность: {stats['forward_calls']/stats['total_time']:.2f} шагов/сек")
```

### Пример 10: Сравнение Режимов Обработки

```python
import time

# Тест параллельной обработки
lattice.config.parallel_processing = True
lattice.reset_states()

start_time = time.time()
for _ in range(5):
    lattice.forward()
parallel_time = time.time() - start_time

# Тест последовательной обработки
lattice.config.parallel_processing = False
lattice.reset_states()

start_time = time.time()
for _ in range(5):
    lattice.forward()
sequential_time = time.time() - start_time

print(f"Параллельная обработка: {parallel_time:.4f} сек")
print(f"Последовательная обработка: {sequential_time:.4f} сек")
print(f"Ускорение: {sequential_time/parallel_time:.2f}x")
```

## Биологические Аналогии

### Пример 11: Моделирование Нейронной Активности

```python
# Моделируем возбуждение нейронной сети
lattice.reset_states()

# "Нейронный стимул" - сильный сигнал на одну область
stimulus = torch.zeros(input_face_size, input_size)
stimulus[:4] = torch.ones(4, input_size) * 3.0  # Стимул в углу

print("Моделирование нейронной активности:")
print("Подача стимула...")

# Первый шаг - подача стимула
lattice.forward(stimulus)

# Последующие шаги - распространение активности
for step in range(1, 6):
    states = lattice.forward()

    # Анализируем активность
    total_activity = states.abs().sum().item()
    max_activity = states.abs().max().item()

    print(f"Шаг {step}: общая активность = {total_activity:.2f}, "
          f"макс активность = {max_activity:.2f}")

print("Моделирование завершено")
```

## Отладка и Диагностика

### Пример 12: Проверка Топологии

```python
# Проверяем топологию решетки
topology_stats = lattice.topology.validate_topology()

print("Статистика топологии:")
print(f"Всего клеток: {topology_stats['total_cells']}")
print(f"Граничные условия: {topology_stats['boundary_conditions']}")
print(f"Распределение соседей: {topology_stats['neighbor_counts']}")
print(f"Симметрия связей: {topology_stats['symmetry_check']}")
print(f"Связность проверена: {topology_stats['connectivity_check']}")
```

### Пример 13: Визуализация Состояний

```python
import matplotlib.pyplot as plt

# Получаем срез решетки для визуализации
states = lattice.get_states()
lattice_3d = states.view(*config.dimensions, lattice.cell_prototype.state_size)

# Визуализируем средний слой по Z
middle_z = config.dimensions[2] // 2
slice_data = lattice_3d[:, :, middle_z, 0]  # Первый компонент состояния

plt.figure(figsize=(8, 6))
plt.imshow(slice_data.detach().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar(label='Значение состояния')
plt.title(f'Срез решетки (Z = {middle_z})')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.show()
```

## Заключение

Модуль `lattice_3d` предоставляет полную функциональность для создания и управления трехмерными решетками клеток. Основные возможности:

- ✅ Создание решеток произвольного размера
- ✅ Различные граничные условия
- ✅ Параллельное и последовательное обновление
- ✅ Интерфейсы ввода/вывода через грани
- ✅ Управление состояниями клеток
- ✅ Статистика производительности
- ✅ Интеграция с модулем cell_prototype

Все примеры протестированы и готовы к использованию.
