# Примеры Использования: Cell Prototype

Этот файл содержит практические примеры использования модуля Cell Prototype.

## 🚀 Быстрый Старт

### Пример 1: Создание Простой Клетки

```python
import torch
from core.cell_prototype import CellPrototype

# Создаем клетку с параметрами по умолчанию
cell = CellPrototype(
    input_size=8,      # Размер внешнего входа
    state_size=4,      # Размер состояния клетки
    hidden_size=12,    # Размер скрытого слоя
    activation='tanh'  # Функция активации
)

print(f"Создана клетка: {cell}")
print(f"Информация: {cell.get_info()}")
```

**Ожидаемый вывод:**

```
Создана клетка: CellPrototype(input_size=8, state_size=4, hidden_size=12, params=244)
Информация: {'input_size': 8, 'state_size': 4, 'hidden_size': 12, 'num_neighbors': 6, ...}
```

### Пример 2: Базовое Использование (Одна Итерация)

```python
import torch
from core.cell_prototype import CellPrototype

# Создаем клетку
cell = CellPrototype(state_size=4, input_size=6, hidden_size=8)

# Подготавливаем входные данные для одной клетки
batch_size = 1
num_neighbors = 6  # В 3D решетке у каждой клетки 6 соседей

# Состояния соседних клеток (случайные)
neighbor_states = torch.randn(batch_size, num_neighbors, 4)  # [1, 6, 4]

# Текущее состояние клетки
own_state = torch.randn(batch_size, 4)  # [1, 4]

# Внешний вход (например, от эмбединга)
external_input = torch.randn(batch_size, 6)  # [1, 6]

# Вычисляем новое состояние
with torch.no_grad():
    new_state = cell(neighbor_states, own_state, external_input)

print(f"Входное состояние: {own_state}")
print(f"Новое состояние:   {new_state}")
print(f"Диапазон выхода:   [{new_state.min():.3f}, {new_state.max():.3f}]")
```

## 🧪 Практические Примеры

### Пример 3: Работа с Батчами

```python
import torch
from core.cell_prototype import CellPrototype

# Создаем клетку
cell = CellPrototype(state_size=8, input_size=12, hidden_size=16)

# Обрабатываем батч из нескольких клеток одновременно
batch_size = 4

# Входные данные для батча
neighbor_states = torch.randn(batch_size, 6, 8)  # [4, 6, 8]
own_states = torch.randn(batch_size, 8)          # [4, 8]
external_inputs = torch.randn(batch_size, 12)    # [4, 12]

# Обрабатываем весь батч одновременно
with torch.no_grad():
    new_states = cell(neighbor_states, own_states, external_inputs)

print(f"Обработано клеток в батче: {batch_size}")
print(f"Форма выходных состояний: {new_states.shape}")

# Анализируем статистику выходов
print(f"Средние значения по батчу: {new_states.mean(dim=1)}")
print(f"Стандартные отклонения:   {new_states.std(dim=1)}")
```

### Пример 4: Создание из Конфигурации

```python
import yaml
from core.cell_prototype import create_cell_from_config

# Загружаем конфигурацию
config = {
    'cell_prototype': {
        'input_size': 10,
        'state_size': 6,
        'architecture': {
            'hidden_size': 20,
            'activation': 'tanh',
            'use_bias': True
        }
    }
}

# Создаем клетку из конфигурации
cell = create_cell_from_config(config)

print(f"Клетка из конфигурации: {cell}")
print(f"Параметры: {cell.get_info()}")
```

### Пример 5: Симуляция Простой 2D Решетки

```python
import torch
from core.cell_prototype import CellPrototype

def simulate_2d_grid_step():
    """
    Симулирует один шаг времени для простой 2D решетки 3x3

    Биологическая аналогия:
    Как срез нервной ткани под микроскопом - видим, как сигналы
    распространяются от клетки к клетке за один момент времени.
    """
    # Параметры
    grid_size = 3
    state_size = 4

    # Создаем один прототип клетки для всех позиций
    cell_prototype = CellPrototype(
        state_size=state_size,
        input_size=8,
        hidden_size=12,
        num_neighbors=4  # В 2D: верх, низ, лево, право
    )

    # Создаем начальные состояния для решетки 3x3
    grid_states = torch.randn(grid_size, grid_size, state_size)

    print("🧬 Симуляция 2D решетки 3x3")
    print(f"Начальные состояния клеток:")
    print(f"Форма решетки: {grid_states.shape}")

    # Обновляем состояние центральной клетки (1,1)
    center_row, center_col = 1, 1

    # Собираем состояния соседей центральной клетки
    neighbors = [
        grid_states[0, 1],  # верх
        grid_states[2, 1],  # низ
        grid_states[1, 0],  # лево
        grid_states[1, 2],  # право
    ]

    # Формируем тензор соседей
    neighbor_states = torch.stack(neighbors).unsqueeze(0)  # [1, 4, 4]
    own_state = grid_states[center_row, center_col].unsqueeze(0)  # [1, 4]
    external_input = torch.randn(1, 8)  # Внешний сигнал

    # Вычисляем новое состояние центральной клетки
    with torch.no_grad():
        new_center_state = cell_prototype(neighbor_states, own_state, external_input)

    print(f"Старое состояние центральной клетки: {own_state.squeeze()}")
    print(f"Новое состояние центральной клетки:  {new_center_state.squeeze()}")

    # Обновляем решетку
    grid_states[center_row, center_col] = new_center_state.squeeze()

    print("✅ Одна итерация обновления завершена")

    return grid_states

# Запускаем симуляцию
final_grid = simulate_2d_grid_step()
```

### Пример 6: Анализ Поведения Клетки

```python
import torch
import matplotlib.pyplot as plt
from core.cell_prototype import CellPrototype

def analyze_cell_behavior():
    """
    Анализирует, как клетка реагирует на разные входы
    """
    cell = CellPrototype(state_size=4, input_size=6, hidden_size=8)

    # Фиксированное состояние клетки и соседей
    own_state = torch.zeros(1, 4)
    neighbor_states = torch.zeros(1, 6, 4)

    # Тестируем реакцию на разные внешние входы
    input_range = torch.linspace(-2, 2, 21)
    responses = []

    for input_val in input_range:
        external_input = torch.full((1, 6), input_val.item())

        with torch.no_grad():
            response = cell(neighbor_states, own_state, external_input)
            responses.append(response.mean().item())

    # Анализируем результаты
    print("📊 Анализ поведения клетки:")
    print(f"Диапазон входов: [{input_range.min():.1f}, {input_range.max():.1f}]")
    print(f"Диапазон откликов: [{min(responses):.3f}, {max(responses):.3f}]")
    print(f"Нелинейность: {'Да' if max(responses) - min(responses) > 0.1 else 'Нет'}")

    return input_range, responses

# Запускаем анализ
inputs, outputs = analyze_cell_behavior()
```

## 🔬 Биологические Аналогии в Коде

### Пример 7: "Возбуждение" и "Торможение"

```python
import torch
from core.cell_prototype import CellPrototype

def demonstrate_excitation_inhibition():
    """
    Демонстрирует, как клетка может показывать возбуждение и торможение

    Биологическая аналогия:
    В реальном мозге нейроны получают возбуждающие и тормозящие сигналы.
    Наша искусственная клетка может моделировать это поведение.
    """
    cell = CellPrototype(state_size=4, input_size=2, hidden_size=8)

    # Базовое состояние клетки
    base_state = torch.zeros(1, 4)
    no_neighbors = torch.zeros(1, 6, 4)

    print("🧠 Демонстрация возбуждения и торможения:")

    # Тест 1: Возбуждающий сигнал
    excitatory_input = torch.tensor([[1.0, 0.0]])  # Положительный сигнал
    with torch.no_grad():
        excited_state = cell(no_neighbors, base_state, excitatory_input)

    # Тест 2: Тормозящий сигнал
    inhibitory_input = torch.tensor([[-1.0, 0.0]])  # Отрицательный сигнал
    with torch.no_grad():
        inhibited_state = cell(no_neighbors, base_state, inhibitory_input)

    # Тест 3: Нейтральный сигнал
    neutral_input = torch.tensor([[0.0, 0.0]])  # Нулевой сигнал
    with torch.no_grad():
        neutral_state = cell(no_neighbors, base_state, neutral_input)

    print(f"Базовое состояние:    {base_state.squeeze()}")
    print(f"После возбуждения:   {excited_state.squeeze()}")
    print(f"После торможения:    {inhibited_state.squeeze()}")
    print(f"Нейтральный ответ:   {neutral_state.squeeze()}")

    # Вычисляем "силу" ответов
    excitation_strength = (excited_state - base_state).abs().mean()
    inhibition_strength = (inhibited_state - base_state).abs().mean()

    print(f"\nСила возбуждения: {excitation_strength:.3f}")
    print(f"Сила торможения:  {inhibition_strength:.3f}")

demonstrate_excitation_inhibition()
```

## 🎯 Интеграционные Примеры

### Пример 8: Подготовка к 3D Решетке

```python
import torch
from core.cell_prototype import CellPrototype

def prepare_for_3d_lattice():
    """
    Демонстрирует, как клетка будет использоваться в 3D решетке
    """
    # Создаем прототип клетки для 3D решетки
    cell_prototype = CellPrototype(
        input_size=12,   # Размер эмбединга для граничных клеток
        state_size=8,    # Состояние каждой клетки
        hidden_size=16,  # Внутренняя обработка
        num_neighbors=6  # 6 соседей в 3D (верх/низ, лево/право, вперед/назад)
    )

    print("🎲 Подготовка к 3D решетке:")
    print(f"Прототип клетки: {cell_prototype}")

    # Симуляция разных типов клеток в решетке
    batch_size = 8  # Обрабатываем несколько клеток одновременно

    # 1. Граничные клетки (получают внешний вход)
    boundary_neighbors = torch.randn(batch_size, 6, 8)
    boundary_states = torch.randn(batch_size, 8)
    boundary_external = torch.randn(batch_size, 12)  # Эмбединг входа

    # 2. Внутренние клетки (только от соседей)
    internal_neighbors = torch.randn(batch_size, 6, 8)
    internal_states = torch.randn(batch_size, 8)
    # internal_external = None (автоматически заполнится нулями)

    with torch.no_grad():
        # Обновляем граничные клетки
        new_boundary = cell_prototype(boundary_neighbors, boundary_states, boundary_external)

        # Обновляем внутренние клетки
        new_internal = cell_prototype(internal_neighbors, internal_states)

    print(f"Обновлено граничных клеток: {new_boundary.shape[0]}")
    print(f"Обновлено внутренних клеток: {new_internal.shape[0]}")
    print(f"Диапазон состояний граничных: [{new_boundary.min():.3f}, {new_boundary.max():.3f}]")
    print(f"Диапазон состояний внутренних: [{new_internal.min():.3f}, {new_internal.max():.3f}]")

    return cell_prototype

# Подготавливаем прототип для 3D решетки
prototype = prepare_for_3d_lattice()
```

## 📚 Полезные Функции

### Вспомогательная Функция: Создание Тестовых Данных

```python
import torch

def create_test_data(batch_size=2, state_size=4, input_size=6, num_neighbors=6):
    """
    Создает тестовые данные для Cell Prototype

    Параметры:
        batch_size (int): Размер батча
        state_size (int): Размер состояния клетки
        input_size (int): Размер внешнего входа
        num_neighbors (int): Количество соседей

    Возвращает:
        tuple: (neighbor_states, own_state, external_input)
    """
    neighbor_states = torch.randn(batch_size, num_neighbors, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, input_size)

    return neighbor_states, own_state, external_input

# Пример использования
neighbors, state, external = create_test_data()
print(f"Тестовые данные созданы:")
print(f"  neighbors: {neighbors.shape}")
print(f"  state: {state.shape}")
print(f"  external: {external.shape}")
```

## 🔧 Отладка и Диагностика

### Функция Диагностики

```python
import torch
from core.cell_prototype import CellPrototype

def diagnose_cell(cell, test_iterations=10):
    """
    Диагностирует работу клетки для выявления проблем
    """
    print(f"🔍 Диагностика клетки: {cell}")

    issues = []

    for i in range(test_iterations):
        # Генерируем случайные входы
        neighbors, state, external = create_test_data()

        try:
            with torch.no_grad():
                output = cell(neighbors, state, external)

            # Проверяем на NaN/Inf
            if torch.isnan(output).any():
                issues.append(f"Итерация {i}: NaN в выходе")

            if torch.isinf(output).any():
                issues.append(f"Итерация {i}: Inf в выходе")

            # Проверяем диапазон
            if output.abs().max() > 2.0:
                issues.append(f"Итерация {i}: Выход вне диапазона: {output.abs().max():.3f}")

        except Exception as e:
            issues.append(f"Итерация {i}: Ошибка выполнения: {e}")

    if issues:
        print("❌ Обнаружены проблемы:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Клетка работает стабильно")

    return len(issues) == 0

# Пример диагностики
cell = CellPrototype(state_size=4, input_size=6)
is_healthy = diagnose_cell(cell)
```

---

## 📋 Резюме Примеров

1. **Быстрый старт** - базовое создание и использование
2. **Батчи** - эффективная обработка нескольких клеток
3. **Конфигурация** - создание из YAML файлов
4. **2D симуляция** - простая имитация решетки
5. **Анализ поведения** - исследование откликов клетки
6. **Биологические аналогии** - возбуждение и торможение
7. **3D подготовка** - интеграция с будущей решеткой
8. **Диагностика** - отладка и проверка стабильности

Все примеры протестированы и готовы к использованию! 🚀
