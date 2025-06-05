# Метаданные: Simple 2D Demo

## 📦 Информация о Модуле

**Название**: Simple 2D Demo  
**Версия**: 1.0.0  
**Тип**: Демонстрационный модуль  
**Статус**: ✅ Полностью готов  
**Последнее обновление**: Декабрь 2024

## 🔗 Зависимости

### Модульные Зависимости

```python
# Внутренние модули проекта
from core import CellPrototype, create_cell_from_config
```

**Критическая зависимость**: `core.cell_prototype`

- **Назначение**: Единый прототип клетки для всей решетки
- **Используемые функции**: `create_cell_from_config()`
- **Статус**: ✅ Доступен и проверен

### Внешние Зависимости

```python
# Стандартные библиотеки
import sys, os
from pathlib import Path

# Научные вычисления
import numpy as np          # Математические операции
import torch               # Нейронные сети и тензоры
import torch.nn as nn      # Слои нейронных сетей

# Визуализация
import matplotlib.pyplot as plt                    # Основные графики
import matplotlib.animation as animation           # Анимации
from matplotlib.colors import LinearSegmentedColormap  # Цветовые схемы

# Интерактивность (Jupyter Notebook)
import ipywidgets as widgets                       # Интерактивные виджеты
from IPython.display import display, HTML, clear_output  # Jupyter дисплей
```

### UI/DOM Зависимости

- **Консольное приложение**: matplotlib визуализация
- **Jupyter Notebook**: интерактивные виджеты и дисплей
- **Браузерный интерфейс**: через Jupyter для интерактивности

## 📤 Экспортируемый API

### Основные Классы

```python
class Simple2DLattice:
    """Основная 2D решетка из клеток"""
    def __init__(width, height, cell_config=None)
    def step(external_input=None)
    def reset()
    def get_activity_map()
    def get_neighbors(row, col)

class PatternGenerator:
    """Генератор входных паттернов"""
    @staticmethod
    def point_source(width, height, x, y, intensity=1.0)
    @staticmethod
    def wave_source(width, height, side='left', intensity=1.0)
    @staticmethod
    def pulse_pattern(width, height, center_x, center_y, radius, intensity=1.0)

class Demo2DVisualizer:
    """Система визуализации"""
    def __init__(lattice)
    def plot_current_state(title, save_path=None)
    def create_animation(save_path=None, fps=10)
```

### Демонстрационные Функции

```python
def run_wave_demo() -> (lattice, animation)
def run_pulse_demo() -> (lattice, animation)
def run_interference_demo() -> (lattice, animation)
def main() -> None
```

### Полный Экспорт

```python
__all__ = [
    'Simple2DLattice',      # Основная решетка
    'PatternGenerator',     # Генератор паттернов
    'Demo2DVisualizer',     # Визуализатор
    'run_wave_demo',        # Демонстрация волны
    'run_pulse_demo',       # Демонстрация импульса
    'run_interference_demo', # Демонстрация интерференции
    'main'                  # Главная функция
]
```

## 🎯 Основные Возможности

### Функциональные Блоки

1. **Симуляция**: 2D решетка с синхронным обновлением
2. **Визуализация**: Статические изображения и анимации
3. **Демонстрации**: Три готовых сценария
4. **Интерактивность**: Пользовательский интерфейс с паузами

### Конфигурационные Параметры

```python
# Конфигурация клетки (по умолчанию)
default_cell_config = {
    'input_size': 8,      # 4 соседа + состояние + внешний вход
    'hidden_size': 16,    # Размер скрытого слоя
    'output_size': 4,     # Размер состояния клетки
    'activation': 'tanh', # Функция активации
    'use_bias': True      # Использовать смещение
}

# Параметры решетки
lattice_params = {
    'width': 10-20,      # Рекомендуемая ширина
    'height': 10-15,     # Рекомендуемая высота
    'boundary': 'mirror' # Граничные условия
}

# Параметры визуализации
viz_params = {
    'colormap': 'neural',    # Кастомная цветовая схема
    'fps': 4-10,            # Скорость анимации
    'save_format': 'gif'    # Формат сохранения
}
```

## 🔄 Взаимодействие с Проектом

### Входящие Зависимости

- **Импортирует из**: `core.cell_prototype`
- **Использует**: Конфигурационную систему проекта
- **Требует**: PyTorch и базовые научные библиотеки

### Исходящие Зависимости

- **Экспортирует для**: Основного приложения
- **Предоставляет**: Демонстрационную среду
- **Готовит к**: Пониманию 3D версии

### Интеграционные Точки

```python
# Использование в main.py
from demos.simple_2d import main as demo_2d_main

# Использование отдельных компонентов
from demos.simple_2d import Simple2DLattice, PatternGenerator

# Создание кастомных экспериментов
lattice = Simple2DLattice(width=15, height=15)
pattern = PatternGenerator.wave_source(15, 15, side='left')
```

## 📊 Технические Характеристики

### Производительность

- **Время инициализации**: < 1 секунда
- **Время одного шага**: O(width × height)
- **Память**: O(width × height × state_size)
- **Рекомендуемый размер**: до 20×20 для интерактивности

### Ограничения

- **Последовательная обработка**: не векторизовано
- **2D только**: нет 3D функциональности
- **Matplotlib**: может быть медленным для больших решеток

### Возможности Расширения

- ✅ Легко добавить новые паттерны
- ✅ Простое изменение граничных условий
- ✅ Возможность добавления метрик
- ✅ Готовность к векторизации

## 🎨 Визуальные Особенности

### Цветовая Схема

```python
colors = ['#000033', '#000080', '#0000FF', '#4080FF', '#80C0FF', '#FFFFFF']
# Темно-синий → Синий → Голубой → Белый
```

### Типы Визуализации

1. **Статические изображения**: текущее состояние
2. **Анимации**: временная эволюция
3. **Интерактивные графики**: matplotlib backend
4. **Сохранение**: PNG (статика) + GIF (анимация)

## 🚀 Пути Использования

### Образовательные Цели

- Демонстрация концепции новичкам
- Визуальное объяснение клеточных автоматов
- Подготовка к изучению 3D версии

### Исследовательские Цели

- Быстрое тестирование идей
- Отладка алгоритмов распространения
- Анализ паттернов активности

### Разработческие Цели

- Проверка интеграции с core модулями
- Тестирование архитектурных решений
- Прототипирование новых возможностей

---

**🎯 Итог**: Полностью самодостаточный модуль для демонстрации концепции клеточной нейронной сети с отличной документацией и готовностью к использованию.
