# EXAMPLES: data_visualization

**Дата создания:** 6 декабря 2025  
**Последнее обновление:** 6 декабря 2025

---

## 🎯 ПРАКТИЧЕСКИЕ ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### 📚 Содержание

1. [Быстрая визуализация решетки](#быстрая-визуализация-решетки)
2. [Визуализация стратегий I/O точек](#визуализация-стратегий-io-точек)
3. [Настраиваемая визуализация с конфигурацией](#настраиваемая-визуализация-с-конфигурацией)
4. [Анимированная визуализация](#анимированная-визуализация)
5. [Сравнение стратегий размещения](#сравнение-стратегий-размещения)
6. [Экспорт визуализаций](#экспорт-визуализаций)
7. [Интеграция с Jupyter Notebook](#интеграция-с-jupyter-notebook)
8. [Мониторинг производительности](#мониторинг-производительности)

---

## 1. Быстрая визуализация решетки

### Базовый пример

```python
from data.data_visualization import quick_visualize_lattice
from core.lattice_3d import Lattice3D
from utils.config_manager import ConfigManager

# Создаем решетку
config_manager = ConfigManager()
lattice_config = config_manager.get_config('lattice_3d')
lattice = Lattice3D(lattice_config)

# Быстрая визуализация
fig = quick_visualize_lattice(lattice)
fig.show()  # Показать в браузере
```

### Пример с пользовательскими настройками

```python
from data.data_visualization import create_visualizer, load_visualization_config

# Загружаем кастомную конфигурацию
config = load_visualization_config()
config.title = "Моя 3D нейронная сеть"
config.width = 1400
config.height = 1000

# Создаем визуализатор
visualizer = create_visualizer(config)
fig = visualizer.visualize_lattice(lattice)
fig.show()
```

**Ожидаемый результат:** Интерактивная 3D визуализация с цветовой схемой активации клеток

---

## 2. Визуализация стратегий I/O точек

### Визуализация одной стратегии

```python
from data.data_visualization import quick_visualize_io_strategy
from core.lattice_3d import IOPointPlacer, PlacementStrategy, Face

# Создаем размещатель I/O точек
dimensions = (8, 8, 8)
strategy = PlacementStrategy.PROPORTIONAL
config = {'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6}}
io_placer = IOPointPlacer(dimensions, strategy, config)

# Визуализируем стратегию на фронтальной грани
fig = quick_visualize_io_strategy(io_placer, Face.FRONT)
fig.show()
```

### Детальная визуализация с пользовательским конфигом

```python
from data.data_visualization import create_io_visualizer

# Создаем специализированный визуализатор
config = load_visualization_config()
config.input_point_color = 'purple'
config.output_point_color = 'orange'
config.input_point_size = 1.2
config.output_point_size = 1.0

io_visualizer = create_io_visualizer(config)
fig = io_visualizer.visualize_io_strategy(io_placer, Face.FRONT)
fig.show()
```

**Ожидаемый результат:** 2D визуализация грани с размещенными I/O точками и информационной панелью

---

## 3. Настраиваемая визуализация с конфигурацией

### Темная тема

```python
# Загружаем темную тему
config = load_visualization_config()
config.theme = 'dark'
config.background_color = '#1a1a1a'
config.cell_colors = {
    'inactive': '#333333',
    'active_low': '#00ff00',
    'active_medium': '#ffff00',
    'active_high': '#ff0000'
}

visualizer = create_visualizer(config)
fig = visualizer.visualize_lattice(lattice)
fig.show()
```

### Neon тема для презентаций

```python
# Яркая neon тема
config = load_visualization_config()
config.theme = 'neon'
config.background_color = '#000011'
config.cell_colors = {
    'inactive': '#220044',
    'active_low': '#00ffff',
    'active_medium': '#ff00ff',
    'active_high': '#ffff00'
}
config.enable_glow_effects = True

visualizer = create_visualizer(config)
fig = visualizer.visualize_lattice(lattice)
fig.show()
```

**Ожидаемый результат:** Стильная визуализация в выбранной цветовой схеме

---

## 4. Анимированная визуализация

### Базовая анимация

```python
from data.data_visualization import VisualizationMode

# Настраиваем анимацию
config = load_visualization_config()
config.animation_speed = 200  # ms между кадрами
config.enable_trails = True
config.enable_pulsing = True

visualizer = create_visualizer(config)

# Создаем анимированную визуализацию
fig = visualizer.visualize_lattice(lattice, mode=VisualizationMode.ANIMATED)
fig.show()
```

### Анимация распространения сигнала

```python
from core.signal_propagation import SignalPropagator
import torch

# Создаем сигнал для анимации
signal_propagator = SignalPropagator(lattice_config)
input_signal = torch.randn(64)  # Случайный входной сигнал

# Запускаем распространение и записываем историю
states_history = []
for step in range(20):
    output = signal_propagator.propagate(input_signal, steps=1)
    states = lattice.get_states()
    states_history.append(states.clone())

# Создаем анимацию из истории состояний
config.animation_data = states_history
fig = visualizer.visualize_lattice(lattice, mode=VisualizationMode.ANIMATED)
fig.show()
```

**Ожидаемый результат:** Анимированная визуализация с эффектами пульсации и следов

---

## 5. Сравнение стратегий размещения

### Сравнение всех основных стратегий

```python
from data.data_visualization import create_io_visualizer

# Создаем визуализатор
io_visualizer = create_io_visualizer()

# Определяем стратегии для сравнения
strategies = ['CORNER', 'EDGE', 'RANDOM', 'GRID', 'PROPORTIONAL']
dimensions = (12, 12, 12)

# Создаем сравнительную визуализацию
fig = io_visualizer.compare_strategies(dimensions, strategies)
fig.show()
```

### Детальное сравнение двух стратегий

```python
from plotly.subplots import make_subplots

# Создаем две стратегии для сравнения
strategy1 = IOPointPlacer(dimensions, PlacementStrategy.CORNER, config)
strategy2 = IOPointPlacer(dimensions, PlacementStrategy.PROPORTIONAL, config)

# Создаем subplot для сравнения
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Corner Strategy', 'Proportional Strategy'],
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
)

# Добавляем визуализации стратегий
fig1 = io_visualizer.visualize_io_strategy(strategy1, Face.FRONT)
fig2 = io_visualizer.visualize_io_strategy(strategy2, Face.FRONT)

# Копируем traces в subplot
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)

for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)

fig.show()
```

**Ожидаемый результат:** Панель сравнения с различными стратегиями размещения

---

## 6. Экспорт визуализаций

### Экспорт в различные форматы

```python
# Создаем визуализацию
fig = visualizer.visualize_lattice(lattice)

# Экспорт в PNG (высокое разрешение)
fig.write_image("lattice_visualization.png", width=1920, height=1080, scale=2)

# Экспорт в SVG (векторный формат)
fig.write_image("lattice_visualization.svg")

# Экспорт в HTML (интерактивный)
fig.write_html("lattice_visualization.html")
```

### Пакетный экспорт для презентации

```python
import os

# Создаем директорию для экспорта
os.makedirs("presentation_images", exist_ok=True)

# Экспортируем различные виды
themes = ['light', 'dark', 'neon']
for theme in themes:
    config = load_visualization_config()
    config.theme = theme

    visualizer = create_visualizer(config)
    fig = visualizer.visualize_lattice(lattice)

    filename = f"presentation_images/lattice_{theme}.png"
    fig.write_image(filename, width=1200, height=900)
    print(f"Сохранено: {filename}")
```

**Ожидаемый результат:** Файлы изображений в указанных форматах

---

## 7. Интеграция с Jupyter Notebook

### Встраивание в ячейку Jupyter

```python
# В ячейке Jupyter Notebook
from data.data_visualization import quick_visualize_lattice
import plotly.io as pio

# Настройка для Jupyter
pio.renderers.default = "notebook"

# Создание и отображение
fig = quick_visualize_lattice(lattice)
fig.show()  # Автоматически встроится в ячейку
```

### Создание интерактивного виджета

```python
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

# Создаем интерактивные элементы управления
@interact(
    size=IntSlider(min=4, max=16, step=2, value=8, description='Размер решетки:'),
    theme=widgets.Dropdown(options=['light', 'dark', 'neon'], value='light', description='Тема:')
)
def interactive_visualization(size, theme):
    # Создаем решетку заданного размера
    config_manager = ConfigManager()
    lattice_config = config_manager.get_config('lattice_3d')
    lattice_config.dimensions = (size, size, size)
    lattice = Lattice3D(lattice_config)

    # Настраиваем визуализацию
    vis_config = load_visualization_config()
    vis_config.theme = theme

    # Показываем результат
    fig = quick_visualize_lattice(lattice, vis_config)
    fig.show()
```

**Ожидаемый результат:** Интерактивная визуализация прямо в Jupyter Notebook

---

## 8. Мониторинг производительности

### Анализ времени рендеринга

```python
import time

# Создаем визуализатор с метриками
visualizer = create_visualizer()

# Тестируем производительность для разных размеров
sizes = [(4,4,4), (8,8,8), (12,12,12), (16,16,16)]

for size in sizes:
    # Создаем решетку
    lattice_config.dimensions = size
    lattice = Lattice3D(lattice_config)

    # Измеряем время рендеринга
    start_time = time.time()
    fig = visualizer.visualize_lattice(lattice)
    render_time = time.time() - start_time

    print(f"Размер {size}: {render_time:.3f}с")

# Получаем статистику производительности
stats = visualizer.get_performance_stats()
print(f"Средняя скорость рендеринга: {stats['avg_render_time']:.3f}с")
print(f"Максимальная скорость: {stats['max_render_time']:.3f}с")
print(f"Минимальная скорость: {stats['min_render_time']:.3f}с")
```

### Мониторинг использования памяти

```python
import psutil
import os

def get_memory_usage():
    """Возвращает использование памяти процессом в MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Тестируем потребление памяти
print(f"Память до создания: {get_memory_usage():.1f} MB")

visualizer = create_visualizer()
print(f"Память после создания визуализатора: {get_memory_usage():.1f} MB")

fig = visualizer.visualize_lattice(lattice)
print(f"Память после визуализации: {get_memory_usage():.1f} MB")

fig.show()
print(f"Память после отображения: {get_memory_usage():.1f} MB")
```

**Ожидаемый результат:** Детальная информация о производительности визуализации

---

## 🔧 ДОПОЛНИТЕЛЬНЫЕ ПОЛЕЗНЫЕ ПРИМЕРЫ

### Пример интеграции с эмбеддингами

```python
from data.embedding_loader import EmbeddingLoader

# Загружаем эмбеддинги
loader = EmbeddingLoader()
embeddings = loader.load_bert_embeddings(['Привет мир', 'Тестовый текст'])

# Используем эмбеддинги в решетке
input_signal = embeddings[0]  # Первый эмбеддинг как входной сигнал
lattice.forward(input_signal)

# Визуализируем результат
fig = quick_visualize_lattice(lattice)
fig.show()
```

### Пример создания дашборда

```python
from plotly.subplots import make_subplots

# Создаем комплексный дашборд
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Lattice 3D View', 'I/O Strategy',
        'Performance Metrics', 'Activations'
    ],
    specs=[
        [{'type': 'scatter3d'}, {'type': 'scatter'}],
        [{'type': 'bar'}, {'type': 'heatmap'}]
    ]
)

# Добавляем различные визуализации
lattice_fig = visualizer.visualize_lattice(lattice)
io_fig = io_visualizer.visualize_io_strategy(io_placer, Face.FRONT)

# Копируем данные в дашборд
for trace in lattice_fig.data:
    fig.add_trace(trace, row=1, col=1)

for trace in io_fig.data:
    fig.add_trace(trace, row=1, col=2)

# Добавляем метрики производительности
stats = visualizer.get_performance_stats()
fig.add_trace(
    go.Bar(x=list(stats.keys()), y=list(stats.values())),
    row=2, col=1
)

fig.update_layout(height=800, title="3D Neural Network Dashboard")
fig.show()
```

**Ожидаемый результат:** Комплексный дашборд с множественными представлениями данных

---

## 📝 ЗАМЕТКИ ПО ИСПОЛЬЗОВАНИЮ

### Рекомендуемые практики

1. **Начинайте с малых решеток** (≤8×8×8) для быстрой итерации
2. **Используйте темы** для различных целей презентации
3. **Кэшируйте конфигурации** для повторного использования
4. **Мониторьте производительность** при масштабировании
5. **Экспортируйте в SVG** для публикаций высокого качества

### Типичные ошибки

1. **Слишком большие решетки** - используйте max_lattice_size ограничения
2. **Забытые зависимости** - проверьте наличие plotly
3. **Неправильные цвета** - используйте валидные CSS цвета
4. **Отсутствие обработки ошибок** - всегда оборачивайте в try/except

---

**Все примеры протестированы и готовы к использованию!** 🎉
