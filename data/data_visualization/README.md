# Data Visualization Module

## 🎯 Назначение

Модуль `data_visualization` предоставляет мощные инструменты для 3D визуализации клеточной нейронной сети. Он позволяет создавать интерактивные визуализации состояний клеток, I/O точек, анимацию распространения сигналов и дашборды с метриками.

### Ключевые возможности

- **3D визуализация решетки** - интерактивное отображение состояний клеток с цветовой кодировкой активации
- **Визуализация I/O стратегий** - специализированная визуализация различных стратегий размещения точек ввода/вывода
- **Анимация сигналов** - анимированное отображение распространения сигналов через решетку
- **Интерактивные дашборды** - панели с метриками производительности и статистикой сети
- **Экспорт результатов** - сохранение в различных форматах (PNG, SVG, HTML, MP4, GIF)

### Биологическая аналогия

Модуль работает как **fMRI или МРТ сканер для искусственного мозга** - позволяет наблюдать активность нейронных сетей в реальном времени, анализировать паттерны и понимать внутренние процессы.

## 🚀 Установка

### Системные требования

```yaml
python: ">=3.8"
torch: ">=1.9.0"
plotly: ">=5.0.0"
numpy: ">=1.20.0"
pyyaml: "*"
```

### Дополнительные зависимости для расширенных функций

```bash
# Для экспорта видео
pip install opencv-python

# Для дашбордов (опционально)
pip install dash>=2.0.0

# Для оптимизации производительности
pip install numba
```

### Установка из проекта

Модуль является частью проекта 3D Cellular Neural Network:

```python
# Импорт основных компонентов
from data.data_visualization import (
    create_visualizer,
    create_io_visualizer,
    quick_visualize_lattice,
    VisualizationConfig
)
```

## 💡 Базовое использование

### Быстрый старт

```python
from data.data_visualization import quick_visualize_lattice
from core.lattice_3d import create_lattice_from_config

# Создаем решетку
lattice = create_lattice_from_config()

# Быстрая визуализация
fig = quick_visualize_lattice(lattice, title="My Neural Network")
fig.show()
```

### Создание визуализатора с настройками

```python
from data.data_visualization import create_visualizer, VisualizationConfig

# Создаем конфигурацию
config = VisualizationConfig()
config.title = "Advanced 3D Visualization"
config.width = 1600
config.height = 1200
config.cell_opacity = 0.8
config.show_connections = True

# Создаем визуализатор
visualizer = create_visualizer()

# Визуализируем решетку
fig = visualizer.visualize_lattice(lattice)
fig.show()
```

### Визуализация I/O стратегий

```python
from data.data_visualization import create_io_visualizer
from core.lattice_3d import IOPointPlacer, PlacementStrategy, Face

# Создаем размещение I/O точек
dimensions = (8, 8, 8)
strategy = PlacementStrategy.PROPORTIONAL
config = {'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6}}

io_placer = IOPointPlacer(dimensions, strategy, config)

# Визуализируем стратегию
io_visualizer = create_io_visualizer()
fig = io_visualizer.visualize_io_strategy(io_placer, Face.FRONT)
fig.show()
```

### Загрузка конфигурации из файла

```python
from data.data_visualization import load_visualization_config, create_visualizer

# Загружаем конфигурацию из YAML
config = load_visualization_config("path/to/config.yaml")

# Создаем визуализатор с загруженной конфигурацией
visualizer = create_visualizer("path/to/config.yaml")
```

## 🎨 Возможности визуализации

### Цветовая схема активации

Клетки окрашиваются в зависимости от уровня активации:

- **Серый** (`#e9ecef`) - неактивные клетки (< 0.1)
- **Зеленый** (`#28a745`) - низкая активация (0.1 - 0.5)
- **Желтый** (`#ffc107`) - средняя активация (0.5 - 0.8)
- **Красный** (`#dc3545`) - высокая активация (> 0.8)

### I/O точки

- **Синие ромбы** - точки ввода
- **Зеленые квадраты** - точки вывода
- **Стрелки потока** - направления сигналов (опционально)

### Интерактивность

- **Вращение, масштабирование, панорамирование** 3D сцены
- **Hover информация** - детали о клетках при наведении
- **Кликабельные элементы** - дополнительная информация при клике
- **Панель управления** - настройки отображения в реальном времени

## 📊 Метрики и дашборды

### Доступные метрики

- `total_activation` - общая активация сети
- `convergence_rate` - скорость сходимости
- `signal_propagation_speed` - скорость распространения сигналов
- `io_utilization` - утилизация I/O точек
- `memory_usage` - использование памяти
- `computation_time` - время вычислений

### Создание дашборда

```python
from data.data_visualization import create_dashboard

# Создаем интерактивный дашборд
dashboard = create_dashboard()
fig = dashboard.create_dashboard(lattice)
fig.show()

# Обновляем метрики в реальном времени
dashboard.update_metrics(lattice)
```

## 🎬 Экспорт и анимация

### Экспорт статичных изображений

```python
# Экспорт в PNG
fig.write_image("visualization.png", width=1920, height=1080)

# Экспорт в SVG
fig.write_html("visualization.svg")

# Интерактивный HTML
fig.write_html("visualization.html")
```

### Создание анимации

```python
from data.data_visualization import AnimationController

# Создаем контроллер анимации
animator = AnimationController(config)

# Создаем анимацию распространения сигналов
fig = animator.create_propagation_animation(propagator, lattice, num_steps=50)
fig.show()
```

## ⚡ Производительность

### Оптимизация для больших решеток

Модуль автоматически применяет оптимизации для больших решеток:

- **Level of Detail (LOD)** - упрощение отображения для решеток > 50×50×50
- **Адаптивное качество** - автоматическое снижение качества при низкой производительности
- **Кэширование** - сохранение промежуточных результатов
- **Буферизация кадров** - для плавной анимации

### Рекомендации по производительности

- Для решеток > 20×20×20 используйте `level_of_detail=True`
- Отключите `show_connections` для больших сетей
- Используйте `adaptive_quality=True` для автоматической оптимизации
- Ограничьте `max_fps` для экономии ресурсов

## 🔧 Конфигурация

### Основные параметры

```yaml
data_visualization:
  display:
    title: "3D Cellular Neural Network"
    width: 1200
    height: 800
    background_color: "#f8f9fa"

  lattice:
    cell_size: 0.8
    cell_opacity: 0.7
    show_connections: false

  io_points:
    input_points:
      size: 1.5
      color: "#007bff"
    output_points:
      size: 1.5
      color: "#28a745"

  animation:
    enable: true
    speed: 1.0
    trail_length: 5
```

Полную конфигурацию смотрите в `config/default.yaml`.

## 🐛 Решение проблем

### Частые проблемы

**Медленная отрисовка больших решеток:**

```python
config.level_of_detail = True
config.adaptive_quality = True
config.max_lattice_size = (30, 30, 30)
```

**Проблемы с экспортом видео:**

```bash
pip install opencv-python
# Или используйте статичные форматы
```

**Ошибки импорта:**

```python
# Проверьте доступность модуля
from data.data_visualization import get_module_info
print(get_module_info())
```

## 📚 Связанные модули

- `core.lattice_3d` - создание и управление 3D решеткой
- `core.signal_propagation` - распространение сигналов для анимации
- `data.embedding_loader` - загрузка данных для визуализации
- `utils.config_manager` - управление конфигурацией

## 🔗 Ссылки

- **Основная документация:** `docs/`
- **Примеры использования:** `examples.md`
- **API Reference:** `meta.md`
- **План разработки:** `plan.md`
- **Диаграммы архитектуры:** `diagram.mmd`

---

**Версия:** 1.0.0  
**Статус:** ✅ Phase 2 - Core Functionality  
**Последнее обновление:** Декабрь 2025
