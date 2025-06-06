# Lattice 3D Module

## Обзор

Модуль `lattice_3d` реализует трехмерную решетку "умных клеток" для нейронной сети клеточного типа. Это ключевой компонент архитектуры, который организует клетки-прототипы в пространственную структуру и управляет их взаимодействием.

## Основные Концепции

### Биологическая Аналогия

Представьте кору головного мозга как трехмерную ткань из взаимосвязанных нейронов:

- Каждый нейрон связан с ближайшими соседями
- Все нейроны одного типа (используют один прототип)
- Сигналы распространяются по связям между соседями
- Обработка происходит параллельно во всей ткани

### Техническая Реализация

- **3D решетка**: Трехмерная сетка позиций для размещения клеток
- **Топология соседства**: Каждая клетка связана с 6 соседями (±X, ±Y, ±Z)
- **Параллельная обработка**: Все клетки обновляются одновременно
- **Граничные условия**: Настраиваемое поведение на границах решетки

## Архитектура Модуля

### Основные Компоненты

1. **Lattice3D** - Главный класс решетки

   - Управление сеткой клеток
   - Топология соседства
   - Синхронизация состояний

2. **NeighborTopology** - Система соседства

   - Вычисление соседних позиций
   - Граничные условия
   - Валидация связей

3. **LatticeConfig** - Конфигурация решетки

   - Размеры решетки (X, Y, Z)
   - Тип граничных условий
   - Параметры клеток

4. **IOPointPlacer** - 🆕 Система размещения I/O точек

   - Пропорциональное автоматическое масштабирование (7.8-15.6%)
   - 5 стратегий размещения: PROPORTIONAL, RANDOM, CORNERS, CORNERS_CENTER, FULL_FACE
   - Биологически обоснованная постоянная плотность рецепторов

5. **PlacementStrategy** - 🆕 Стратегии размещения точек ввода/вывода
   - Автоматическое масштабирование от 4×4×4 до 128×128×128
   - Конфигурируемые параметры покрытия и ограничения

### Зависимости

- **cell_prototype**: Использует CellPrototype для создания клеток
- **PyTorch**: Для тензорных операций и GPU поддержки
- **NumPy**: Для математических операций с координатами

## Использование

### Базовый Пример

```python
from core.lattice_3d import Lattice3D, LatticeConfig, PlacementStrategy

# Создание конфигурации с пропорциональной I/O стратегией
config = LatticeConfig(
    dimensions=(8, 8, 8),
    boundary_conditions='walls',
    placement_strategy=PlacementStrategy.PROPORTIONAL,
    io_strategy_config={
        'coverage_ratio': {'min_percentage': 7.8, 'max_percentage': 15.6},
        'absolute_limits': {'min_points': 5, 'max_points': 0},
        'seed': 42
    }
)

# Создание решетки
lattice = Lattice3D(config)

# Получение информации о I/O точках
io_info = lattice.get_io_point_info()
print(f"Входных точек: {io_info['input_points']['count']} ({io_info['input_points']['coverage_percentage']:.1f}%)")

# Подготовка входных данных для пропорциональных точек
num_input_points = io_info['input_points']['count']
input_size = lattice.cell_prototype.input_size
external_inputs = torch.randn(num_input_points, input_size)

# Обновление состояний всех клеток
new_states = lattice.forward(external_inputs)

# Получение только выходных точек (вместо всех клеток)
output_states = lattice.get_output_states()
```

### 🆕 Стратегии Размещения I/O Точек

```python
from core.lattice_3d import IOPointPlacer, PlacementStrategy, Face

# Пропорциональная стратегия (рекомендуется)
placer = IOPointPlacer(
    lattice_dimensions=(16, 16, 16),
    strategy=PlacementStrategy.PROPORTIONAL,
    config={
        'coverage_ratio': {'min_percentage': 10.0, 'max_percentage': 15.0},
        'absolute_limits': {'min_points': 8, 'max_points': 0}
    },
    seed=42
)

# Получение точек ввода/вывода
input_points = placer.get_input_points(Face.FRONT)
output_points = placer.get_output_points(Face.BACK)

print(f"Входных точек: {len(input_points)} из {16*16} возможных")
print(f"Покрытие: {len(input_points)/(16*16)*100:.1f}%")

# Другие стратегии
strategies = [
    PlacementStrategy.CORNERS,        # 4 угла грани
    PlacementStrategy.CORNERS_CENTER, # 4 угла + центр
    PlacementStrategy.RANDOM,         # 25% случайных точек
    PlacementStrategy.FULL_FACE,      # Все точки грани
]
```

### Масштабирование I/O Точек

```python
# Автоматическое масштабирование для разных размеров
sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]

for size in sizes:
    placer = IOPointPlacer(size, PlacementStrategy.PROPORTIONAL, config)
    input_points = placer.get_input_points(Face.FRONT)
    face_area = size[0] * size[1]
    coverage = len(input_points) / face_area * 100

    print(f"{size[0]}×{size[1]}×{size[2]}: {len(input_points)} точек ({coverage:.1f}%)")
# Вывод:
# 8×8×8: 5 точек (7.8%)
# 16×16×16: 25 точек (9.8%)
# 32×32×32: 95 точек (9.3%)
# 64×64×64: 350 точек (8.5%)
```

## Возможности

### Граничные Условия

1. **Walls (Стенки)** - Границы блокируют сигналы
2. **Periodic (Периодические)** - Решетка замыкается в тор
3. **Absorbing (Поглощающие)** - Сигналы затухают на границах
4. **Reflecting (Отражающие)** - Сигналы отражаются от границ

### Топология Соседства

- **6-связность**: Стандартная кубическая решетка (±X, ±Y, ±Z)
- **Валидация**: Автоматическая проверка корректности связей
- **Эффективность**: Оптимизированное вычисление соседей

### Масштабируемость

- **Произвольные размеры**: От малых тестовых (3×3×3) до больших (100×100×100)
- **GPU ускорение**: Параллельная обработка на видеокарте
- **Батчинг**: Обработка нескольких решеток одновременно

## Конфигурация

Конфигурация осуществляется через YAML файлы:

```yaml
# config/default.yaml
lattice_3d:
  dimensions: [8, 8, 8]
  boundary_conditions: "walls"
  parallel_processing: true
  gpu_enabled: true

  # 🆕 Конфигурация I/O стратегии
  io_strategy:
    placement_method: "proportional" # proportional, random, corners, full_face
    coverage_ratio:
      min_percentage: 7.8 # Минимальный % покрытия грани
      max_percentage: 15.6 # Максимальный % покрытия грани
    absolute_limits:
      min_points: 5 # Минимальное количество точек
      max_points: 0 # Максимальное количество точек (0 = без ограничений)
    seed: 42 # Для воспроизводимости

  # Размер-специфичные настройки
  size_specific:
    "16x16": { min_points: 20, max_points: 40 }
    "32x32": { min_points: 80, max_points: 160 }

  cell_prototype:
    input_size: 64
    state_size: 32
    neighbor_channels: 6
```

## Интеграция

### С Cell Prototype

```python
# Решетка автоматически создает клетки на основе прототипа
lattice = Lattice3D(config)  # Использует настройки из cell_prototype
```

### С Signal Propagation

```python
# Подготовка к модулю распространения сигналов
temporal_states = lattice.get_temporal_interface()
```

## Диагностика и Отладка

### Визуализация Состояний

```python
# Получение срезов для визуализации
xy_slice = lattice.get_slice(axis='z', position=4)
connectivity_map = lattice.get_connectivity_map()
```

### Статистика Активности

```python
stats = lattice.get_activity_stats()
# {'mean_activation': 0.45, 'active_cells': 85, 'signal_strength': 0.67}
```

### Проверка Целостности

```python
lattice.validate_topology()  # Проверяет корректность связей
lattice.check_states()       # Проверяет валидность состояний
```

## Производительность

### Оптимизации

- **Векторизация**: Все операции используют тензорные операции PyTorch
- **GPU поддержка**: Автоматическое использование CUDA если доступно
- **Ленивые вычисления**: Соседство вычисляется только при необходимости
- **Кэширование**: Повторно используемые структуры кэшируются

### Рекомендации по Размерам

- **Тестирование**: 3×3×3 до 8×8×8
- **Эксперименты**: 16×16×16 до 32×32×32
- **Производство**: 64×64×64 и выше (с GPU)

## Расширения

### Будущие Возможности

- **Иерархические решетки**: Многоуровневые структуры
- **Адаптивная топология**: Динамическое изменение связей
- **Смешанная точность**: Оптимизация памяти
- **Распределенные вычисления**: Большие решетки на кластерах

### Экспериментальные Режимы

- **Нерегулярные решетки**: Нестандартные топологии
- **Различные типы клеток**: Гетерогенные решетки
- **Динамические размеры**: Изменяющиеся во времени решетки

## Связанные Модули

- **`cell_prototype`**: Прототип отдельной клетки
- **`signal_propagation`**: Временная динамика распространения
- **`data_visualization`**: Визуализация состояний решетки
- **`config_manager`**: Управление конфигурациями

## Статус Разработки

- **Версия**: 0.1.0 (Phase 1 - Foundation)
- **Статус**: 📋 В разработке
- **Совместимость**: Python 3.8+, PyTorch 1.9+
- **Тестирование**: Ручное (следуя принципам проекта)

## Примеры Применения

### Простое Распространение Сигнала

```python
# Создание небольшой решетки для демонстрации
lattice = Lattice3D(LatticeConfig(dimensions=(5, 5, 5)))
lattice.set_input_signal(torch.ones(32), position=(0, 2, 2))

# Наблюдение распространения по шагам
for i in range(10):
    lattice.step()
    print(f"Step {i}: signal strength = {lattice.get_total_activity()}")
```

### Обработка Последовательности

```python
# Подача последовательности эмбедингов
embeddings = load_embeddings(sequence_length=20)
outputs = []

for embedding in embeddings:
    lattice.set_input_face(embedding)
    lattice.propagate(steps=5)
    output = lattice.get_output_face()
    outputs.append(output)
```

---

_Модуль реализует ключевую архитектурную концепцию проекта - организацию "умных клеток" в 3D пространстве для параллельной обработки информации по аналогии с биологическими нейронными сетями._
