# Energy Flow Architecture - Подробный план реализации

## Концепция

Энергетическая архитектура на основе 3D решетки, где энергия (представленная RNN-моделями) течет через простые нейроны-автоматы. Ключевая идея - параллельная обработка независимых энергетических потоков вместо последовательной обработки клеток.

## Архитектура системы

### Структура проекта
```
energy_flow/
├── config/
│   ├── __init__.py
│   ├── energy_config.py       # Конфигурация энергетической системы
│   └── base_config.py         # Базовая конфигурация (адаптированная из new_rebuild)
├── core/
│   ├── __init__.py
│   ├── energy_carrier.py      # RNN-based энергетические потоки
│   ├── simple_neuron.py       # Простой нейрон-автомат
│   ├── energy_lattice.py      # 3D решетка для потоков
│   └── flow_processor.py      # Механизм распространения энергии
├── training/
│   ├── __init__.py
│   └── energy_trainer.py      # Тренировочный цикл
├── utils/
│   ├── __init__.py
│   ├── logging.py            # (копируем из new_rebuild)
│   ├── device_manager.py     # (адаптируем из new_rebuild)
│   └── helpers.py            # Вспомогательные функции
└── examples/
    └── simple_training.py     # Пример использования
```

## Компоненты системы

### 1. EnergyCarrier (energy_carrier.py)
**Назначение**: Представление энергии в виде RNN-модели (LSTM/GRU)

**Характеристики**:
- ~10M параметров
- LSTM с hidden_size=1024, num_layers=3
- Входной эмбеддинг: 768D (стандартный размер)
- Внутреннее состояние: скрытое состояние LSTM
- Выходной эмбеддинг: 768D

**Ключевые методы**:
- `__init__(embedding_dim=768, hidden_size=1024, num_layers=3)`
- `forward(current_position, neuron_output)` → next_position, energy_value
- `can_spawn()` → bool (может ли создать новый поток)
- `spawn()` → новый EnergyCarrier с уменьшенной энергией

### 2. SimpleNeuron (simple_neuron.py)
**Назначение**: Простой нейрон-автомат в каждой клетке решетки

**Характеристики**:
- ~1000 параметров
- Архитектура: 32 входа → 64 скрытых → 16 выходов
- Общие веса для всех нейронов в решетке

**Ключевые методы**:
- `__init__(input_dim=32, hidden_dim=64, output_dim=16)`
- `process(energy_value, position)` → transformation_vector
- `compute_next_position(current_pos, transformation)` → next_pos

### 3. EnergyLattice (energy_lattice.py)
**Назначение**: 3D решетка для управления потоками

**Характеристики**:
- Размеры: width × height × depth
- Хранит активные потоки и их позиции
- Управляет бюджетом потоков

**Размеры по режимам**:
- DEBUG: 20×20×10 (4,000 клеток)
- EXPERIMENT: 50×50×20 (50,000 клеток)
- OPTIMIZED: 100×100×50 (500,000 клеток)

**Ключевые методы**:
- `__init__(width, height, depth, max_flows=1000)`
- `place_initial_energy(embeddings)` → размещение на входной стороне
- `get_active_flows()` → список активных потоков
- `collect_output_energy()` → сбор энергии с выходной стороны

### 4. FlowProcessor (flow_processor.py)
**Назначение**: Механизм распространения энергии

**Ключевые особенности**:
- Энергия может двигаться только вперед (по оси Z)
- Может прыгать на любую клетку впереди
- При высокой энергии может создавать новые потоки
- Параллельная обработка всех потоков

**Ключевые методы**:
- `__init__(lattice, neuron, device_manager)`
- `step()` → один шаг распространения всех потоков
- `propagate_flow(flow, position)` → новая позиция или None (затухание)

## Конфигурация системы

### EnergyConfig (energy_config.py)
```python
@dataclass
class EnergyConfig:
    # Размеры решетки
    lattice_width: int
    lattice_height: int
    lattice_depth: int
    
    # Параметры энергии
    max_active_flows: int = 1000
    energy_threshold: float = 0.1  # Минимальная энергия для продолжения
    spawn_threshold: float = 0.8   # Порог для создания новых потоков
    max_spawn_per_step: int = 10   # Максимум новых потоков за шаг
    
    # Параметры моделей
    carrier_hidden_size: int = 1024
    carrier_num_layers: int = 3
    neuron_hidden_dim: int = 64
    
    # Обучение
    learning_rate: float = 1e-4
    batch_size: int = 32
```

### Режимы работы
```python
DEBUG_CONFIG = EnergyConfig(
    lattice_width=20, lattice_height=20, lattice_depth=10,
    max_active_flows=100, batch_size=8
)

EXPERIMENT_CONFIG = EnergyConfig(
    lattice_width=50, lattice_height=50, lattice_depth=20,
    max_active_flows=500, batch_size=16
)

OPTIMIZED_CONFIG = EnergyConfig(
    lattice_width=100, lattice_height=100, lattice_depth=50,
    max_active_flows=1000, batch_size=32
)
```

## Механизм работы

### 1. Инициализация
- Создаем решетку заданного размера
- Инициализируем общий SimpleNeuron
- Подготавливаем пул для EnergyCarrier

### 2. Прямой проход
1. Размещаем входные эмбеддинги на входной стороне куба (z=0)
2. Для каждого эмбеддинга создаем EnergyCarrier
3. На каждом шаге:
   - Все активные потоки обрабатываются параллельно
   - Каждый поток взаимодействует с нейроном в текущей позиции
   - Нейрон определяет трансформацию
   - EnergyCarrier определяет новую позицию (только вперед)
   - При высокой энергии могут создаваться новые потоки
4. Собираем энергию с выходной стороны (z=depth-1)

### 3. Обучение
- Сравниваем выходные эмбеддинги с целевыми
- Вычисляем loss (MSE или cosine similarity)
- Обратное распространение через:
  - Веса всех активных EnergyCarrier
  - Веса общего SimpleNeuron
- Оптимизация через Adam

## Особенности реализации

### Параллелизм
- Все EnergyCarrier обрабатываются параллельно в батчах
- Используем torch.nn.parallel для эффективной обработки
- Векторизованные операции где возможно

### Управление памятью
- Пул предаллоцированных EnergyCarrier
- Переиспользование неактивных потоков
- Автоматическая очистка по порогу энергии

### Эмерджентность
- Минимум жестко заданных правил
- Путь энергии определяется обучением
- Естественное формирование "каналов" через решетку

## Интеграция с существующими компонентами

### Из new_rebuild переиспользуем:
1. **Система логирования** (utils/logging.py)
   - Custom debug levels
   - Контекстное логирование

2. **DeviceManager** (адаптированный)
   - Управление GPU/CPU
   - Мониторинг памяти

3. **Базовые хелперы**
   - Position3D для навигации
   - Batch processing утилиты

## Примерный API использования

```python
from energy_flow.config import create_experiment_config
from energy_flow.core import EnergyLattice, SimpleNeuron, FlowProcessor
from energy_flow.training import EnergyTrainer

# Конфигурация
config = create_experiment_config()

# Создание компонентов
lattice = EnergyLattice(config)
neuron = SimpleNeuron(config)
processor = FlowProcessor(lattice, neuron, config)

# Обучение
trainer = EnergyTrainer(processor, config)
trainer.train(input_embeddings, target_embeddings, epochs=100)
```

## Метрики и мониторинг

### Ключевые метрики:
- Количество активных потоков
- Средняя энергия потоков
- Процент достигших выхода
- Loss (MSE/cosine similarity)
- Утилизация GPU памяти

### Логирование:
- DEBUG_ENERGY: детали распространения энергии
- DEBUG_SPAWN: создание новых потоков
- DEBUG_CONVERGENCE: статистика достижения выхода

## Этапы реализации

1. **Базовая инфраструктура** (config, logging, device)
2. **SimpleNeuron** - простейший компонент
3. **EnergyCarrier** - ядро системы
4. **EnergyLattice** - управление пространством
5. **FlowProcessor** - механизм распространения
6. **EnergyTrainer** - обучение
7. **Простой пример** - проверка работоспособности

## Оптимизации (на будущее)

1. **Batch processing**: группировка потоков по позициям
2. **Sparse operations**: только активные клетки
3. **Multi-GPU**: распределение решетки по устройствам
4. **Checkpoint/restore**: сохранение состояния обучения
5. **Адаптивный бюджет**: динамическое управление max_flows