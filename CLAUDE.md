### Контекст проекта

- Исследовательский проект на одного разработчика
- Структура: AA/ , AA/new_rebuild/ (легаси), AA/archive/ (старые версии) AA/energy_flow/ (активная разработка)
- Тесты запускаются из корня AA/

## Принципы работы

**Приоритет: вдумчивость, постепенность, эффективность**

- Сначала разобраться в проблеме, потом действовать
- Простой вариант → тесты → оптимизация при необходимости

### Основные принципы

- Централизованные конфигурации и логирование
- Минимальные церемонии, максимальная эффективность
- Современные языковые возможности
- Прямолинейные решения вместо сложных абстракций
- Проект исследовательский, не продакшн
- Без fallback - лучше ошибка чем костыли
- RTX 5090 32GB - используем по максимуму

### Что исключаем

- **НЕТ** CLI автоматизации (только Python API)
- **НЕТ** Множественных конфигураций
- **НЕТ** Legacy совместимости
- **НЕТ** Динамических конфигураций

## Система конфигурации

### 3 режима работы

- **DEBUG** - быстрые тесты (15x15x15 решетка, максимум логов)
- **EXPERIMENT** - исследования (30x30x30, сбалансировано)
- **OPTIMIZED** - финальные прогоны (100x100x100, минимум логов)

```python
from new_rebuild.config import create_debug_config, set_project_config
config = create_debug_config()
set_project_config(config)
```

### Защита от hardcoded

- `@no_hardcoded` - декоратор для функций
- `strict_no_hardcoded()` - автоматическая замена
- `HardcodedValueError` - исключение с инструкцией
- `allow_hardcoded` - временное отключение

### Custom Debug Levels

- `DEBUG_VERBOSE` (11) - подробный вывод
- `DEBUG_CACHE` (12) - кэширование
- `DEBUG_SPATIAL` (13) - пространственная оптимизация
- `DEBUG_FORWARD` (14) - forward pass
- `DEBUG_MEMORY` (15) - управление памятью
- `DEBUG_TRAINING` (16) - прогресс обучения
- `DEBUG_INIT` (17) - инициализация
- `DEBUG_ENERGY` (18) - энергетические потоки (energy_flow)
- `DEBUG_SPAWN` (19) - создание новых потоков
- `DEBUG_CONVERGENCE` (20) - статистика достижения выхода

## Архитектура energy_flow

### Концепция

Энергетическая архитектура, где RNN-модели ("энергия") распространяются через 3D решетку простых нейронов. Ключевое отличие - параллельная обработка независимых потоков вместо последовательной обработки клеток.

### config/ - Конфигурация энергетической системы

- **`energy_config.py`** - `EnergyConfig` с параметрами решетки и потоков
- **`base_config.py`** - Адаптированная базовая конфигурация

### core/ - Ядро энергетической архитектуры

#### energy_carrier.py - RNN-based энергетические потоки

- LSTM с ~10M параметров (hidden_size=1024, 3 слоя)
- Может создавать новые потоки при высокой энергии
- Определяет следующую позицию (только вперед)

#### simple_neuron.py - Простой нейрон-автомат

- ~1000 параметров (32→64→16)
- Общие веса для всех клеток
- Трансформирует энергию и определяет направление

#### energy_lattice.py - 3D решетка для потоков

- Управление активными потоками
- Размещение входной энергии
- Сбор выходной энергии

#### flow_processor.py - Механизм распространения

- Параллельная обработка всех потоков
- Энергия движется только вперед (по Z)
- Управление бюджетом потоков

### training/ - Обучение

- **`energy_trainer.py`** - Тренировочный цикл через сравнение выходных эмбеддингов

### Размеры решетки для energy_flow

- **DEBUG**: 20x20x10 (толщина 10 слоев)
- **EXPERIMENT**: 50x50x20
- **OPTIMIZED**: 100x100x50

### Использование energy_flow

```python
from energy_flow.config import create_experiment_config
from energy_flow.core import EnergyLattice, SimpleNeuron, FlowProcessor
from energy_flow.training import EnergyTrainer

config = create_experiment_config()
lattice = EnergyLattice(config)
neuron = SimpleNeuron(config)
processor = FlowProcessor(lattice, neuron, config)

trainer = EnergyTrainer(processor, config)
trainer.train(input_embeddings, target_embeddings)
```
