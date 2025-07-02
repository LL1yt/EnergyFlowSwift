# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a research project implementing a **3D Cellular Neural Network** inspired by biological brain structures. The system uses cellular automata-like structures arranged in a 3D lattice where each cell runs the same neural network prototype but processes signals from neighboring cells.

### Контекст проекта

- Исследовательский проект на одного разработчика
- Структура: AA/ (легаси), AA/new_rebuild/ (активная разработка), AA/archive/ (старые версии)
- Тесты запускаются из корня AA/
- new_rebuild\Working_solutions_from_the_Legacy_project_that_can_be_used.md - можно использовать как примеры

временное решение для тестов: вообще, возможно, что у нас некоторая путаница из-за перехода на обновленный конфиг. и если мы даже временные для тестов значения используем, то есть решение все равно прописовать их в центральном конфиге и комментировать рабочие конфиги, а когда тесты проходят успешно - комментировать конфиги для теста и возращаться к рабочему варианту, если он отличался.

### Детальная структура `new_rebuild`

Директория `new_rebuild` содержит основную, переработанную архитектуру проекта с модульной структурой.

#### **`new_rebuild/` (корень)**

- **`__init__.py`** (1.8KB, 60 строк) - Главный модуль с экспортом основных компонентов. Включает ProjectConfig, BaseCell, VectorizedGNNCell и фабричные функции. Версия 0.1.0.
- **`Working_solutions_from_the_Legacy_project_that_can_be_used.md`** (4.7KB) - Документация с рабочими решениями из legacy проекта для переиспользования.

#### **`new_rebuild/config/` - Централизованная конфигурация**

- **`__init__.py`** - Экспорт SimpleProjectConfig как основного ProjectConfig, включает все компоненты конфигурации и фабричные функции. Поддерживает 3 режима конфигурации: DEBUG, EXPERIMENT, OPTIMIZED.
- **`config_components.py`** (большой файл) - Модульные компоненты конфигурации через композицию:
  - `LatticeSettings` - размеры решетки, стратегии размещения
  - `ModelSettings` - параметры GNN модели (state_size, hidden_dim, neighbor_count)
  - `TrainingSettings` - настройки обучения (learning_rate, batch_size, optimizer)
  - `CNFSettings` - настройки Continuous Normalizing Flows
  - `EulerSettings` - параметры интегрирования
  - `CacheSettings` - конфигурация кэширования с GPU поддержкой
  - `SpatialSettings` - пространственная оптимизация
  - `VectorizedSettings` - векторизованные операции
  - `DeviceSettings` - управление GPU/CPU
  - `LoggingSettings` - централизованное логирование с поддержкой custom debug levels
  - `ExpertSettings` - настройки MoE экспертов
  - `EmbeddingSettings` и `TrainingEmbeddingSettings` - работа с эмбеддингами
  - `TrainingOptimizerSettings` - параметры оптимизатора и обучения
  - `EmbeddingMappingSettings` - настройки маппинга эмбеддингов
  - `MemoryManagementSettings` - управление памятью
  - `ArchitectureConstants` - архитектурные константы
  - `AlgorithmicStrategies` - стратегии и строковые константы
- **`simple_config.py`** - Основной класс `SimpleProjectConfig` с композицией компонентов. Предоставляет единую точку доступа ко всем настройкам проекта.

#### **`new_rebuild/core/` - Ядро архитектуры**

- **`__init__.py`** - Экспорт основных компонентов: BaseCell, VectorizedGNNCell, create_cell, Lattice3D, create_lattice.

##### **`new_rebuild/core/cells/` - Клетки нейронной сети**

- **`__init__.py`** (4.2KB, 122 строки) - Фабрика клеток с поддержкой различных типов: vectorized_gnn, base_cell. Включает валидацию параметров и создание клеток с логированием.
- **`base_cell.py`** (3.2KB, 94 строки) - Абстрактный базовый класс `BaseCell` для всех типов клеток. Определяет интерфейс forward(), методы для сброса памяти и получения информации о параметрах.
- **`vectorized_gnn.py`** (16KB, 393 строки) - Основная векторизованная GNN клетка `VectorizedGNNCell`. Использует attention механизм для агрегации соседей, поддерживает внешние входы, оптимизирована для GPU с batch processing.

##### **`new_rebuild/core/moe/` - Mixture of Experts архитектура**

- **`__init__.py`** (3.7KB, 82 строки) - Экспорт всех MoE компонентов и фабричных функций для создания connection processor и экспертов.
- **`moe_processor.py`** (28KB, 617 строк) - Основной `MoEConnectionProcessor` с тремя экспертами:
  - Local Expert (SimpleLinear, 10% связей)
  - Functional Expert (HybridGNN_CNF, 55% связей)
  - Distant Expert (GPUEnhancedCNF, 35% связей)
  - Включает gating network и connection classifier с кэшированием
- **`gating_network.py`** (5.3KB, 122 строки) - `GatingNetwork` для адаптивного взвешивания экспертов на основе состояний клеток.
- **`connection_classifier.py`** (33KB, 736 строк) - `UnifiedConnectionClassifier` для классификации связей по типам (LOCAL/FUNCTIONAL/DISTANT) с GPU кэшированием.
- **`connection_cache.py`** (32KB, 735 строк) - Система кэширования классификации связей с GPU ускорением для RTX 5090.
- **`connection_types.py`** (1.7KB, 44 строки) - Enum'ы для типов связей и категорий.
- **`simple_linear_expert.py`** (12KB, 235 строк) - Простой линейный эксперт для локальных связей.
- **`hybrid_gnn_cnf_expert.py`** (12KB, 284 строки) - Гибридный эксперт, комбинирующий GNN и CNF для функциональных связей.
- **`functional_similarity.py`** (4.2KB, 101 строка) - Вычисление функционального сходства между клетками.
- **`distance_calculator.py`** (3.8KB, 94 строки) - Расчет расстояний в 3D решетке с различными метриками.

##### **`new_rebuild/core/cnf/` - Continuous Normalizing Flows**

- **`__init__.py`** (3.0KB, 88 строк) - Экспорт GPU-оптимизированных CNF компонентов. Deprecated старые компоненты заменены на GPU Enhanced версии.
- **`gpu_enhanced_cnf.py`** (26KB, 666 строк) - `GPUEnhancedCNF` с векторизованным Neural ODE, batch processing и адаптивной интеграцией. Поддерживает различные режимы обработки батчей.
- **`gpu_optimized_euler_solver.py`** (44KB, 1110 строк) - `GPUOptimizedEulerSolver` с адаптивными методами интеграции, Lipschitz-based step control и GPU ускорением.

##### **`new_rebuild/core/lattice/` - 3D решетка**

- **`__init__.py`** (5.6KB, 152 строки) - Экспорт компонентов решетки: Lattice3D, Position3D, spatial optimization классы.
- **`lattice.py`** (15KB, 351 строка) - Основной класс `Lattice3D` для MoE архитектуры. Управляет GNN клетками, MoE processor и spatial optimization.
- **`position.py`** (4.1KB, 92 строки) - `Position3D` helper для работы с 3D координатами и индексами.
- **`enums.py`** (1.8KB, 49 строк) - Enum'ы для Face, PlacementStrategy и других констант решетки.
- **`io.py`** (10KB, 253 строки) - `IOPointPlacer` для размещения входных и выходных точек на гранях куба.
- **`gpu_spatial_hashing.py`** (22KB, 509 строк) - GPU-ускоренное пространственное хеширование с Morton encoding.
- **`vectorized_spatial_processor.py`** (16KB, 371 строка) - Векторизованная обработка пространственных запросов.

##### **`new_rebuild/core/lattice/spatial_optimization/` - Пространственная оптимизация**

- **`__init__.py`** (4.8KB, 125 строк) - Экспорт unified spatial optimizer и связанных компонентов.
- **`unified_spatial_optimizer.py`** (22KB, 589 строк) - `UnifiedSpatialOptimizer` - единый оптимизатор для всех типов пространственных операций.
- **`adaptive_chunker.py`** (30KB, 721 строка) - `AdaptiveGPUChunker` для умного разбиения больших решеток на GPU-оптимальные блоки.
- **`gpu_spatial_processor.py`** (36KB, 847 строк) - `GPUSpatialProcessor` для высокопроизводительной обработки пространственных запросов на GPU.
- **`memory_manager.py`** (8.4KB, 212 строк) - `MemoryPoolManager` для управления памятью в пространственных операциях.

##### **`new_rebuild/core/training/` - Компоненты обучения**

- **`__init__.py`** (747B, 26 строк) - Экспорт EmbeddingTrainer.
- **`embedding_trainer.py`** (24KB, 556 строк) - Основной `EmbeddingTrainer` для обучения 3D куба на эмбеддингах. Реализует полный цикл: эмбединги → куб → эмбединги → текст.
- **`embedding_lattice_mapper.py`** (20KB, 420 строк) - Компоненты для маппинга эмбеддингов в решетку и обратно.

##### **`new_rebuild/core/inference/` - Компоненты инференса**

- **`__init__.py`** (756B, 25 строк) - Экспорт text decoder компонентов.
- **`text_decoder.py`** (26KB, 645 строк) - `SimpleTextDecoder` и `JointTextDecoder` для преобразования эмбеддингов обратно в текст с кэшированием.

##### **`new_rebuild/core/common/` - Общие компоненты**

- **`__init__.py`** (699B, 18 строк) - Экспорт EmbeddingTransformer и интерфейсов.
- **`embedding_transformer.py`** (14KB, 310 строк) - `EmbeddingTransformer` для преобразования размерностей эмбеддингов (768D ↔ lattice dimensions).
- **`interfaces.py`** (6.4KB, 177 строк) - Абстрактные интерфейсы для различных компонентов системы.

#### **`new_rebuild/utils/` - Утилиты**

- **`__init__.py`** (2.2KB, 79 строк) - Экспорт логирования и device management функций.
- **`logging.py`** (24KB, 602 строки) - Централизованная система логирования с caller tracking, anti-duplication фильтрами, контекстным форматированием и поддержкой custom debug levels (DEBUG_VERBOSE, DEBUG_CACHE, DEBUG_SPATIAL, DEBUG_FORWARD, DEBUG_MEMORY, DEBUG_TRAINING, DEBUG_INIT).
- **`device_manager.py`** (17KB, 434 строки) - `DeviceManager` для управления GPU/CPU, автоматического определения оптимального устройства и мониторинга памяти.
- **`model_cache.py`** (14KB, 379 строк) - Система кэширования моделей с поддержкой различных бэкендов.
- **`hardcoded_checker.py`** - Система защиты от hardcoded значений с декораторами `@no_hardcoded`, функциями `strict_no_hardcoded()` и исключением `HardcodedValueError`.

#### **`new_rebuild/docs/` - Документация**

- **`plan_training.md`** - Детальный план реализации обучения на эмбеддингах LLM. Описывает архитектуру teacher-student подхода с DistilBERT.
- **`todo.md`** - Список задач на будущее и моментов для запоминания.
- **`custom_debug_levels_guide.md`** - описание новой системы логирования

### Принципы работы

**приоритет на вдумчивость, постепенность и эффективность(сначала разобраться в проблеме и понять, что к чему, а птом действовать. мы никуда не спешим) - применяем простой и эффективный вариант, доводим через тесты до рабочего состояния и потом оптимизируем и усложняем, если это улучшит производительность**

- Минимальные церемонии, максимальная эффективность, в том смысле, что можно пожертвовать перфекционизмом в угоду простому и эффективному решению
- Используй современные языковые возможности
- Предпочитай прямолинейные решения сложным абстракциям
- при этом нас НЕ интересует продакшн и все что с этим связано, когда речь идет о больших коммерческих проектах с большим числом сотрудников
- **Конфигурации** - удачные параметры из экспериментов
- **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
- Использовать централизованное логирование, что бы можно было отследить какой модуль был вызван и кем
- **ИСКЛЮЧАЕМ CLI автоматизация**(за редким исключением, когда без этого нельзя и пользователь укажет отдельно) - оставляем только Python API
- **ИСКЛЮЧАЕМ Множественные конфигурации** - оставляем только один new_rebuild\config\project_config.py
- **ИСКЛЮЧАЕМ Legacy совместимость** - оставляем чистый код
- **ИСКЛЮЧАЕМ Динамические конфигурации**(за редким исключением, когда без этого нельзя и пользователь укажет отдельно) - оставляем статичные конфигурации
- так как у нас исследование, то нам важно решать любые проблемы лично, а не делать заглушки в виде fallback, так что лучше пусть будет ошибка, которая приведет к полному падению программы, чем костыли непонятные
- в наличии есть nvidia 5090 32gb памяти, так что при возможности нужно это использовать
- **Конфигурации**: 1. Всегда используйте значения из config вместо hardcoded 2. Применяйте @no_hardcoded декоратор к новым функциям 3. Используйте strict_no_hardcoded() для автоматической замены
- **Конфигурации**: дополнительная информация по режимам конфигурации: docs\CONFIG_MODES_SUMMARY.md
- **Система логирования**: Используйте custom debug levels для точной настройки вывода логов: `debug_cache()`, `debug_spatial()`, `debug_forward()`, `debug_memory()`, `debug_training()`, `debug_init()`, `debug_verbose()`. Подробности в new_rebuild/docs/custom_debug_levels_guide.md

### Система конфигурации (НОВИНКА 2025)

**3 режима конфигурации для разных этапов разработки:**

- **DEBUG** - для быстрых тестов и отладки
  - Маленькая решетка (8x8x8)
  - Минимальные параметры
  - Максимальное логирование с custom debug levels
  
- **EXPERIMENT** - для исследований
  - Средняя решетка (15x15x15)
  - Сбалансированные параметры
  - Умеренное логирование
  
- **OPTIMIZED** - для финальных прогонов
  - Большая решетка (30x30x30)
  - Максимальные параметры
  - Минимальное логирование, все оптимизации включены

**Использование режимов:**
```python
from new_rebuild.config import create_debug_config, create_experiment_config, create_optimized_config, set_project_config

# Для отладки
config = create_debug_config()
set_project_config(config)

# Для экспериментов
config = create_experiment_config()
set_project_config(config)

# Для финальных прогонов
config = create_optimized_config()
set_project_config(config)
```

**Система защиты от hardcoded значений:**

- `@no_hardcoded` - декоратор для функций
- `strict_no_hardcoded()` - автоматическая замена hardcoded значений на значения из конфига
- `HardcodedValueError` - исключение с подробной информацией о том, где искать параметр в конфиге
- `allow_hardcoded` - контекстный менеджер для временного отключения проверок

**Custom Debug Levels система логирования:**

Специальные уровни логирования между DEBUG (10) и INFO (20):
- `DEBUG_VERBOSE` (11) - самый подробный вывод
- `DEBUG_CACHE` (12) - операции кэширования
- `DEBUG_SPATIAL` (13) - пространственная оптимизация
- `DEBUG_FORWARD` (14) - детали forward pass
- `DEBUG_MEMORY` (15) - управление памятью GPU
- `DEBUG_TRAINING` (16) - прогресс обучения
- `DEBUG_INIT` (17) - инициализация компонентов

Использование: `logger.debug_cache()`, `logger.debug_spatial()`, etc.

### Ключевые оптимизации производительности

**Connection Cache System (НОВИНКА 2024)**

Система pre-computed кэширования классификации связей для массивного ускорения больших решеток:

- **Pre-computed расстояния**: Статические distance matrices для всех пар клеток
- **Базовая классификация**: LOCAL/DISTANT связи вычисляются один раз при старте
- **Functional candidates**: Средние расстояния для динамической similarity проверки
- **Disk persistence**: Кэш сохраняется на диск с hash-ключами конфигурации
- **Automatic management**: Автоматическая перестройка при изменении параметров
