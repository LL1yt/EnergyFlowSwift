# План реализации обучения на эмбедингах LLM для 3D Cellular Neural Network

## Обзор архитектуры обучения

Цель: Обучить 3D решетку клеточных нейронных сетей используя эмбединги от существующих LLM моделей (начнем с DistilBERT 768D).
(открытость для улучшений в процессе)

### Ключевые компоненты системы

1. **Преобразование эмбедингов**: 768D → (для тестов начнем с 8х8 для прогонки быстрой, главное помнить, что решетка слишком маленькая для некоторых тестов и тогда планировать тесты на потом для больших решеток в new_rebuild/docs/todo.md) 1369D (37×37) → обработка кубом → 1369D → 768D
2. **Обучение через teacher-student подход**: DistilBERT как учитель, 3D куб как ученик
3. **Модуль декодирования**: Преобразование эмбедингов в текст для валидации
4. **Централизованная конфигурация**: Все параметры в едином конфиге

## Этап 1: Расширение конфигурации

### 1.1 Добавление настроек эмбедингов в config_components.py

```python
@dataclass
class EmbeddingSettings:
    """Настройки для работы с эмбедингами"""
    # Основные параметры
    teacher_model: str = "distilbert-base-uncased"
    teacher_embedding_dim: int = 768
    cube_surface_dim: int = 8  # Для куба 8×8×8
    # cube_surface_dim: int = 37  # Для куба 37×37×37
    cube_embedding_dim: int = 64  # 8×8
    # cube_embedding_dim: int = 1369  # 37×37

    # Параметры преобразования
    transformation_type: str = "linear"  # linear, attention, autoencoder
    use_layer_norm: bool = True
    dropout_rate: float = 0.1

    # Кэширование
    cache_embeddings: bool = True
    cache_dir: str = "cache/embeddings"
    max_cache_size_gb: float = 10.0

@dataclass
class TrainingEmbeddingSettings:
    """Расширенные настройки обучения для эмбедингов"""
    # Фазы обучения
    warmup_epochs: int = 10
    main_epochs: int = 100
    fine_tune_epochs: int = 50

    # Loss функции и веса
    reconstruction_loss_weight: float = 1.0
    similarity_loss_weight: float = 0.5
    diversity_loss_weight: float = 0.1
    emergence_loss_weight: float = 0.2

    # Curriculum learning
    use_curriculum_learning: bool = True
    curriculum_start_difficulty: float = 0.1
    curriculum_end_difficulty: float = 1.0

    # Параметры батчей
    embedding_batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # Тестовые параметры (закомментируем после тестов)
    test_mode: bool = True
    test_lattice_dim: int = 37
    test_dataset_size: int = 1000
    test_validation_split: float = 0.2
```

### 1.2 Интеграция в SimpleProjectConfig

Добавить новые поля в основной конфиг с поддержкой централизованного логирования.

## Этап 2: Структура модулей обучения

### 2.1 Создание директории new_rebuild/core/training/

```
new_rebuild/core/training/
├── __init__.py
├── embedding_transformer.py    # Преобразование размерностей эмбедингов
├── embedding_decoder.py        # Декодирование в текст
├── embedding_trainer.py        # Основной тренер
├── loss_functions.py          # Специализированные loss функции
├── data_loaders.py           # Загрузчики данных с кэшированием
└── training_utils.py         # Вспомогательные функции
```

### 2.2 Модуль преобразования эмбедингов (embedding_transformer.py)

```python
class EmbeddingTransformer(nn.Module):
    """Преобразование между размерностями эмбедингов с сохранением семантики"""

    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)

        # 768D → 1369D (на поверхность куба)
        self.to_cube = self._build_transformer(
            config.embedding.teacher_embedding_dim,
            config.embedding.cube_embedding_dim
        )

        # 1369D → 768D (обратно к teacher)
        self.from_cube = self._build_transformer(
            config.embedding.cube_embedding_dim,
            config.embedding.teacher_embedding_dim
        )
```

### 2.3 Интеграция с MoE экспертами

Модификация существующих экспертов для работы с эмбедингами:

```python
class EmbeddingAwareMoEProcessor(MoEConnectionProcessor):
    """Расширенный MoE процессор с поддержкой эмбедингов"""

    def forward(self, cell_states, connections, embedding_input=None):
        # Интеграция эмбединга в состояния клеток
        if embedding_input is not None:
            surface_cells = self._get_surface_cells()
            self._inject_embedding(surface_cells, embedding_input)

        # Стандартная обработка
        return super().forward(cell_states, connections)
```

## Этап 3: Процесс обучения

### 3.1 Основной цикл обучения

1. **Подготовка данных**

   - Загрузка текстов из датасета
   - Генерация эмбедингов через DistilBERT
   - Кэширование для ускорения

2. **Forward pass**

   - Преобразование 768D → 1369D
   - Размещение на передней поверхности куба (37×37)
   - Обработка через все слои куба
   - Считывание с задней поверхности
   - Преобразование 1369D → 768D

3. **Loss вычисление**

   - Reconstruction loss: MSE между выходом и целевым эмбедингом
   - Similarity loss: Косинусное сходство
   - Emergence loss: Метрики паттернов активации

4. **Backward pass**
   - Градиенты только для MoE экспертов
   - Декодер остается замороженным

### 3.2 Валидация через декодирование

```python
class ValidationDecoder:
    """Декодер для проверки качества обучения"""

    def validate_epoch(self, cube_output_embeddings):
        # Преобразуем эмбединги обратно в текст
        decoded_texts = self.decode_embeddings(cube_output_embeddings)

        # Логируем примеры для отслеживания прогресса
        self.logger.info(f"Примеры декодированных текстов: {decoded_texts[:5]}")

        # Вычисляем метрики качества
        return self.compute_metrics(decoded_texts)
```

## Этап 4: Интеграция с централизованными системами

### 4.1 Логирование

Использование существующей системы логирования для всех модулей:

```python
from new_rebuild.utils.logging import get_logger, log_performance, LogContext

logger = get_logger(__name__)

with LogContext("embedding_training", epoch=epoch, phase="forward"):
    logger.info(f"Обработка батча {batch_idx}")
    # Код обучения
    log_performance("batch_processing", duration, loss=loss.item())
```

### 4.2 Управление устройствами

Использование DeviceManager для всех операций с GPU:

```python
from new_rebuild.utils.device_manager import get_device_manager

device_manager = get_device_manager()
model = device_manager.transfer_module(embedding_transformer)
```

## Этап 5: Тестирование и оптимизация

### 5.1 Создание тест-настройщика

Модуль для интерактивной настройки параметров:

```python
class ConfigurationTuner:
    """Интерактивный настройщик параметров конфигурации"""

    def tune_parameter(self, param_name: str, test_values: list):
        results = []
        for value in test_values:
            # Изменяем параметр
            self._set_config_value(param_name, value)

            # Запускаем мини-тест
            metrics = self._run_quick_test()

            # Собираем результаты
            results.append({
                'value': value,
                'performance': metrics['throughput'],
                'quality': metrics['loss'],
                'memory': metrics['memory_usage']
            })

        return self._analyze_results(results)
```

### 5.2 Мониторинг производительности

- Отслеживание скорости обучения
- Использование GPU памяти
- Cache hit rates
- Emergence метрики

## Этап 6: Постепенное усложнение

### 6.1 Фаза 1: Базовое обучение

- Простая реконструкция эмбедингов
- Фиксированный размер куба 37×37×37
- Линейные преобразования

### 6.2 Фаза 2: Улучшения

- Добавление attention механизмов
- Curriculum learning
- Triplet loss для семантического выравнивания

### 6.3 Фаза 3: Масштабирование

- Поддержка различных размеров кубов
- Transfer learning между размерами
- Multi-teacher обучение

## Потенциальные проблемы и решения

### Проблема 1: Изменение размерности при разных кубах

**Решение**: Использовать адаптивные преобразователи и сохранять мапинги размерностей

### Проблема 2: Потеря семантики при преобразовании

**Решение**: Многоуровневые loss функции и residual connections

### Проблема 3: Скорость обучения

**Решение**: Агрессивное кэширование, GPU оптимизации, gradient checkpointing

## Метрики успеха

1. **Reconstruction качество**: MSE < 0.01
2. **Semantic similarity**: Косинусное сходство > 0.9
3. **Emergence patterns**: Наличие специализации клеток
4. **Декодирование**: Осмысленные фразы после 50 эпох

## Следующие шаги после базовой реализации

1. Интеграция системы управления памятью из docs/
2. Полный цикл обучения через текст (без teacher эмбедингов)
3. Multi-modal обучение (текст + изображения)
4. Иерархические кубы для сложных задач

Дополнительные соображения и улучшения

1. Проблема катастрофического забывания

При обучении на эмбедингах разных доменов куб может "забывать" предыдущие знания. Предлагаю:

- Elastic Weight Consolidation (EWC) - сохранять важность весов для предыдущих задач
- Progressive Neural Networks подход - заморозка части экспертов после обучения на домене
- Memory replay buffer - периодически подмешивать старые эмбединги

2. Эффективность преобразования размерностей

Линейное преобразование 768D → 1369D может терять информацию. Рекомендую:

- Иерархическое кодирование: 768D → 512D → 1024D → 1369D с residual connections
- Variational Information Bottleneck - сохранение максимальной взаимной информации
- Learned positional encoding для 37×37 поверхности куба

3. Эмерджентность vs Управляемость

Баланс между спонтанной специализацией и контролируемым обучением:

- Sparse activation patterns - поощрять разреженную активацию через L1 регуляризацию
- Information flow metrics - отслеживать пути распространения информации в кубе
- Cell specialization scores - метрики для измерения функциональной дифференциации

4. Оптимизация для больших кубов

Для масштабирования beyond 37×37×37:

- Hierarchical cubes - вложенные кубы разных масштабов (как в мозге: колонки → области)
- Adaptive computation - динамическое включение/выключение регионов
- Gradient checkpointing с умным выбором checkpoint слоев

5. Улучшенная валидация

Вместо простого декодирования в текст:

- Semantic similarity matrices - сравнение внутренних представлений с teacher model
- Probing tasks - специальные задачи для проверки понимания (NER, sentiment, etc.)
- Attention visualization - визуализация путей информации через куб

6. Ускорение сходимости

- Warmup with synthetic data - начать с простых синтетических паттернов
- Knowledge distillation temperature scheduling - динамическая температура для softmax
- Adaptive learning rates per expert - разные LR для local/functional/distant экспертов

7. Робастность к размеру куба

Для переноса между размерами:

- Scale-invariant representations - нормализация по размеру решетки
- Fractal encoding - самоподобные паттерны на разных масштабах
- Universal base embeddings - промежуточное представление независимое от размера

8. Практические советы

- Начните с 8×8×8 для быстрой итерации, потом масштабируйте
- Логируйте embedding norms - для отслеживания коллапса представлений
- Используйте mixed precision - FP16 для forward, FP32 для градиентов
- Batch normalization per expert - стабилизация обучения разнородных экспертов

9. Архитектурное предложение

Добавить "Router Cells" - специальные клетки на поверхности для:

- Маршрутизации входящей информации
- Агрегации выходных сигналов
- Метаобучения оптимальных путей

10. Метрики эмерджентности

Критически важно измерять:

- Mutual Information между слоями
- Effective Dimensionality внутренних представлений
- Compositional Complexity - способность комбинировать концепты

Эти улучшения можно внедрять постепенно после базовой реализации. Особенно рекомендую начать с пунктов 2, 5 и 8 как наиболее практичных.
