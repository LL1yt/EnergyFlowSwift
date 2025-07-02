# Система конфигурации с 3 режимами и защитой от hardcoded значений

## 🎯 Реализованные возможности

### 1. **Три режима конфигурации**

- **DEBUG** - для быстрых тестов и отладки
  - Маленькая решетка (8x8x8)
  - Минимальные параметры
  - Максимальное логирование
  
- **EXPERIMENT** - для исследований
  - Средняя решетка (15x15x15)
  - Сбалансированные параметры
  - Умеренное логирование
  
- **OPTIMIZED** - для финальных прогонов
  - Большая решетка (30x30x30)
  - Максимальные параметры
  - Минимальное логирование
  - Включены оптимизации производительности

### 2. **Централизованные параметры**

Добавлены новые компоненты конфигурации для замены hardcoded значений:

- `TrainingOptimizerSettings` - параметры оптимизатора и обучения
- `EmbeddingMappingSettings` - настройки маппинга эмбеддингов
- `MemoryManagementSettings` - управление памятью
- `ArchitectureConstants` - архитектурные константы
- `AlgorithmicStrategies` - стратегии и строковые константы

### 3. **Система защиты от hardcoded значений**

Новый модуль `new_rebuild/utils/hardcoded_checker.py` предоставляет:

- **`check_hardcoded_value()`** - проверка отдельных значений
- **`@no_hardcoded`** - декоратор для функций
- **`strict_no_hardcoded()`** - автоматическая замена на значения из конфига
- **`HardcodedValueError`** - исключение с подробной информацией
- **`allow_hardcoded`** - контекстный менеджер для временного отключения

## 📝 Использование

### Создание конфигурации с нужным режимом:

```python
from new_rebuild.config import (
    create_debug_config,
    create_experiment_config,
    create_optimized_config,
    set_project_config
)

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

### Использование параметров из конфига:

```python
from new_rebuild.config import get_project_config

config = get_project_config()

# Вместо hardcoded значений
learning_rate = config.training_optimizer.learning_rate
dropout = config.architecture.cnf_dropout_rate
max_neighbors = config.architecture.spatial_max_neighbors
```

### Защита от hardcoded значений:

```python
from new_rebuild.utils import no_hardcoded, strict_no_hardcoded

# Декоратор для функций
@no_hardcoded
def train_model(lr=None, dropout=None):
    # Функция выбросит ошибку если передать hardcoded значения
    pass

# Автоматическая замена
hidden_dim = strict_no_hardcoded(64, "model.hidden_dim")
# Автоматически возьмет значение из config.model.hidden_dim
```

## 🔄 Миграция существующего кода

### Шаг 1: Найти hardcoded значения

```python
# Плохо ❌
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)
dropout = nn.Dropout(0.1)
batch_size = 64
```

### Шаг 2: Заменить на значения из конфига

```python
# Хорошо ✅
config = get_project_config()
optimizer = torch.optim.Adam(
    params, 
    lr=config.training_optimizer.learning_rate,
    weight_decay=config.training_optimizer.weight_decay
)
dropout = nn.Dropout(config.architecture.cnf_dropout_rate)
batch_size = config.training.batch_size
```

### Шаг 3: Добавить проверки

```python
@no_hardcoded  # Добавить декоратор к новым функциям
def my_function(...):
    pass
```

## 💡 Преимущества

1. **Централизованное управление** - все параметры в одном месте
2. **Легкое переключение режимов** - один вызов функции
3. **Защита от ошибок** - автоматическое обнаружение hardcoded значений
4. **Удобная миграция** - постепенный переход с помощью `strict_no_hardcoded`
5. **Подробные ошибки** - система подсказывает где искать параметр в конфиге

## 🚀 Следующие шаги

1. Постепенно мигрировать все модули на использование конфига
2. Добавлять `@no_hardcoded` к новым функциям
3. Расширять список централизованных параметров по мере необходимости
4. Использовать соответствующий режим для каждой задачи

## 📊 Статус реализации

- ✅ Создана система 3 режимов конфигурации
- ✅ Добавлены централизованные параметры для hardcoded значений
- ✅ Реализована система проверки hardcoded значений
- ✅ Созданы примеры использования и миграции
- 🔄 Требуется постепенная миграция существующих модулей