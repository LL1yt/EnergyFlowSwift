# Energy Flow Dataset Module

Оптимизированный модуль для создания и управления датасетами в energy_flow архитектуре.

## 🎯 Основные возможности

- **Автоматическая проверка и загрузка модели-учителя** (DistilBERT)
- **Загрузка готовых эмбеддингов** из .pt файлов  
- **Генерация данных из SNLI** датасета
- **Унифицированный API** для всех источников данных
- **Интеграция с EnergyTrainer** и системой конфигурации
- **GPU оптимизация** и кэширование эмбеддингов

## 🚀 Быстрый старт

```python
from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.dataset import (
    create_dataset_config_from_energy,
    create_dataset_manager
)

# 1. Настройка конфигурации
energy_config = create_debug_config()
set_energy_config(energy_config)

dataset_config = create_dataset_config_from_energy(energy_config)

# 2. Создание менеджера датасета
dataset_manager = create_dataset_manager(dataset_config, energy_config)

# 3. Проверка готовности системы
validation = dataset_manager.validate_setup()
if validation['overall_status']:
    print("✅ System ready for training")
    
    # 4. Создание DataLoader
    dataloader = dataset_manager.create_dataloader(batch_size=16)
    
    # 5. Использование в обучении
    for batch in dataloader:
        input_texts = batch['input_text']
        target_texts = batch['target_text'] 
        input_embeddings = batch['input_embedding']
        target_embeddings = batch['target_embedding']
        # ... обучение
```

## 📁 Структура модуля

```
dataset/
├── __init__.py           # Основные экспорты
├── config.py            # DatasetConfig и настройки
├── manager.py           # DatasetManager - центральный класс
├── utils.py             # Утилиты диагностики
├── providers/           # Провайдеры данных
│   ├── base_provider.py      # Базовый класс
│   ├── teacher_model.py      # DistilBERT управление  
│   ├── snli_provider.py      # SNLI датасет
│   └── precomputed_provider.py # Готовые эмбеддинги
```

## ⚙️ Конфигурация

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    # Модель-учитель
    teacher_model: str = "distilbert-base-uncased"
    use_local_model: bool = True
    
    # Источники данных (в порядке приоритета)
    dataset_sources: List[str] = field(default_factory=lambda: ["precomputed", "snli"])
    
    # Параметры загрузки
    batch_size: int = 32
    max_samples_per_source: Optional[int] = None
    normalize_embeddings: bool = True
    
    # Кэширование
    embedding_cache_enabled: bool = True
    cache_batch_size: int = 64
```

### Адаптация под режимы

Конфигурация автоматически адаптируется под режим energy_flow:

- **DEBUG**: `max_samples_per_source=1000`, быстрая загрузка
- **EXPERIMENT**: `max_samples_per_source=5000`, сбалансированная загрузка  
- **OPTIMIZED**: без ограничений, максимальная производительность

## 🔧 Провайдеры данных

### TeacherModelProvider
Управление моделью-учителем (DistilBERT):
```python
# Автоматическая проверка локальной модели
if not teacher_provider.is_available():
    teacher_provider.download_model_if_needed()

# Генерация эмбеддингов с кэшированием
embeddings = teacher_provider.encode_texts(texts)
```

### PrecomputedProvider  
Загрузка готовых .pt файлов:
- Формат `generate_snli_embedding_dataset.py`
- Формат `unified_dataset_loader.py` 
- Простые тензоры

### SNLIProvider
Генерация данных из SNLI датасета:
- Настраиваемая фракция датасета (по умолчанию 20%)
- Автоматическая фильтрация и валидация
- Статистика по label распределению

## 📊 Диагностика и валидация

### Полная проверка системы
```python
validation = dataset_manager.validate_setup()
print(f"Teacher model: {'✅' if validation['teacher_model'] else '❌'}")
print(f"Data providers: {sum(validation['providers'].values())}/{len(validation['providers'])}")
print(f"Overall status: {'🎉' if validation['overall_status'] else '⚠️'}")
```

### Статистика датасета
```python
stats = dataset_manager.get_statistics()
print(f"Total samples: {stats['total_samples']:,}")
print(f"Sources: {', '.join(stats['providers_used'])}")
print(f"Embedding dim: {stats['embedding_dimension']}")
```

### Текстовый отчет
```python
from energy_flow.dataset.utils import create_dataset_summary_report
report = create_dataset_summary_report(dataset_manager)
print(report)
```

## 🔗 Интеграция с EnergyTrainer

Новый dataset модуль полностью совместим с EnergyTrainer:

```python
from energy_flow.training import EnergyTrainer

# Создание trainer'а
trainer = EnergyTrainer(energy_config)

# Подготовка данных
dataloader = dataset_manager.create_dataloader()

# Обучение
for batch in dataloader:
    step_metrics = trainer.train_step(
        input_texts=batch['input_text'],
        target_texts=batch['target_text'], 
        teacher_input_embeddings=batch['input_embedding'],
        teacher_target_embeddings=batch['target_embedding']
    )
```

## 📝 Примеры использования

### Базовый пример
```bash
python energy_flow/examples/dataset_example.py
```

### Полное обучение с интеграцией
```bash
python energy_flow/examples/training_with_dataset.py
```

## 🛠️ Устранение проблем

### Модель-учитель не найдена
```python
# Автоматическая загрузка
if not dataset_manager.ensure_teacher_model():
    print("❌ Failed to setup teacher model")
    
# Или вручную через провайдер
teacher_provider.download_model_if_needed()
```

### Нет готовых данных
1. Запустите legacy скрипты для генерации:
   ```bash
   python generate_snli_embedding_dataset.py --fraction 0.1
   ```

2. Или используйте только SNLI провайдер:
   ```python
   dataset_config.dataset_sources = ["snli"]
   ```

### Проблемы с памятью
```python
# Уменьшите размеры для DEBUG режима
dataset_config.max_samples_per_source = 500
dataset_config.cache_batch_size = 32
dataset_config.batch_size = 8
```

## 🎯 Производительность

- **Кэширование эмбеддингов**: автоматическое для повторяющихся текстов
- **Батчевая обработка**: оптимизированная загрузка данных
- **GPU оптимизация**: автоматическое использование CUDA
- **Ленивая загрузка**: данные загружаются по требованию

## 🔄 Миграция с legacy

Старые скрипты остаются совместимы, но новый модуль предоставляет:
- Единый API для всех источников данных
- Автоматическую интеграцию с конфигурацией
- Улучшенную диагностику и валидацию
- Прямую совместимость с EnergyTrainer

Постепенный переход:
1. Используйте новый `DatasetManager` для новых экспериментов
2. Legacy скрипты продолжают работать для существующих пайплайнов
3. Миграция завершается заменой вызовов в основных тренировочных скриптах