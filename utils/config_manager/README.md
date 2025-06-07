# ConfigManager - Централизованное управление конфигурацией

## 📋 Обзор

ConfigManager обеспечивает централизованное управление конфигурацией для проекта 3D Cellular Neural Network. Поддерживает модульную архитектуру, валидацию, hot reloading и environment-specific настройки.

## 🚀 Основные возможности

- **Модульная архитектура** - автоматическое обнаружение конфигураций модулей
- **Иерархическое наследование** - base config + module configs + environment overrides
- **Hot reloading** - автоматическая перезагрузка при изменении файлов
- **Валидация** - схемы валидации и правила проверки
- **Thread-safe** - безопасная работа в многопоточной среде
- **Dot-notation** - удобный доступ к вложенным настройкам
- **Environment overrides** - различные настройки для dev/test/prod

## 📦 Установка и использование

### Быстрый старт

```python
from utils.config_manager import ConfigManager, create_config_manager

# Создание ConfigManager
config = create_config_manager()

# Получение конфигурации
lattice_config = config.get_config('lattice')
depth = config.get_config('lattice', 'dimensions.depth')

# Установка значений
config.set_config('training', 'batch_size', 32)
config.set_config('training', learning_rate=0.001, num_epochs=100)

# Работа с секциями
training_section = config.get_section('training')
training_section.set('optimizer.type', 'Adam')
print(training_section.get('batch_size', 32))
```

### Продвинутое использование

```python
from utils.config_manager import ConfigManager, ConfigManagerSettings

# Настройка ConfigManager
settings = ConfigManagerSettings(
    base_config_path="config/main_config.yaml",
    environment="production",
    enable_hot_reload=True,
    enable_validation=True
)

with ConfigManager(settings) as config:
    # Валидация конфигурации
    errors = config.validate_all()
    if errors:
        print("Configuration errors:", errors)

    # Экспорт конфигурации
    config.export_config("backup_config.yaml", format="yaml")

    # Статистика использования
    stats = config.get_stats()
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}")
```

## 📁 Структура модуля

```
utils/config_manager/
├── __init__.py              # Экспорты модуля
├── config_manager.py        # Основной ConfigManager класс
├── config_section.py        # ConfigSection для работы с секциями
├── config_validator.py      # Система валидации
├── config_schema.py         # Схемы конфигурации
├── README.md               # Документация (этот файл)
├── plan.md                 # План реализации
├── meta.md                 # Метаданные модуля
├── errors.md               # Документированные ошибки
├── diagram.mmd             # Архитектурная диаграмма
└── examples.md             # Примеры использования
```

## ⚙️ Конфигурация

### Настройки ConfigManager

```python
@dataclass
class ConfigManagerSettings:
    base_config_path: str = "config/main_config.yaml"
    environment: str = "development"  # development, testing, production
    enable_hot_reload: bool = True
    hot_reload_interval: float = 1.0  # секунды
    enable_validation: bool = True
    enable_caching: bool = True
    config_search_paths: List[str] = [
        "config/",
        "core/*/config/",
        "data/*/config/",
        "inference/*/config/",
        "training/*/config/",
    ]
```

### Структура конфигурационных файлов

```yaml
# config/main_config.yaml (базовая конфигурация)
project:
  name: "3D Cellular Neural Network"
  version: "0.1.0"

lattice:
  dimensions:
    depth: 5
    height: 5
    width: 5

training:
  batch_size: 4
  learning_rate: 0.001

# Environment-specific overrides
development:
  training:
    batch_size: 2

production:
  training:
    batch_size: 32
```

## 🔧 API Reference

### ConfigManager

#### Основные методы

- **`get_config(section, key, default)`** - получение конфигурации
- **`set_config(section, key, value, **kwargs)`\*\* - установка конфигурации
- **`reload_config(section=None)`** - перезагрузка конфигурации
- **`get_section(section_name)`** - получение секции как объекта
- **`validate_all()`** - валидация всех конфигураций
- **`export_config(path, format, section)`** - экспорт конфигурации

#### Статистика и мониторинг

- **`get_stats()`** - статистика использования
- **`shutdown()`** - корректное завершение работы

### ConfigSection

Удобный wrapper для работы с отдельными секциями:

```python
section = config.get_section('training')

# Dot-notation доступ
batch_size = section.get('optimizer.batch_size', 32)
section.set('optimizer.type', 'Adam')

# Dict-like интерфейс
section['learning_rate'] = 0.001
if 'num_epochs' in section:
    print(section['num_epochs'])

# Множественное обновление
section.update({
    'batch_size': 64,
    'optimizer.weight_decay': 0.0001
})
```

### Валидация

#### Создание валидаторов

```python
from utils.config_manager import ConfigValidator, ConfigValidatorBuilder

# Предустановленные валидаторы
lattice_validator = ConfigValidatorBuilder.create_lattice_validator()

# Пользовательский валидатор
validator = ConfigValidator("custom_section")
validator.field("timeout").required().type_check(int).range_check(1, 3600)
validator.field("host").required().type_check(str).regex(r"^[a-zA-Z0-9.-]+$")

# Валидация
errors = validator.validate(config_data)
if errors:
    print("Validation errors:", errors)
```

#### Правила валидации

- **`required()`** - обязательное поле
- **`type_check(type)`** - проверка типа
- **`range_check(min, max)`** - проверка диапазона
- **`choices(list)`** - выбор из списка
- **`regex(pattern)`** - проверка регулярным выражением
- **`custom(func, description)`** - пользовательская валидация

### Схемы конфигурации

```python
from utils.config_manager import ConfigSchema, SchemaBuilder

# Создание схемы
schema = ConfigSchema("my_config", "Описание конфигурации")
schema.int_field("port", min_value=1024, max_value=65535, default=8080)
schema.string_field("host", pattern=r"^[a-zA-Z0-9.-]+$", default="localhost")
schema.bool_field("debug", default=False)

# Валидация по схеме
errors = schema.validate(config_data)

# Применение значений по умолчанию
config_with_defaults = schema.apply_defaults(config_data)
```

## 🎯 Интеграция с проектом

### Замена существующих config loaders

ConfigManager может заменить существующие локальные загрузчики конфигурации:

```python
# Старый способ (в каждом модуле)
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Новый способ (централизованно)
from utils.config_manager import get_global_config_manager

config = get_global_config_manager()
lattice_config = config.get_config('lattice_3d')
```

### Интеграция в main.py

```python
from utils.config_manager import create_config_manager, set_global_config_manager

def main():
    # Инициализация глобального ConfigManager
    config = create_config_manager(
        environment="development",
        enable_hot_reload=True
    )
    set_global_config_manager(config)

    # Использование в приложении
    lattice_config = config.get_config('lattice_3d')
    # ... остальная логика
```

## 📊 Производительность

### Кэширование

- **Умный кэш** - результаты кэшируются в памяти
- **Hot reloading** - минимальные накладные расходы
- **Thread-safe** - безопасная работа в многопоточности

### Метрики

```python
stats = config.get_stats()
print(f"""
ConfigManager Statistics:
- Cache hit rate: {stats['cache_hit_rate']:.2%}
- Cached sections: {stats['cached_sections']}
- Config loads: {stats['config_loads']}
- Hot reloads: {stats['hot_reloads']}
""")
```

## 🛠️ Разработка и тестирование

### Запуск тестов

```bash
python test_config_manager_basic.py
```

### Отладка

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# ConfigManager будет выводить подробную информацию
config = create_config_manager()
```

## 🔄 Migration Guide

### Переход от локальных config loaders

1. **Определите секции** - разбейте конфигурацию на логические секции
2. **Создайте схемы** - определите схемы валидации для каждой секции
3. **Обновите код** - замените локальные загрузчики на ConfigManager
4. **Добавьте валидацию** - используйте встроенную систему валидации

### Пример миграции

```python
# До миграции
class EmbeddingLoader:
    def __init__(self, config_path="config/embedding_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

# После миграции
class EmbeddingLoader:
    def __init__(self, config_manager=None):
        self.config = config_manager or get_global_config_manager()
        self.embedding_config = self.config.get_section('embedding_loader')
```

## 🚨 Best Practices

1. **Используйте секции** - группируйте связанные настройки
2. **Добавляйте валидацию** - определяйте схемы для критичных настроек
3. **Environment overrides** - используйте для различных окружений
4. **Dot-notation** - для доступа к вложенным настройкам
5. **Context manager** - для корректного завершения работы
6. **Глобальный экземпляр** - для упрощения доступа из модулей

## 📚 См. также

- **`plan.md`** - детальный план реализации
- **`meta.md`** - метаданные и зависимости модуля
- **`examples.md`** - дополнительные примеры использования
- **`diagram.mmd`** - архитектурная диаграмма модуля
