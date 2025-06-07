# ConfigManager - Централизованное управление конфигурацией

## 📋 Обзор

ConfigManager обеспечивает **enterprise-level** централизованное управление конфигурацией для проекта 3D Cellular Neural Network. Поддерживает модульную архитектуру, расширенную валидацию, версионирование конфигураций и hot reloading.

## 🚀 Основные возможности

### **🔧 Базовые возможности**

- **Модульная архитектура** - автоматическое обнаружение конфигураций модулей
- **Иерархическое наследование** - base config + module configs + environment overrides
- **Hot reloading** - автоматическая перезагрузка при изменении файлов
- **Thread-safe** - безопасная работа в многопоточной среде
- **Dot-notation** - удобный доступ к вложенным настройкам
- **Environment overrides** - различные настройки для dev/test/prod

### **🆕 Enterprise-level возможности**

- **🔍 Enhanced Validation** - многоуровневая валидация с severity levels (ERROR/WARNING/INFO/HINT)
- **📋 JSON Schema Validation** - валидация через JSON Schema с автоматической загрузкой
- **📚 Config Versioning** - полное версионирование конфигураций с change tracking
- **🔄 Rollback Support** - откат к предыдущим версиям конфигураций
- **📊 Migration System** - система миграций между версиями конфигураций
- **📈 Comprehensive Reporting** - детальная отчетность о состоянии конфигураций

## 📦 Установка и использование

### Быстрый старт

```python
from utils.config_manager import ConfigManager, create_config_manager

# Создание ConfigManager с enhanced возможностями
config = create_config_manager(
    enable_enhanced_validation=True,
    enable_versioning=True
)

# Получение конфигурации
lattice_config = config.get_config('lattice')
depth = config.get_config('lattice_3d', 'dimensions.depth')

# Установка значений
config.set_config('training', 'batch_size', 32)
config.set_config('training', learning_rate=0.001, num_epochs=100)

# 🆕 Enhanced validation
validation_result = config.validate_enhanced('lattice_3d')
if validation_result.has_errors:
    print("Validation errors:", validation_result.errors)

# 🆕 Create config version
version = config.create_config_version("Updated training settings")
print(f"Created version: {version}")
```

### 🆕 Enhanced Usage - Versioning & Validation

```python
from utils.config_manager import ConfigManager, ConfigManagerSettings

# 🆕 Enhanced настройка ConfigManager
settings = ConfigManagerSettings(
    base_config_path="config/main_config.yaml",
    environment="production",
    enable_hot_reload=True,
    enable_validation=True,
    # Enhanced features
    enable_enhanced_validation=True,
    enable_versioning=True,
    versions_dir="config/versions",
    schemas_dir="config/schemas"
)

with ConfigManager(settings) as config:
    # 🆕 Enhanced validation с детальными результатами
    validation_results = config.validate_enhanced()
    for section, result in validation_results.items():
        print(f"Section {section}:")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Validation time: {result.validation_time:.2f}ms")

    # 🆕 Config versioning
    config.set_config('training', 'batch_size', 64)
    version = config.create_config_version("Increased batch size", user="developer")

    # 🆕 Version history
    versions = config.list_config_versions()
    for v in versions:
        print(f"Version {v['version']}: {v['description']} ({len(v['changes'])} changes)")

    # 🆕 Rollback
    if config.rollback_to_version("1.0.1"):
        print("Successfully rolled back to version 1.0.1")

    # 🆕 Comprehensive validation report
    report = config.get_validation_report()
    print(f"Total errors: {report['summary']['total_errors']}")
    print(f"Total warnings: {report['summary']['total_warnings']}")
```

## 📁 Структура модуля

```
utils/config_manager/
├── __init__.py              # Экспорты модуля
├── config_manager.py        # Основной ConfigManager класс
├── config_section.py        # ConfigSection для работы с секциями
├── config_validator.py      # Базовая система валидации
├── config_schema.py         # Схемы конфигурации
├── 🆕 enhanced_validator.py  # Enhanced validation с severity levels
├── 🆕 config_versioning.py   # Система версионирования конфигураций
├── README.md               # Документация (этот файл)
├── plan.md                 # План реализации
├── meta.md                 # Метаданные модуля
├── errors.md               # Документированные ошибки
├── diagram.mmd             # Архитектурная диаграмма
└── examples.md             # Примеры использования
```

## ⚙️ Конфигурация

### 🆕 Enhanced настройки ConfigManager

```python
@dataclass
class ConfigManagerSettings:
    # Базовые настройки
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

    # 🆕 Enhanced возможности
    enable_versioning: bool = True
    versions_dir: str = "config/versions"
    schemas_dir: str = "config/schemas"
    enable_enhanced_validation: bool = True
    enable_auto_migration: bool = True
    config_version: str = "1.0.0"
```

### 🆕 JSON Schema файлы

```yaml
# config/schemas/lattice_3d.json
{
  "type": "object",
  "properties":
    {
      "dimensions":
        {
          "type": "object",
          "properties":
            {
              "depth": { "type": "integer", "minimum": 1, "maximum": 100 },
              "height": { "type": "integer", "minimum": 1, "maximum": 100 },
              "width": { "type": "integer", "minimum": 1, "maximum": 100 },
            },
          "required": ["depth", "height", "width"],
        },
      "connectivity":
        {
          "type": "object",
          "properties":
            {
              "propagation_steps":
                { "type": "integer", "minimum": 1, "maximum": 1000 },
            },
        },
    },
  "required": ["dimensions", "connectivity"],
}
```

## 🔧 API Reference

### ConfigManager

#### 🆕 Enhanced методы

- **`validate_enhanced(section=None)`** - расширенная валидация с severity levels
- **`create_config_version(description, user)`** - создание версии конфигурации
- **`rollback_to_version(version)`** - откат к версии
- **`list_config_versions()`** - список всех версий
- **`get_validation_report()`** - comprehensive отчет о валидации
- **`load_schema_for_section(section, schema_file)`** - загрузка JSON Schema

#### Базовые методы

- **`get_config(section, key, default)`** - получение конфигурации
- **`set_config(section, key, value, **kwargs)`\*\* - установка конфигурации
- **`reload_config(section=None)`** - перезагрузка конфигурации
- **`get_section(section_name)`** - получение секции как объекта
- **`validate_all()`** - базовая валидация всех конфигураций
- **`export_config(path, format, section)`** - экспорт конфигурации

#### Статистика и мониторинг

- **`get_stats()`** - статистика использования
- **`shutdown()`** - корректное завершение работы

### 🆕 Enhanced Validation

#### ValidationResult класс

```python
@dataclass
class ValidationResult:
    section: str
    is_valid: bool
    errors: List[str]          # SEVERITY.ERROR
    warnings: List[str]        # SEVERITY.WARNING
    hints: List[str]          # SEVERITY.HINT
    info: List[str]           # SEVERITY.INFO
    validation_time: float    # миллисекунды
    fields_validated: List[str]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        # Полная информация в словаре
```

#### Enhanced Validators

```python
from utils.config_manager import EnhancedConfigValidator, ValidationSeverity

# Создание enhanced validator
validator = EnhancedConfigValidator("lattice_3d")

# Добавление правил
validator.add_rule(SchemaValidationRule("config/schemas/lattice_3d.json"))
validator.add_rule(DependencyValidationRule("dimensions.depth", ["connectivity.propagation_steps"]))
validator.add_rule(ConditionalValidationRule(
    condition=lambda cfg: cfg.get("mode") == "advanced",
    then_rules=[RequiredFieldRule("advanced_settings")]
))

# Валидация
result = validator.validate(config_data)
print(f"Validation completed in {result.validation_time:.2f}ms")
```

### 🆕 Config Versioning

#### ConfigVersionManager

```python
from utils.config_manager import ConfigVersionManager, ChangeType

# Создание версии
version = version_manager.create_version(
    config_data=current_config,
    description="Updated lattice settings",
    user="developer"
)

# Просмотр изменений
changes = version_manager.get_changes_since_version("1.0.0")
for change in changes:
    print(f"{change.change_type}: {change.path} = {change.new_value}")

# Rollback
config_data = version_manager.rollback_to_version("1.0.1")
```

#### Change Tracking

```python
@dataclass
class ConfigChange:
    path: str                    # "training.batch_size"
    change_type: ChangeType      # ADDED, MODIFIED, DELETED, RENAMED
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = None
    user: str = None
    description: str = None
```

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

## 🎯 Интеграция с проектом

### 🆕 Enhanced интеграция в main.py

```python
from utils.config_manager import create_config_manager, set_global_config_manager

def main():
    # 🆕 Инициализация enhanced ConfigManager
    config = create_config_manager(
        environment="development",
        enable_hot_reload=True,
        enable_enhanced_validation=True,
        enable_versioning=True
    )
    set_global_config_manager(config)

    # 🆕 Загрузка JSON схем
    config.load_schema_for_section('lattice_3d')
    config.load_schema_for_section('training')

    # 🆕 Валидация на старте
    validation_results = config.validate_enhanced()
    for section, result in validation_results.items():
        if result.has_errors:
            print(f"❌ Configuration errors in {section}:")
            for error in result.errors:
                print(f"  - {error}")
            return False

    # 🆕 Создание начальной версии
    config.create_config_version("Application startup", user="system")

    # Использование в приложении
    lattice_config = config.get_config('lattice_3d')
    # ... остальная логика
```

### Замена существующих config loaders

```python
# Старый способ (в каждом модуле)
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 🆕 Новый способ (централизованно с validation)
from utils.config_manager import get_global_config_manager

config = get_global_config_manager()
lattice_config = config.get_config('lattice_3d')

# 🆕 С валидацией
validation_result = config.validate_enhanced('lattice_3d')
if validation_result.has_errors:
    raise ConfigurationError(f"Invalid lattice_3d config: {validation_result.errors}")
```

## 📊 Производительность

### Кэширование

- **Умный кэш** - результаты кэшируются в памяти
- **Hot reloading** - минимальные накладные расходы
- **Thread-safe** - безопасная работа в многопоточности
- **🆕 Schema caching** - JSON схемы кэшируются для повышения производительности
- **🆕 Validation caching** - результаты валидации кэшируются

### 🆕 Enhanced метрики

```python
stats = config.get_stats()
print(f"""
ConfigManager Enhanced Statistics:
- Cache hit rate: {stats['cache_hit_rate']:.2%}
- Cached sections: {stats['cached_sections']}
- Config loads: {stats['config_loads']}
- Hot reloads: {stats['hot_reloads']}
- Enhanced validations: {stats['enhanced_validations']}
- Schema loads: {stats['schema_loads']}
- Config versions: {stats['config_versions']}
""")

# 🆕 Validation report
report = config.get_validation_report()
print(f"""
Validation Report:
- Total sections: {report['summary']['total_sections']}
- Total errors: {report['summary']['total_errors']}
- Total warnings: {report['summary']['total_warnings']}
- Enhanced validators: {report['summary']['enhanced_validators']}
""")
```

## 🛠️ Разработка и тестирование

### Запуск тестов

```bash
# Базовые тесты
python test_config_manager_basic.py

# 🆕 Enhanced тесты
python demos/demo_enhanced_config_manager.py
```

### 🆕 Создание JSON Schema

```python
# Для создания schema для новой секции
from utils.config_manager import create_json_schema

schema = create_json_schema("my_section", {
    "timeout": {"type": "integer", "minimum": 1, "maximum": 3600},
    "host": {"type": "string", "pattern": "^[a-zA-Z0-9.-]+$"},
    "debug": {"type": "boolean", "default": False}
})

# Сохранение схемы
with open("config/schemas/my_section.json", "w") as f:
    json.dump(schema, f, indent=2)
```

## 🚨 Best Practices

### 🆕 Enhanced Best Practices

1. **Используйте JSON Schema** - создавайте схемы для всех критичных секций
2. **Версионируйте изменения** - создавайте версии при значимых изменениях
3. **Мониторьте validation** - регулярно проверяйте validation reports
4. **Используйте rollback** - не бойтесь откатываться при проблемах
5. **Документируйте изменения** - добавляйте описания к версиям

### Традиционные Best Practices

1. **Используйте секции** - группируйте связанные настройки
2. **Добавляйте валидацию** - определяйте схемы для критичных настроек
3. **Environment overrides** - используйте для различных окружений
4. **Dot-notation** - для доступа к вложенным настройкам
5. **Context manager** - для корректного завершения работы
6. **Глобальный экземпляр** - для упрощения доступа из модулей

## 🔄 Migration Guide

### 🆕 Переход на Enhanced ConfigManager

1. **Включите enhanced features** в settings
2. **Создайте JSON schemas** для ваших секций
3. **Добавьте версионирование** в критичные моменты
4. **Настройте validation monitoring**
5. **Используйте rollback** при проблемах

### Пример enhanced миграции

```python
# До enhanced версии
class EmbeddingLoader:
    def __init__(self, config_path="config/embedding_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

# 🆕 После enhanced версии
class EmbeddingLoader:
    def __init__(self, config_manager=None):
        self.config = config_manager or get_global_config_manager()

        # Enhanced validation
        validation_result = self.config.validate_enhanced('embedding_loader')
        if validation_result.has_errors:
            raise ConfigurationError(f"Invalid config: {validation_result.errors}")

        self.embedding_config = self.config.get_section('embedding_loader')

        # Create version when config changes
        if self.embedding_config.get('auto_version', True):
            self.config.create_config_version("EmbeddingLoader initialized")
```

## 📚 См. также

- **`plan.md`** - детальный план реализации с отметками о завершении
- **`meta.md`** - метаданные и зависимости модуля (обновлено)
- **`examples.md`** - дополнительные примеры использования enhanced features
- **`diagram.mmd`** - архитектурная диаграмма модуля (обновлено)
- **`errors.md`** - документированные ошибки и их решения

## 🎉 Статус модуля

**✅ PRODUCTION READY** - Все enhanced возможности полностью реализованы и протестированы!

- ✅ Schema Validation (JSON Schema)
- ✅ Enhanced Validation (Severity levels)
- ✅ Config Versioning & Change Tracking
- ✅ Rollback Support
- ✅ Migration System
- ✅ Comprehensive Reporting
- ✅ Полная интеграция с существующим API
- ✅ Backward compatibility

**🚀 Готов к использованию во всех модулях проекта 3D Cellular Neural Network!**
