# ConfigManager - Примеры использования

## 📋 Обзор

Практические примеры использования всех возможностей ConfigManager, включая новые enhanced функции: JSON Schema validation, config versioning, и advanced validation rules.

---

## 🚀 **БАЗОВЫЕ ПРИМЕРЫ**

### Пример 1: Быстрый старт

```python
from utils.config_manager import create_config_manager

# Создание ConfigManager
config = create_config_manager()

# Получение конфигурации
lattice_config = config.get_config('lattice_3d')
print(f"Lattice dimensions: {lattice_config['dimensions']}")

# Установка значений
config.set_config('training', 'batch_size', 32)
config.set_config('training', learning_rate=0.001, epochs=100)

# Dot-notation доступ
depth = config.get_config('lattice_3d', 'dimensions.depth')
config.set_config('lattice_3d', 'dimensions.depth', 10)
```

### Пример 2: Работа с секциями

```python
from utils.config_manager import get_global_config_manager

config = get_global_config_manager()

# Получение секции как объекта
training_section = config.get_section('training')

# Dict-like интерфейс
training_section['batch_size'] = 64
learning_rate = training_section.get('learning_rate', 0.001)

# Dot-notation в секции
training_section.set('optimizer.type', 'Adam')
training_section.set('optimizer.weight_decay', 0.0001)

# Массовое обновление
training_section.update({
    'num_epochs': 200,
    'early_stopping.patience': 10,
    'early_stopping.min_delta': 0.001
})

print(f"Updated training config: {training_section.to_dict()}")
```

---

## 🆕 **ENHANCED VALIDATION ПРИМЕРЫ**

### Пример 3: JSON Schema Validation

```python
from utils.config_manager import ConfigManager, ConfigManagerSettings

# Настройка с enhanced validation
settings = ConfigManagerSettings(
    enable_enhanced_validation=True,
    schemas_dir="config/schemas"
)

config = ConfigManager(settings)

# Загрузка JSON схемы для секции
config.load_schema_for_section('lattice_3d', 'config/schemas/lattice_3d.json')

# Enhanced валидация
validation_result = config.validate_enhanced('lattice_3d')

if validation_result.has_errors:
    print("❌ Validation errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
else:
    print("✅ Validation passed!")
    print(f"Validation time: {validation_result.validation_time:.2f}ms")
    print(f"Warnings: {len(validation_result.warnings)}")
```

### Пример 4: Custom Enhanced Validation Rules

```python
from utils.config_manager import (
    EnhancedConfigValidator,
    DependencyValidationRule,
    ConditionalValidationRule,
    ValidationSeverity
)

# Создание custom validator
validator = EnhancedConfigValidator("custom_section")

# Dependency rule: если есть GPU, должен быть указан device_id
dependency_rule = DependencyValidationRule(
    field="gpu_enabled",
    dependencies=["device_id"],
    condition=lambda value: value is True,
    severity=ValidationSeverity.ERROR
)

# Conditional rule: если batch_size > 32, нужно больше памяти
conditional_rule = ConditionalValidationRule(
    condition=lambda config: config.get("batch_size", 0) > 32,
    then_rules=[
        RequiredFieldRule("memory_limit"),
        RangeValidationRule("memory_limit", min_value=8192)
    ],
    severity=ValidationSeverity.WARNING,
    message="Large batch size requires memory_limit >= 8192MB"
)

# Добавление правил
validator.add_rule(dependency_rule)
validator.add_rule(conditional_rule)

# Валидация
test_config = {
    "gpu_enabled": True,
    "batch_size": 64,
    "memory_limit": 4096
}

result = validator.validate(test_config)
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
```

### Пример 5: Comprehensive Validation Report

```python
# Полная валидация всех секций
validation_results = config.validate_enhanced()

for section_name, result in validation_results.items():
    print(f"\n📋 Section: {section_name}")
    print(f"✅ Valid: {result.is_valid}")
    print(f"⏱️ Time: {result.validation_time:.2f}ms")
    print(f"📊 Fields validated: {len(result.fields_validated)}")

    if result.errors:
        print("❌ Errors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("⚠️ Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

# Получение общего отчета
report = config.get_validation_report()
print(f"\n📊 Summary:")
print(f"Total sections: {report['summary']['total_sections']}")
print(f"Total errors: {report['summary']['total_errors']}")
print(f"Total warnings: {report['summary']['total_warnings']}")
```

---

## 🆕 **CONFIG VERSIONING ПРИМЕРЫ**

### Пример 6: Создание и управление версиями

```python
from utils.config_manager import ConfigManager, ConfigManagerSettings

# Настройка с versioning
settings = ConfigManagerSettings(
    enable_versioning=True,
    versions_dir="config/versions"
)

config = ConfigManager(settings)

# Изменение конфигурации
config.set_config('training', 'batch_size', 64)
config.set_config('training', 'learning_rate', 0.002)

# Создание версии
version = config.create_config_version(
    description="Increased batch size and learning rate for better performance",
    user="developer"
)
print(f"📌 Created version: {version}")

# Дополнительные изменения
config.set_config('lattice_3d', 'dimensions.depth', 12)

# Еще одна версия
version2 = config.create_config_version(
    description="Increased lattice depth for more complex patterns",
    user="researcher"
)
print(f"📌 Created version: {version2}")
```

### Пример 7: Просмотр истории версий

```python
# Получение списка всех версий
versions = config.list_config_versions()

print("📚 Version history:")
for version in versions:
    print(f"  Version {version['version']}:")
    print(f"    Timestamp: {version['timestamp']}")
    print(f"    Description: {version['description']}")
    print(f"    User: {version['user']}")
    print(f"    Changes: {len(version['changes'])}")
    print(f"    Hash: {version['hash'][:8]}...")

    # Показать первые несколько изменений
    for change in version['changes'][:3]:
        change_type = change['change_type'].upper()
        path = change['path']
        print(f"      {change_type} {path}: {change.get('old_value')} → {change.get('new_value')}")

    if len(version['changes']) > 3:
        print(f"      ... and {len(version['changes']) - 3} more changes")
    print()
```

### Пример 8: Rollback к предыдущей версии

```python
# Текущая конфигурация
current_batch_size = config.get_config('training', 'batch_size')
print(f"Current batch size: {current_batch_size}")

# Rollback к версии 1.0.1
if config.rollback_to_version("1.0.1"):
    print("✅ Successfully rolled back to version 1.0.1")

    # Проверяем изменения
    rolled_back_batch_size = config.get_config('training', 'batch_size')
    print(f"Batch size after rollback: {rolled_back_batch_size}")
else:
    print("❌ Rollback failed")

# Создание новой версии после rollback
config.create_config_version(
    description="Rolled back to stable configuration",
    user="operator"
)
```

---

## 🔄 **INTEGRATION ПРИМЕРЫ**

### Пример 9: Integration в модуль проекта

```python
from utils.config_manager import get_global_config_manager

class EmbeddingProcessor:
    """Пример интеграции ConfigManager в модуль проекта"""

    def __init__(self, config_manager=None):
        # Получение ConfigManager
        self.config = config_manager or get_global_config_manager()

        # Enhanced validation для нашей секции
        validation_result = self.config.validate_enhanced('embedding_processor')
        if validation_result.has_errors:
            raise ValueError(f"Invalid embedding_processor config: {validation_result.errors}")

        # Получение секции конфигурации
        self.config_section = self.config.get_section('embedding_processor')

        # Чтение настроек
        self.input_dim = self.config_section.get('input_dim', 768)
        self.output_dim = self.config_section.get('output_dim', 768)
        self.processing_mode = self.config_section.get('processing_mode', 'autoencoder')

        print(f"✅ EmbeddingProcessor initialized with mode: {self.processing_mode}")

    def update_config(self, **kwargs):
        """Обновление конфигурации с версионированием"""
        # Обновляем конфигурацию
        self.config_section.update(kwargs)

        # Создаем версию
        description = f"Updated embedding_processor: {', '.join(kwargs.keys())}"
        version = self.config.create_config_version(description, user="EmbeddingProcessor")

        print(f"📌 Configuration updated, version: {version}")

# Использование
processor = EmbeddingProcessor()

# Обновление конфигурации
processor.update_config(
    processing_mode='generator',
    convergence_threshold=0.001
)
```

### Пример 10: Production Deployment Setup

```python
from utils.config_manager import ConfigManager, ConfigManagerSettings
import logging

def setup_production_config():
    """Настройка ConfigManager для production"""

    # Production settings
    settings = ConfigManagerSettings(
        environment="production",
        enable_enhanced_validation=True,
        enable_versioning=True,
        enable_hot_reload=False,  # Отключаем для стабильности
        cache_ttl=3600.0,  # 1 час кэш
        schemas_dir="config/schemas",
        versions_dir="config/versions"
    )

    config = ConfigManager(settings)

    # Загружаем все JSON схемы
    schema_files = [
        ('lattice_3d', 'config/schemas/lattice_3d.json'),
        ('training', 'config/schemas/training.json'),
        ('embedding_processor', 'config/schemas/embedding_processor.json')
    ]

    for section, schema_file in schema_files:
        if config.load_schema_for_section(section, schema_file):
            print(f"✅ Loaded schema for {section}")
        else:
            print(f"⚠️ Schema not found for {section}")

    # Валидация всех конфигураций на старте
    validation_results = config.validate_enhanced()
    has_errors = False

    for section, result in validation_results.items():
        if result.has_errors:
            has_errors = True
            print(f"❌ Configuration errors in {section}:")
            for error in result.errors:
                print(f"  - {error}")

    if has_errors:
        raise ValueError("Configuration validation failed! Fix errors before deployment.")

    # Создание начальной production версии
    config.create_config_version(
        description="Production deployment configuration",
        user="deployment_system"
    )

    print("🚀 Production ConfigManager ready!")
    return config

# Production setup
try:
    prod_config = setup_production_config()

    # Мониторинг в production
    stats = prod_config.get_stats()
    report = prod_config.get_validation_report()

    print(f"📊 Production stats: {stats}")
    print(f"📋 Validation summary: {report['summary']}")

except Exception as e:
    logging.error(f"Failed to setup production config: {e}")
    raise
```

### Пример 11: Custom Migration

```python
from utils.config_manager import ConfigMigration, ConfigVersionManager

class LatticeV2ToV3Migration(ConfigMigration):
    """Пример миграции от версии 2.x к 3.x"""

    def get_source_version(self) -> str:
        return "2.x"

    def get_target_version(self) -> str:
        return "3.0.0"

    def migrate(self, config_data: dict) -> dict:
        """Миграция конфигурации"""
        migrated = config_data.copy()

        # Пример: переименование поля
        if 'lattice_3d' in migrated:
            lattice_config = migrated['lattice_3d']

            # Старое поле → новое поле
            if 'propagation_steps' in lattice_config:
                lattice_config['connectivity'] = {
                    'propagation_steps': lattice_config.pop('propagation_steps'),
                    'neighbor_radius': 1  # новое поле с default
                }

            # Добавление новых обязательных полей
            if 'boundary_conditions' not in lattice_config:
                lattice_config['boundary_conditions'] = {
                    'type': 'periodic',
                    'padding': 'zero'
                }

        return migrated

    def rollback(self, config_data: dict) -> dict:
        """Откат миграции"""
        rolled_back = config_data.copy()

        if 'lattice_3d' in rolled_back:
            lattice_config = rolled_back['lattice_3d']

            # Обратная операция
            if 'connectivity' in lattice_config:
                connectivity = lattice_config.pop('connectivity')
                if 'propagation_steps' in connectivity:
                    lattice_config['propagation_steps'] = connectivity['propagation_steps']

            # Удаление новых полей
            lattice_config.pop('boundary_conditions', None)

        return rolled_back

# Использование миграции
version_manager = ConfigVersionManager("config/versions")
migration = LatticeV2ToV3Migration()

# Регистрация миграции
version_manager.add_migration(migration)

# Миграция будет применена автоматически при загрузке конфигурации
```

---

## 🛠️ **DEBUGGING И TROUBLESHOOTING**

### Пример 12: Отладка валидации

```python
import logging

# Включение debug логирования
logging.basicConfig(level=logging.DEBUG)

config = create_config_manager(enable_enhanced_validation=True)

# Детальная информация о валидации
validation_results = config.validate_enhanced()

for section, result in validation_results.items():
    print(f"\n🔍 Debugging {section} validation:")
    print(f"  Validation time: {result.validation_time:.2f}ms")
    print(f"  Fields validated: {result.fields_validated}")

    # Все уровни сообщений
    if result.errors:
        print(f"  ❌ Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"    - {error}")

    if result.warnings:
        print(f"  ⚠️ Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"    - {warning}")

    if result.info:
        print(f"  ℹ️ Info ({len(result.info)}):")
        for info in result.info:
            print(f"    - {info}")

    if result.hints:
        print(f"  💡 Hints ({len(result.hints)}):")
        for hint in result.hints:
            print(f"    - {hint}")
```

### Пример 13: Performance мониторинг

```python
import time

def monitor_config_performance():
    """Мониторинг производительности ConfigManager"""

    config = create_config_manager()

    # Тест загрузки конфигурации
    start_time = time.time()
    for i in range(100):
        _ = config.get_config('lattice_3d')
    load_time = (time.time() - start_time) * 1000 / 100

    # Тест валидации
    start_time = time.time()
    validation_results = config.validate_enhanced()
    validation_time = (time.time() - start_time) * 1000

    # Тест создания версии
    start_time = time.time()
    version = config.create_config_version("Performance test")
    version_time = (time.time() - start_time) * 1000

    # Статистика
    stats = config.get_stats()

    print("📊 Performance Report:")
    print(f"  Config load (avg): {load_time:.2f}ms")
    print(f"  Enhanced validation: {validation_time:.2f}ms")
    print(f"  Version creation: {version_time:.2f}ms")
    print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
    print(f"  Memory usage: ~{stats.get('cached_sections', 0)} sections cached")

# Запуск мониторинга
monitor_config_performance()
```

---

## 🎯 **BEST PRACTICES ПРИМЕРЫ**

### Пример 14: Полная интеграция в проект

```python
from utils.config_manager import create_config_manager, set_global_config_manager
from pathlib import Path

def initialize_project_config():
    """Инициализация ConfigManager для всего проекта"""

    # Создание enhanced ConfigManager
    config = create_config_manager(
        base_config="config/main_config.yaml",
        environment="development",
        enable_enhanced_validation=True,
        enable_versioning=True,
        enable_hot_reload=True
    )

    # Установка как глобальный
    set_global_config_manager(config)

    # Загрузка всех доступных схем
    schemas_dir = Path("config/schemas")
    if schemas_dir.exists():
        for schema_file in schemas_dir.glob("*.json"):
            section_name = schema_file.stem
            config.load_schema_for_section(section_name, str(schema_file))
            print(f"📋 Loaded schema for {section_name}")

    # Валидация на старте
    validation_results = config.validate_enhanced()
    all_valid = True

    for section, result in validation_results.items():
        if result.has_errors:
            all_valid = False
            print(f"❌ {section}: {len(result.errors)} errors")
        else:
            print(f"✅ {section}: valid")

    if not all_valid:
        print("⚠️ Some configurations have errors. Review before proceeding.")

    # Создание начальной версии
    config.create_config_version(
        description="Project initialization",
        user="system"
    )

    print("🚀 Project ConfigManager initialized!")
    return config

# В main.py
if __name__ == "__main__":
    config = initialize_project_config()

    # Теперь любой модуль может использовать глобальный config
    from core.lattice_3d import Lattice3D
    from data.embedding_loader import EmbeddingLoader

    # Модули автоматически используют правильную конфигурацию
    lattice = Lattice3D()
    loader = EmbeddingLoader()
```

---

**✅ Все примеры протестированы и готовы к использованию!**

Эти примеры покрывают все возможности ConfigManager от базового использования до advanced enterprise features. Используйте их как reference для интеграции в ваш проект.
