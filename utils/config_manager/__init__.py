"""
Централизованный Config Manager для 3D Cellular Neural Network

Основные возможности:
- Иерархическая загрузка конфигураций (main + модульные)
- Валидация и мерж конфигураций
- Environment-specific overrides
- Hot reloading конфигураций
- Кэширование и оптимизация
"""

from .config_manager import (
    ConfigManager,
    ConfigManagerSettings,
    create_config_manager,
    get_global_config_manager,
    set_global_config_manager,
)
from .config_section import ConfigSection, TypedConfigSection
from .config_validator import (
    ConfigValidator,
    ConfigValidatorBuilder,
    DEFAULT_VALIDATORS,
)
from .config_schema import (
    ConfigSchema,
    FieldSchema,
    FieldType,
    SchemaBuilder,
    DEFAULT_SCHEMAS,
)

# 🆕 Новые модули для версионирования и enhanced validation
from .config_versioning import (
    ConfigVersionManager,
    ConfigChange,
    ConfigVersion,
    ConfigMigration,
    ChangeType,
    DEFAULT_MIGRATIONS,
)
from .enhanced_validator import (
    EnhancedConfigValidator,
    EnhancedFieldValidator,
    ValidationResult,
    ValidationSeverity,
    SchemaManager,
    SchemaValidationRule,
    DependencyValidationRule,
    ConditionalValidationRule,
    StructureValidationRule,
    ENHANCED_VALIDATORS,
)

# Dynamic Config
from .dynamic_config import DynamicConfigManager

__all__ = [
    # Основные классы
    "ConfigManager",
    "ConfigManagerSettings",
    "ConfigSection",
    "TypedConfigSection",
    "ConfigValidator",
    "ConfigValidatorBuilder",
    "ConfigSchema",
    "FieldSchema",
    "FieldType",
    "SchemaBuilder",
    # 🆕 Версионирование
    "ConfigVersionManager",
    "ConfigChange",
    "ConfigVersion",
    "ConfigMigration",
    "ChangeType",
    # 🆕 Enhanced Validation
    "EnhancedConfigValidator",
    "EnhancedFieldValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SchemaManager",
    "SchemaValidationRule",
    "DependencyValidationRule",
    "ConditionalValidationRule",
    "StructureValidationRule",
    # Helper функции
    "create_config_manager",
    "get_global_config_manager",
    "set_global_config_manager",
    # Предустановленные валидаторы и схемы
    "DEFAULT_VALIDATORS",
    "DEFAULT_SCHEMAS",
    "DEFAULT_MIGRATIONS",
    "ENHANCED_VALIDATORS",
    # Dynamic Config
    "DynamicConfigManager",
]
