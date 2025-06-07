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
    set_global_config_manager
)
from .config_section import ConfigSection, TypedConfigSection
from .config_validator import (
    ConfigValidator, 
    ConfigValidatorBuilder,
    DEFAULT_VALIDATORS
)
from .config_schema import (
    ConfigSchema, 
    FieldSchema, 
    FieldType,
    SchemaBuilder,
    DEFAULT_SCHEMAS
)

__all__ = [
    # Основные классы
    'ConfigManager',
    'ConfigManagerSettings',
    'ConfigSection',
    'TypedConfigSection',
    'ConfigValidator',
    'ConfigValidatorBuilder', 
    'ConfigSchema',
    'FieldSchema',
    'FieldType',
    'SchemaBuilder',
    
    # Helper функции
    'create_config_manager',
    'get_global_config_manager',
    'set_global_config_manager',
    
    # Предустановленные валидаторы и схемы
    'DEFAULT_VALIDATORS',
    'DEFAULT_SCHEMAS',
] 