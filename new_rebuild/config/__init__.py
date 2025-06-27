"""
Clean 3D Cellular Neural Network - Configuration Module
=======================================================

Централизованная система конфигурации для clean архитектуры.
"""

# Новая упрощенная конфигурация является основной
from .simple_config import (
    SimpleProjectConfig,
    get_project_config,
    set_project_config,
    reset_project_config,
    create_simple_config,
    create_research_config_simple,
    create_production_config_simple,
)

# Экспорт SimpleProjectConfig под основным именем ProjectConfig для совместимости
ProjectConfig = SimpleProjectConfig

from .config_components import (
    LatticeSettings,
    ModelSettings,
    TrainingSettings,
    CNFSettings,
    EulerSettings,
    CacheSettings,
    SpatialSettings,
    VectorizedSettings,
    DeviceSettings,
    LoggingSettings,
    MemorySettings,
    ExperimentSettings,
    PerformanceSettings,
    ValidationSettings,
)

# Legacy configuration (больше не используется напрямую)
# from .project_config_legacy import * # Оставляем для возможной отладки

__all__ = [
    # Новая упрощенная конфигурация
    "ProjectConfig",  # Основной класс конфигурации
    "SimpleProjectConfig",  # Явный экспорт нового класса
    "get_project_config",
    "set_project_config",
    "reset_project_config",
    "create_simple_config",
    "create_research_config_simple",
    "create_production_config_simple",
    # Компоненты конфигурации
    "LatticeSettings",
    "ModelSettings",
    "TrainingSettings",
    "CNFSettings",
    "EulerSettings",
    "CacheSettings",
    "SpatialSettings",
    "VectorizedSettings",
    "DeviceSettings",
    "LoggingSettings",
    "MemorySettings",
    "ExperimentSettings",
    "PerformanceSettings",
    "ValidationSettings",
]
