"""
Clean 3D Cellular Neural Network - Configuration Module
=======================================================

Централизованная система конфигурации для clean архитектуры.
"""

# New simplified configuration (recommended)
from .simple_config import (
    SimpleProjectConfig,
    get_project_config as get_simple_config,
    set_project_config as set_simple_config,
    reset_project_config as reset_simple_config,
    create_simple_config,
    create_research_config_simple,
    create_production_config_simple,
)

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

# Legacy configuration (for backward compatibility)
from .project_config import (
    ProjectConfig,
    LatticeConfig,
    GNNConfig,
    TrainingConfig,
    BiologyConfig,
    NeighborConfig,
    PlasticityConfig,
    CNFConfig,
    SpatialOptimConfig,
    MemoryConfig,
    LoggingConfig,
    DeviceConfig,
    InitConfig,
    ExpertConfig,
    LocalExpertConfig,
    FunctionalExpertConfig,
    DistantExpertConfig,
    GatingConfig,
    EulerSolverConfig,
    AdaptiveChunkerConfig,
    UnifiedSpatialOptimizerConfig,
    Lattice3DConfig,
    get_project_config,
    set_project_config,
    reset_global_config,
)

__all__ = [
    # New simplified config (recommended)
    "SimpleProjectConfig",
    "get_simple_config",
    "set_simple_config", 
    "reset_simple_config",
    "create_simple_config",
    "create_research_config_simple",
    "create_production_config_simple",
    # New config components
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
    # Legacy config (for backward compatibility)
    "ProjectConfig",
    "LatticeConfig",
    "GNNConfig",
    "TrainingConfig",
    "BiologyConfig",
    "NeighborConfig",
    "PlasticityConfig",
    "CNFConfig",
    "SpatialOptimConfig",
    "MemoryConfig",
    "LoggingConfig",
    "DeviceConfig",
    "InitConfig",
    "ExpertConfig",
    "LocalExpertConfig",
    "FunctionalExpertConfig",
    "DistantExpertConfig",
    "GatingConfig",
    "EulerSolverConfig",
    "AdaptiveChunkerConfig",
    "UnifiedSpatialOptimizerConfig",
    "Lattice3DConfig",
    "get_project_config",
    "set_project_config",
    "reset_global_config",
]
