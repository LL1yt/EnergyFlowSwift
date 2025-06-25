"""
Clean 3D Cellular Neural Network - Configuration Module
=======================================================

Централизованная система конфигурации для clean архитектуры.
"""

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
    # Main Config
    "ProjectConfig",
    # Modular Configs
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
    # Helper Functions
    "get_project_config",
    "set_project_config",
    "reset_global_config",
]
