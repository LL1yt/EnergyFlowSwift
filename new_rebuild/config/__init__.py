"""
Clean 3D Cellular Neural Network - Configuration Module
=======================================================

Централизованная система конфигурации для clean архитектуры.
"""

from .project_config import (
    ProjectConfig,
    get_project_config,
    set_project_config,
)

__all__ = [
    "ProjectConfig",
    "get_project_config",
    "set_project_config",
]
