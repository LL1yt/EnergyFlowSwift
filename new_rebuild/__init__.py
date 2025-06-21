"""
Clean 3D Cellular Neural Network Implementation
==============================================

Чистая реализация 3D клеточной нейронной сети с:
- Минимальными вычислениями
- Максимальной эмерджентностью
- Биологической корректностью

Основные компоненты:
- config: Централизованная конфигурация
- core: Базовые компоненты (клетки, решетка)
- lattice: 3D решетка (будет добавлена)
- training: Система обучения (будет добавлена)
"""

from .config import ProjectConfig, get_project_config, set_project_config
from .core import BaseCell, NCACell, GMLPCell, CellFactory

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "ProjectConfig",
    "get_project_config",
    "set_project_config",
    # Core components
    "BaseCell",
    "NCACell",
    "GMLPCell",
    "CellFactory",
]
