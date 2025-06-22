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
- moe: Mixture of Experts архитектура (Phase 4.5)
- cnf: Continuous Normalizing Flows
- utils: Утилиты и логирование
"""

from .config import ProjectConfig, get_project_config, set_project_config

# Основные компоненты
try:
    from .core import BaseCell, NCACell, GNNCell, CellFactory

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

# Deprecated компоненты
try:
    from .core import GMLPCell

    _GMLP_AVAILABLE = True
except ImportError:
    _GMLP_AVAILABLE = False

__version__ = "0.1.0"

# Основные экспорты
__all__ = [
    # Configuration
    "ProjectConfig",
    "get_project_config",
    "set_project_config",
]

# Добавляем core компоненты если доступны
if _CORE_AVAILABLE:
    __all__.extend(
        [
            "BaseCell",
            "NCACell",
            "GNNCell",
            "CellFactory",
        ]
    )

# Добавляем deprecated компоненты если доступны
if _GMLP_AVAILABLE:
    __all__.append("GMLPCell")  # DEPRECATED - используйте GNNCell
