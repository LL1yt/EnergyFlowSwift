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
# СТРОГАЯ ПРОВЕРКА - БЕЗ FALLBACK
from .core import BaseCell, VectorizedGNNCell, create_cell

_CORE_AVAILABLE = True

# Deprecated компоненты больше не поддерживаются
# GMLPCell удален - используйте VectorizedGNNCell
_GMLP_AVAILABLE = False

__version__ = "0.1.0"

# Основные экспорты
__all__ = [
    # Configuration
    "ProjectConfig",
    "get_project_config",
    "set_project_config",
]

# Добавляем core компоненты
__all__.extend(
    [
        "BaseCell",
        # "NCACell",  # DEPRECATED - удалено
        "VectorizedGNNCell",  # Векторизованная GNN клетка
        "create_cell",  # Фабрика клеток
    ]
)

# Deprecated компоненты удалены
# GMLPCell больше не поддерживается - используйте VectorizedGNNCell
