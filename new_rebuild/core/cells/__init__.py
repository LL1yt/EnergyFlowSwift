"""
Clean 3D Cellular Neural Network - Cells Module
===============================================

НОВАЯ АРХИТЕКТУРА: MoE (Mixture of Experts)
Клетки заменены на экспертов в core/moe/

СТАТУС КОМПОНЕНТОВ:
- GNNCell: АКТИВНЫЙ (используется в Functional Expert)
- NCACell: DEPRECATED (заменен на GatingNetwork в MoE)
- HybridCell*, GMLPCell: DEPRECATED (заменены на MoE)
- MoE архитектура: в отдельном модуле core/moe/
"""

from .base_cell import BaseCell, CellFactory
from .gnn_cell import GNNCell

# DEPRECATED компоненты (для обратной совместимости)
"""
try:
    from .nca_cell import NCACell  # DEPRECATED: заменен на GatingNetwork

    _NCA_AVAILABLE = True
except ImportError:
    _NCA_AVAILABLE = False

try:
    from .gmlp_cell import GMLPCell  # DEPRECATED: заменен на GNN

    _GMLP_AVAILABLE = True
except ImportError:
    _GMLP_AVAILABLE = False
"""
# Основные активные компоненты
__all__ = [
    "BaseCell",
    "GNNCell",  # Единственная активная клетка (для Functional Expert)
    "CellFactory",
]
