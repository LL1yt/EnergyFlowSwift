"""
Clean 3D Cellular Neural Network - Core Module
==============================================

НОВАЯ АРХИТЕКТУРА: MoE (Mixture of Experts)
Основные компоненты для MoE архитектуры:
- MoE: Mixture of Experts - ОСНОВНАЯ АРХИТЕКТУРА (в отдельном модуле)
- Lattice: 3D решетка с полной функциональностью
- GNNCell: единственная активная клетка (для Functional Expert)
- LightweightCNF: continuous dynamics (для Distant Expert)

DEPRECATED:
- NCA Cell: заменен на GatingNetwork в MoE
- Hybrid архитектуры: заменены на MoE
"""

# Основные компоненты
from .cells import BaseCell, GNNCell, CellFactory
from .lattice import Lattice3D, create_lattice  # NeighborTopology


# Основные экспорты - MoE архитектура
__all__ = [
    "BaseCell",
    "GNNCell",  # Единственная активная клетка (для Functional Expert)
    "CellFactory",
    "Lattice3D",
    "create_lattice",
]
