"""
Clean 3D Cellular Neural Network - Cells Module
===============================================

Клетки (нейроны) для 3D решетки.
Перенос из Legacy core/cell_prototype/architectures/ с оптимизацией.
"""

from .base_cell import BaseCell, CellFactory
from .nca_cell import NCACell
from .gnn_cell import GNNCell
from .gmlp_cell import GMLPCell  # Legacy совместимость
from .hybrid_cell import HybridCell
from .nca_modulator import NCAModulator
from .modulated_gnn import ModulatedGNNCell
from .hybrid_cell_v2 import HybridCellV2

__all__ = [
    "BaseCell",
    "NCACell",
    "GNNCell",
    "GMLPCell",  # DEPRECATED
    "HybridCell",
    "NCAModulator",
    "ModulatedGNNCell",
    "HybridCellV2",
    "CellFactory",
]
