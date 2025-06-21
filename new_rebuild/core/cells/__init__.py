"""
Clean 3D Cellular Neural Network - Cells Module
===============================================

Клетки (нейроны) для 3D решетки.
Перенос из Legacy core/cell_prototype/architectures/ с оптимизацией.
"""

from .base_cell import BaseCell, CellFactory
from .nca_cell import NCACell
from .gmlp_cell import GMLPCell

__all__ = [
    "BaseCell",
    "NCACell",
    "GMLPCell",
    "CellFactory",
]
