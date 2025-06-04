"""
Основные компоненты 3D Клеточной Нейронной Сети

Этот пакет содержит базовые компоненты для построения и работы
с 3D клеточной нейронной сетью.

Модули:
    cell_prototype: Прототип "умной клетки" ✅ ГОТОВ
    lattice_3d: 3D решетка клеток 📋 В РАЗРАБОТКЕ
    signal_propagation: Распространение сигналов (планируется)
"""

# Импорт готовых модулей
from .cell_prototype import (
    CellPrototype,
    create_cell_from_config,
)

from .lattice_3d import (
    Lattice3D,
    LatticeConfig,
    Position3D,
    NeighborTopology,
    BoundaryCondition,
    Face,
    load_lattice_config,
    create_lattice_from_config,
    validate_lattice_config,
    Coordinates3D,
    Dimensions3D,
)

__all__ = [
    # Cell Prototype модуль
    'CellPrototype',
    'create_cell_from_config',
    
    # Lattice 3D модуль
    'Lattice3D',
    'LatticeConfig',
    'Position3D', 
    'NeighborTopology',
    'BoundaryCondition',
    'Face',
    'load_lattice_config',
    'create_lattice_from_config',
    'validate_lattice_config',
    'Coordinates3D',
    'Dimensions3D',
]

__version__ = '0.1.0' 