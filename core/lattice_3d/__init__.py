"""
Модуль 3D Решетки (3D Lattice)
===============================

Этот пакет содержит все компоненты, необходимые для создания,
конфигурирования и симуляции трехмерной клеточной нейронной сети.

Основные экспортируемые компоненты:
- `Lattice3D`: Главный класс, управляющий решеткой.
- `LatticeConfig`: Датакласс для конфигурации решетки.
- `create_lattice_from_config`: Фабричная функция для создания Lattice3D.
- `Face`, `BoundaryCondition`, `PlacementStrategy`: Основные перечисления.
"""

# Импортируем основные классы и функции, чтобы они были доступны
# на уровне пакета (from core.lattice_3d import ...)

from .config import LatticeConfig, load_lattice_config
from .lattice import Lattice3D, create_lattice_from_config
from .topology import NeighborTopology
from .io import IOPointPlacer
from .position import Position3D
from .enums import Face, BoundaryCondition, PlacementStrategy
from .spatial_hashing import SpatialHashGrid, MortonEncoder

# Определяем, что будет импортировано при `from . import *`
__all__ = [
    # Из lattice.py
    "Lattice3D",
    "create_lattice_from_config",
    # Из config.py
    "LatticeConfig",
    "load_lattice_config",
    # Из topology.py
    "NeighborTopology",
    # Из io.py
    "IOPointPlacer",
    # Из position.py
    "Position3D",
    # Из enums.py
    "Face",
    "BoundaryCondition",
    "PlacementStrategy",
    # Из spatial_hashing.py
    "SpatialHashGrid",
    "MortonEncoder",
]
