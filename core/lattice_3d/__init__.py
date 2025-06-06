"""
Модуль Lattice 3D - Трехмерная Решетка Клеток

Экспорт основных компонентов для работы с 3D решеткой "умных клеток".
"""

from .main import (
    # Основные классы
    Lattice3D,
    LatticeConfig,
    Position3D,
    NeighborTopology,
    IOPointPlacer,  # НОВОЕ
    
    # Енумы
    BoundaryCondition,
    Face,
    PlacementStrategy,  # НОВОЕ
    
    # Функции загрузки и создания
    load_lattice_config,
    create_lattice_from_config,
    validate_lattice_config,
    
    # Типы
    Coordinates3D,
    Dimensions3D,
)

__all__ = [
    # Основные классы
    'Lattice3D',
    'LatticeConfig', 
    'Position3D',
    'NeighborTopology',
    'IOPointPlacer',  # НОВОЕ
    
    # Енумы
    'BoundaryCondition',
    'Face',
    'PlacementStrategy',  # НОВОЕ
    
    # Функции
    'load_lattice_config',
    'create_lattice_from_config',
    'validate_lattice_config',
    
    # Типы
    'Coordinates3D',
    'Dimensions3D',
] 