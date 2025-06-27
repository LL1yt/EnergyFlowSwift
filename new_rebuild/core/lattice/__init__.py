"""
Модуль 3D Решетки (3D Lattice)
===============================

Этот пакет содержит все компоненты, необходимые для создания,
конфигурирования и симуляции трехмерной клеточной нейронной сети.

Основные экспортируемые компоненты:
- `Lattice3D`, `create_lattice`: Основная решетка и фабричная функция.
- `NeighborTopology`: Управление соседством клеток.
- `Face`, `BoundaryCondition`, `PlacementStrategy`: Основные перечисления.
- `Position3D`, `Coordinates3D`, `Dimensions3D`: Работа с координатами.
- `MortonEncoder`, `SpatialHashGrid`: Пространственные оптимизации.
- `IOPointPlacer`: Размещение точек ввода/вывода.
- `VectorizedSpatialProcessor`: Высокопроизводительная векторизованная обработка.
"""

# Импортируем основные классы и функции
from .enums import Face, BoundaryCondition, PlacementStrategy, NeighborStrategy
from .position import Position3D, Coordinates3D, Dimensions3D

# from .spatial_hashing import SpatialHashGrid, MortonEncoder
# from .spatial_optimization import (
# DEPRECATED: используйте UnifiedSpatialOptimizer
# SpatialOptimizer,
# create_spatial_optimizer,
# estimate_memory_requirements,
# MoESpatialOptimizer,
# create_moe_spatial_optimizer,
# estimate_moe_memory_requirements,
# )

# NEW: Unified Spatial Optimizer - рекомендуется для новых проектов
from .spatial_optimization.unified_spatial_optimizer import (
    UnifiedSpatialOptimizer,
    create_unified_spatial_optimizer,
    OptimizationConfig,
    OptimizationMode,
    estimate_unified_memory_requirements,
)
from .io import IOPointPlacer

# from .topology import NeighborTopology
from .lattice import Lattice3D, create_lattice

# Добавляем импорты для новых GPU spatial компонентов
from .gpu_spatial_hashing import (
    AdaptiveGPUSpatialHash,
    GPUSpatialHashGrid,
    GPUMortonEncoder,
    GPUSpatialHashingStats,
)

# Импорт векторизованного пространственного процессора
try:
    from .vectorized_spatial_processor import VectorizedSpatialProcessor

    VECTORIZED_SPATIAL_AVAILABLE = True
except ImportError:
    VECTORIZED_SPATIAL_AVAILABLE = False


# Фабрика для создания оптимального пространственного процессора
def create_spatial_processor(dimensions, processor_type: str = "auto", **kwargs):
    """
    Фабрика для создания оптимального пространственного процессора

    Args:
        dimensions: Размерности решетки
        processor_type: Тип процессора ('vectorized', 'unified', 'auto')
        **kwargs: Дополнительные параметры

    Returns:
        Экземпляр пространственного процессора
    """
    from ...config import get_project_config

    config = get_project_config()

    # Автоопределение типа
    if processor_type == "auto":
        if VECTORIZED_SPATIAL_AVAILABLE and config.vectorized.enabled:
            processor_type = "vectorized"
        else:
            processor_type = "unified"

    if processor_type == "vectorized":
        if not VECTORIZED_SPATIAL_AVAILABLE:
            raise ImportError("Vectorized Spatial Processor not available")
        return VectorizedSpatialProcessor(dimensions, **kwargs)

    elif processor_type == "unified":
        return create_unified_spatial_optimizer(dimensions, **kwargs)

    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def get_recommended_spatial_processor() -> str:
    """Возвращает рекомендуемый тип пространственного процессора"""
    from ...config import get_project_config

    config = get_project_config()

    if VECTORIZED_SPATIAL_AVAILABLE and config.vectorized.enabled:
        return "vectorized"
    return "unified"


# Определяем, что будет импортировано при `from . import *`
__all__ = [
    # Из lattice.py
    "Lattice3D",
    "create_lattice",
    # Из topology.py
    # "NeighborTopology",
    # Из enums.py
    "Face",
    "BoundaryCondition",
    "PlacementStrategy",
    "NeighborStrategy",
    # Из position.py
    "Position3D",
    "Coordinates3D",
    "Dimensions3D",
    # Из spatial_hashing.py
    # "SpatialHashGrid",
    # "MortonEncoder",
    # Из spatial_optimization.py (DEPRECATED)
    "SpatialOptimizer",
    "create_spatial_optimizer",
    "estimate_memory_requirements",
    "MoESpatialOptimizer",
    "create_moe_spatial_optimizer",
    "estimate_moe_memory_requirements",
    # NEW: Unified Spatial Optimizer
    "UnifiedSpatialOptimizer",
    "create_unified_spatial_optimizer",
    "OptimizationConfig",
    "OptimizationMode",
    "estimate_unified_memory_requirements",
    # Из io.py
    "IOPointPlacer",
    # Фабрики для векторизованных компонентов
    "create_spatial_processor",
    "get_recommended_spatial_processor",
]

# Условный экспорт векторизованных компонентов
if VECTORIZED_SPATIAL_AVAILABLE:
    __all__.append("VectorizedSpatialProcessor")
