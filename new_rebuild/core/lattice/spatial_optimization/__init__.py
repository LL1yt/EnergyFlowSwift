#!/usr/bin/env python3
"""
Spatial Optimization Module
===========================

Модуляризированная система пространственной оптимизации
для эффективной обработки больших 3D решеток.

ОБНОВЛЕНИЕ (28 декабря 2025):
- Все конфигурационные классы перенесены в центральный config
- ChunkInfo и SpatialOptimConfig теперь в project_config.py
"""

# Базовые классы
from .spatial_optimizer import SpatialOptimizer
from .moe_spatial_optimizer import MoESpatialOptimizer

# Вспомогательные компоненты
from .chunker import LatticeChunker
from .memory_manager import MemoryPoolManager
from .hierarchical_index import HierarchicalSpatialIndex
from .parallel_processor import ParallelSpatialProcessor

# Конфигурационные классы (теперь из центрального конфига)
from ....config.project_config import ChunkInfo, create_spatial_config_for_lattice
from ....config.project_config import ProjectConfig


def get_spatial_config_from_project(project_config: ProjectConfig) -> dict:
    """Получить spatial optimization конфигурацию из ProjectConfig"""
    return project_config.get_spatial_optim_config()


# Фабричные функции (из moe_spatial_optimizer.py)
from .moe_spatial_optimizer import (
    create_moe_spatial_optimizer,
    estimate_moe_memory_requirements,
)
from .spatial_optimizer import create_spatial_optimizer, estimate_memory_requirements

# Экспорты для обратной совместимости
__all__ = [
    # Основные классы
    "SpatialOptimizer",
    "MoESpatialOptimizer",
    # Вспомогательные компоненты
    "LatticeChunker",
    "MemoryPoolManager",
    "HierarchicalSpatialIndex",
    "ParallelSpatialProcessor",
    # Конфигурационные классы (из центрального конфига)
    "ChunkInfo",
    "create_spatial_config_for_lattice",
    "get_spatial_config_from_project",
    # Фабричные функции
    "create_moe_spatial_optimizer",
    "create_spatial_optimizer",
    "estimate_moe_memory_requirements",
    "estimate_memory_requirements",
]
