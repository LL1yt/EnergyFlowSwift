#!/usr/bin/env python3
"""
Spatial Optimization Module
===========================

Модуляризированная система пространственной оптимизации
для эффективной обработки больших 3D решеток.

ОБНОВЛЕНИЕ (28 декабря 2025):
- Все конфигурационные классы перенесены в центральный config
- ChunkInfo и SpatialOptimConfig теперь в project_config.py
- 🚀 ДОБАВЛЕНЫ GPU SPATIAL OPTIMIZATION КОМПОНЕНТЫ

⚠️ MIGRATION GUIDE:
===================
DEPRECATED → РЕКОМЕНДУЕМАЯ ЗАМЕНА:
- LatticeChunker → AdaptiveGPUChunker (GPU acceleration, adaptive memory)
- ParallelSpatialProcessor → GPUSpatialProcessor (GPU + async processing)
- SpatialHashGrid → GPUSpatialHashGrid или AdaptiveGPUSpatialHash (GPU batch operations)
- MortonEncoder → GPUMortonEncoder (GPU vectorized operations)

Старые компоненты помечены как DEPRECATED и будут удалены в версии 2.0.
Для новых проектов используйте GPU аналоги!
"""

# Базовые классы
# DEPRECATED: используйте UnifiedSpatialOptimizer
from .spatial_optimizer import SpatialOptimizer
from .moe_spatial_optimizer import MoESpatialOptimizer

# NEW: Unified Spatial Optimizer
from .unified_spatial_optimizer import (
    UnifiedSpatialOptimizer,
    create_unified_spatial_optimizer,
    OptimizationConfig,
    OptimizationMode,
    estimate_unified_memory_requirements,
)

# ⚠️ DEPRECATED вспомогательные компоненты (используйте GPU аналоги)
# from .chunker import LatticeChunker  # DEPRECATED: use AdaptiveGPUChunker
from .memory_manager import MemoryPoolManager  # Still used
from .hierarchical_index import HierarchicalSpatialIndex  # Still used

# from .parallel_processor import (
#     ParallelSpatialProcessor,
# )  # DEPRECATED: use GPUSpatialProcessor

# Конфигурационные классы (теперь из центрального конфига)
from ....config.project_config import ChunkInfo, create_spatial_config_for_lattice
from ....config.project_config import ProjectConfig

# GPU Spatial Optimization Components (из guide)
from .adaptive_chunker import (
    AdaptiveGPUChunker,
    AdaptiveChunkInfo,
    ChunkProcessingTask,
    AdaptiveMemoryPredictor,
    ChunkScheduler,
)

from .gpu_spatial_processor import GPUSpatialProcessor, SpatialQuery, SpatialQueryResult

# GPU Spatial Hashing (из core/lattice/)
from ..gpu_spatial_hashing import (
    GPUMortonEncoder,
    GPUSpatialHashGrid,
    AdaptiveGPUSpatialHash,
    GPUSpatialHashingStats,
)


def get_spatial_config_from_project(project_config: ProjectConfig) -> dict:
    """Получить spatial optimization конфигурацию из ProjectConfig"""
    return project_config.get_spatial_optim_config()


# Фабричные функции (из moe_spatial_optimizer.py)
from .moe_spatial_optimizer import (
    create_moe_spatial_optimizer,
    estimate_moe_memory_requirements,
)
from .spatial_optimizer import create_spatial_optimizer, estimate_memory_requirements

# Экспорты для обратной совместимости и новые компоненты
__all__ = [
    # DEPRECATED: Основные классы (используйте UnifiedSpatialOptimizer)
    "SpatialOptimizer",  # DEPRECATED
    "MoESpatialOptimizer",  # DEPRECATED
    # NEW: Unified Spatial Optimizer
    "UnifiedSpatialOptimizer",
    "create_unified_spatial_optimizer",
    "OptimizationConfig",
    "OptimizationMode",
    "estimate_unified_memory_requirements",
    # ⚠️ DEPRECATED вспомогательные компоненты
    # "LatticeChunker",  # DEPRECATED: use AdaptiveGPUChunker
    "MemoryPoolManager",  # Still used
    "HierarchicalSpatialIndex",  # Still used
    # "ParallelSpatialProcessor",  # DEPRECATED: use GPUSpatialProcessor
    # Конфигурационные классы (из центрального конфига)
    "ChunkInfo",
    "create_spatial_config_for_lattice",
    "get_spatial_config_from_project",
    # Фабричные функции
    "create_moe_spatial_optimizer",
    "create_spatial_optimizer",
    "estimate_moe_memory_requirements",
    "estimate_memory_requirements",
    # GPU Spatial Optimization Components
    "AdaptiveGPUChunker",
    "AdaptiveChunkInfo",
    "ChunkProcessingTask",
    "AdaptiveMemoryPredictor",
    "ChunkScheduler",
    "GPUSpatialProcessor",
    "SpatialQuery",
    "SpatialQueryResult",
    "GPUMortonEncoder",
    "GPUSpatialHashGrid",
    "AdaptiveGPUSpatialHash",
    "GPUSpatialHashingStats",
]
