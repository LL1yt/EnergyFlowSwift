#!/usr/bin/env python3
"""
Unified Spatial Optimizer - Унифицированная система пространственной оптимизации
==============================================================================

Объединяет функциональность SpatialOptimizer и MoESpatialOptimizer в единую
высокопроизводительную систему с полной GPU поддержкой.

Ключевые особенности:
- Единый API для всех типов пространственной оптимизации
- Полная GPU-acceleration с fallback на CPU
- Интеграция MoE архитектуры
- GPUMortonEncoder для оптимизированного spatial indexing
- Adaptive chunking и memory management
- Real-time performance monitoring

Автор: 3D Cellular Neural Network Project
Версия: 3.0.0 (2024-12-27)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from ....config import get_project_config
from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager
from ..position import Position3D
from .gpu_spatial_processor import GPUSpatialProcessor
from .adaptive_chunker import AdaptiveGPUChunker
from ..gpu_spatial_hashing import AdaptiveGPUSpatialHash, GPUMortonEncoder

logger = get_logger(__name__)

Coordinates3D = Tuple[int, int, int]


class OptimizationMode(Enum):
    """Режимы оптимизации"""

    GPU_ONLY = "gpu_only"


class ConnectionType(Enum):
    """Типы соединений для MoE архитектуры"""

    LOCAL = "local"
    FUNCTIONAL = "functional"
    DISTANT = "distant"


@dataclass
class OptimizationConfig:
    """Конфигурация для UnifiedSpatialOptimizer"""

    enable_moe: bool = True
    enable_morton_encoding: bool = True
    enable_adaptive_chunking: bool = True
    max_memory_gb: float = 8.0
    target_performance_ms: float = 10.0


@dataclass
class SpatialOptimizationResult:
    """Результат пространственной оптимизации"""

    new_states: torch.Tensor
    processing_time_ms: float
    memory_usage_mb: float
    neighbors_found: int
    gpu_utilization: float
    mode_used: OptimizationMode
    cache_hit_rate: float = 0.0
    chunks_processed: int = 0


class BaseSpatialProcessor(ABC):
    """Базовый интерфейс для всех spatial processors"""

    @abstractmethod
    def find_neighbors(
        self, coords: Union[Coordinates3D, torch.Tensor], radius: float
    ) -> List[int]:
        """Найти соседей в радиусе"""
        pass

    @abstractmethod
    def process_lattice(
        self, states: torch.Tensor, processor_fn: Callable
    ) -> torch.Tensor:
        """Обработать решетку"""
        pass

    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности"""
        pass


class GPUSpatialProcessorWrapper(BaseSpatialProcessor):
    """Wrapper для GPU Spatial Processor"""

    def __init__(self, dimensions: Coordinates3D, config: dict):
        self.dimensions = dimensions
        self.config = config
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Создаем GPU компоненты
        self.gpu_processor = GPUSpatialProcessor(dimensions, config)
        self.pos_helper = Position3D(dimensions)

        # GPU-специфичные компоненты
        self.morton_encoder = GPUMortonEncoder(dimensions)
        self.adaptive_hash = AdaptiveGPUSpatialHash(
            dimensions, config.get("memory_pool_size_gb", 8.0) * 1024 * 0.6
        )

        logger.info(f"🚀 GPU Spatial Processor готов на {self.device}")

    def find_neighbors(
        self, coords: Union[Coordinates3D, torch.Tensor], radius: float
    ) -> List[int]:
        """GPU-accelerated поиск соседей"""
        # Конвертируем координаты в tensor
        if isinstance(coords, (tuple, list)):
            coords_tensor = torch.tensor(
                [list(coords)], dtype=torch.float32, device=self.device
            )
        else:
            coords_tensor = self.device_manager.ensure_device(coords)
            if coords_tensor.dim() == 1:
                coords_tensor = coords_tensor.unsqueeze(0)

        try:
            # Используем синхронный поиск через GPU processor
            result = self.gpu_processor.query_neighbors_sync(
                coords_tensor, radius, timeout=10.0
            )

            if result and result.neighbor_lists:
                neighbors = result.neighbor_lists[0].cpu().tolist()
                return neighbors
            else:
                return []

        except Exception as e:
            logger.warning(f"⚠️ GPU поиск не удался: {e}, fallback на adaptive hash")

            # Fallback на adaptive hash
            try:
                neighbor_lists = self.adaptive_hash.query_radius_batch(
                    coords_tensor, radius
                )
                if neighbor_lists:
                    return neighbor_lists[0].cpu().tolist()
                return []
            except Exception as e2:
                logger.error(f"❌ Adaptive hash тоже не удался: {e2}")
                return []

    def process_lattice(
        self, states: torch.Tensor, processor_fn: Callable
    ) -> torch.Tensor:
        """GPU обработка решетки с chunking"""
        states = self.device_manager.ensure_device(states)
        num_cells = states.shape[0]
        output_states = states.clone()

        # Используем adaptive chunker для больших решеток
        cfg = get_project_config().unified_optimizer
        max_batch_size = cfg.max_test_batches * cfg.batch_size_multiplier

        if num_cells > max_batch_size:
            # Chunked processing
            for batch_start in range(0, num_cells, max_batch_size):
                batch_end = min(batch_start + max_batch_size, num_cells)
                batch_indices = list(range(batch_start, batch_end))

                batch_output = self._process_batch_gpu(
                    states, batch_indices, processor_fn
                )
                output_states[batch_start:batch_end] = batch_output
        else:
            # Single batch processing
            batch_indices = list(range(num_cells))
            output_states = self._process_batch_gpu(states, batch_indices, processor_fn)

        return output_states

    def _process_batch_gpu(
        self, states: torch.Tensor, batch_indices: List[int], processor_fn: Callable
    ) -> torch.Tensor:
        """Обработка batch на GPU"""
        batch_size = len(batch_indices)
        output_batch = torch.zeros_like(states[batch_indices])

        cfg = get_project_config().unified_optimizer
        for i, cell_idx in enumerate(batch_indices):
            coords = self.pos_helper.to_3d_coordinates(cell_idx)
            neighbors = self.find_neighbors(
                coords, self.config.get("max_search_radius", cfg.default_search_radius)
            )

            if neighbors:
                neighbor_states = states[neighbors]
                new_state = processor_fn(
                    states[cell_idx].unsqueeze(0), neighbor_states, cell_idx, neighbors
                )
                output_batch[i] = new_state.squeeze(0)
            else:
                output_batch[i] = states[cell_idx]

        return output_batch

    def get_performance_stats(self) -> Dict[str, Any]:
        """Статистика GPU processor"""
        gpu_stats = self.gpu_processor.get_performance_stats()
        device_stats = self.device_manager.get_memory_stats()

        return {
            "mode": "gpu_accelerated",
            "gpu_processor": gpu_stats,
            "device": device_stats,
            "morton_encoder": {
                "enabled": True,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        }


class UnifiedSpatialOptimizer:
    """
    Унифицированная система пространственной оптимизации (GPU-only).
    """

    def __init__(
        self,
        dimensions: Coordinates3D,
        config: Optional[OptimizationConfig] = None,
        moe_processor: Optional[nn.Module] = None,
    ):
        self.config = config or get_project_config().unified_optimizer
        self.dimensions = dimensions
        self.device_manager = get_device_manager()
        self.pos_helper = Position3D(dimensions)
        self.moe_processor = moe_processor
        self.mode = self._determine_optimal_mode()

        if not self.device_manager.is_cuda():
            raise RuntimeError("UnifiedSpatialOptimizer requires a CUDA-enabled GPU.")

        # Инициализируем только GPU компоненты
        self.gpu_processor = GPUSpatialProcessor(self.dimensions, self.config)
        self.chunker = (
            AdaptiveGPUChunker(self.dimensions, self.config)
            if self.config.enable_adaptive_chunking
            else None
        )

        self.perf_history = []
        self._setup_moe_integration()

        logger.info(
            f"🚀 UnifiedSpatialOptimizer initialized in GPU_ONLY mode for dimensions {dimensions}"
        )

    def _determine_optimal_mode(self) -> OptimizationMode:
        """Определяет оптимальный режим. Теперь всегда GPU_ONLY или ошибка."""
        if not self.device_manager.is_cuda():
            raise RuntimeError(
                "Cannot initialize UnifiedSpatialOptimizer: CUDA device not available."
            )
        return OptimizationMode.GPU_ONLY

    def _setup_moe_integration(self):
        """Настройка MoE процессора, если он есть."""
        if self.moe_processor:
            self.moe_processor = self.device_manager.transfer_module(self.moe_processor)
            logger.info("✅ MoE Processor integrated with UnifiedSpatialOptimizer.")

    def find_neighbors_optimized(
        self, coords: Union[Coordinates3D, torch.Tensor], radius: float
    ) -> List[int]:
        """GPU-ускоренный поиск соседей."""
        return self.gpu_processor.find_neighbors(coords, radius)

    def optimize_lattice_forward(
        self, states: torch.Tensor, processor_fn: Optional[Callable] = None
    ) -> SpatialOptimizationResult:
        """
        Выполняет один шаг forward pass через решетку, используя GPU.
        """
        start_time = time.time()
        num_cells = states.shape[0]

        # Определяем функцию обработки
        if processor_fn is None:
            if self.moe_processor:
                processor_fn = self._create_moe_processor_fn()
            else:
                processor_fn = self._create_default_processor_fn()

        mem_before = self.device_manager.get_memory_stats().get("allocated_mb", 0)

        # Обработка всегда через GPU
        new_states = self.gpu_processor.process_lattice(
            states, processor_fn, self.chunker
        )

        self.device_manager.synchronize()
        processing_time_ms = (time.time() - start_time) * 1000

        result = self._create_optimization_result(
            new_states, processing_time_ms, self.mode, num_cells, mem_before
        )
        self._record_performance(processing_time_ms, self.mode, num_cells)

        return result

    def _create_moe_processor_fn(self) -> Callable:
        """Создает processor function для MoE архитектуры"""

        def moe_processor(current_state, neighbor_states, cell_idx, neighbor_indices):
            try:
                # Подготавливаем входы для MoE
                if current_state.dim() == 1:
                    current_state = current_state.unsqueeze(0)

                if len(neighbor_indices) == 0:
                    empty_neighbors = torch.empty(
                        1, 0, current_state.shape[-1], device=current_state.device
                    )
                else:
                    if neighbor_states.dim() == 1:
                        neighbor_states = neighbor_states.unsqueeze(0).unsqueeze(0)
                    elif neighbor_states.dim() == 2:
                        neighbor_states = neighbor_states.unsqueeze(0)
                    empty_neighbors = neighbor_states

                # Вызываем MoE processor
                result = self.moe_processor(
                    current_state=current_state,
                    neighbor_states=empty_neighbors,
                    cell_idx=cell_idx,
                    neighbor_indices=neighbor_indices,
                    spatial_optimizer=self,
                )

                # Извлекаем результат
                if isinstance(result, dict) and "new_state" in result:
                    return result["new_state"].squeeze(0)
                elif isinstance(result, torch.Tensor):
                    return result.squeeze(0)
                else:
                    return current_state.squeeze(0)

            except Exception as e:
                import traceback
                logger.error(f"⚠️ MoE processor error: {e}")
                logger.error(f"📍 Full traceback:\n{traceback.format_exc()}")
                logger.error(f"🔍 Context: cell_idx={cell_idx}, current_state.shape={getattr(current_state, 'shape', 'N/A')}")
                return (
                    current_state.squeeze(0)
                    if current_state.dim() > 1
                    else current_state
                )

        return moe_processor

    def _create_default_processor_fn(self) -> Callable:
        """Создает базовую processor function"""

        def default_processor(
            current_state, neighbor_states, cell_idx, neighbor_indices
        ):
            if len(neighbor_indices) == 0:
                return current_state

            # Простое усреднение с текущим состоянием
            if neighbor_states.dim() == 1:
                neighbor_states = neighbor_states.unsqueeze(0)

            mean_neighbor = neighbor_states.mean(dim=0)

            # Взвешенное усреднение: 70% текущее, 30% соседи
            new_state = 0.7 * current_state + 0.3 * mean_neighbor

            return new_state

        return default_processor

    def _create_optimization_result(
        self,
        new_states: torch.Tensor,
        processing_time_ms: float,
        mode: OptimizationMode,
        num_cells: int,
        mem_before: float,
    ) -> SpatialOptimizationResult:
        cfg = get_project_config().unified_optimizer

        # Memory usage
        memory_usage_mb = (
            self.device_manager.get_memory_stats().get("allocated_mb", 0) - mem_before
        )

        # GPU utilization
        gpu_utilization = 1.0

        # Cache hit rate (приблизительная оценка)
        cache_hit_rate = 0.0
        if hasattr(self.gpu_processor, "get_performance_stats"):
            gpu_stats = self.gpu_processor.get_performance_stats()
            cache_hit_rate = (
                gpu_stats.get("gpu_processor", {})
                .get("processor", {})
                .get("cache_hit_rate", 0.0)
            )

        return SpatialOptimizationResult(
            new_states=new_states,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            neighbors_found=num_cells * cfg.neighbors_found_factor,
            gpu_utilization=gpu_utilization,
            mode_used=mode,
            cache_hit_rate=cache_hit_rate,
            chunks_processed=max(1, num_cells // cfg.chunks_processed_div),
        )

    def _record_performance(
        self, time_ms: float, mode: OptimizationMode, data_size: int
    ):
        """Записывает производительность для адаптивной оптимизации"""
        self.perf_history.append(time_ms)

        # Оставляем только последние 100 записей
        if len(self.perf_history) > 100:
            self.perf_history = self.perf_history[-100:]

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Возвращает полную статистику по оптимизатору."""
        stats = {
            "mode": self.mode.value,
            "performance_history_ms": [round(t, 2) for t in self.perf_history[-100:]],
            "avg_perf_ms": np.mean(self.perf_history) if self.perf_history else 0,
            "gpu_processor": self.gpu_processor.get_performance_stats(),
        }

        if self.chunker:
            stats["chunker"] = self.chunker.get_comprehensive_stats()

        if self.moe_processor and hasattr(self.moe_processor, "get_usage_stats"):
            stats["moe_processor"] = self.moe_processor.get_usage_stats()

        return stats

    def optimize_performance(self):
        """Принудительная оптимизация производительности"""
        logger.info("🔧 Запуск принудительной оптимизации UnifiedSpatialOptimizer")

        # Оптимизируем GPU компоненты
        self.gpu_processor.optimize_performance()

        # Очищаем историю производительности
        self.perf_history = self.perf_history[-20:]  # Оставляем только последние 20

        # Принудительная очистка памяти
        self.device_manager.cleanup()

        logger.info("✅ Принудительная оптимизация завершена")

    def cleanup(self):
        """Освобождение ресурсов"""
        logger.info("🛑 Завершение работы UnifiedSpatialOptimizer")

        self.gpu_processor.shutdown()

        # Финальная очистка
        self.device_manager.cleanup()

        logger.info("✅ UnifiedSpatialOptimizer завершен")


# === FACTORY FUNCTIONS ===


def create_unified_spatial_optimizer(
    dimensions: Coordinates3D,
    config: Optional[OptimizationConfig] = None,
    moe_processor: Optional[nn.Module] = None,
) -> UnifiedSpatialOptimizer:
    """
    Фабричная функция для создания унифицированного оптимизатора

    Args:
        dimensions: Размеры решетки
        config: Конфигурация оптимизации
        moe_processor: MoE processor для интеграции

    Returns:
        Настроенный UnifiedSpatialOptimizer
    """
    if config is None:
        config = OptimizationConfig()

    logger.info(f"🏭 Создание UnifiedSpatialOptimizer для {dimensions}")

    return UnifiedSpatialOptimizer(
        dimensions=dimensions, config=config, moe_processor=moe_processor
    )


def estimate_unified_memory_requirements(
    dimensions: Coordinates3D, config: Optional[OptimizationConfig] = None
) -> Dict[str, float]:
    """
    Оценка требований к памяти для унифицированной системы

    Args:
        dimensions: Размеры решетки
        config: Конфигурация оптимизации

    Returns:
        Словарь с оценками памяти в GB
    """
    cfg = get_project_config().unified_optimizer
    if config is None:
        config = OptimizationConfig()

    total_cells = np.prod(dimensions)

    # Базовые требования
    cell_states_gb = (
        total_cells * cfg.moe_expert_state_size * 4 / (1024**3)
    )  # float32 состояния

    # GPU компоненты (если включены)
    gpu_requirements = {
        "gpu_spatial_hash_gb": total_cells * cfg.gpu_spatial_hash_bytes / (1024**3),
        "gpu_morton_encoder_gb": total_cells * cfg.gpu_morton_encoder_bytes / (1024**3),
        "gpu_chunker_gb": config.max_memory_gb * cfg.gpu_chunker_memory_fraction,
        "gpu_tensor_overhead_gb": cell_states_gb * cfg.gpu_tensor_overhead_fraction,
    }

    # MoE компоненты (если включены)
    moe_requirements = {}
    if config.enable_moe:
        moe_requirements = {
            "moe_expert_states_gb": total_cells
            * cfg.moe_expert_state_size
            * 4
            * cfg.moe_expert_count
            / (1024**3),
            "moe_connection_classification_gb": total_cells
            * cfg.moe_connection_neighbors
            * 4
            / (1024**3),
        }

    # Общие требования
    base_memory = cell_states_gb
    gpu_memory = sum(gpu_requirements.values())
    moe_memory = sum(moe_requirements.values())

    total_memory_gb = base_memory + gpu_memory + moe_memory

    result = {
        "cell_states_gb": base_memory,
        **gpu_requirements,
        **moe_requirements,
        "total_memory_gb": total_memory_gb,
        "recommended_gpu_memory_gb": total_memory_gb
        * cfg.recommended_gpu_memory_fraction,
        "recommended_system_memory_gb": gpu_memory
        * cfg.recommended_system_memory_fraction,
    }

    return result
