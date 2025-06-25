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

try:
    from ....config.project_config import get_project_config, ChunkInfo
    from ....utils.logging import get_logger
    from ....utils.device_manager import get_device_manager
    from ..position import Position3D
    from .hierarchical_index import HierarchicalSpatialIndex
    from ..spatial_hashing import SpatialHashGrid
    from .gpu_spatial_processor import GPUSpatialProcessor, SpatialQueryResult
    from .adaptive_chunker import AdaptiveGPUChunker
    from ..gpu_spatial_hashing import AdaptiveGPUSpatialHash, GPUMortonEncoder
except ImportError:
    # Fallback для прямого запуска
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from config.project_config import get_project_config, ChunkInfo
    from utils.logging import get_logger
    from utils.device_manager import get_device_manager
    from core.lattice.position import Position3D

logger = get_logger(__name__)

Coordinates3D = Tuple[int, int, int]


class OptimizationMode(Enum):
    """Режимы оптимизации"""
    AUTO = "auto"  # Автоматический выбор
    CPU_ONLY = "cpu_only"  # Только CPU
    GPU_ONLY = "gpu_only"  # Только GPU
    HYBRID = "hybrid"  # Гибридный режим


class ConnectionType(Enum):
    """Типы соединений для MoE архитектуры"""
    LOCAL = "local"
    FUNCTIONAL = "functional"
    DISTANT = "distant"


@dataclass
class OptimizationConfig:
    """Конфигурация для UnifiedSpatialOptimizer"""
    mode: OptimizationMode = OptimizationMode.AUTO
    enable_moe: bool = True
    enable_morton_encoding: bool = True
    enable_adaptive_chunking: bool = True
    max_memory_gb: float = 8.0
    target_performance_ms: float = 10.0
    fallback_enabled: bool = True


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
        self, 
        coords: Union[Coordinates3D, torch.Tensor], 
        radius: float
    ) -> List[int]:
        """Найти соседей в радиусе"""
        pass
    
    @abstractmethod
    def process_lattice(
        self, 
        states: torch.Tensor, 
        processor_fn: Callable
    ) -> torch.Tensor:
        """Обработать решетку"""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности"""
        pass


class CPUFallbackProcessor(BaseSpatialProcessor):
    """CPU fallback processor для обратной совместимости"""
    
    def __init__(self, dimensions: Coordinates3D, config: dict):
        self.dimensions = dimensions
        self.config = config
        self.pos_helper = Position3D(dimensions)
        
        # Создаем базовые структуры
        self.spatial_index = HierarchicalSpatialIndex(dimensions, config)
        max_dim = max(dimensions)
        cell_size = max(1, max_dim // 32)
        self.spatial_grid = SpatialHashGrid(dimensions, cell_size)
        
        # Заполняем индексы
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]
        coords_list = []
        indices_list = []
        
        for idx in range(total_cells):
            coords = self.pos_helper.to_3d_coordinates(idx)
            coords_list.append(coords)
            indices_list.append(idx)
            self.spatial_grid.insert(coords, idx)
        
        self.spatial_index.insert_batch(coords_list, indices_list)
        self.stats = {"total_queries": 0, "total_time_ms": 0.0}
    
    def find_neighbors(self, coords: Union[Coordinates3D, torch.Tensor], radius: float) -> List[int]:
        """CPU реализация поиска соседей"""
        if isinstance(coords, torch.Tensor):
            coords = tuple(coords.cpu().numpy().astype(int))
        
        start_time = time.time()
        neighbors = list(self.spatial_grid.query_radius(coords, radius))
        
        # Убираем саму точку
        center_idx = self.pos_helper.to_linear_index(coords)
        if center_idx in neighbors:
            neighbors.remove(center_idx)
        
        # Обновляем статистику
        query_time = (time.time() - start_time) * 1000
        self.stats["total_queries"] += 1
        self.stats["total_time_ms"] += query_time
        
        return neighbors
    
    def process_lattice(self, states: torch.Tensor, processor_fn: Callable) -> torch.Tensor:
        """CPU обработка решетки"""
        num_cells = states.shape[0]
        output_states = states.clone()
        
        for cell_idx in range(num_cells):
            coords = self.pos_helper.to_3d_coordinates(cell_idx)
            neighbors = self.find_neighbors(coords, self.config.get("max_search_radius", 10.0))
            
            if neighbors:
                new_state = processor_fn(
                    states[cell_idx], states[neighbors], cell_idx, neighbors
                )
                output_states[cell_idx] = new_state
        
        return output_states
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Статистика CPU processor"""
        avg_time = (self.stats["total_time_ms"] / max(1, self.stats["total_queries"]))
        return {
            "mode": "cpu_fallback",
            "total_queries": self.stats["total_queries"],
            "avg_query_time_ms": avg_time,
            "memory_usage_mb": 0.0,  # CPU не отслеживаем
        }


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
            dimensions, 
            config.get("memory_pool_size_gb", 8.0) * 1024 * 0.6
        )
        
        logger.info(f"🚀 GPU Spatial Processor готов на {self.device}")
    
    def find_neighbors(self, coords: Union[Coordinates3D, torch.Tensor], radius: float) -> List[int]:
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
                neighbor_lists = self.adaptive_hash.query_radius_batch(coords_tensor, radius)
                if neighbor_lists:
                    return neighbor_lists[0].cpu().tolist()
                return []
            except Exception as e2:
                logger.error(f"❌ Adaptive hash тоже не удался: {e2}")
                return []
    
    def process_lattice(self, states: torch.Tensor, processor_fn: Callable) -> torch.Tensor:
        """GPU обработка решетки с chunking"""
        states = self.device_manager.ensure_device(states)
        num_cells = states.shape[0]
        output_states = states.clone()
        
        # Используем adaptive chunker для больших решеток
        project_config = get_project_config()
        max_batch_size = getattr(project_config, "max_test_batches", 3) * 1000
        
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
        self, 
        states: torch.Tensor, 
        batch_indices: List[int], 
        processor_fn: Callable
    ) -> torch.Tensor:
        """Обработка batch на GPU"""
        batch_size = len(batch_indices)
        output_batch = torch.zeros_like(states[batch_indices])
        
        for i, cell_idx in enumerate(batch_indices):
            coords = self.pos_helper.to_3d_coordinates(cell_idx)
            neighbors = self.find_neighbors(coords, self.config.get("max_search_radius", 10.0))
            
            if neighbors:
                neighbor_states = states[neighbors]
                new_state = processor_fn(
                    states[cell_idx].unsqueeze(0), 
                    neighbor_states, 
                    cell_idx, 
                    neighbors
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
                "device": str(self.device)
            }
        }


class UnifiedSpatialOptimizer:
    """
    Унифицированная система пространственной оптимизации
    
    Объединяет функциональность SpatialOptimizer и MoESpatialOptimizer
    с полной GPU поддержкой и адаптивным выбором оптимальной стратегии.
    """
    
    def __init__(
        self, 
        dimensions: Coordinates3D, 
        config: Optional[OptimizationConfig] = None,
        moe_processor: Optional[nn.Module] = None
    ):
        self.dimensions = dimensions
        self.config = config or OptimizationConfig()
        self.moe_processor = moe_processor
        
        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Выбираем оптимальный режим работы
        self.active_mode = self._determine_optimal_mode()
        
        # Создаем процессоры
        self._initialize_processors()
        
        # MoE специфичные настройки
        if self.config.enable_moe and moe_processor:
            self._setup_moe_integration()
        
        # Performance monitoring
        self.performance_history = []
        self.adaptive_threshold_ms = self.config.target_performance_ms
        
        logger.info(f"🔧 UnifiedSpatialOptimizer инициализирован:")
        logger.info(f"   📊 Размеры: {dimensions}")
        logger.info(f"   🎯 Режим: {self.active_mode.value}")
        logger.info(f"   🤖 MoE: {'включен' if self.config.enable_moe else 'выключен'}")
        logger.info(f"   🚀 GPU: {self.device}")
    
    def _determine_optimal_mode(self) -> OptimizationMode:
        """Определяет оптимальный режим работы"""
        if self.config.mode != OptimizationMode.AUTO:
            return self.config.mode
        
        # Автоматический выбор на основе условий
        total_cells = np.prod(self.dimensions)
        available_memory_gb = self.device_manager.get_available_memory_gb()
        
        if not self.device_manager.is_cuda():
            logger.info("🖥️ CUDA недоступна, используем CPU режим")
            return OptimizationMode.CPU_ONLY
        
        if total_cells > 1000000 and available_memory_gb < 4.0:
            logger.info("⚖️ Большая решетка + мало памяти, используем гибридный режим")
            return OptimizationMode.HYBRID
        
        if available_memory_gb >= 4.0:
            logger.info("🚀 Достаточно GPU памяти, используем GPU режим")
            return OptimizationMode.GPU_ONLY
        
        return OptimizationMode.HYBRID
    
    def _initialize_processors(self):
        """Инициализирует процессоры в зависимости от режима"""
        project_config = get_project_config()
        base_config = project_config.get_spatial_optim_config()
        
        # Всегда создаем CPU fallback
        self.cpu_processor = CPUFallbackProcessor(self.dimensions, base_config)
        
        # GPU processor если нужен
        if self.active_mode in [OptimizationMode.GPU_ONLY, OptimizationMode.HYBRID]:
            try:
                self.gpu_processor = GPUSpatialProcessorWrapper(self.dimensions, base_config)
                self.has_gpu = True
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать GPU processor: {e}")
                self.has_gpu = False
                if self.active_mode == OptimizationMode.GPU_ONLY:
                    logger.info("🔄 Переключаемся на CPU режим")
                    self.active_mode = OptimizationMode.CPU_ONLY
        else:
            self.has_gpu = False
    
    def _setup_moe_integration(self):
        """Настраивает интеграцию с MoE архитектурой"""
        project_config = get_project_config()
        
        self.connection_distributions = {
            ConnectionType.LOCAL: project_config.local_connections_ratio,
            ConnectionType.FUNCTIONAL: project_config.functional_connections_ratio,
            ConnectionType.DISTANT: project_config.distant_connections_ratio,
        }
        
        # Переносим MoE processor на правильное устройство
        if hasattr(self.moe_processor, "to"):
            self.moe_processor.to(self.device)
        
        logger.info(f"🤖 MoE интеграция настроена: {self.connection_distributions}")
    
    def find_neighbors_optimized(
        self, 
        coords: Union[Coordinates3D, torch.Tensor], 
        radius: float
    ) -> List[int]:
        """
        Оптимизированный поиск соседей с автоматическим выбором алгоритма
        
        Args:
            coords: Координаты точки поиска
            radius: Радиус поиска
            
        Returns:
            Список индексов найденных соседей
        """
        start_time = time.time()
        
        try:
            # Выбираем процессор на основе текущего режима
            if self.active_mode == OptimizationMode.CPU_ONLY or not self.has_gpu:
                neighbors = self.cpu_processor.find_neighbors(coords, radius)
                mode_used = OptimizationMode.CPU_ONLY
            
            elif self.active_mode == OptimizationMode.GPU_ONLY:
                neighbors = self.gpu_processor.find_neighbors(coords, radius)
                mode_used = OptimizationMode.GPU_ONLY
            
            else:  # HYBRID mode
                # Пробуем GPU, fallback на CPU при ошибке
                try:
                    neighbors = self.gpu_processor.find_neighbors(coords, radius)
                    mode_used = OptimizationMode.GPU_ONLY
                except Exception as e:
                    logger.debug(f"GPU fallback: {e}")
                    neighbors = self.cpu_processor.find_neighbors(coords, radius)
                    mode_used = OptimizationMode.CPU_ONLY
            
            query_time_ms = (time.time() - start_time) * 1000
            
            # Адаптивная оптимизация режима
            self._record_performance(query_time_ms, mode_used, len(neighbors))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в find_neighbors_optimized: {e}")
            # Финальный fallback
            return self.cpu_processor.find_neighbors(coords, radius)
    
    def optimize_lattice_forward(
        self,
        states: torch.Tensor,
        processor_fn: Optional[Callable] = None
    ) -> SpatialOptimizationResult:
        """
        Унифицированная оптимизация решетки с поддержкой MoE
        
        Args:
            states: Состояния клеток [num_cells, state_size]
            processor_fn: Пользовательская функция обработки (опционально)
            
        Returns:
            Результат оптимизации с полной статистикой
        """
        start_time = time.time()
        
        # Убеждаемся что states на правильном устройстве
        if self.active_mode in [OptimizationMode.GPU_ONLY, OptimizationMode.HYBRID] and self.has_gpu:
            states = self.device_manager.ensure_device(states)
        
        # Выбираем процессор
        if processor_fn is None:
            if self.config.enable_moe and self.moe_processor:
                processor_fn = self._create_moe_processor_fn()
            else:
                processor_fn = self._create_default_processor_fn()
        
        # Выполняем оптимизацию
        try:
            if self.active_mode == OptimizationMode.CPU_ONLY or not self.has_gpu:
                new_states = self.cpu_processor.process_lattice(states, processor_fn)
                mode_used = OptimizationMode.CPU_ONLY
                
            elif self.active_mode == OptimizationMode.GPU_ONLY:
                new_states = self.gpu_processor.process_lattice(states, processor_fn)
                mode_used = OptimizationMode.GPU_ONLY
                
            else:  # HYBRID
                try:
                    new_states = self.gpu_processor.process_lattice(states, processor_fn)
                    mode_used = OptimizationMode.GPU_ONLY
                except Exception as e:
                    logger.warning(f"⚠️ GPU processing failed, fallback: {e}")
                    new_states = self.cpu_processor.process_lattice(states, processor_fn)
                    mode_used = OptimizationMode.CPU_ONLY
        
        except Exception as e:
            logger.error(f"❌ Критическая ошибка обработки: {e}")
            new_states = states.clone()  # Безопасный fallback
            mode_used = OptimizationMode.CPU_ONLY
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Собираем статистику
        result = self._create_optimization_result(
            new_states, processing_time_ms, mode_used, states.shape[0]
        )
        
        # Адаптивная оптимизация
        self._record_performance(processing_time_ms, mode_used, states.shape[0])
        
        return result
    
    def _create_moe_processor_fn(self) -> Callable:
        """Создает processor function для MoE архитектуры"""
        def moe_processor(current_state, neighbor_states, cell_idx, neighbor_indices):
            try:
                # Подготавливаем входы для MoE
                if current_state.dim() == 1:
                    current_state = current_state.unsqueeze(0)
                
                if len(neighbor_indices) == 0:
                    empty_neighbors = torch.empty(1, 0, current_state.shape[-1], device=current_state.device)
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
                    spatial_optimizer=self
                )
                
                # Извлекаем результат
                if isinstance(result, dict) and "new_state" in result:
                    return result["new_state"].squeeze(0)
                elif isinstance(result, torch.Tensor):
                    return result.squeeze(0)
                else:
                    return current_state.squeeze(0)
                    
            except Exception as e:
                logger.warning(f"⚠️ MoE processor error: {e}")
                return current_state.squeeze(0) if current_state.dim() > 1 else current_state
        
        return moe_processor
    
    def _create_default_processor_fn(self) -> Callable:
        """Создает базовую processor function"""
        def default_processor(current_state, neighbor_states, cell_idx, neighbor_indices):
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
        mode_used: OptimizationMode,
        num_cells: int
    ) -> SpatialOptimizationResult:
        """Создает результат оптимизации"""
        
        # Memory usage
        memory_usage_mb = 0.0
        if mode_used == OptimizationMode.GPU_ONLY and self.has_gpu:
            device_stats = self.device_manager.get_memory_stats()
            memory_usage_mb = device_stats.get("allocated_mb", 0.0)
        
        # GPU utilization
        gpu_utilization = 1.0 if mode_used == OptimizationMode.GPU_ONLY else 0.0
        
        # Cache hit rate (приблизительная оценка)
        cache_hit_rate = 0.0
        if self.has_gpu and hasattr(self, 'gpu_processor'):
            gpu_stats = self.gpu_processor.get_performance_stats()
            cache_hit_rate = gpu_stats.get("gpu_processor", {}).get("processor", {}).get("cache_hit_rate", 0.0)
        
        return SpatialOptimizationResult(
            new_states=new_states,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            neighbors_found=num_cells * 20,  # Приблизительная оценка
            gpu_utilization=gpu_utilization,
            mode_used=mode_used,
            cache_hit_rate=cache_hit_rate,
            chunks_processed=max(1, num_cells // 1000)
        )
    
    def _record_performance(
        self, 
        time_ms: float, 
        mode: OptimizationMode, 
        data_size: int
    ):
        """Записывает производительность для адаптивной оптимизации"""
        self.performance_history.append({
            "time_ms": time_ms,
            "mode": mode,
            "data_size": data_size,
            "timestamp": time.time()
        })
        
        # Оставляем только последние 100 записей
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Адаптивное переключение режима в HYBRID mode
        if (self.config.mode == OptimizationMode.AUTO and 
            self.active_mode == OptimizationMode.HYBRID):
            self._adaptive_mode_optimization()
    
    def _adaptive_mode_optimization(self):
        """Адаптивная оптимизация режима работы"""
        if len(self.performance_history) < 10:
            return
        
        recent_history = self.performance_history[-10:]
        
        # Анализируем производительность GPU vs CPU
        gpu_times = [h["time_ms"] for h in recent_history if h["mode"] == OptimizationMode.GPU_ONLY]
        cpu_times = [h["time_ms"] for h in recent_history if h["mode"] == OptimizationMode.CPU_ONLY]
        
        if len(gpu_times) >= 3 and len(cpu_times) >= 3:
            avg_gpu = np.mean(gpu_times)
            avg_cpu = np.mean(cpu_times)
            
            # Переключаемся на более быстрый режим
            if avg_gpu < avg_cpu * 0.8:  # GPU значительно быстрее
                if self.active_mode != OptimizationMode.GPU_ONLY:
                    logger.info("🚀 Переключение на GPU_ONLY режим (производительность)")
                    self.active_mode = OptimizationMode.GPU_ONLY
            elif avg_cpu < avg_gpu * 0.8:  # CPU значительно быстрее
                if self.active_mode != OptimizationMode.CPU_ONLY:
                    logger.info("🖥️ Переключение на CPU_ONLY режим (производительность)")
                    self.active_mode = OptimizationMode.CPU_ONLY
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Получить полную статистику системы"""
        stats = {
            "unified_optimizer": {
                "dimensions": self.dimensions,
                "active_mode": self.active_mode.value,
                "moe_enabled": self.config.enable_moe,
                "morton_enabled": self.config.enable_morton_encoding,
                "performance_history_length": len(self.performance_history)
            }
        }
        
        # CPU stats
        stats["cpu_processor"] = self.cpu_processor.get_performance_stats()
        
        # GPU stats если доступен
        if self.has_gpu:
            stats["gpu_processor"] = self.gpu_processor.get_performance_stats()
            stats["device"] = self.device_manager.get_memory_stats()
        
        # Performance analysis
        if self.performance_history:
            recent = self.performance_history[-20:]  # Последние 20 операций
            stats["performance_analysis"] = {
                "avg_time_ms": np.mean([h["time_ms"] for h in recent]),
                "mode_distribution": {
                    mode.value: len([h for h in recent if h["mode"] == mode]) 
                    for mode in OptimizationMode
                },
                "target_performance_ms": self.adaptive_threshold_ms
            }
        
        return stats
    
    def optimize_performance(self):
        """Принудительная оптимизация производительности"""
        logger.info("🔧 Запуск принудительной оптимизации UnifiedSpatialOptimizer")
        
        # Оптимизируем GPU компоненты
        if self.has_gpu:
            self.gpu_processor.gpu_processor.optimize_performance()
        
        # Очищаем историю производительности
        self.performance_history = self.performance_history[-20:]  # Оставляем только последние 20
        
        # Принудительная очистка памяти
        self.device_manager.cleanup()
        
        logger.info("✅ Принудительная оптимизация завершена")
    
    def cleanup(self):
        """Освобождение ресурсов"""
        logger.info("🛑 Завершение работы UnifiedSpatialOptimizer")
        
        if self.has_gpu:
            self.gpu_processor.gpu_processor.shutdown()
        
        # Финальная очистка
        self.device_manager.cleanup()
        
        logger.info("✅ UnifiedSpatialOptimizer завершен")


# === FACTORY FUNCTIONS ===

def create_unified_spatial_optimizer(
    dimensions: Coordinates3D,
    config: Optional[OptimizationConfig] = None,
    moe_processor: Optional[nn.Module] = None
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
        dimensions=dimensions,
        config=config,
        moe_processor=moe_processor
    )


def estimate_unified_memory_requirements(
    dimensions: Coordinates3D,
    config: Optional[OptimizationConfig] = None
) -> Dict[str, float]:
    """
    Оценка требований к памяти для унифицированной системы
    
    Args:
        dimensions: Размеры решетки
        config: Конфигурация оптимизации
        
    Returns:
        Словарь с оценками памяти в GB
    """
    if config is None:
        config = OptimizationConfig()
    
    total_cells = np.prod(dimensions)
    
    # Базовые требования
    cell_states_gb = total_cells * 32 * 4 / (1024**3)  # float32 состояния
    
    # CPU компоненты (всегда присутствуют)
    cpu_requirements = {
        "cpu_spatial_index_gb": total_cells * 16 / (1024**3),
        "cpu_neighbor_cache_gb": total_cells * 26 * 4 / (1024**3)
    }
    
    # GPU компоненты (если включены)
    gpu_requirements = {}
    if config.mode in [OptimizationMode.AUTO, OptimizationMode.GPU_ONLY, OptimizationMode.HYBRID]:
        gpu_requirements = {
            "gpu_spatial_hash_gb": total_cells * 8 / (1024**3),
            "gpu_morton_encoder_gb": total_cells * 4 / (1024**3),
            "gpu_chunker_gb": config.max_memory_gb * 0.1,
            "gpu_tensor_overhead_gb": cell_states_gb * 0.3
        }
    
    # MoE компоненты (если включены)
    moe_requirements = {}
    if config.enable_moe:
        moe_requirements = {
            "moe_expert_states_gb": total_cells * 32 * 4 * 3 / (1024**3),  # 3 эксперта
            "moe_connection_classification_gb": total_cells * 26 * 4 / (1024**3)
        }
    
    # Общие требования
    base_memory = cell_states_gb
    cpu_memory = sum(cpu_requirements.values())
    gpu_memory = sum(gpu_requirements.values())
    moe_memory = sum(moe_requirements.values())
    
    total_memory_gb = base_memory + cpu_memory + gpu_memory + moe_memory
    
    result = {
        "cell_states_gb": base_memory,
        **cpu_requirements,
        **gpu_requirements,
        **moe_requirements,
        "total_memory_gb": total_memory_gb,
        "recommended_gpu_memory_gb": total_memory_gb * 1.4,  # 40% запас
        "recommended_system_memory_gb": cpu_memory * 1.5,   # CPU fallback
    }
    
    return result