#!/usr/bin/env python3
"""
Централизованная конфигурация для Clean 3D Cellular Neural Network
=================================================================

Единственный источник истины для всей архитектуры.
Основано на принципах из utils/centralized_config.py, но упрощено для clean архитектуры.

Принцип: Модульность > Монолитность. Конфигурация разделена на логические блоки.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
import logging
import torch
import math

from ..utils.device_manager import DeviceManager, get_device_manager


# === МОДУЛЬНЫЕ КОНФИГУРАЦИИ ===


@dataclass
class LatticeConfig:
    """Конфигурация 3D решетки"""

    dimensions: Tuple[int, int, int] = (5, 5, 5)  # (x, y, z)
    # target_dimensions: Tuple[int, int, int] = (666, 666, 333) # для научных опытов


@dataclass
class GNNConfig:
    """Конфигурация GNN (замена gMLP)"""

    state_size: int = 32
    message_dim: int = 16
    hidden_dim: int = 32
    neighbor_count: int = 26  # Синхронизация с NCA (legacy)
    external_input_size: int = 8
    target_params: int = 8000
    activation: str = "gelu"
    use_attention: bool = True
    aggregation: str = "attention"
    num_layers: int = 1


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""

    learning_rate: float = 0.001
    batch_size: int = 4
    embedding_dim: int = 768  # DistilBERT
    phrase_based_training: bool = True


@dataclass
class BiologyConfig:
    """Конфигурация биологических принципов"""

    shared_weights: bool = True
    tissue_simulation: bool = True
    receptor_coverage: float = 1.0
    signal_propagation: bool = True


@dataclass
class NeighborConfig:
    """Конфигурация поиска и классификации соседей"""

    finding_strategy: str = "tiered"
    dynamic_count: bool = True
    base_neighbor_count: int = 26  # Legacy
    max_neighbors: int = 20000  # Биологический лимит

    # Adaptive Radius - ИСПРАВЛЕНО для лучшего распределения связей
    adaptive_radius_enabled: bool = True
    adaptive_radius_ratio: float = (
        0.4  # Увеличено с 0.3 до 0.4 для больше DISTANT связей
    )
    adaptive_radius_max: float = 500.0
    adaptive_radius_min: float = 1.0  # Уменьшено с 0.1 до 1.0

    # Tiered Topology
    local_tier: float = 0.1
    functional_tier: float = 0.55
    distant_tier: float = 0.35
    local_grid_cell_size: int = 8


@dataclass
class PlasticityConfig:
    """Конфигурация пластичности"""

    enabled: bool = True
    rule: str = "combined"  # STDP + BCM + competitive
    competitive_learning_enabled: bool = True
    metaplasticity_enabled: bool = True
    clustering_enabled: bool = False  # Пока отключено


@dataclass
class CNFConfig:
    """Конфигурация Lightweight Continuous Normalizing Flows"""

    enabled: bool = True
    functional_connections: bool = True
    distant_connections: bool = True
    integration_steps: int = 3  # 3-step Euler
    adaptive_step_size: bool = True
    target_params_per_connection: int = 3000
    batch_processing_mode: str = "ADAPTIVE_BATCH"  # или "batch" если потребуется
    max_batch_size: int = 1024  # разумный дефолт для GPU/CPU
    adaptive_method: str = (
        "LIPSCHITZ_BASED"  # или "rk4", "euler", "adaptive" если потребуется
    )


@dataclass
class SpatialOptimConfig:
    """Конфигурация пространственной оптимизации (chunking, memory)"""

    chunk_size: int = 64
    chunk_overlap: int = 8
    max_chunks_in_memory: int = 4
    memory_pool_size_gb: float = 12.0
    garbage_collect_frequency: int = 100
    prefetch_chunks: bool = True
    levels: int = 3  # Hierarchical indexing levels
    min_cells_per_node: int = 1000
    max_search_radius: float = 50.0
    num_worker_threads: int = 4
    batch_size_per_thread: int = 10000
    enable_async_processing: bool = True
    enable_profiling: bool = True
    log_memory_usage: bool = True


@dataclass
class VectorizedConfig:
    """Конфигурация векторизованных компонентов (основные оптимизации производительности)"""

    # Основные настройки векторизации
    enabled: bool = True  # Использовать векторизованные компоненты по умолчанию
    force_vectorized: bool = (
        True  # Принудительно использовать только векторизованные версии
    )

    # Batch processing настройки
    adaptive_batch_size: bool = True  # Адаптивный размер батча на основе GPU памяти
    min_batch_size: int = 100  # Минимальный размер батча
    max_batch_size: int = 8000  # Максимальный размер батча для GPU
    cpu_batch_size: int = 1000  # Размер батча для CPU

    # Memory optimization
    tensor_reuse: bool = True  # Переиспользование тензоров для экономии памяти
    memory_efficient: bool = True  # Включить memory-efficient режим
    prefetch_neighbors: bool = True  # Предзагрузка соседей

    # Performance monitoring
    enable_profiling: bool = True  # Включить профилирование производительности
    log_performance_stats: bool = True  # Логировать статистику производительности
    benchmark_mode: bool = False  # Режим бенчмарка (более подробные метрики)

    # Fallback настройки
    fallback_to_sequential: bool = (
        False  # Откат к sequential версии при ошибках (ОТКЛЮЧЕНО)
    )
    strict_vectorization: bool = True  # Строгий режим: только векторизованные операции


@dataclass
class MemoryConfig:
    """Конфигурация оптимизации памяти"""

    efficient: bool = True
    use_checkpointing: bool = True
    mixed_precision: bool = True


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""

    enabled: bool = True
    debug_mode: bool = True
    log_level: str = "INFO"


@dataclass
class DeviceConfig:
    """Конфигурация устройства (CPU/GPU)"""

    device: str = "auto"
    prefer_cuda: bool = True
    debug_mode: bool = True
    memory_cleanup_threshold: int = 100
    memory_safety_buffer: float = 0.15


@dataclass
class InitConfig:
    """Конфигурация инициализации"""

    seed: int = 42
    initialization_method: str = "xavier"


@dataclass
class EulerSolverConfig:
    """Глобальная конфигурация для GPU Optimized Euler Solver"""

    adaptive_method: str = "LIPSCHITZ_BASED"
    base_dt: float = 0.1
    min_dt: float = 0.001
    max_dt: float = 0.5
    lipschitz_safety_factor: float = 0.8
    stability_threshold: float = 10.0
    memory_efficient: bool = True
    max_batch_size: int = 1000
    error_tolerance: float = 1e-3
    enable_profiling: bool = True


@dataclass
class AdaptiveChunkerConfig:
    """Конфигурация адаптивного чанкера для оптимизации памяти и производительности"""

    optimal_batch_size: int = 1000
    preferred_device: str = "cuda"
    memory_per_cell_base: int = 64
    memory_overhead_factor: float = 1.3
    max_history: int = 1000
    min_available_memory_mb: float = 500.0
    cuda_fallback_available_mb: float = 8000.0
    cpu_fallback_available_mb: float = 16000.0
    safe_memory_buffer: float = 0.8
    max_concurrent_chunks: int = 4
    max_chunks_in_memory: int = 4
    min_chunk_size: int = 8
    chunk_size_fallback_div: int = 8
    optimal_batch_size_small: int = 100
    optimal_batch_size_medium: int = 2500
    optimal_batch_size_large: int = 2500
    memory_pressure_high: float = 0.8
    memory_pressure_low: float = 0.3
    processing_priority_high_delta: int = 10
    processing_priority_low_delta: int = 20


@dataclass
class UnifiedSpatialOptimizerConfig:
    """Конфигурация унифицированного пространственного оптимизатора"""

    fallback_memory_mb: int = 4000
    fallback_memory_gb: float = 4.0
    large_lattice_threshold: int = 1000000
    min_gpu_memory_gb: float = 4.0
    max_test_batches: int = 3
    batch_size_multiplier: int = 1000
    default_search_radius: float = 10.0
    neighbors_found_factor: int = 20
    chunks_processed_div: int = 1000
    moe_expert_state_size: int = 32
    moe_expert_count: int = 3
    moe_connection_neighbors: int = 26
    cpu_spatial_index_bytes: int = 16
    cpu_neighbor_cache_neighbors: int = 26
    cpu_neighbor_cache_bytes: int = 4
    gpu_spatial_hash_bytes: int = 8
    gpu_morton_encoder_bytes: int = 4
    gpu_chunker_memory_fraction: float = 0.1
    gpu_tensor_overhead_fraction: float = 0.3
    recommended_gpu_memory_fraction: float = 1.4
    recommended_system_memory_fraction: float = 1.5


# === MoE КОНФИГУРАЦИИ (остаются отдельными для ясности) ===


@dataclass
class LocalExpertConfig:
    """Конфигурация для Local Expert"""

    params: int = 2059
    type: str = "linear"
    alpha: float = 0.1
    beta: float = 0.9
    neighbor_agg_hidden1: int = 32
    neighbor_agg_hidden2: int = 16
    processor_hidden: int = 64
    max_neighbors_buffer: int = 100
    use_attention: bool = True
    default_batch_size: int = (
        1  # Batch размер для fallback случаев, когда размерность теряется
    )


@dataclass
class FunctionalExpertConfig:
    """Конфигурация для Functional Expert (GNN)"""

    params: int = 8233
    type: str = "gnn"


@dataclass
class DistantExpertConfig:
    """Конфигурация для Distant Expert (CNF)"""

    params: int = 4000
    type: str = "cnf"


@dataclass
class GatingConfig:
    """Конфигурация для Gating Network"""

    state_size: int = 32
    params: int = 808
    num_experts: int = 3
    activation: str = "gelu"
    hidden_dim: int = 11


@dataclass
class ExpertConnectionConfig:
    """Конфигурация для распределения связей по экспертам - ИСПРАВЛЕНО"""

    local_ratio: float = 0.10
    functional_ratio: float = 0.55
    distant_ratio: float = 0.35
    # ИСПРАВЛЕННЫЕ ПОРОГИ для adaptive_radius ~6.0 (15*0.4)
    local_distance_threshold: float = 1.8  # 30% от радиуса (1.8/6.0)
    functional_distance_threshold: float = 4.0  # 67% от радиуса (4.0/6.0)
    distant_distance_threshold: float = 5.5  # 92% от радиуса (5.5/6.0)
    functional_similarity_threshold: float = 0.3


@dataclass
class ConnectionCacheConfig:
    """Конфигурация для Connection Cache оптимизации"""

    # Основные настройки кэширования
    enabled: bool = True  # Включить/выключить кэширование связей
    enable_performance_monitoring: bool = True  # Включить мониторинг производительности
    enable_detailed_stats: bool = False  # Детальная статистика (может замедлить работу)

    # Настройки автоматического включения/выключения
    auto_enable_threshold: int = (
        10000  # Автоматически включать кэш для решеток >10k клеток
    )
    auto_disable_threshold: int = (
        1000  # Автоматически выключать кэш для решеток <1k клеток
    )

    # Настройки производительности
    force_cache_rebuild: bool = False  # Принудительно пересоздать кэш при старте
    save_to_disk: bool = True  # Сохранять кэш на диск
    load_from_disk: bool = True  # Загружать кэш с диска

    # Настройки памяти
    max_cache_size_mb: float = 100.0  # Максимальный размер кэша в MB
    clear_cache_on_memory_pressure: bool = True  # Очищать кэш при нехватке памяти

    # Настройки для малых решеток
    small_lattice_fallback: bool = True  # Использовать fallback для малых решеток
    benchmark_small_lattices: bool = False  # Бенчмарк малых решеток (для отладки)


@dataclass
class ExpertConfig:
    """Главная конфигурация для MoE"""

    enabled: bool = True
    gating: GatingConfig = field(default_factory=GatingConfig)
    local: LocalExpertConfig = field(default_factory=LocalExpertConfig)
    functional: FunctionalExpertConfig = field(default_factory=FunctionalExpertConfig)
    distant: DistantExpertConfig = field(default_factory=DistantExpertConfig)
    connections: ExpertConnectionConfig = field(default_factory=ExpertConnectionConfig)
    cache: ConnectionCacheConfig = field(default_factory=ConnectionCacheConfig)


# === КОНФИГУРАЦИЯ ДЛЯ УСТАРЕВШИХ ПАРАМЕТРОВ ===


@dataclass
class DeprecatedConfig:
    """
    Параметры, которые больше не используются активно.
    Хранятся для справки и legacy-совместимости.
    """

    architecture_type: str = "moe"  # Теперь определяется наличием `expert` конфига
    gmlp_params: int = 113000  # Заменено GNN
    hybrid_gnn_cnf_expert_params: int = 12233
    # ... можно добавить другие устаревшие параметры по мере их выявления
    # effective_neighbors - заменен на calculate_dynamic_neighbors


@dataclass
class ConnectionInfoConfig:
    """Централизованная конфигурация для информации о связи между клетками"""

    strength: float = 1.0
    functional_similarity: Optional[float] = None


@dataclass
class Lattice3DConfig:
    spatial_mode: str = "GPU_ONLY"
    enable_moe: bool = True
    enable_morton_encoding: bool = True
    target_performance_ms: float = 50.0
    fallback_enabled: bool = False


# === ГЛАВНЫЙ КОНФИГУРАЦИОННЫЙ КЛАСС ===


@dataclass
class ProjectConfig:
    """
    Единственный источник истины для всей архитектуры.
    Собран из модульных блоков для чистоты и ясности.
    """

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    biology: BiologyConfig = field(default_factory=BiologyConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    cnf: CNFConfig = field(default_factory=CNFConfig)
    spatial: SpatialOptimConfig = field(default_factory=SpatialOptimConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    init: InitConfig = field(default_factory=InitConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    deprecated: DeprecatedConfig = field(default_factory=DeprecatedConfig)
    euler: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    connection: ConnectionInfoConfig = field(default_factory=ConnectionInfoConfig)
    adaptive_chunker: AdaptiveChunkerConfig = field(
        default_factory=AdaptiveChunkerConfig
    )
    unified_spatial_optimizer: UnifiedSpatialOptimizerConfig = field(
        default_factory=UnifiedSpatialOptimizerConfig
    )
    unified_optimizer: UnifiedSpatialOptimizerConfig = field(
        default_factory=UnifiedSpatialOptimizerConfig
    )
    lattice3d: Lattice3DConfig = field(default_factory=Lattice3DConfig)
    vectorized: VectorizedConfig = field(default_factory=VectorizedConfig)

    # --- Вычисляемые и Runtime-свойства ---
    device_manager: DeviceManager = field(init=False)
    total_cells: int = field(init=False)
    current_device: str = field(init=False)
    max_neighbors: int = field(init=False)

    def __post_init__(self):
        """Валидация и автоматическая настройка после инициализации"""
        self.device_manager = get_device_manager(
            prefer_cuda=self.device.prefer_cuda, debug_mode=self.device.debug_mode
        )
        self.current_device = self.device_manager.get_device_str()

        self.total_cells = (
            self.lattice.dimensions[0]
            * self.lattice.dimensions[1]
            * self.lattice.dimensions[2]
        )

        self.max_neighbors = self.calculate_dynamic_neighbors()

        if self.logging.debug_mode:
            self.log_initialization()

    def log_initialization(self):
        """Вывод информации о конфигурации при инициализации"""
        logging.info("🔧 ProjectConfig initialized with modular structure:")
        logging.info(
            f"   Lattice: {self.lattice.dimensions} = {self.total_cells} cells"
        )
        logging.info(f"   Device: {self.current_device} (via DeviceManager)")

        # Векторизация статус
        if self.vectorized.enabled:
            cell_type = self.get_cell_type()
            optimal_batch = self.calculate_optimal_batch_size()
            logging.info(
                f"🚀 VECTORIZED MODE: {cell_type.upper()} (batch: {optimal_batch})"
            )
            if self.vectorized.force_vectorized:
                logging.info("⚡ Forced vectorization enabled - maximum performance!")
        else:
            logging.info(
                "🐌 Legacy mode - consider enabling vectorization for better performance"
            )

        if self.expert.enabled:
            logging.info("   Architecture: MoE (Mixture of Experts)")
            logging.info(f"   MoE Gating params: {self.expert.gating.params}")
            logging.info(f"   GNN Expert params: {self.expert.functional.params}")
            logging.info(f"   CNF Expert params: {self.expert.distant.params}")
        device_stats = self.device_manager.get_memory_stats()
        logging.info(f"   Memory stats: {device_stats}")

    # === МЕТОДЫ ДОСТУПА (сохраняем для совместимости, но рекомендуем прямой доступ) ===

    def get_gnn_config(self) -> Dict[str, Any]:
        """Получить GNN конфигурацию. Рекомендуется: `config.gnn`."""
        return {
            "state_size": self.gnn.state_size,
            "neighbor_count": self.gnn.neighbor_count,
            "message_dim": self.gnn.message_dim,
            "hidden_dim": self.gnn.hidden_dim,
            "external_input_size": self.gnn.external_input_size,
            "target_params": self.gnn.target_params,
            "activation": self.gnn.activation,
            "use_attention": self.gnn.use_attention,
            "aggregation": self.gnn.aggregation,
            "num_layers": self.gnn.num_layers,
            "dropout": 0.0,
        }

    def get_lattice_config(self) -> Dict[str, Any]:
        """Получить конфигурацию 3D решетки. Рекомендуется: `config.lattice`."""
        return {
            "dimensions": self.lattice.dimensions,
            "total_cells": self.total_cells,
            "device": self.current_device,
            "enable_logging": self.logging.enabled,
            "seed": self.init.seed,
            "initialization_method": self.init.initialization_method,
        }

    def get_cnf_config(self) -> Dict[str, Any]:
        """Получить конфигурацию Lightweight CNF. Рекомендуется: `config.cnf`."""
        return {
            "enable_cnf": self.cnf.enabled,
            "functional_connections": self.cnf.functional_connections,
            "distant_connections": self.cnf.distant_connections,
            "integration_steps": self.cnf.integration_steps,
            "adaptive_step_size": self.cnf.adaptive_step_size,
            "target_params_per_connection": self.cnf.target_params_per_connection,
        }

    def get_neighbor_strategy_config(self) -> Dict[str, Any]:
        """Получить конфигурацию стратегии соседства."""
        return {
            "local_tier": self.neighbors.local_tier,
            "functional_tier": self.neighbors.functional_tier,
            "distant_tier": self.neighbors.distant_tier,
            "local_grid_cell_size": self.neighbors.local_grid_cell_size,
            "adaptive_radius_enabled": self.neighbors.adaptive_radius_enabled,
            "adaptive_radius_ratio": self.neighbors.adaptive_radius_ratio,
            "adaptive_radius_max": self.neighbors.adaptive_radius_max,
            "adaptive_radius_min": self.neighbors.adaptive_radius_min,
            "adaptive_radius": self.calculate_adaptive_radius(),
        }

    def get_gating_config(self) -> Dict[str, Any]:
        """Получить конфигурацию GatingNetwork. Рекомендуется: `config.expert.gating`."""
        return {
            "state_size": self.expert.gating.state_size,
            "num_experts": self.expert.gating.num_experts,
            "target_params": self.expert.gating.params,
            "activation": self.expert.gating.activation,
            "hidden_dim": self.expert.gating.hidden_dim,
        }

    def get_moe_config(self) -> Dict[str, Any]:
        """Получить полную конфигурацию MoE архитектуры. Рекомендуется: `config.expert`."""
        return {
            "enable_moe": self.expert.enabled,
            "gating_config": self.get_gating_config(),
            "experts": {
                "local": {
                    "type": self.expert.local.type,
                    "params": self.expert.local.params,
                    "ratio": self.expert.connections.local_ratio,
                },
                "functional": {
                    "type": self.expert.functional.type,
                    "params": self.expert.functional.params,
                    "ratio": self.expert.connections.functional_ratio,
                },
                "distant": {
                    "type": self.expert.distant.type,
                    "params": self.expert.distant.params,
                    "ratio": self.expert.connections.distant_ratio,
                },
            },
            "thresholds": {
                "local_distance": self.expert.connections.local_distance_threshold,
                "functional_similarity": self.expert.connections.functional_similarity_threshold,
            },
        }

    def get_spatial_optim_config(self) -> Dict[str, Any]:
        """Получить конфигурацию spatial optimization. Рекомендуется: `config.spatial`."""
        max_dim = max(self.lattice.dimensions)
        return {
            "chunk_size": (
                min(self.spatial.chunk_size, max_dim // 2)
                if self.total_cells > 100_000
                else 32
            ),
            "chunk_overlap": (
                self.spatial.chunk_overlap if self.total_cells > 50_000 else 4
            ),
            "max_chunks_in_memory": (
                self.spatial.max_chunks_in_memory if self.total_cells > 100_000 else 2
            ),
            "memory_pool_size_gb": (
                self.spatial.memory_pool_size_gb
                if self.current_device == "cuda"
                else 4.0
            ),
            "garbage_collect_frequency": self.spatial.garbage_collect_frequency,
            "prefetch_chunks": self.spatial.prefetch_chunks,
            "spatial_levels": self.spatial.levels,
            "min_cells_per_node": self.spatial.min_cells_per_node,
            "max_search_radius": self.calculate_adaptive_radius(),
            "num_worker_threads": (
                self.spatial.num_worker_threads if self.current_device == "cuda" else 2
            ),
            "batch_size_per_thread": self.spatial.batch_size_per_thread,
            "enable_async_processing": self.spatial.enable_async_processing
            and self.current_device == "cuda",
            "enable_profiling": self.spatial.enable_profiling,
            "log_memory_usage": self.spatial.log_memory_usage,
        }

    # === ВЫЧИСЛЯЕМЫЕ МЕТОДЫ ===

    def calculate_dynamic_neighbors(self) -> int:
        """Динамический расчет количества соседей на основе размера решетки."""
        if not self.neighbors.dynamic_count:
            return self.neighbors.base_neighbor_count

        if self.total_cells <= 216:  # 6x6x6
            return 26
        elif self.total_cells <= 4096:  # 16x16x16
            return 500
        elif self.total_cells <= 19683:  # 27x27x27
            return 5000
        elif self.total_cells <= 262144:  # 64x64x64
            return 10000
        else:
            return min(self.neighbors.max_neighbors, self.total_cells)

    def calculate_adaptive_radius(self) -> float:
        """Вычисляет адаптивный радиус поиска соседей с улучшенной логикой."""
        if not self.neighbors.adaptive_radius_enabled:
            return self.neighbors.adaptive_radius_max

        max_dimension = max(self.lattice.dimensions)

        # ИСПРАВЛЕННАЯ ЛОГИКА: больше радиус для лучшего распределения связей
        if max_dimension <= 5:
            adaptive_radius = 2.0  # Было 1.5
        elif max_dimension <= 10:
            adaptive_radius = 3.5  # Было 2.0
        elif max_dimension <= 27:
            adaptive_radius = 6.0  # Было 3.0 -> теперь найдет DISTANT связи
        elif max_dimension <= 100:
            adaptive_radius = 8.0  # Было 4.0
        else:
            adaptive_radius = min(12.0, 3.0 + math.log10(max_dimension))  # Было 8.0

        return max(
            self.neighbors.adaptive_radius_min,
            min(self.neighbors.adaptive_radius_max, adaptive_radius),
        )

    # === DEVICE MANAGEMENT МЕТОДЫ ===

    def get_device_manager(self) -> DeviceManager:
        return self.device_manager

    def ensure_tensor_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.device_manager.ensure_device(tensor)

    def allocate_tensor(self, shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
        return self.device_manager.allocate_tensor(shape, **kwargs)

    def transfer_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return self.device_manager.transfer_module(module)

    def cleanup_memory(self):
        self.device_manager.cleanup()

    def get_device_config(self) -> Dict[str, Any]:
        """Получить конфигурацию устройства для использования в модулях."""
        return {
            "device": self.current_device,
            "device_manager": self.device_manager,
            "prefer_cuda": self.device.prefer_cuda,
            "debug_mode": self.device.debug_mode,
            "memory_cleanup_threshold": self.device.memory_cleanup_threshold,
            "memory_safety_buffer": self.device.memory_safety_buffer,
            "is_cuda": self.device_manager.is_cuda(),
            "memory_stats": self.device_manager.get_memory_stats(),
        }

    def get_vectorized_config(self) -> Dict[str, Any]:
        """Получить конфигурацию векторизованных компонентов. Рекомендуется: `config.vectorized`."""
        return {
            "enabled": self.vectorized.enabled,
            "force_vectorized": self.vectorized.force_vectorized,
            "adaptive_batch_size": self.vectorized.adaptive_batch_size,
            "optimal_batch_size": self.calculate_optimal_batch_size(),
            "min_batch_size": self.vectorized.min_batch_size,
            "max_batch_size": self.vectorized.max_batch_size,
            "cpu_batch_size": self.vectorized.cpu_batch_size,
            "tensor_reuse": self.vectorized.tensor_reuse,
            "memory_efficient": self.vectorized.memory_efficient,
            "prefetch_neighbors": self.vectorized.prefetch_neighbors,
            "enable_profiling": self.vectorized.enable_profiling,
            "log_performance_stats": self.vectorized.log_performance_stats,
            "benchmark_mode": self.vectorized.benchmark_mode,
            "fallback_to_sequential": self.vectorized.fallback_to_sequential,
            "strict_vectorization": self.vectorized.strict_vectorization,
            "device": self.current_device,
            "is_cuda": self.device_manager.is_cuda(),
        }

    def calculate_optimal_batch_size(self) -> int:
        """Вычисляет оптимальный размер батча для векторизованных операций."""
        if not self.vectorized.adaptive_batch_size:
            return (
                self.vectorized.max_batch_size
                if self.device_manager.is_cuda()
                else self.vectorized.cpu_batch_size
            )

        if self.device_manager.is_cuda():
            memory_stats = self.device_manager.get_memory_stats()
            available_mb = memory_stats.get("available_mb", 8000)

            if available_mb > 16000:  # >16GB
                return min(self.total_cells, self.vectorized.max_batch_size)
            elif available_mb > 8000:  # >8GB
                return min(self.total_cells, self.vectorized.max_batch_size // 2)
            else:  # <8GB
                return min(self.total_cells, self.vectorized.max_batch_size // 4)
        else:
            return min(self.total_cells, self.vectorized.cpu_batch_size)

    def should_use_vectorized(self) -> bool:
        """Определяет, следует ли использовать векторизованные компоненты."""
        if self.vectorized.force_vectorized:
            return True
        return self.vectorized.enabled

    def get_cell_type(self) -> str:
        """Возвращает тип клетки для использования (только vectorized)."""
        # Всегда возвращаем векторизованную версию
        return "vectorized_gnn"

    def should_use_connection_cache(self) -> bool:
        """Определяет, следует ли использовать кэширование связей."""
        if not self.expert.cache.enabled:
            return False

        # Автоматическое определение на основе размера решетки
        if self.total_cells >= self.expert.cache.auto_enable_threshold:
            return True
        elif self.total_cells <= self.expert.cache.auto_disable_threshold:
            return False if self.expert.cache.small_lattice_fallback else True
        else:
            return True  # Средние размеры - используем кэш

    def get_connection_cache_config(self) -> Dict[str, Any]:
        """Получить конфигурацию кэширования связей."""
        return {
            "enabled": self.should_use_connection_cache(),
            "enable_performance_monitoring": self.expert.cache.enable_performance_monitoring,
            "enable_detailed_stats": self.expert.cache.enable_detailed_stats,
            "force_cache_rebuild": self.expert.cache.force_cache_rebuild,
            "save_to_disk": self.expert.cache.save_to_disk,
            "load_from_disk": self.expert.cache.load_from_disk,
            "max_cache_size_mb": self.expert.cache.max_cache_size_mb,
            "clear_cache_on_memory_pressure": self.expert.cache.clear_cache_on_memory_pressure,
            "small_lattice_fallback": self.expert.cache.small_lattice_fallback,
            "benchmark_small_lattices": self.expert.cache.benchmark_small_lattices,
            "total_cells": self.total_cells,
            "lattice_dimensions": self.lattice.dimensions,
        }


# === ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ===
_global_project_config: Optional["ProjectConfig"] = None


def get_project_config() -> "ProjectConfig":
    """
    Получить глобальную конфигурацию проекта.
    Создает новую если не существует.
    """
    global _global_project_config
    if _global_project_config is None:
        _global_project_config = ProjectConfig()
    return _global_project_config


def set_project_config(config: "ProjectConfig"):
    """Установить глобальную конфигурацию проекта."""
    global _global_project_config
    _global_project_config = config


def reset_global_config():
    """Сброс глобальной конфигурации (принудительное пересоздание)."""
    global _global_project_config
    _global_project_config = None  # Принудительно сбрасываем


# === Утилиты и хелперы (если нужны) ===


@dataclass
class ChunkInfo:
    """Информация о chunk'е решетки для spatial optimization"""

    chunk_id: int
    start_coords: Tuple[int, int, int]
    end_coords: Tuple[int, int, int]
    cell_indices: List[int]
    neighbor_chunks: List[int]
    memory_size_mb: float
    processing_time_ms: float = 0.0
