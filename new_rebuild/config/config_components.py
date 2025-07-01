#!/usr/bin/env python3
"""
Модульные компоненты конфигурации для 3D Cellular Neural Network
==============================================================

Использует композицию вместо наследования для гибкости настроек.
Каждый компонент отвечает за свою область функциональности.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import torch


# === ОСНОВНЫЕ КОМПОНЕНТЫ ===


@dataclass
class LatticeSettings:
    """Настройки 3D решетки"""

    # dimensions: Tuple[int, int, int] = (10, 10, 10)
    dimensions: Tuple[int, int, int] = (8, 8, 8)  # для прогоночных тестов
    # dimensions: Tuple[int, int, int] = (37, 37, 37)  # Для тестов с большой решеткой
    adaptive_radius_enabled: bool = True
    adaptive_radius_ratio: float = 0.2
    adaptive_radius_max: float = 100.0
    adaptive_radius_min: float = 0.1

    # Новые параметры для классификации соединений
    local_distance_ratio: float = (
        0.1  # - это промежуток от 0 до 10% связей(0.1 от максимального радиуса для конкретной решетки); связано с local_distance_ratio, так что важно не забыть их так же изменить
    )
    functional_distance_ratio: float = (
        0.65  # - это промежуток от 10% до 65% связей(0.55 от максимального радиуса для конкретной решетки); связано с functional_distance_ratio, так что важно не забыть их так же изменить
    )
    distant_distance_ratio: float = (
        1.0  # - это промежуток от 65% до 100% связей(0.35 от максимального радиуса для конкретной решетки); связано с distant_distance_ratio, так что важно не забыть их так же изменить
    )
    functional_similarity_threshold: float = 0.3

    # Настройки пространственной оптимизации (для MoE архитектуры)
    enable_morton_encoding: bool = True
    target_performance_ms: float = 16.67  # Target 60 FPS

    @property
    def total_cells(self) -> int:
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    @property
    def max_radius(self) -> float:
        """Максимальный радиус для данной решетки"""
        max_dim = max(self.dimensions)
        return max_dim * self.adaptive_radius_ratio

    @property
    def local_distance_threshold(self) -> float:
        """Автоматический расчет порога для local связей"""
        return self.max_radius * self.local_distance_ratio

    @property
    def functional_distance_threshold(self) -> float:
        """Автоматический расчет порога для functional связей"""
        return self.max_radius * self.functional_distance_ratio

    @property
    def distant_distance_threshold(self) -> float:
        """Автоматический расчет порога для distant связей"""
        return self.max_radius * self.distant_distance_ratio


@dataclass
class ModelSettings:
    """Настройки модели GNN"""

    # === РАЗМЕРЫ ДЛЯ РЕАЛЬНОГО ОБУЧЕНИЯ 8x8x8 ===
    state_size: int = 64  # Увеличено для emergent behavior
    message_dim: int = 32  # Больше для лучшего обмена информацией
    hidden_dim: int = 128  # Для RTX 5090 можем позволить больше
    # hidden_dim: int = 32          # Старое значение для маленьких тестов

    # === ДИНАМИЧЕСКИЙ NEIGHBOR COUNT ===
    # neighbor_count: int = 26      # Статическое значение - DEPRECATED
    neighbor_count: int = -1  # -1 означает динамическое определение
    # Для 8x8x8 куба будет ~26 соседей, для больших кубов больше

    external_input_size: int = 16  # Увеличено для эмбеддингов
    # external_input_size: int = 8  # Старое значение

    target_params: int = 50000  # Увеличено для реального обучения
    # target_params: int = 8000     # Старое значение для быстрых тестов

    activation: str = "gelu"
    use_attention: bool = True
    aggregation: str = "attention"
    num_layers: int = 1


@dataclass
class NeighborSettings:
    """Настройки поиска и классификации соседей"""

    # Стратегия поиска соседей
    finding_strategy: str = "tiered"
    dynamic_count: bool = True
    # base_neighbor_count: int = 26
    max_neighbors: int = 20000  # Биологический лимит

    # Adaptive Radius настройки
    adaptive_radius_enabled: bool = True
    adaptive_radius_ratio: float = 0.2
    adaptive_radius_max: float = 500.0
    adaptive_radius_min: float = 0.1

    # Tiered Topology - пропорции связей по типам
    local_tier: float = (
        0.1  # 10% связей; связано с local_distance_ratio, так что важно не забыть их так же изменить
    )
    functional_tier: float = (
        0.55  # 55% связей; связано с functional_distance_ratio, так что важно не забыть их так же изменить
    )
    distant_tier: float = (
        0.35  # 35% связей; связано с distant_distance_ratio, так что важно не забыть их так же изменить
    )

    # Пороги расстояний теперь вычисляются автоматически из LatticeSettings
    # через метод get_distance_thresholds(lattice_settings)
    functional_similarity_threshold: float = 0.3

    # Локальная сетка для тестов и оптимизации
    local_grid_cell_size: int = 8

    def get_distance_thresholds(
        self, lattice_settings: "LatticeSettings"
    ) -> Dict[str, float]:
        """Получение автоматически вычисленных порогов расстояний"""
        return {
            "local": lattice_settings.local_distance_threshold,
            "functional": lattice_settings.functional_distance_threshold,
            "distant": lattice_settings.distant_distance_threshold,
            "functional_similarity": self.functional_similarity_threshold,
        }


@dataclass
class LocalExpertSettings:
    """Настройки для Local Expert"""

    params: int = 2059
    type: str = "linear"
    alpha: float = 0.1
    beta: float = 0.9
    neighbor_agg_hidden1: int = 32
    neighbor_agg_hidden2: int = 16
    processor_hidden: int = 64
    max_neighbors_buffer: int = 100
    use_attention: bool = True
    default_batch_size: int = 1


@dataclass
class FunctionalExpertSettings:
    """Настройки для Functional Expert"""

    params: int = 8000
    type: str = "hybrid_gnn_cnf"
    gnn_ratio: float = 0.6  # 60% параметров для GNN
    cnf_ratio: float = 0.4  # 40% параметров для CNF
    use_attention: bool = True
    message_dim: int = 16
    hidden_dim: int = 32
    integration_steps: int = 3
    adaptive_step_size: bool = True


@dataclass
class DistantExpertSettings:
    """Настройки для Distant Expert (CNF)"""

    params: int = 4000
    type: str = "cnf"
    integration_steps: int = 3
    adaptive_step_size: bool = True
    batch_processing_mode: str = "ADAPTIVE_BATCH"
    max_batch_size: int = 1024
    adaptive_method: str = "LIPSCHITZ_BASED"
    memory_efficient: bool = True


@dataclass
class GatingNetworkSettings:
    """Настройки для Gating Network"""

    params: int = 808
    state_size: int = 32
    num_experts: int = 3
    hidden_dim: int = 64
    activation: str = "softmax"
    use_temperature: bool = True
    temperature: float = 1.0


@dataclass
class ExpertSettings:
    """Настройки для всех экспертов MoE системы"""

    local: LocalExpertSettings = field(default_factory=LocalExpertSettings)
    functional: FunctionalExpertSettings = field(
        default_factory=FunctionalExpertSettings
    )
    distant: DistantExpertSettings = field(default_factory=DistantExpertSettings)
    gating: GatingNetworkSettings = field(default_factory=GatingNetworkSettings)

    # Настройки кэширования связей
    cache: "CacheSettings" = None  # Будет инициализировано позже

    # Пропорции распределения связей
    connection_ratios: Dict[str, float] = field(
        default_factory=lambda: {
            "local": 0.1,
            "functional": 0.55,
            "distant": 0.35,
        }  # - это пропорции распределения связей; связано с connection_ratios, так что важно не забыть их так же изменить
    )


@dataclass
class TrainingSettings:
    """Настройки обучения"""

    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 50
    validation_split: float = 0.2
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100


@dataclass
class CNFSettings:
    """Настройки Continuous Normalizing Flows"""

    enabled: bool = True
    functional_connections: bool = True
    distant_connections: bool = True
    integration_steps: int = 3
    adaptive_step_size: bool = True
    target_params_per_connection: int = 3000
    batch_processing_mode: str = "ADAPTIVE_BATCH"
    max_batch_size: int = 1024
    adaptive_method: str = "LIPSCHITZ_BASED"


@dataclass
class EulerSettings:
    """Настройки Euler Solver"""

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
class CacheSettings:
    """Настройки кэширования"""

    enabled: bool = True
    enable_performance_monitoring: bool = True
    enable_detailed_stats: bool = False
    auto_enable_threshold: int = 3000
    auto_disable_threshold: int = 1000
    small_lattice_fallback: bool = False  # Использовать кэширование для маленьких решеток. во-первых, потому что мы тестируем на маленьких решетках, а во-вторых, потому что мы не используем fallback'и
    use_gpu_acceleration: bool = True
    gpu_batch_size: int = 10000
    prefer_gpu_for_large_lattices: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Эти значения вычисляются автоматически из LatticeSettings
    # но нужны для совместимости с legacy кодом
    local_radius: float = 1.0  # Будет переопределено из lattice settings
    functional_similarity_threshold: float = 0.3


@dataclass
class SpatialSettings:
    """Настройки пространственной оптимизации"""

    chunk_size: int = 64
    chunk_overlap: int = 8
    max_chunks_in_memory: int = 4
    memory_pool_size_gb: float = 12.0
    garbage_collect_frequency: int = 100
    prefetch_chunks: bool = True
    levels: int = 3
    min_cells_per_node: int = 1000
    max_search_radius: float = 50.0
    num_worker_threads: int = 4
    batch_size_per_thread: int = 10000
    enable_async_processing: bool = True
    enable_profiling: bool = True
    log_memory_usage: bool = True
    # toroidal_topology: bool = True  # Включить торическую топологию для увеличения соседей у граничных клеток


@dataclass
class UnifiedOptimizerSettings:
    """Настройки унифицированного оптимизатора"""

    performance_monitoring_enabled: bool = True
    detailed_timing: bool = False
    cache_statistics: bool = True
    memory_tracking: bool = True
    optimization_mode: str = "auto"  # auto, aggressive, balanced, conservative
    adaptive_mode_switching: bool = True
    target_performance_threshold_ms: float = 16.67  # 60 FPS
    fallback_threshold_ms: float = 33.33  # 30 FPS
    enable_adaptive_chunking: bool = True  # Включить адаптивное разбиение на чанки

    # Для создания optimization result
    neighbors_found_factor: float = 0.5  # коэффициент для расчета найденных соседей
    chunks_processed_div: int = 64  # делитель для расчета обработанных чанков


@dataclass
class VectorizedSettings:
    """Настройки векторизации"""

    enabled: bool = True
    force_vectorized: bool = True
    chunk_size: int = 1000
    parallel_processing: bool = True
    use_compiled_kernels: bool = True
    memory_efficient_mode: bool = True
    batch_norm_enabled: bool = True
    dropout_rate: float = 0.1


@dataclass
class InitSettings:
    """Настройки инициализации"""

    seed: int = 42
    reproducible: bool = True
    init_method: str = "xavier"  # xavier, kaiming, normal, uniform
    gain: float = 1.0


@dataclass
class DeviceSettings:
    """Настройки устройства"""

    prefer_cuda: bool = True
    # debug_mode: bool = False  # УДАЛЕНО - используем централизованный debug_mode из LoggingSettings
    fallback_cpu: bool = False # не используем в проекте fallbackи
    memory_fraction: float = 0.9
    allow_tf32: bool = True
    deterministic: bool = False


@dataclass
class LoggingSettings:
    """Настройки логирования"""

    level: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    # level: str = "DEBUG"  # Для тестов можно использовать DEBUG
    debug_mode: bool = False  # По умолчанию используем level, а не debug_mode
    log_to_file: bool = True
    log_file: str = "logs/cnf_debug.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_profiling: bool = True
    performance_tracking: bool = True


@dataclass
class MemorySettings:
    """Настройки управления памятью"""

    optimal_batch_size: int = 1000
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


@dataclass
class AdaptiveChunkerSettings:
    """Настройки адаптивного разбиения на чанки для GPU обработки"""

    max_chunks_in_memory: int = 8
    max_concurrent_chunks: int = 4
    chunk_overlap: int = 8
    min_chunk_size: int = 32
    # max_chunk_size: int = 256
    max_chunk_size: int = 64  # Для тестовых (8,8,8) решеток
    memory_safety_factor: float = 0.75
    enable_prefetching: bool = True
    prefetch_queue_size: int = 2
    max_history: int = 1000  # для AdaptiveMemoryPredictor

    # Для AdaptiveChunkInfo
    optimal_batch_size: int = 1000
    preferred_device: str = "cuda"  # auto, cuda, cpu

    # Для оптимизации параметров чанков
    optimal_batch_size_small: int = 100
    optimal_batch_size_medium: int = 500
    optimal_batch_size_large: int = 1000
    memory_pressure_high: float = 0.8
    memory_pressure_low: float = 0.3
    processing_priority_low_delta: int = 10
    processing_priority_high_delta: int = 10


# === СПЕЦИАЛИЗИРОВАННЫЕ КОМПОНЕНТЫ ===


@dataclass
class EmbeddingSettings:
    """Настройки для работы с эмбедингами"""

    # === НАСТРОЙКИ ДЛЯ РЕАЛЬНОГО ОБУЧЕНИЯ ===
    teacher_model: str = "distilbert-base-uncased"
    input_dim: int = 768  # DistilBERT dimension
    output_dim: int = 64  # Для куба 8x8x8 (512/8 = 64)
    teacher_embedding_dim: int = 768  # Старая совместимость

    # Параметры проекции для real training
    use_projection: bool = True
    projection_layers: List[int] = field(default_factory=lambda: [768, 256, 64])
    # projection_layers: List[int] = field(default_factory=lambda: [768, 128, 64])  # Альтернатива

    # Параметры преобразования
    transformation_type: str = "linear"  # linear, attention, autoencoder, hierarchical
    use_layer_norm: bool = True
    dropout_rate: float = 0.1  # Для стабильности
    # dropout_rate: float = 0.2           # Более агрессивное при overfitting
    use_residual_connections: bool = True

    # === КЭШИРОВАНИЕ ДЛЯ RTX 5090 ===
    cache_embeddings: bool = True
    cache_dir: str = "cache/embeddings"
    max_cache_size_gb: float = 24.0  # Увеличено для RTX 5090
    # max_cache_size_gb: float = 10.0     # Старое значение

    # Декодирование
    decoder_model: str = (
        "distilbert-base-uncased"  # Модель для декодирования обратно в текст
    )
    decoder_cache_enabled: bool = True
    max_decode_length: int = 512

    # Локальное кэширование моделей (для RTX 5090)
    local_models_dir: str = "models/local_cache"
    auto_download_models: bool = True  # Автоматически загружать если нет локально
    prefer_local_models: bool = True  # Предпочитать локальные модели


@dataclass
class TrainingEmbeddingSettings:
    """Расширенные настройки обучения для эмбедингов"""

    # === НАСТРОЙКИ ДЛЯ РЕАЛЬНОГО ОБУЧЕНИЯ 8x8x8 ===
    # test_mode: bool = True              # ТЕСТОВЫЙ РЕЖИМ
    test_mode: bool = False  # РЕАЛЬНОЕ ОБУЧЕНИЕ

    # Эпохи обучения для реального training
    num_epochs: int = 1  # Основные эпохи для первого эксперимента
    # warmup_epochs: int = 10             # Старые настройки
    # main_epochs: int = 100
    # fine_tune_epochs: int = 50

    # Интервалы сохранения и валидации
    validation_interval: int = 1  # Валидация каждую эпоху
    save_checkpoint_every: int = 5  # Checkpoint каждые 5 эпох
    log_interval: int = 10  # Лог каждые 10 батчей
    early_stopping_patience: int = 10  # Early stopping

    # === LOSS WEIGHTS ДЛЯ РЕАЛЬНОГО ОБУЧЕНИЯ ===
    reconstruction_weight: float = 1.0  # Основная задача
    similarity_weight: float = 0.5  # Семантическая похожесть
    diversity_weight: float = 0.2  # Разнообразие представлений
    emergence_weight: float = 0.1  # Emergent behavior

    # Старые названия для совместимости (DEPRECATED)
    # reconstruction_loss_weight: float = 1.0
    # similarity_loss_weight: float = 0.5
    # diversity_loss_weight: float = 0.1
    # emergence_loss_weight: float = 0.2

    # === СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ ДЛЯ ЭМБЕДДИНГОВ ===
    target_embedding_dim: int = 64  # Сжимаем 768 → 64 для куба 8x8x8
    teacher_model: str = "distilbert-base-uncased"
    use_teacher_forcing: bool = True
    lattice_steps: int = 5  # Количество шагов распространения

    # Curriculum learning
    use_curriculum_learning: bool = False  # Отключено для первого эксперимента
    # use_curriculum_learning: bool = True # Включить после успешных тестов
    curriculum_start_difficulty: float = 0.1
    curriculum_end_difficulty: float = 1.0
    curriculum_schedule: str = "linear"

    # === БАТЧИ ДЛЯ RTX 5090 ===
    embedding_batch_size: int = 16  # Оптимально для 32GB памяти
    # embedding_batch_size: int = 32      # Старое значение
    gradient_accumulation_steps: int = 2  # Уменьшено для стабильности

    # Валидация
    enable_semantic_validation: bool = True
    enable_probing_tasks: bool = False  # Отключено для первого эксперимента
    # enable_probing_tasks: bool = True   # Включить после стабилизации
    probing_tasks: List[str] = None
    visualization_enabled: bool = False
    visualization_frequency: int = 10

    # === ОГРАНИЧЕНИЯ ДАТАСЕТА ===
    # Общий лимит на количество сэмплов в датасете (для всех источников вместе)
    # None = без ограничений, int = максимальное количество сэмплов
    # max_total_samples: Optional[int] = None  # Для реального обучения без ограничений
    max_total_samples: int = 50  # Раскомментировать для ограниченного обучения
    
    # GPU память, резервируемая для обучения (GB)
    gpu_memory_reserve_gb: float = 20.0  # Оставляем 20GB для модели и обучения
    
    # === ТЕСТОВЫЕ ПАРАМЕТРЫ (ВРЕМЕННО) ===
    # Эти параметры используются только когда test_mode = True
    test_lattice_dim: int = 8
    # test_lattice_dim: int = 37
    test_dataset_size: int = (
        5  # Для прогоночного обучения используем только 658 сэмплов (из dialogue cache)
    )
    # test_dataset_size: int = 1000  # Старое значение для быстрых тестов
    test_validation_split: float = 0.2
    test_quick_iterations: int = 10

    def __post_init__(self):
        if self.probing_tasks is None:
            self.probing_tasks = ["sentiment", "similarity"]


@dataclass
class ExperimentSettings:
    """Настройки для экспериментов"""

    experiment_name: str = "default"
    save_results: bool = True
    results_dir: str = "results"
    random_seed: int = 42
    reproducible: bool = True
    track_metrics: bool = True
    save_checkpoints: bool = True
    visualize_results: bool = False


@dataclass
class PerformanceSettings:
    """Настройки производительности"""

    enable_jit: bool = False
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    dataloader_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    benchmark_mode: bool = False
    profiling_enabled: bool = False


@dataclass
class ValidationSettings:
    """Настройки валидации"""

    validate_config: bool = True
    strict_validation: bool = False
    warn_on_deprecated: bool = True
    auto_fix_conflicts: bool = True
    check_memory_requirements: bool = True
    estimate_compute_time: bool = False


@dataclass
class ConnectionSettings:
    """Настройки связей между клетками"""
    
    # Базовые параметры связей
    strength: float = 1.0  # Стандартная сила связи
    functional_similarity: float = 0.3  # Порог функционального сходства
    
    # Дополнительные параметры для будущего расширения
    decay_factor: float = 0.9  # Коэффициент затухания связи
    min_strength: float = 0.1  # Минимальная сила связи
    max_strength: float = 5.0  # Максимальная сила связи


# === ФУНКЦИИ КОМПОЗИЦИИ ===


def create_basic_config() -> Dict[str, Any]:
    """Создает базовую конфигурацию с минимальными настройками"""
    return {
        "lattice": LatticeSettings(),
        "model": ModelSettings(),
        "training": TrainingSettings(),
        "device": DeviceSettings(),
        "logging": LoggingSettings(),
    }


def create_research_config() -> Dict[str, Any]:
    """Создает конфигурацию для исследований"""
    config = create_basic_config()

    # Инициализируем expert settings с правильной ссылкой на cache
    cache_settings = CacheSettings()
    expert_settings = ExpertSettings()
    expert_settings.cache = cache_settings

    config.update(
        {
            "cnf": CNFSettings(),
            "euler": EulerSettings(),
            "cache": cache_settings,
            "spatial": SpatialSettings(),
            "vectorized": VectorizedSettings(),
            "experiment": ExperimentSettings(),
            "performance": PerformanceSettings(),
            "neighbors": NeighborSettings(),
            "expert": expert_settings,
        }
    )
    return config


def create_production_config() -> Dict[str, Any]:
    """Создает конфигурацию для production"""
    config = create_basic_config()

    # Режим production - минимум логов
    config["logging"].debug_mode = False  # Единственное место управления debug_mode
    config["logging"].level = "WARNING"
    # config["device"].debug_mode = False  # УДАЛЕНО - больше не существует

    # Отключаем детализированные логи
    config["cache"].enable_detailed_stats = False
    config["spatial"].enable_profiling = False
    config["unified_optimizer"].performance_monitoring_enabled = False

    # Оптимизируем производительность
    config["vectorized"].force_vectorized = True
    config["cache"].use_gpu_acceleration = True

    return config


def validate_config_components(config: Dict[str, Any]) -> bool:
    """Валидирует компоненты конфигурации на совместимость"""
    try:
        # Проверяем базовые требования
        if "lattice" not in config or "model" not in config:
            return False

        lattice = config["lattice"]
        model = config["model"]

        from ..utils.logging import get_logger

        logger = get_logger(__name__)
        if (
            lattice.total_cells > 10000
            and not config.get("cache", CacheSettings()).enabled
        ):
            logger.warning("WARNING: Large lattice without cache may be slow")

        # Проверяем совместимость модели
        if model.state_size < 8:
            logger.warning("WARNING: Very small state_size may limit model capacity")

        return True

    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return False


def merge_config_updates(
    base_config: Dict[str, Any], updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Безопасно объединяет обновления конфигурации"""
    result = base_config.copy()

    for key, value in updates.items():
        if key in result:
            if hasattr(result[key], "__dict__") and hasattr(value, "__dict__"):
                # Обновляем dataclass
                for attr, new_val in value.__dict__.items():
                    if hasattr(result[key], attr):
                        setattr(result[key], attr, new_val)
            else:
                result[key] = value
        else:
            result[key] = value

    return result
