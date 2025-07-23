#!/usr/bin/env python3
"""
Модульные компоненты конфигурации для 3D Cellular Neural Network
==============================================================

Использует композицию вместо наследования для гибкости настроек.
Каждый компонент отвечает за свою область функциональности.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
from enum import Enum
import torch


# === РЕЖИМЫ КОНФИГУРАЦИИ ===


class ConfigMode(Enum):
    """Режимы работы конфигурации для исследовательского проекта"""

    DEBUG = "debug"  # Прогоночные тесты и отладка ошибок
    EXPERIMENT = "experiment"  # Эксперименты и исследования
    OPTIMIZED = "optimized"  # Финальный оптимизированный конфиг


@dataclass
class ModeSettings:
    """Настройки режима работы"""

    mode: ConfigMode = ConfigMode.DEBUG
    auto_apply_overrides: bool = True
    log_mode_info: bool = True


# === ОСНОВНЫЕ КОМПОНЕНТЫ ===


@dataclass
class LatticeSettings:
    """Настройки 3D решетки"""

    # Основные параметры - устанавливаются через пресеты
    dimensions: Tuple[int, int, int]

    # Адаптивный радиус - константы алгоритма (не дублирование!)
    adaptive_radius_enabled: bool = True
    adaptive_radius_ratio: float = 0.2
    adaptive_radius_max: float = 100.0
    adaptive_radius_min: float = 1.0

    # Классификация соединений - алгоритмические пропорции (не дублирование!)
    local_distance_ratio: float = (
        0.4  # 10% ближайших связей (от 0.0 до 0.1) для примера, может отличаться от текущих настроек
    )
    functional_distance_ratio: float = (
        0.80  # до 65% связей для functional (от 0.1 до 0.65)
    )
    distant_distance_ratio: float = 0.99  # до 100% всех связей (от 0.65 до 1.0)
    functional_similarity_threshold: float = 0.3
    # 0 ≤ LOCAL < local_distance_ratio*Adaptive_radius; local_distance_ratio*Adaptive_radius ≤ FUNCTIONAL: ≤ functional_distance_ratio*Adaptive_radius; functional_distance_ratio*Adaptive_radius < DISTANT ≤ distant_distance_ratio*Adaptive_radius

    # Пространственная оптимизация - алгоритмические константы
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
    """Настройки модели - общие параметры для всех экспертов"""

    # Основной параметр - устанавливается через пресеты
    state_size: int  # ОБЩИЙ для всех экспертов!

    # Динамический neighbor count - алгоритмическая константа
    neighbor_count: int = -1  # -1 означает динамическое определение
    external_input_size: int = 16

    # Архитектурные параметры - константы алгоритма
    # ПРИМЕЧАНИЕ: hidden_dim, message_dim, target_params перенесены в настройки экспертов
    activation: str = "gelu"
    use_attention: bool = True
    aggregation: str = "attention"
    num_layers: int = 1
    num_heads: int = 4
    use_layer_norm: bool = True
    dropout_rate: float = 0.1


@dataclass
class NeighborSettings:
    """Настройки поиска и классификации соседей"""

    # Стратегия поиска соседей
    finding_strategy: str = "tiered"
    dynamic_count: bool = True
    max_neighbors: int = 20000  # Биологический лимит

    # Adaptive Radius (дублирует LatticeSettings - убираем)
    # Используем значения из LatticeSettings

    # Tiered Topology - пропорции связей по типам (соответствуют LatticeSettings)
    local_tier: float = 0.1  # 10% связей
    functional_tier: float = 0.55  # 55% связей
    distant_tier: float = 0.35  # 35% связей

    # Порог функционального сходства
    functional_similarity_threshold: float = 0.3

    # Локальная сетка для оптимизации
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

    params: int  # Устанавливается через пресеты

    # Алгоритмические константы
    type: str = "linear"
    alpha: float = 0.1
    beta: float = 0.9

    # Параметры для настройки количества параметров модели
    neighbor_agg_hidden1: int = 32  # Можно менять для изменения размера модели
    neighbor_agg_hidden2: int = 16  # Можно менять для изменения размера модели
    processor_hidden: int = 64  # Можно менять для изменения размера модели

    max_neighbors_buffer: int = 200
    use_attention: bool = True
    default_batch_size: int = 1


@dataclass
class FunctionalExpertSettings:
    """Настройки для Functional Expert"""

    params: int  # Устанавливается через пресеты

    # Алгоритмические константы
    type: str = "hybrid_gnn_cnf"
    gnn_ratio: float = 0.6  # 60% параметров для GNN
    cnf_ratio: float = 0.4  # 40% параметров для CNF
    use_attention: bool = True

    # Параметры для настройки количества параметров модели
    message_dim: int = 16  # Можно менять для изменения размера модели
    hidden_dim: int = 32  # Можно менять для изменения размера модели

    integration_steps: int = 3
    adaptive_step_size: bool = True


@dataclass
class DistantExpertSettings:
    """Настройки для Distant Expert (CNF)"""

    params: int  # Устанавливается через пресеты

    # Алгоритмические константы
    type: str = "cnf"
    integration_steps: int = 3
    adaptive_step_size: bool = True
    batch_processing_mode: str = "ADAPTIVE_BATCH"
    max_batch_size: int = 1024
    adaptive_method: str = "LIPSCHITZ_BASED"
    memory_efficient: bool = True

    # Параметры для VectorizedNeuralODE (можно менять для изменения размера модели)
    ode_hidden_dim: Optional[int] = None  # None = max(16, state_size // 2)
    ode_dropout_rate: float = 0.1
    ode_damping_strength: float = 0.1
    ode_time_embedding_dim_ratio: float = 0.25  # hidden_dim // 4


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

    # Параметры, устанавливаемые через пресеты
    batch_size: int
    max_epochs: int
    num_epochs: int
    early_stopping_patience: int
    checkpoint_frequency: int

    # Алгоритмические константы
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_split: float = 0.2
    save_checkpoints: bool = True


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

    # Параметры, устанавливаемые через пресеты
    enable_detailed_stats: bool

    # Алгоритмические константы
    enabled: bool = True
    enable_performance_monitoring: bool = True
    small_lattice_fallback: bool = False  # Не используем fallback'и

    # GPU кэширование - константы конфигурации
    use_gpu_acceleration: bool = True
    gpu_batch_size: int = 10000
    prefer_gpu_for_large_lattices: bool = True
    gpu_memory_fraction: float = 0.8
    gpu_cache_size_mb: int = 1024

    # Размеры кэшей - константы конфигурации
    neighbor_cache_size: int = 100000
    connection_cache_enabled: bool = True
    persistent_cache_dir: str = "cache"

    # Legacy совместимость (вычисляются автоматически из LatticeSettings)
    local_radius: float = 1.0  # Будет переопределено
    functional_similarity_threshold: float = 0.3  # Дублирует LatticeSettings


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
    fallback_cpu: bool = False  # не используем в проекте fallbackи
    memory_fraction: float = 0.9
    allow_tf32: bool = True
    deterministic: bool = False
    device: str = "cuda"  # auto, cuda, cuda:0, cpu
    dtype: str = "float32"  # float32, float16, bfloat16
    compile_model: bool = False  # torch.compile для PyTorch 2.0+


@dataclass
class LoggingSettings:
    """Настройки логирования"""

    # Параметры, устанавливаемые через пресеты
    level: str
    debug_mode: bool
    enable_profiling: bool
    performance_tracking: bool
    debug_categories: List[str] = field(default_factory=list)

    # Константы конфигурации
    log_to_file: bool = True
    log_file: str = "logs/cnf_debug.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    enable_file_logging: bool = True

    # Предустановленные наборы категорий - алгоритмические константы
    CACHE_DEBUG = ["cache"]
    SPATIAL_DEBUG = ["spatial", "memory"]
    TRAINING_DEBUG = ["training", "forward"]
    INIT_DEBUG = ["init"]
    ALL_DEBUG = ["cache", "spatial", "forward", "memory", "training", "init", "verbose"]


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
    min_chunk_size: int = 8
    max_chunk_size: int = 16
    # max_chunk_size: int = 64  # Для тестовых (8,8,8) решеток
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
    teacher_dim: int = 768  # Размерность эмбеддингов от teacher модели (DistilBERT)
    input_dim: int = 768  # Входная размерность (совпадает с teacher_dim)
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

    # Параметры, устанавливаемые через пресеты
    test_mode: bool
    num_epochs: int
    max_total_samples: int

    # Алгоритмические константы - интервалы сохранения и валидации
    validation_interval: int = 1  # Валидация каждую эпоху
    save_checkpoint_every: int = 5  # Checkpoint каждые 5 эпох
    log_interval: int = 10  # Лог каждые 10 батчей
    early_stopping_patience: int = 10  # Early stopping

    # Loss weights для обучения - алгоритмические константы
    reconstruction_weight: float = 1.0  # Основная задача
    similarity_weight: float = 0.5  # Семантическая похожесть
    diversity_weight: float = 0.2  # Разнообразие представлений
    emergence_weight: float = 0.1  # Emergent behavior

    # Специфичные параметры для эмбеддингов - алгоритмические константы
    target_embedding_dim: int = 64  # Сжимаем 768 → 64
    teacher_model: str = "distilbert-base-uncased"
    use_teacher_forcing: bool = True
    lattice_steps: int = 5  # Количество шагов распространения

    # Curriculum learning - алгоритмические константы
    use_curriculum_learning: bool = False
    curriculum_start_difficulty: float = 0.1
    curriculum_end_difficulty: float = 1.0
    curriculum_schedule: str = "linear"

    # Батчи для RTX 5090 - константы конфигурации
    embedding_batch_size: int = 64  # Увеличено с 16 для лучшей GPU утилизации
    gradient_accumulation_steps: int = 2
    gpu_memory_reserve_gb: float = 20.0

    # Валидация - алгоритмические константы
    enable_semantic_validation: bool = True
    enable_probing_tasks: bool = False  # Отключено пока
    probing_tasks: List[str] = None
    visualization_enabled: bool = False
    visualization_frequency: int = 10

    # Тестовые параметры - алгоритмические константы
    test_lattice_dim: int = 8
    test_dataset_size: int = 5
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

    # Настройки для тестирования
    num_forward_passes: int = 1  # Количество forward pass'ов для теста стабильности
    stability_threshold: float = 0.1  # Порог стабильности (10% по умолчанию)


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


# === ЦЕНТРАЛИЗОВАННЫЕ ПАРАМЕТРЫ ДЛЯ HARDCODED ЗНАЧЕНИЙ ===


@dataclass
class TrainingOptimizerSettings:
    """Централизованные параметры оптимизатора и обучения"""

    # Оптимизатор
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Scheduler
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2

    # Gradient clipping
    gradient_clip_max_norm: float = 1.0

    # Частота логирования
    log_batch_frequency: int = 10

    # Веса для агрегации состояний
    surface_contribution_weight: float = 0.7
    volume_contribution_weight: float = 0.3


@dataclass
class EmbeddingMappingSettings:
    """Настройки маппинга эмбеддингов в решетку"""

    surface_coverage: float = 0.8
    lattice_steps: int = 5
    convergence_threshold: float = 1e-4

    # Веса для loss функций
    lattice_loss_weight: float = 0.1
    spatial_consistency_weight: float = 0.05

    # Архитектурные параметры
    attention_num_heads: int = 4
    dropout_rate: float = 0.1

    # Инициализация
    position_embedding_scale: float = 0.1

    # Размеры для uniform placement
    target_surface_points: int = 64


@dataclass
class MemoryManagementSettings:
    """Настройки управления памятью и производительностью"""

    # Garbage collection - будет переопределен из пресетов
    cleanup_threshold: int = 1000  # Значение по умолчанию, переопределяется пресетами

    # Минимальные требования
    min_gpu_memory_gb: float = 8.0

    # Лимиты логирования
    tensor_transfer_log_limit: int = 5

    # Размеры данных
    embedding_size_bytes: int = 768 * 4

    # Резервирование памяти
    training_memory_reserve_gb: float = 20.0
    memory_safety_factor: float = 0.8

    # Интервал maintenance tasks в секундах
    maintenance_interval_seconds: int = 666

    # DataLoader настройки
    dataloader_workers: int = 8
    prefetch_factor: int = 6

    # GPU memory safety
    gpu_memory_safety_factor: float = 0.85
    cpu_memory_safety_factor: float = 0.85


@dataclass
class ArchitectureConstants:
    """Архитектурные константы и параметры"""

    # MoE параметры экспертов
    moe_functional_params: int = 8000
    moe_distant_params: int = 4000
    moe_num_experts: int = 3
    moe_gating_hidden_dim: int = 64

    # CNF параметры
    cnf_max_batch_size: int = 100
    cnf_hidden_dim_ratio: float = 0.5  # hidden_dim = state_size * ratio
    cnf_min_hidden_dim: int = 16
    cnf_dropout_rate: float = 0.1
    cnf_damping_strength: float = 0.1
    cnf_target_params: int = 3000

    # Пространственная оптимизация
    spatial_cell_size: int = 2
    spatial_max_neighbors: int = 20000  # Биологический лимит
    spatial_log_frequency: int = 500
    spatial_chunk_overlap: int = 8

    # Алгоритмические константы
    spatial_consistency_range: int = 27  # 3x3x3 cube
    max_comparison_cells: int = 100

    # Stack search depth
    stack_search_depth: int = 15

    # EMA веса для функционального сходства
    ema_weight_current: float = 0.9
    ema_weight_history: float = 0.1

    # Teacher embedding dimension (например, от DistilBERT)
    teacher_embedding_dim: int = 768


@dataclass
class AlgorithmicStrategies:
    """Строковые константы и стратегии алгоритмов"""

    # Placement strategies
    placement_strategies: List[str] = field(
        default_factory=lambda: ["faces", "edges", "corners", "uniform"]
    )

    # Extraction strategies
    extraction_strategies: List[str] = field(
        default_factory=lambda: [
            "surface_mean",
            "weighted_surface",
            "volume_projection",
        ]
    )

    # CNF batch processing modes
    cnf_processing_modes: List[str] = field(
        default_factory=lambda: ["single", "batch", "adaptive"]
    )

    # Default strategies
    default_placement_strategy: str = "faces"
    default_extraction_strategy: str = "surface_mean"
    default_cnf_mode: str = "adaptive"


@dataclass
class ModePresets:
    """Предустановленные значения для разных режимов конфигурации"""

    @dataclass
    class DebugPreset:
        """Настройки для DEBUG режима - быстрые тесты и отладка"""

        # Lattice
        lattice_dimensions: Tuple[int, int, int] = (15, 15, 15)

        # Adaptive radius settings for small lattice
        lattice_adaptive_radius_ratio: float = (
            0.4  # Much smaller ratio for debug mode (15% of max dimension)
        )

        # Model
        model_state_size: int = 24  # Общий для всех экспертов

        # Training
        training_batch_size: int = 32  # Увеличено с 8 для лучшей GPU утилизации
        training_num_epochs: int = 5
        training_early_stopping_patience: int = 5
        training_checkpoint_frequency: int = 5
        training_max_samples: int = 50

        # Experts (уменьшены для DEBUG)
        expert_local_params: int = 1500  # было 2000
        expert_functional_params: int = 4000  # было 5000
        expert_distant_params: int = 2500  # было 3000

        # Дополнительные параметры экспертов для DEBUG
        # Local expert
        expert_local_neighbor_agg_hidden1: int = 12  # меньше для быстрых тестов
        expert_local_neighbor_agg_hidden2: int = 12  # меньше для быстрых тестов
        expert_local_processor_hidden: int = 12  # меньше для быстрых тестов

        # Functional expert
        expert_functional_hidden_dim: int = 12  # меньше для быстрых тестов
        expert_functional_message_dim: int = 8  # меньше для быстрых тестов

        # Distant expert (CNF)
        expert_distant_ode_hidden_dim: int = 48  # меньше для быстрых тестов
        expert_distant_ode_dropout_rate: float = 0.05  # меньше dropout для тестов

        # Gating network
        expert_gating_params: int = 800  # примерно
        expert_gating_hidden_dim: int = 24  # для достижения ~800 параметров

        # Memory & Performance
        memory_reserve_gb: float = 2.0
        dataloader_workers: int = 2
        cleanup_threshold: int = 5000  # Редкие cleanups для DEBUG

        # Logging
        logging_level: str = "DEBUG_INIT"
        logging_debug_mode: bool = False  # Отключено для лучшей производительности
        logging_enable_profiling: bool = True

        # Validation settings для тестов
        validation_num_forward_passes: int = (
            3  # Несколько проходов для теста стабильности
        )
        validation_stability_threshold: float = (
            0.15  # 15% допустимая вариация в debug режиме
        )

    @dataclass
    class ExperimentPreset:
        """Настройки для EXPERIMENT режима - исследования и эксперименты"""

        # Lattice
        lattice_dimensions: Tuple[int, int, int] = (30, 30, 30)

        # Model
        model_state_size: int = 64  # Общий для всех экспертов

        # Training
        training_batch_size: int = 48  # Увеличено с 16 для лучшей GPU утилизации
        training_num_epochs: int = 100
        training_early_stopping_patience: int = 10
        training_checkpoint_frequency: int = 25
        training_max_samples: int = 1000

        # Experts
        expert_local_params: int = 4000
        expert_functional_params: int = 8000
        expert_distant_params: int = 4000

        # Дополнительные параметры экспертов для EXPERIMENT
        # Local expert
        expert_local_neighbor_agg_hidden1: int = 32  # стандартные значения
        expert_local_neighbor_agg_hidden2: int = 16
        expert_local_processor_hidden: int = 64

        # Functional expert
        expert_functional_hidden_dim: int = 32  # стандартные значения
        expert_functional_message_dim: int = 16

        # Distant expert (CNF)
        expert_distant_ode_hidden_dim: Optional[int] = (
            None  # авто: max(16, state_size // 2)
        )
        expert_distant_ode_dropout_rate: float = 0.1  # стандартный dropout

        # Gating network
        expert_gating_params: int = 2000  # целевое значение
        expert_gating_hidden_dim: int = 64  # стандартное значение

        # Memory & Performance
        memory_reserve_gb: float = 10.0
        dataloader_workers: int = 4
        cleanup_threshold: int = 1000  # Умеренные cleanups для EXPERIMENT

        # Logging
        logging_level: str = "INFO"
        logging_debug_mode: bool = False
        logging_enable_profiling: bool = True

        # Validation settings для экспериментов
        validation_num_forward_passes: int = 5  # Больше проходов для экспериментов
        validation_stability_threshold: float = 0.1  # 10% стандартный порог

    @dataclass
    class OptimizedPreset:
        """Настройки для OPTIMIZED режима - финальные прогоны"""

        # Lattice
        lattice_dimensions: Tuple[int, int, int] = (50, 50, 50)

        # Model
        model_state_size: int = 128  # Общий для всех экспертов

        # Training
        training_batch_size: int = 64  # Увеличено с 32 для лучшей GPU утилизации
        training_num_epochs: int = 1000
        training_early_stopping_patience: int = 50
        training_checkpoint_frequency: int = 100
        training_max_samples: Optional[int] = None  # Без ограничений

        # Experts
        expert_local_params: int = 8000
        expert_functional_params: int = 15000
        expert_distant_params: int = 8000

        # Дополнительные параметры экспертов для OPTIMIZED
        # Local expert
        expert_local_neighbor_agg_hidden1: int = 64  # увеличены для производительности
        expert_local_neighbor_agg_hidden2: int = 32
        expert_local_processor_hidden: int = 128

        # Functional expert
        expert_functional_hidden_dim: int = 64  # увеличены для производительности
        expert_functional_message_dim: int = 32

        # Distant expert (CNF)
        expert_distant_ode_hidden_dim: int = 64  # увеличено для производительности
        expert_distant_ode_dropout_rate: float = 0.15  # больше regularization

        # Gating network
        expert_gating_params: int = 3000  # больше для сложных решений
        expert_gating_hidden_dim: int = 128  # увеличено для производительности

        # Memory & Performance
        memory_reserve_gb: float = 20.0
        dataloader_workers: int = 8
        cleanup_threshold: int = 10000  # Минимальные cleanups для OPTIMIZED

        # Logging
        logging_level: str = "WARNING"
        logging_debug_mode: bool = False
        logging_enable_profiling: bool = False

        # Validation settings для финальных прогонов
        validation_num_forward_passes: int = 10  # Много проходов для точной проверки
        validation_stability_threshold: float = 0.05  # 5% более строгий порог

    # Экземпляры пресетов
    debug: DebugPreset = field(default_factory=DebugPreset)
    experiment: ExperimentPreset = field(default_factory=ExperimentPreset)
    optimized: OptimizedPreset = field(default_factory=OptimizedPreset)


# === SPATIAL OPTIMIZATION HELPERS ===


@dataclass
class ChunkInfo:
    """Информация о чанке для пространственной оптимизации"""

    start: Tuple[int, int, int]
    end: Tuple[int, int, int]
    size: Tuple[int, int, int]
    chunk_id: int
    total_cells: int
    overlap: int = 0

    def contains_position(self, x: int, y: int, z: int) -> bool:
        """Проверка, содержит ли чанк данную позицию"""
        return (
            self.start[0] <= x < self.end[0]
            and self.start[1] <= y < self.end[1]
            and self.start[2] <= z < self.end[2]
        )


def create_spatial_config_for_lattice(
    lattice_dimensions: Tuple[int, int, int],
) -> Dict[str, Any]:
    """Создание конфигурации пространственной оптимизации для заданной решетки"""
    total_cells = lattice_dimensions[0] * lattice_dimensions[1] * lattice_dimensions[2]

    # Адаптивное определение размера чанка на основе общего размера решетки
    if total_cells <= 1000:
        chunk_size = min(8, min(lattice_dimensions))
    elif total_cells <= 10000:
        chunk_size = min(16, min(lattice_dimensions))
    elif total_cells <= 100000:
        chunk_size = min(32, min(lattice_dimensions))
    else:
        chunk_size = min(64, min(lattice_dimensions))

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": max(2, chunk_size // 8),
        "max_chunks_in_memory": 8 if total_cells > 10000 else 4,
        "enable_prefetching": total_cells > 1000,
        "prefetch_queue_size": 4 if total_cells > 10000 else 2,
        "memory_safety_factor": 0.75,
        "optimization_mode": "aggressive" if total_cells > 100000 else "balanced",
    }
