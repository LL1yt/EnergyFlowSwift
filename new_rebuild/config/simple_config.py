#!/usr/bin/env python3
"""
Упрощенная централизованная конфигурация - Clean Architecture
============================================================

Использует композицию вместо глубокой вложенности dataclass'ов.
Простая, расширяемая и поддерживаемая архитектура конфигурации.

Принципы:
- Композиция > Наследование
- Простота > Сложность
- Производительность > Перфекционизм
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

from .config_components import (
    # Режимы конфигурации
    ConfigMode,
    ModeSettings,
    # Основные компоненты
    LatticeSettings,
    ModelSettings,
    TrainingSettings,
    CNFSettings,
    EulerSettings,
    CacheSettings,
    SpatialSettings,
    UnifiedOptimizerSettings,
    VectorizedSettings,
    InitSettings,
    DeviceSettings,
    LoggingSettings,
    MemorySettings,
    AdaptiveChunkerSettings,
    ExperimentSettings,
    PerformanceSettings,
    ValidationSettings,
    ConnectionSettings,
    EmbeddingSettings,
    TrainingEmbeddingSettings,
    NeighborSettings,
    ExpertSettings,
    LocalExpertSettings,
    FunctionalExpertSettings,
    DistantExpertSettings,
    GatingNetworkSettings,
    # Новые компоненты для hardcoded значений
    TrainingOptimizerSettings,
    EmbeddingMappingSettings,
    MemoryManagementSettings,
    ArchitectureConstants,
    # Spatial optimization helpers
    ChunkInfo,
    create_spatial_config_for_lattice,
    AlgorithmicStrategies,
    ModePresets,
    # Функции
    create_basic_config,
    create_research_config,
    validate_config_components,
)
from ..utils.device_manager import DeviceManager, get_device_manager


@dataclass
class SimpleProjectConfig:
    """
    Упрощенная конфигурация проекта использующая композицию

    Вместо глубокой вложенности dataclass'ов использует
    простые компоненты через композицию.
    """

    # Режим работы конфигурации
    mode: ModeSettings = field(default_factory=ModeSettings)
    
    # Основные компоненты (будут инициализированы в __post_init__)
    lattice: LatticeSettings = field(init=False)
    model: ModelSettings = field(init=False)
    training: TrainingSettings = field(init=False)
    logging: LoggingSettings = field(init=False)
    cache: CacheSettings = field(init=False)
    training_embedding: TrainingEmbeddingSettings = field(init=False)
    
    # Остальные компоненты с обычной инициализацией
    init: InitSettings = field(default_factory=InitSettings)
    device: DeviceSettings = field(default_factory=DeviceSettings)

    # Дополнительные компоненты (опциональные)
    cnf: Optional[CNFSettings] = field(default_factory=CNFSettings)
    euler: Optional[EulerSettings] = field(default_factory=EulerSettings)
    spatial: Optional[SpatialSettings] = field(default_factory=SpatialSettings)
    unified_optimizer: Optional[UnifiedOptimizerSettings] = field(
        default_factory=UnifiedOptimizerSettings
    )
    vectorized: Optional[VectorizedSettings] = field(default_factory=VectorizedSettings)
    memory: Optional[MemorySettings] = field(default_factory=MemorySettings)
    adaptive_chunker: Optional[AdaptiveChunkerSettings] = field(
        default_factory=AdaptiveChunkerSettings
    )

    # Экспериментальные компоненты
    experiment: Optional[ExperimentSettings] = None
    performance: Optional[PerformanceSettings] = None
    validation: Optional[ValidationSettings] = None

    # Компоненты для работы с эмбедингами
    embedding: Optional[EmbeddingSettings] = field(default_factory=EmbeddingSettings)

    # MoE компоненты
    neighbors: Optional[NeighborSettings] = field(default_factory=NeighborSettings)
    expert: Optional[ExpertSettings] = field(init=False)  # Инициализируем в __post_init__
    connection: Optional[ConnectionSettings] = field(default_factory=ConnectionSettings)
    
    # Централизованные параметры (для миграции hardcoded значений)
    training_optimizer: Optional[TrainingOptimizerSettings] = field(default_factory=TrainingOptimizerSettings)
    embedding_mapping: Optional[EmbeddingMappingSettings] = field(default_factory=EmbeddingMappingSettings)
    memory_management: Optional[MemoryManagementSettings] = field(default_factory=MemoryManagementSettings)
    architecture: Optional[ArchitectureConstants] = field(default_factory=ArchitectureConstants)
    strategies: Optional[AlgorithmicStrategies] = field(default_factory=AlgorithmicStrategies)
    
    # Предустановленные значения для режимов
    mode_presets: Optional[ModePresets] = field(default_factory=ModePresets)

    # Runtime компоненты (вычисляются автоматически)
    device_manager: Optional[DeviceManager] = field(init=False, default=None)

    def __post_init__(self):
        """Автоматическая настройка после инициализации"""
        # СНАЧАЛА инициализируем компоненты с пресетами
        self._initialize_components_from_presets()
        
        # Применяем настройки режима ПЕРЕД всем остальным
        if self.mode.auto_apply_overrides:
            self._apply_mode_settings()
            
        # ВАЖНО: Сначала настраиваем логирование на основе наших настроек
        from ..utils.logging import setup_logging
        setup_logging(
            debug_mode=self.logging.debug_mode,
            level=self.logging.level,  # Передаем уровень из конфигурации
            log_file=self.logging.log_file if self.logging.log_to_file else None,
            enable_deduplication=False,
            enable_context=True,
            debug_categories=self.logging.debug_categories  # Передаем категории debug
        )
        
        # Инициализация device manager с централизованным debug_mode из logging настроек
        self.device_manager = get_device_manager(
            prefer_cuda=self.device.prefer_cuda, 
            debug_mode=self.logging.debug_mode
        )

        # Связываем cache с expert settings
        if self.expert and self.cache:
            self.expert.cache = self.cache

        # СТРОГАЯ ВАЛИДАЦИЯ - всегда выполняется
        # В соответствии с принципом: "лучше явная ошибка, чем скрытая проблема"
        self._validate_configuration()

        # Логирование инициализации
        if self.logging.debug_mode:
            self._log_initialization()
            
    def _apply_mode_settings(self):
        """Применить настройки в зависимости от режима"""
        global _global_migration_warned
        
        # В DEBUG режиме сбрасываем флаг, чтобы показывать предупреждения каждый раз
        if self.mode.mode == ConfigMode.DEBUG:
            _global_migration_warned = False
        
        # Показываем предупреждение только один раз за всю сессию (кроме DEBUG режима)
        if not _global_migration_warned:
            import warnings
            warnings.warn(
                "\n⚠️ НАПОМИНАНИЕ: Режимы конфигурации используют централизованные пресеты!\n"
                "✅ Это хорошо, но помните:\n"
                "   1. Всегда используйте значения из config вместо hardcoded в вашем коде\n"
                "   2. Применяйте @no_hardcoded декоратор к новым функциям\n"
                "   3. Используйте strict_no_hardcoded() для автоматической замены\n"
                "📝 Пресеты находятся в config.mode_presets.{debug|experiment|optimized}",
                UserWarning,
                stacklevel=4
            )
            _global_migration_warned = True
            
        if self.mode.mode == ConfigMode.DEBUG:
            self._apply_debug_mode()
        elif self.mode.mode == ConfigMode.EXPERIMENT:
            self._apply_experiment_mode()
        elif self.mode.mode == ConfigMode.OPTIMIZED:
            self._apply_optimized_mode()
            
        # Логируем информацию о режиме
        if self.mode.log_mode_info:
            logging.info(f"🎯 Config mode: {self.mode.mode.value.upper()}")
            
    def _initialize_components_from_presets(self):
        """Инициализируем компоненты с значениями из пресетов"""
        preset = self._get_current_preset()
        
        # Создаем компоненты с обязательными параметрами из пресетов
        self.lattice = LatticeSettings(dimensions=preset.lattice_dimensions)
        
        self.model = ModelSettings(
            state_size=preset.model_state_size
        )
        
        self.training = TrainingSettings(
            batch_size=preset.training_batch_size,
            max_epochs=preset.training_num_epochs,
            num_epochs=preset.training_num_epochs,
            early_stopping_patience=preset.training_early_stopping_patience,
            checkpoint_frequency=preset.training_checkpoint_frequency
        )
        
        self.logging = LoggingSettings(
            level=preset.logging_level,
            debug_mode=preset.logging_debug_mode,
            enable_profiling=preset.logging_enable_profiling,
            performance_tracking=preset.logging_enable_profiling
        )
        
        self.cache = CacheSettings(
            enable_detailed_stats=preset.logging_debug_mode  # Связано с debug режимом
        )
        
        self.training_embedding = TrainingEmbeddingSettings(
            test_mode=preset.logging_debug_mode,  # DEBUG = test_mode
            num_epochs=preset.training_num_epochs,
            max_total_samples=preset.training_max_samples
        )
        
        # Expert settings
        local_expert = LocalExpertSettings(
            params=preset.expert_local_params,
            neighbor_agg_hidden1=getattr(preset, 'expert_local_neighbor_agg_hidden1', 32),
            neighbor_agg_hidden2=getattr(preset, 'expert_local_neighbor_agg_hidden2', 16),
            processor_hidden=getattr(preset, 'expert_local_processor_hidden', 64)
        )
        functional_expert = FunctionalExpertSettings(
            params=preset.expert_functional_params,
            hidden_dim=getattr(preset, 'expert_functional_hidden_dim', 32),
            message_dim=getattr(preset, 'expert_functional_message_dim', 16)
        )
        distant_expert = DistantExpertSettings(
            params=preset.expert_distant_params,
            ode_hidden_dim=getattr(preset, 'expert_distant_ode_hidden_dim', None),
            ode_dropout_rate=getattr(preset, 'expert_distant_ode_dropout_rate', 0.1)
        )
        
        # Gating network settings
        gating_network = GatingNetworkSettings(
            params=getattr(preset, 'expert_gating_params', 808),
            state_size=preset.model_state_size,  # Используем общий state_size
            hidden_dim=getattr(preset, 'expert_gating_hidden_dim', 64)
        )
        
        self.expert = ExpertSettings(
            local=local_expert,
            functional=functional_expert,
            distant=distant_expert,
            gating=gating_network
        )
        
        # Validation settings из пресетов
        self.validation = ValidationSettings(
            num_forward_passes=getattr(preset, 'validation_num_forward_passes', 1),
            stability_threshold=getattr(preset, 'validation_stability_threshold', 0.1)
        )
        
    def _get_current_preset(self):
        """Получить текущий пресет на основе режима"""
        if not hasattr(self, 'mode_presets') or self.mode_presets is None:
            from .config_components import ModePresets
            self.mode_presets = ModePresets()
            
        if self.mode.mode == ConfigMode.DEBUG:
            return self.mode_presets.debug
        elif self.mode.mode == ConfigMode.EXPERIMENT:
            return self.mode_presets.experiment
        elif self.mode.mode == ConfigMode.OPTIMIZED:
            return self.mode_presets.optimized
        else:
            return self.mode_presets.debug  # По умолчанию
            
    def _apply_debug_mode(self):
        """Режим отладки - дополнительные настройки"""
        # Основные параметры уже установлены в _initialize_components_from_presets
        preset = self.mode_presets.debug
        
        # Дополнительные настройки для DEBUG режима
        self.logging.debug_categories = ['cache', 'init', 'training']
        
        # Architecture (для совместимости с legacy кодом)
        self.architecture.moe_functional_params = preset.expert_functional_params
        self.architecture.moe_distant_params = preset.expert_distant_params
        
        # Memory & Performance
        self.memory_management.training_memory_reserve_gb = preset.memory_reserve_gb
        self.memory_management.dataloader_workers = preset.dataloader_workers
        self.memory_management.cleanup_threshold = preset.cleanup_threshold
        
        # Override adaptive radius for small debug lattice
        if hasattr(preset, 'lattice_adaptive_radius_ratio'):
            self.lattice.adaptive_radius_ratio = preset.lattice_adaptive_radius_ratio
        
    def _apply_experiment_mode(self):
        """Режим экспериментов - дополнительные настройки"""
        # Основные параметры уже установлены в _initialize_components_from_presets
        preset = self.mode_presets.experiment
        
        # Дополнительные настройки для EXPERIMENT режима
        self.logging.debug_categories = self.logging.TRAINING_DEBUG
        
        # Architecture (для совместимости с legacy кодом)
        self.architecture.moe_functional_params = preset.expert_functional_params
        self.architecture.moe_distant_params = preset.expert_distant_params
        
        # Memory & Performance
        self.memory_management.training_memory_reserve_gb = preset.memory_reserve_gb
        self.memory_management.dataloader_workers = preset.dataloader_workers
        self.memory_management.cleanup_threshold = preset.cleanup_threshold
        
    def _apply_optimized_mode(self):
        """Финальный оптимизированный режим - дополнительные настройки"""
        # Основные параметры уже установлены в _initialize_components_from_presets
        preset = self.mode_presets.optimized
        
        # Дополнительные настройки для OPTIMIZED режима
        self.logging.performance_tracking = False
        self.logging.debug_categories = []
        
        # Architecture (для совместимости с legacy кодом)
        self.architecture.moe_functional_params = preset.expert_functional_params
        self.architecture.moe_distant_params = preset.expert_distant_params
        
        # Memory & Performance
        self.memory_management.training_memory_reserve_gb = preset.memory_reserve_gb
        self.memory_management.dataloader_workers = preset.dataloader_workers
        self.memory_management.cleanup_threshold = preset.cleanup_threshold
        
        # Включить оптимизации производительности
        if self.performance is None:
            self.performance = PerformanceSettings()
        self.performance.enable_jit = True
        self.performance.benchmark_mode = True

    def _validate_configuration(self):
        """Строгая валидация конфигурации без fallback'ов"""
        # Импортируем валидатор
        from .config_validator import ConfigValidator
        
        # Запускаем полную валидацию
        ConfigValidator.validate_full_config(self)
        
        # Дополнительная бизнес-логика валидации
        try:
            # Проверяем большие решетки без кэша
            if self.lattice.total_cells > 5000 and (
                not self.cache or not self.cache.enabled
            ):
                logging.warning(
                    f"Large lattice ({self.lattice.total_cells} cells) "
                    "without cache may be slow"
                )

            # Проверяем совместимость размеров
            if self.model.state_size < 8:
                logging.warning("Very small state_size may limit model capacity")

            # Проверяем GPU настройки
            if (
                self.cache
                and self.cache.use_gpu_acceleration
                and not self.device.prefer_cuda
            ):
                logging.warning("GPU acceleration enabled but CUDA not preferred")

        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")

    def _log_initialization(self):
        """Логирование информации о конфигурации"""
        logging.info("🔧 SimpleProjectConfig initialized:")
        logging.info(
            f"   📊 Lattice: {self.lattice.dimensions} = {self.lattice.total_cells} cells"
        )
        logging.info(
            f"   🧠 Model: {self.model.state_size}D state (общий для всех экспертов)"
        )
        logging.info(f"   🖥️ Device: {self.device_manager.get_device_str()}")

        if self.cache and self.cache.enabled:
            logging.info(
                f"   💾 Cache: enabled (GPU: {self.cache.use_gpu_acceleration})"
            )
        if self.cnf and self.cnf.enabled:
            logging.info(f"   🌊 CNF: enabled ({self.cnf.adaptive_method})")
        if self.embedding:
            logging.info(
                f"   🎯 Embeddings: {self.embedding.teacher_model} ({self.embedding.teacher_embedding_dim}D → {self.cube_embedding_dim}D)"
            )

    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.lattice.total_cells

    @property
    def current_device(self) -> str:
        """Текущее устройство"""
        return self.device_manager.get_device_str() if self.device_manager else "cpu"

    @property
    def max_neighbors(self) -> int:
        """Максимальное количество соседей (для обратной совместимости)"""
        return self.neighbors.max_neighbors if self.neighbors else 20000

    @property
    def cube_surface_dim(self) -> int:
        """Размерность поверхности куба (первые две размерности решетки)"""
        return self.lattice.dimensions[0]  # Предполагаем кубическую решетку

    @property
    def cube_embedding_dim(self) -> int:
        """Размерность поверхностных эмбедингов (surface_dim²)"""
        surface_dim = self.cube_surface_dim
        return surface_dim * surface_dim

    @property
    def effective_max_chunk_size(self) -> int:
        """Эффективный максимальный размер chunk'а для текущей решетки"""
        config_max = (
            self.adaptive_chunker.max_chunk_size if self.adaptive_chunker else 64
        )
        # Для малых решеток ограничиваем chunk размером в 1/4 решетки по каждой оси
        max_dim = max(self.lattice.dimensions)
        quarter_lattice = max_dim // 4
        return min(config_max, max(quarter_lattice, 4))  # минимум 4 клетки

    @property
    def effective_min_chunk_size(self) -> int:
        """Эффективный минимальный размер chunk'а для текущей решетки"""
        config_min = (
            self.adaptive_chunker.min_chunk_size if self.adaptive_chunker else 32
        )
        # Для малых решеток делаем min_chunk_size = 1/8 от максимального измерения
        max_dim = max(self.lattice.dimensions)
        eighth_lattice = max(max_dim // 8, 2)  # минимум 2 клетки
        return min(config_min, eighth_lattice)

    def calculate_adaptive_radius(self) -> float:
        """
        Вычисляет адаптивный радиус на основе размеров решетки.
        Формула: (max_dim * ratio), ограниченный min/max.
        """
        if not self.lattice.adaptive_radius_enabled:
            return self.lattice.adaptive_radius_max

        max_dim = max(self.lattice.dimensions)
        radius = max_dim * self.lattice.adaptive_radius_ratio

        return float(
            max(
                self.lattice.adaptive_radius_min,
                min(radius, self.lattice.adaptive_radius_max),
            )
        )
    
    def estimate_neighbors_in_radius(self, radius: float) -> int:
        """
        Оценка количества соседей в заданном радиусе для 3D решетки.
        
        Использует формулу объема сферы с учетом дискретности решетки.
        
        Args:
            radius: Радиус для подсчета соседей
            
        Returns:
            Примерное количество соседей
        """
        import math
        
        if radius <= 0:
            return 0
        
        # Объем сферы: 4/3 * π * r³
        volume = (4/3) * math.pi * (radius ** 3)
        
        # Ограничиваем максимальным количеством клеток минус 1 (исключаем саму клетку)
        total_cells = self.lattice.dimensions[0] * self.lattice.dimensions[1] * self.lattice.dimensions[2]
        estimated = min(int(volume), total_cells - 1)
        
        return estimated
    
    def get_neighbor_counts_by_type(self) -> Dict[str, int]:
        """
        Получает примерное количество соседей для каждого типа связей
        на основе адаптивного радиуса и порогов классификации.
        
        Returns:
            Словарь с количеством соседей по типам: {local, functional, distant}
        """
        adaptive_radius = self.calculate_adaptive_radius()
        
        # Вычисляем пороги
        local_threshold = adaptive_radius * self.lattice.local_distance_ratio
        functional_threshold = adaptive_radius * self.lattice.functional_distance_ratio
        distant_threshold = adaptive_radius * self.lattice.distant_distance_ratio
        
        # Оцениваем количество соседей для каждого порога
        local_neighbors = self.estimate_neighbors_in_radius(local_threshold)
        functional_neighbors = self.estimate_neighbors_in_radius(functional_threshold) - local_neighbors
        distant_neighbors = self.estimate_neighbors_in_radius(distant_threshold) - local_neighbors - functional_neighbors
        
        return {
            "local": local_neighbors,
            "functional": functional_neighbors,
            "distant": distant_neighbors,
            "total": local_neighbors + functional_neighbors + distant_neighbors
        }

    def get_component(self, name: str) -> Optional[Any]:
        """Получить компонент конфигурации по имени"""
        return getattr(self, name, None)

    def update_component(self, name: str, **kwargs) -> bool:
        """Обновить настройки компонента"""
        try:
            component = self.get_component(name)
            if component is None:
                return False

            for key, value in kwargs.items():
                if hasattr(component, key):
                    setattr(component, key, value)

            return True
        except Exception:
            return False

    def enable_research_mode(self):
        """Включить режим исследований"""
        # Включаем все дополнительные компоненты
        if self.experiment is None:
            self.experiment = ExperimentSettings()
        if self.performance is None:
            self.performance = PerformanceSettings()
        if self.validation is None:
            self.validation = ValidationSettings()

        # Настраиваем для исследований
        self.logging.debug_mode = False  # Отключено для лучшей производительности
        self.logging.performance_tracking = True
        if self.cache:
            self.cache.enable_detailed_stats = True
        if self.performance:
            self.performance.profiling_enabled = True

    def enable_production_mode(self):
        """Включить режим production"""
        # Отключаем debug режимы - централизованно через LoggingSettings
        self.logging.debug_mode = False
        self.logging.level = "WARNING"
        # self.device.debug_mode = False  # УДАЛЕНО - больше не существует

        # Оптимизируем производительность
        if self.performance is None:
            self.performance = PerformanceSettings()
        self.performance.enable_jit = True
        self.performance.benchmark_mode = True

        # Ограничиваем детализацию
        if self.cache:
            self.cache.enable_detailed_stats = False
        self.logging.performance_tracking = False

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        result = {}

        # Основные компоненты
        for field_name in ["lattice", "model", "training", "init", "device", "logging"]:
            component = getattr(self, field_name)
            result[field_name] = (
                component.__dict__ if hasattr(component, "__dict__") else component
            )

        # Дополнительные компоненты (только если присутствуют)
        for field_name in [
            "cnf",
            "euler",
            "cache",
            "spatial",
            "unified_optimizer",
            "vectorized",
            "memory",
            "adaptive_chunker",
            "experiment",
            "performance",
            "validation",
            "embedding",
            "training_embedding",
            "neighbors",
            "expert",
        ]:
            component = getattr(self, field_name)
            if component is not None:
                result[field_name] = (
                    component.__dict__ if hasattr(component, "__dict__") else component
                )

        return result


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===

_global_config: Optional[SimpleProjectConfig] = None

# Глобальный флаг для предупреждения о миграции (показываем только раз за сессию)
_global_migration_warned: bool = False


def get_project_config() -> SimpleProjectConfig:
    """Получить глобальный экземпляр конфигурации"""
    global _global_config
    if _global_config is None:
        import warnings
        warnings.warn(
            "⚠️ Глобальный конфиг не был инициализирован явно! "
            "Создаем конфиг по умолчанию (DEBUG режим). "
            "Рекомендуется использовать set_project_config() или create_*_config() функции "
            "в начале вашего скрипта для явной инициализации.",
            stacklevel=2
        )
        _global_config = create_debug_config()
    return _global_config


def set_project_config(config: SimpleProjectConfig):
    """Установить глобальный экземпляр конфигурации"""
    global _global_config
    _global_config = config


def reset_project_config():
    """Сбросить глобальную конфигурацию"""
    global _global_config
    _global_config = None


def reset_migration_warning():
    """Сбросить флаг предупреждения о миграции (для тестов)"""
    global _global_migration_warned
    _global_migration_warned = False


# === ФАБРИЧНЫЕ ФУНКЦИИ ===


def create_simple_config(**overrides) -> SimpleProjectConfig:
    """Создать простую конфигурацию с переопределениями"""
    config = SimpleProjectConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_debug_config(**overrides) -> SimpleProjectConfig:
    """Создать конфиг для отладки и быстрых тестов"""
    # Создаем ModeSettings с нужным режимом
    mode_settings = ModeSettings(mode=ConfigMode.DEBUG)
    
    # Создаем конфиг с этим режимом
    config = SimpleProjectConfig(mode=mode_settings)
    
    # Применяем дополнительные переопределения
    for key, value in overrides.items():
        if hasattr(config, key):
            # Если это словарь с настройками компонента
            if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                component = getattr(config, key)
                for k, v in value.items():
                    if hasattr(component, k):
                        setattr(component, k, v)
            else:
                setattr(config, key, value)
    
    return config


def create_experiment_config(**overrides) -> SimpleProjectConfig:
    """Создать конфиг для экспериментов"""
    # Создаем ModeSettings с нужным режимом
    mode_settings = ModeSettings(mode=ConfigMode.EXPERIMENT)
    
    # Создаем конфиг с этим режимом
    config = SimpleProjectConfig(mode=mode_settings)
    
    # Применяем дополнительные переопределения
    for key, value in overrides.items():
        if hasattr(config, key):
            # Если это словарь с настройками компонента
            if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                component = getattr(config, key)
                for k, v in value.items():
                    if hasattr(component, k):
                        setattr(component, k, v)
            else:
                setattr(config, key, value)
    
    return config


def create_optimized_config(**overrides) -> SimpleProjectConfig:
    """Создать оптимизированный конфиг"""
    # Создаем ModeSettings с нужным режимом
    mode_settings = ModeSettings(mode=ConfigMode.OPTIMIZED)
    
    # Создаем конфиг с этим режимом
    config = SimpleProjectConfig(mode=mode_settings)
    
    # Применяем дополнительные переопределения
    for key, value in overrides.items():
        if hasattr(config, key):
            # Если это словарь с настройками компонента
            if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                component = getattr(config, key)
                for k, v in value.items():
                    if hasattr(component, k):
                        setattr(component, k, v)
            else:
                setattr(config, key, value)
    
    return config


def create_research_config_simple(**overrides) -> SimpleProjectConfig:
    """Создать конфигурацию для исследований"""
    config = SimpleProjectConfig()
    config.enable_research_mode()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def create_production_config_simple(**overrides) -> SimpleProjectConfig:
    """Создать конфигурацию для production"""
    config = SimpleProjectConfig()
    config.enable_production_mode()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# === ОБРАТНАЯ СОВМЕСТИМОСТЬ ===

# Алиасы для обратной совместимости с legacy кодом
ProjectConfig = SimpleProjectConfig  # основной алиас


def get_legacy_config():
    """Обертка для legacy кода"""
    return get_project_config()
