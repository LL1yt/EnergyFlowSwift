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
    LatticeSettings,
    ModelSettings,
    TrainingSettings,
    CNFSettings,
    EulerSettings,
    CacheSettings,
    SpatialSettings,
    VectorizedSettings,
    DeviceSettings,
    LoggingSettings,
    MemorySettings,
    ExperimentSettings,
    PerformanceSettings,
    ValidationSettings,
    EmbeddingSettings,
    TrainingEmbeddingSettings,
    NeighborSettings,
    ExpertSettings,
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

    # Основные компоненты (всегда присутствуют)
    lattice: LatticeSettings = field(default_factory=LatticeSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    device: DeviceSettings = field(default_factory=DeviceSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # Дополнительные компоненты (опциональные)
    cnf: Optional[CNFSettings] = field(default_factory=CNFSettings)
    euler: Optional[EulerSettings] = field(default_factory=EulerSettings)
    cache: Optional[CacheSettings] = field(default_factory=CacheSettings)
    spatial: Optional[SpatialSettings] = field(default_factory=SpatialSettings)
    vectorized: Optional[VectorizedSettings] = field(default_factory=VectorizedSettings)
    memory: Optional[MemorySettings] = field(default_factory=MemorySettings)

    # Экспериментальные компоненты
    experiment: Optional[ExperimentSettings] = None
    performance: Optional[PerformanceSettings] = None
    validation: Optional[ValidationSettings] = None

    # Компоненты для работы с эмбедингами
    embedding: Optional[EmbeddingSettings] = field(default_factory=EmbeddingSettings)
    training_embedding: Optional[TrainingEmbeddingSettings] = field(
        default_factory=TrainingEmbeddingSettings
    )

    # MoE компоненты
    neighbors: Optional[NeighborSettings] = field(default_factory=NeighborSettings)
    expert: Optional[ExpertSettings] = field(default_factory=ExpertSettings)

    # Runtime компоненты (вычисляются автоматически)
    device_manager: Optional[DeviceManager] = field(init=False, default=None)

    def __post_init__(self):
        """Автоматическая настройка после инициализации"""
        # Инициализация device manager
        self.device_manager = get_device_manager(
            prefer_cuda=self.device.prefer_cuda, debug_mode=self.device.debug_mode
        )

        # Связываем cache с expert settings
        if self.expert and self.cache:
            self.expert.cache = self.cache

        # Валидация конфигурации если включена
        if self.validation and self.validation.validate_config:
            self._validate_configuration()

        # Логирование инициализации
        if self.logging.debug_mode:
            self._log_initialization()

    def _validate_configuration(self):
        """Валидация конфигурации"""
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
            f"   🧠 Model: {self.model.state_size}D state, {self.model.target_params} params"
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
                f"   🎯 Embeddings: {self.embedding.teacher_model} ({self.embedding.teacher_embedding_dim}D → {self.embedding.cube_embedding_dim}D)"
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
        self.logging.debug_mode = True
        self.logging.performance_tracking = True
        if self.cache:
            self.cache.enable_detailed_stats = True
        if self.performance:
            self.performance.profiling_enabled = True

    def enable_production_mode(self):
        """Включить режим production"""
        # Отключаем debug режимы
        self.logging.debug_mode = False
        self.logging.level = "WARNING"
        self.device.debug_mode = False

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
        for field_name in ["lattice", "model", "training", "device", "logging"]:
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
            "vectorized",
            "memory",
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


def get_project_config() -> SimpleProjectConfig:
    """Получить глобальный экземпляр конфигурации"""
    global _global_config
    if _global_config is None:
        _global_config = SimpleProjectConfig()
    return _global_config


def set_project_config(config: SimpleProjectConfig):
    """Установить глобальный экземпляр конфигурации"""
    global _global_config
    _global_config = config


def reset_project_config():
    """Сбросить глобальную конфигурацию"""
    global _global_config
    _global_config = None


# === ФАБРИЧНЫЕ ФУНКЦИИ ===


def create_simple_config(**overrides) -> SimpleProjectConfig:
    """Создать простую конфигурацию с переопределениями"""
    config = SimpleProjectConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
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
