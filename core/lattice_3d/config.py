"""
Модуль конфигурации для 3D Решетки
====================================

Содержит датакласс LatticeConfig, который определяет все параметры
решетки, и функции для загрузки и валидации этих конфигураций
из YAML-файлов.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
import yaml
from pathlib import Path
import logging

from .enums import BoundaryCondition, Face, PlacementStrategy
from .position import Dimensions3D

# Импорт CellPrototype нужен для type hinting, но может вызвать цикл. импортируем как строку
# from core.cell_prototype import CellPrototype


@dataclass
class LatticeConfig:
    """
    Конфигурация трехмерной решетки клеток.

    Содержит все параметры, необходимые для создания и настройки решетки:
    размеры, граничные условия, параметры производительности и интеграции.
    """

    # Основные параметры решетки
    dimensions: Dimensions3D = (8, 8, 8)
    boundary_conditions: BoundaryCondition = BoundaryCondition.WALLS

    # Производительность
    parallel_processing: bool = True
    gpu_enabled: bool = True
    batch_size: int = 1

    # Инициализация состояний
    initialization_method: str = "normal"
    initialization_std: float = 0.1
    initialization_mean: float = 0.0
    seed: int = 42

    # Топология
    neighbors: int = 6
    validate_connections: bool = True
    cache_neighbors: bool = True
    neighbor_finding_strategy: str = "local"
    neighbor_strategy_config: Optional[Dict[str, Any]] = None

    # Интерфейсы ввода/вывода
    input_face: Face = Face.FRONT
    output_face: Face = Face.BACK
    embedding_mapping: str = "linear"
    placement_strategy: PlacementStrategy = PlacementStrategy.PROPORTIONAL
    io_strategy_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Диагностика
    enable_logging: bool = True
    log_level: str = "INFO"
    track_performance: bool = True
    validate_states: bool = True

    # Оптимизация
    memory_efficient: bool = True
    use_checkpointing: bool = False
    mixed_precision: bool = False

    # Интеграция с cell_prototype
    cell_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    auto_sync_cell_config: bool = True

    # === НОВОЕ: STDP и пластичность ===
    enable_plasticity: bool = False
    plasticity_rule: str = "stdp"  # Тип правила пластичности
    activity_history_size: int = 10  # Размер circular buffer для истории

    # STDP параметры
    stdp_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "activity_threshold": 0.1,  # Порог активности
            "learning_rate": 0.01,  # Скорость обучения
            "A_plus": 0.01,  # LTP амплитуда
            "A_minus": 0.01,  # LTD амплитуда
            "tau_plus": 20,  # LTP time constant
            "tau_minus": 20,  # LTD time constant
            "weight_bounds": [0.1, 2.0],  # Диапазон весов [min, max]
        }
    )

    # === НОВОЕ: Конкурентное обучение (Шаг 3.1) ===
    enable_competitive_learning: bool = False
    competitive_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "winner_boost_factor": 1.05,  # Усиление связи с победителем
            "lateral_inhibition_factor": 0.98,  # Ослабление неактивных связей
            "enable_weight_normalization": True,  # Нормализация весов
            "normalization_method": "proportional",  # Метод нормализации
            "update_frequency": 1,  # Частота применения (каждые N шагов)
            "max_winner_connections": 8,  # Максимум усиленных связей
            "lateral_inhibition_radius": 3.0,  # Радиус латерального торможения
        }
    )

    # === НОВОЕ: Метапластичность - BCM правило (Шаг 3.2) ===
    enable_metaplasticity: bool = False
    bcm_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "tau_theta": 1000.0,  # Временная константа для адаптивных порогов
            "initial_threshold": None,  # Начальный порог (None = наследует activity_threshold)
            "min_threshold": 0.001,  # Минимальный порог активности
            "max_threshold": 0.5,  # Максимальный порог активности
            "bcm_learning_rate_factor": 0.5,  # Коэффициент для BCM относительно основного learning_rate
            "enable_threshold_adaptation": True,  # Включить адаптацию порогов
            "adaptation_frequency": 1,  # Частота адаптации порогов (каждые N шагов)
        }
    )

    # === НОВОЕ: Функциональная кластеризация (Шаг 3.3) ===
    enable_clustering: bool = False
    clustering_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            # Базовая кластеризация
            "enable_clustering": True,
            "similarity_threshold": 0.7,  # Порог сходства для кластеризации
            "max_clusters": 8,  # Максимальное количество кластеров
            "update_frequency": 100,  # Частота обновления кластеров (в шагах)
            "min_cluster_size": 5,  # Минимальный размер кластера
            "stability_threshold": 0.8,  # Порог стабильности кластеров
            # Модификация весов связей
            "intra_cluster_boost": 1.2,  # Усиление связей внутри кластера
            "inter_cluster_dampening": 0.8,  # Ослабление связей между кластерами
            # Интеграция с существующими механизмами
            "priority": 0.5,  # Приоритет кластеризации vs пластичности [0.0-1.0]
            "integration_mode": "additive",  # Режим интеграции весов
            # Координационный интерфейс (готов к расширению)
            "coordination": {
                "coordination_mode": "basic",  # Режим координации
                "enable_user_guidance": False,  # Пользовательское управление
                "enable_learned_coordination": False,  # Обученная координация
                "coordination_strength": 0.5,  # Сила координации
                "adaptation_rate": 0.1,  # Скорость адаптации
                "memory_decay": 0.95,  # Затухание памяти
            },
        }
    )

    def __post_init__(self):
        """Валидация и нормализация конфигурации после создания"""
        self._validate_dimensions()
        self._normalize_enums()
        self._validate_stdp_config()
        self._validate_competitive_config()
        self._validate_bcm_config()
        self._validate_clustering_config()
        self._setup_logging()

    def _validate_dimensions(self):
        """Проверка корректности размеров решетки"""
        if len(self.dimensions) != 3:
            raise ValueError("Dimensions must be a 3-tuple (X, Y, Z)")
        if any(dim < 1 for dim in self.dimensions):
            raise ValueError("All dimensions must be positive")
        if any(dim > 2048 for dim in self.dimensions):  # Увеличено
            logging.warning(
                f"Large dimensions {self.dimensions} may cause performance issues"
            )

    def _normalize_enums(self):
        """Нормализация строковых значений в Enum'ы"""
        if isinstance(self.boundary_conditions, str):
            self.boundary_conditions = BoundaryCondition(
                self.boundary_conditions.lower()
            )
        if isinstance(self.input_face, str):
            self.input_face = Face(self.input_face.lower())
        if isinstance(self.output_face, str):
            self.output_face = Face(self.output_face.lower())
        if isinstance(self.placement_strategy, str):
            self.placement_strategy = PlacementStrategy(self.placement_strategy.lower())

    def _validate_stdp_config(self):
        """Проверка корректности конфигурации STDP"""
        if self.enable_plasticity and not self.stdp_config:
            raise ValueError(
                "STDP configuration is required when enable_plasticity is True"
            )

        if self.enable_plasticity:
            # Валидация STDP параметров
            required_keys = [
                "activity_threshold",
                "learning_rate",
                "A_plus",
                "A_minus",
                "weight_bounds",
            ]
            for key in required_keys:
                if key not in self.stdp_config:
                    logging.warning(f"Missing STDP parameter: {key}, using default")

            # Валидация диапазонов
            if "weight_bounds" in self.stdp_config:
                bounds = self.stdp_config["weight_bounds"]
                if len(bounds) != 2 or bounds[0] >= bounds[1]:
                    raise ValueError("weight_bounds must be [min, max] where min < max")
                if bounds[0] <= 0:
                    raise ValueError("Minimum weight bound must be positive")

            # Валидация activity_history_size
            if self.activity_history_size < 2:
                raise ValueError("activity_history_size must be at least 2 for STDP")
            if self.activity_history_size > 1000:
                logging.warning(
                    f"Large activity_history_size ({self.activity_history_size}) may use excessive memory"
                )

    def _validate_competitive_config(self):
        """Проверка корректности конфигурации конкурентного обучения"""
        if self.enable_competitive_learning and not self.competitive_config:
            raise ValueError(
                "Competitive learning configuration is required when enable_competitive_learning is True"
            )

        if self.enable_competitive_learning:
            # Валидация параметров конкурентного обучения
            required_keys = [
                "winner_boost_factor",
                "lateral_inhibition_factor",
                "update_frequency",
            ]
            for key in required_keys:
                if key not in self.competitive_config:
                    logging.warning(
                        f"Missing competitive learning parameter: {key}, using default"
                    )

            # Валидация диапазонов
            config = self.competitive_config

            if "winner_boost_factor" in config:
                if not (1.0 <= config["winner_boost_factor"] <= 2.0):
                    raise ValueError("winner_boost_factor must be between 1.0 and 2.0")

            if "lateral_inhibition_factor" in config:
                if not (0.5 <= config["lateral_inhibition_factor"] <= 1.0):
                    raise ValueError(
                        "lateral_inhibition_factor must be between 0.5 and 1.0"
                    )

            if "update_frequency" in config:
                if config["update_frequency"] < 1:
                    raise ValueError("update_frequency must be at least 1")

            if "max_winner_connections" in config:
                if config["max_winner_connections"] < 1:
                    raise ValueError("max_winner_connections must be at least 1")
                if config["max_winner_connections"] > self.neighbors:
                    logging.warning(
                        f"max_winner_connections ({config['max_winner_connections']}) "
                        f"exceeds total neighbors ({self.neighbors})"
                    )

            # Требуется STDP для конкурентного обучения
            if not self.enable_plasticity:
                logging.warning(
                    "Competitive learning works best with STDP enabled. "
                    "Consider setting enable_plasticity=True"
                )

    def _validate_bcm_config(self):
        """Проверка корректности конфигурации BCM метапластичности"""
        if self.enable_metaplasticity and not self.bcm_config:
            raise ValueError(
                "BCM configuration is required when enable_metaplasticity is True"
            )

        if self.enable_metaplasticity:
            # Валидация параметров BCM
            required_keys = [
                "tau_theta",
                "min_threshold",
                "max_threshold",
            ]
            for key in required_keys:
                if key not in self.bcm_config:
                    logging.warning(f"Missing BCM parameter: {key}, using default")

            # Валидация диапазонов
            config = self.bcm_config

            if "tau_theta" in config:
                if config["tau_theta"] <= 0:
                    raise ValueError("tau_theta must be positive")
                if config["tau_theta"] < 10:
                    logging.warning(
                        f"Very small tau_theta ({config['tau_theta']}) may cause instability"
                    )

            if "min_threshold" in config and "max_threshold" in config:
                if config["min_threshold"] >= config["max_threshold"]:
                    raise ValueError("min_threshold must be less than max_threshold")
                if config["min_threshold"] <= 0:
                    raise ValueError("min_threshold must be positive")

            if "bcm_learning_rate_factor" in config:
                if not (0.0 < config["bcm_learning_rate_factor"] <= 1.0):
                    raise ValueError(
                        "bcm_learning_rate_factor must be between 0.0 and 1.0"
                    )

            if "adaptation_frequency" in config:
                if config["adaptation_frequency"] < 1:
                    raise ValueError("adaptation_frequency must be at least 1")

            # Требуется STDP для BCM метапластичности
            if not self.enable_plasticity:
                raise ValueError(
                    "BCM metaplasticity requires STDP to be enabled. "
                    "Set enable_plasticity=True"
                )

            # Проверка совместимости с competitive learning
            if self.enable_competitive_learning:
                logging.info(
                    "BCM metaplasticity will work together with competitive learning. "
                    "This is a powerful combination for stable learning."
                )

    def _validate_clustering_config(self):
        """Проверка корректности конфигурации функциональной кластеризации"""
        if self.enable_clustering and not self.clustering_config:
            raise ValueError(
                "Clustering configuration is required when enable_clustering is True"
            )

        if self.enable_clustering:
            # Валидация параметров кластеризации
            required_keys = [
                "similarity_threshold",
                "max_clusters",
                "update_frequency",
                "intra_cluster_boost",
                "inter_cluster_dampening",
                "priority",
                "integration_mode",
            ]
            for key in required_keys:
                if key not in self.clustering_config:
                    logging.warning(
                        f"Missing clustering parameter: {key}, using default"
                    )

            # Валидация диапазонов
            config = self.clustering_config

            if "similarity_threshold" in config:
                if not (0.0 <= config["similarity_threshold"] <= 1.0):
                    raise ValueError("similarity_threshold must be between 0.0 and 1.0")

            if "max_clusters" in config:
                if config["max_clusters"] < 2:
                    raise ValueError("max_clusters must be at least 2")
                if config["max_clusters"] > 50:
                    logging.warning(
                        f"Large max_clusters ({config['max_clusters']}) may cause performance issues"
                    )

            if "update_frequency" in config:
                if config["update_frequency"] < 1:
                    raise ValueError("update_frequency must be at least 1")
                if config["update_frequency"] > 10000:
                    logging.warning(
                        f"Very high update_frequency ({config['update_frequency']}) "
                        "may reduce clustering effectiveness"
                    )

            if "min_cluster_size" in config:
                if config["min_cluster_size"] < 1:
                    raise ValueError("min_cluster_size must be at least 1")

            if "intra_cluster_boost" in config:
                if not (1.0 <= config["intra_cluster_boost"] <= 5.0):
                    raise ValueError("intra_cluster_boost must be between 1.0 and 5.0")

            if "inter_cluster_dampening" in config:
                if not (0.1 <= config["inter_cluster_dampening"] <= 1.0):
                    raise ValueError(
                        "inter_cluster_dampening must be between 0.1 and 1.0"
                    )

            if "priority" in config:
                if not (0.0 <= config["priority"] <= 1.0):
                    raise ValueError("priority must be between 0.0 and 1.0")

            if "integration_mode" in config:
                valid_modes = ["replace", "additive", "multiplicative", "selective"]
                if config["integration_mode"] not in valid_modes:
                    raise ValueError(f"integration_mode must be one of {valid_modes}")

            # Валидация координационной конфигурации
            if "coordination" in config:
                coord_config = config["coordination"]

                if "coordination_mode" in coord_config:
                    valid_coord_modes = ["basic", "user_guided", "learned", "hybrid"]
                    if coord_config["coordination_mode"] not in valid_coord_modes:
                        raise ValueError(
                            f"coordination_mode must be one of {valid_coord_modes}"
                        )

                if "coordination_strength" in coord_config:
                    if not (0.0 <= coord_config["coordination_strength"] <= 1.0):
                        raise ValueError(
                            "coordination_strength must be between 0.0 and 1.0"
                        )

                if "adaptation_rate" in coord_config:
                    if not (0.0 < coord_config["adaptation_rate"] <= 1.0):
                        raise ValueError("adaptation_rate must be between 0.0 and 1.0")

                if "memory_decay" in coord_config:
                    if not (0.0 <= coord_config["memory_decay"] <= 1.0):
                        raise ValueError("memory_decay must be between 0.0 and 1.0")

            # Проверка совместимости с другими механизмами
            if self.enable_plasticity:
                logging.info(
                    "Clustering will work together with plasticity mechanisms. "
                    f"Priority split: clustering={config.get('priority', 0.5):.1f}, "
                    f"plasticity={1-config.get('priority', 0.5):.1f}"
                )

            if self.enable_competitive_learning and self.enable_plasticity:
                logging.info(
                    "Triple integration: clustering + competitive learning + STDP/BCM. "
                    "This is an advanced configuration for emergent behavior."
                )

    def _setup_logging(self):
        """Настройка логирования для модуля"""
        if self.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.log_level.upper(), logging.INFO)
            )

    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LatticeConfig:
        """Создает экземпляр LatticeConfig из словаря."""
        return _build_lattice_config_from_data(data)


def load_lattice_config(config_path: str) -> LatticeConfig:
    """
    Загружает и валидирует конфигурацию решетки из YAML-файла.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise TypeError("Конфигурация должна быть словарем (dictionary)")

    return _build_lattice_config_from_data(config_data)


def _build_lattice_config_from_data(config_data: Dict[str, Any]) -> LatticeConfig:
    """
    Собирает объект LatticeConfig из словаря, извлекая данные из
    вложенных секций 'lattice', 'cell', 'io' и прямых полей.
    """
    lattice_data = config_data.get("lattice", {})
    io_data = config_data.get("io", {})
    cell_data = config_data.get("cell", {})
    cell_prototype_data = config_data.get("cell_prototype", {})

    # Начинаем с основных данных config_data (верхний уровень)
    combined_data = config_data.copy()

    # Объединяем данные, lattice имеет самый высокий приоритет
    combined_data.update(io_data)
    combined_data.update(lattice_data)

    # Используем cell_prototype если есть, иначе cell для совместимости
    if cell_prototype_data:
        combined_data["cell_config"] = {"cell_prototype": cell_prototype_data}
    elif cell_data:
        combined_data["cell_config"] = cell_data

    # Специальная обработка гибридной архитектуры
    if (
        "neuron_architecture" in config_data
        and "connection_architecture" in config_data
    ):
        # Создаем гибридную конфигурацию клеток
        cell_config = {"cell_prototype": {}}

        # Обрабатываем MinimalNCACell
        if config_data.get("neuron_architecture") == "minimal_nca":
            if "minimal_nca_cell" in config_data:
                cell_config["cell_prototype"]["minimal_nca_cell"] = config_data[
                    "minimal_nca_cell"
                ]

        # Обрабатываем GatedMLPCell
        if config_data.get("connection_architecture") == "gated_mlp":
            if "gmlp_cell" in config_data:
                cell_config["cell_prototype"]["gmlp_cell"] = config_data["gmlp_cell"]

        # Добавляем дополнительные параметры
        if "disable_nca_scaling" in config_data:
            cell_config["cell_prototype"]["disable_nca_scaling"] = config_data[
                "disable_nca_scaling"
            ]

        combined_data["cell_config"] = cell_config

    # Специальная обработка neighbor_strategy
    if "neighbor_strategy" in config_data:
        combined_data["neighbor_finding_strategy"] = config_data["neighbor_strategy"]

    # Обработка num_neighbors
    if "num_neighbors" in config_data:
        combined_data["neighbors"] = config_data["num_neighbors"]

    # Обработка tiered_neighbor_config
    if "tiered_neighbor_config" in config_data:
        combined_data["neighbor_strategy_config"] = config_data[
            "tiered_neighbor_config"
        ]

    # Специальная обработка dimensions (словарь -> кортеж)
    if "dimensions" in combined_data and isinstance(combined_data["dimensions"], dict):
        dims_dict = combined_data["dimensions"]
        combined_data["dimensions"] = (
            dims_dict.get("width", 8),
            dims_dict.get("height", 8),
            dims_dict.get("depth", 8),
        )

    # Получаем все известные поля dataclass
    known_fields = {f.name for f in LatticeConfig.__dataclass_fields__.values()}

    # Фильтруем словарь, оставляя только те ключи, которые есть в LatticeConfig
    filtered_data = {k: v for k, v in combined_data.items() if k in known_fields}

    return LatticeConfig(**filtered_data)
