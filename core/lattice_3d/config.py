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

    def __post_init__(self):
        """Валидация и нормализация конфигурации после создания"""
        self._validate_dimensions()
        self._normalize_enums()
        self._validate_stdp_config()
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
    вложенных секций 'lattice', 'cell', 'io'.
    """
    lattice_data = config_data.get("lattice", {})
    io_data = config_data.get("io", {})
    cell_data = config_data.get("cell", {})
    cell_prototype_data = config_data.get("cell_prototype", {})

    # Объединяем данные, lattice имеет самый высокий приоритет
    combined_data = {**io_data, **lattice_data}

    # Используем cell_prototype если есть, иначе cell для совместимости
    if cell_prototype_data:
        combined_data["cell_config"] = {"cell_prototype": cell_prototype_data}
    else:
        combined_data["cell_config"] = cell_data

    # Получаем все известные поля dataclass
    known_fields = {f.name for f in LatticeConfig.__dataclass_fields__.values()}

    # Фильтруем словарь, оставляя только те ключи, которые есть в LatticeConfig
    filtered_data = {k: v for k, v in combined_data.items() if k in known_fields}

    return LatticeConfig(**filtered_data)
