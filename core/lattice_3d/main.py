"""
Модуль Lattice 3D - Трехмерная Решетка Клеток

Этот модуль реализует трехмерную решетку "умных клеток" для нейронной сети
клеточного типа. Основные компоненты:

- LatticeConfig: Конфигурация решетки
- Position3D: Работа с 3D координатами
- NeighborTopology: Система соседства и граничные условия
- Lattice3D: Главный класс решетки

Биологическая аналогия: кора головного мозга как 3D ткань из взаимосвязанных
нейронов, где каждый нейрон связан с ближайшими соседями.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Tuple, List, Dict, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Импорты из других модулей проекта
from core.cell_prototype import CellPrototype, create_cell_from_config


# =============================================================================
# БАЗОВЫЕ ТИПЫ И КОНСТАНТЫ
# =============================================================================


class BoundaryCondition(Enum):
    """Типы граничных условий для решетки"""

    WALLS = "walls"  # Границы блокируют сигналы
    PERIODIC = "periodic"  # Решетка замыкается в тор
    ABSORBING = "absorbing"  # Сигналы затухают на границах
    REFLECTING = "reflecting"  # Сигналы отражаются от границ


class Face(Enum):
    """Грани решетки для ввода/вывода"""

    FRONT = "front"  # Z = 0
    BACK = "back"  # Z = max
    LEFT = "left"  # X = 0
    RIGHT = "right"  # X = max
    TOP = "top"  # Y = max
    BOTTOM = "bottom"  # Y = 0


class PlacementStrategy(Enum):
    """Стратегии размещения точек ввода/вывода"""

    PROPORTIONAL = "proportional"  # Пропорциональное автоматическое масштабирование
    RANDOM = "random"  # Случайное размещение
    CORNERS = "corners"  # Размещение в углах
    CORNERS_CENTER = "corners_center"  # Углы + центр
    FULL_FACE = "full_face"  # Полное покрытие грани (текущая реализация)


# Типы для координат и размеров
Coordinates3D = Tuple[int, int, int]
Dimensions3D = Tuple[int, int, int]


# =============================================================================
# РАЗМЕЩЕНИЕ I/O ТОЧЕК
# =============================================================================


class IOPointPlacer:
    """
    Управление размещением точек ввода/вывода с автоматическим масштабированием.

    Реализует различные стратегии размещения I/O точек на гранях решетки,
    включая биологически обоснованное пропорциональное масштабирование.
    """

    def __init__(
        self,
        lattice_dimensions: Dimensions3D,
        strategy: PlacementStrategy,
        config: Dict[str, Any],
        seed: int = 42,
    ):
        """
        Инициализация размещения I/O точек.

        Args:
            lattice_dimensions: Размеры решетки (X, Y, Z)
            strategy: Стратегия размещения точек
            config: Конфигурация размещения
            seed: Сид для воспроизводимости
        """
        self.dimensions = lattice_dimensions
        self.strategy = strategy
        self.config = config
        self.seed = seed

        # Устанавливаем сид для воспроизводимости
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Кэш для размещенных точек
        self._input_points_cache = {}
        self._output_points_cache = {}

    def calculate_num_points(self, face_area: int) -> Tuple[int, int]:
        """
        Рассчитывает количество точек для пропорциональной стратегии.

        Args:
            face_area: Площадь грани (количество клеток)

        Returns:
            Tuple[int, int]: (минимальное_количество, максимальное_количество)
        """
        if self.strategy != PlacementStrategy.PROPORTIONAL:
            raise ValueError("calculate_num_points только для PROPORTIONAL стратегии")

        coverage_config = self.config.get("coverage_ratio", {})
        min_percentage = coverage_config.get("min_percentage", 7.8)
        max_percentage = coverage_config.get("max_percentage", 15.6)

        # Рассчитываем количество точек
        min_points_calc = max(1, int(face_area * min_percentage / 100))
        max_points_calc = max(min_points_calc, int(face_area * max_percentage / 100))

        # Применяем абсолютные ограничения
        limits_config = self.config.get("absolute_limits", {})
        min_points_abs = limits_config.get("min_points", 1)
        max_points_abs = limits_config.get("max_points", 0)  # 0 = без ограничений

        min_points = max(min_points_calc, min_points_abs)
        max_points = max_points_calc

        if max_points_abs > 0:
            max_points = min(max_points, max_points_abs)

        # ИСПРАВЛЕНИЕ: Убеждаемся, что min_points <= max_points
        if min_points > max_points:
            # Если min_points больше max_points, корректируем
            if max_points_abs > 0:
                # Если есть максимальное ограничение, используем его как min_points
                min_points = max_points
            else:
                # Если нет максимального ограничения, увеличиваем max_points
                max_points = min_points

        return min_points, max_points

    def get_input_points(self, face: Face) -> List[Coordinates3D]:
        """
        Получает координаты точек ввода на указанной грани.

        Args:
            face: Грань решетки для размещения точек ввода

        Returns:
            List[Coordinates3D]: Список 3D координат точек ввода
        """
        # Проверяем кэш
        cache_key = f"input_{face.value}"
        if cache_key in self._input_points_cache:
            return self._input_points_cache[cache_key]

        # Генерируем точки в зависимости от стратегии
        if self.strategy == PlacementStrategy.PROPORTIONAL:
            points = self._generate_proportional_points(face)
        elif self.strategy == PlacementStrategy.FULL_FACE:
            points = self._generate_full_face_points(face)
        elif self.strategy == PlacementStrategy.RANDOM:
            points = self._generate_random_points(face)
        elif self.strategy == PlacementStrategy.CORNERS:
            points = self._generate_corner_points(face)
        elif self.strategy == PlacementStrategy.CORNERS_CENTER:
            points = self._generate_corners_center_points(face)
        else:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")

        # Кэшируем результат
        self._input_points_cache[cache_key] = points
        return points

    def get_output_points(self, face: Face) -> List[Coordinates3D]:
        """
        Получает координаты точек вывода на указанной грани.

        Args:
            face: Грань решетки для размещения точек вывода

        Returns:
            List[Coordinates3D]: Список 3D координат точек вывода
        """
        # Проверяем кэш
        cache_key = f"output_{face.value}"
        if cache_key in self._output_points_cache:
            return self._output_points_cache[cache_key]

        # Для вывода используем те же стратегии, что и для ввода
        if self.strategy == PlacementStrategy.PROPORTIONAL:
            points = self._generate_proportional_points(face)
        elif self.strategy == PlacementStrategy.FULL_FACE:
            points = self._generate_full_face_points(face)
        elif self.strategy == PlacementStrategy.RANDOM:
            points = self._generate_random_points(face)
        elif self.strategy == PlacementStrategy.CORNERS:
            points = self._generate_corner_points(face)
        elif self.strategy == PlacementStrategy.CORNERS_CENTER:
            points = self._generate_corners_center_points(face)
        else:
            raise ValueError(f"Неподдерживаемая стратегия: {self.strategy}")

        # Кэшируем результат
        self._output_points_cache[cache_key] = points
        return points

    def _generate_proportional_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки для пропорциональной стратегии."""
        face_coords = self._get_face_coordinates(face)
        face_area = len(face_coords)

        min_points, max_points = self.calculate_num_points(face_area)

        # Выбираем случайное количество точек в диапазоне
        num_points = np.random.randint(min_points, max_points + 1)

        # Случайно выбираем точки из доступных координат
        selected_indices = np.random.choice(
            len(face_coords), size=num_points, replace=False
        )
        return [face_coords[i] for i in selected_indices]

    def _generate_full_face_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки для полного покрытия грани."""
        return self._get_face_coordinates(face)

    def _generate_random_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует случайно размещенные точки."""
        face_coords = self._get_face_coordinates(face)
        # Используем 25% от площади грани для случайного размещения
        num_points = max(1, len(face_coords) // 4)
        selected_indices = np.random.choice(
            len(face_coords), size=num_points, replace=False
        )
        return [face_coords[i] for i in selected_indices]

    def _generate_corner_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки в углах грани."""
        face_coords = self._get_face_coordinates(face)

        # Находим угловые точки в зависимости от грани
        if face in [Face.FRONT, Face.BACK]:
            # Для граней Z - углы в плоскости XY
            corners = [
                (0, 0),
                (0, self.dimensions[1] - 1),
                (self.dimensions[0] - 1, 0),
                (self.dimensions[0] - 1, self.dimensions[1] - 1),
            ]
            z_val = 0 if face == Face.FRONT else self.dimensions[2] - 1
            return [(x, y, z_val) for x, y in corners]
        elif face in [Face.LEFT, Face.RIGHT]:
            # Для граней X - углы в плоскости YZ
            corners = [
                (0, 0),
                (0, self.dimensions[2] - 1),
                (self.dimensions[1] - 1, 0),
                (self.dimensions[1] - 1, self.dimensions[2] - 1),
            ]
            x_val = 0 if face == Face.LEFT else self.dimensions[0] - 1
            return [(x_val, y, z) for y, z in corners]
        else:  # TOP, BOTTOM
            # Для граней Y - углы в плоскости XZ
            corners = [
                (0, 0),
                (0, self.dimensions[2] - 1),
                (self.dimensions[0] - 1, 0),
                (self.dimensions[0] - 1, self.dimensions[2] - 1),
            ]
            y_val = self.dimensions[1] - 1 if face == Face.TOP else 0
            return [(x, y_val, z) for x, z in corners]

    def _generate_corners_center_points(self, face: Face) -> List[Coordinates3D]:
        """Генерирует точки в углах и центре грани."""
        corner_points = self._generate_corner_points(face)
        center_point = self._get_face_center(face)
        return corner_points + [center_point]

    def _get_face_coordinates(self, face: Face) -> List[Coordinates3D]:
        """Получает все координаты клеток на указанной грани."""
        coords = []

        if face == Face.FRONT:  # Z = 0
            for x in range(self.dimensions[0]):
                for y in range(self.dimensions[1]):
                    coords.append((x, y, 0))
        elif face == Face.BACK:  # Z = max
            for x in range(self.dimensions[0]):
                for y in range(self.dimensions[1]):
                    coords.append((x, y, self.dimensions[2] - 1))
        elif face == Face.LEFT:  # X = 0
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    coords.append((0, y, z))
        elif face == Face.RIGHT:  # X = max
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    coords.append((self.dimensions[0] - 1, y, z))
        elif face == Face.TOP:  # Y = max
            for x in range(self.dimensions[0]):
                for z in range(self.dimensions[2]):
                    coords.append((x, self.dimensions[1] - 1, z))
        elif face == Face.BOTTOM:  # Y = 0
            for x in range(self.dimensions[0]):
                for z in range(self.dimensions[2]):
                    coords.append((x, 0, z))

        return coords

    def _get_face_center(self, face: Face) -> Coordinates3D:
        """Получает координаты центра указанной грани."""
        if face == Face.FRONT:
            return (self.dimensions[0] // 2, self.dimensions[1] // 2, 0)
        elif face == Face.BACK:
            return (
                self.dimensions[0] // 2,
                self.dimensions[1] // 2,
                self.dimensions[2] - 1,
            )
        elif face == Face.LEFT:
            return (0, self.dimensions[1] // 2, self.dimensions[2] // 2)
        elif face == Face.RIGHT:
            return (
                self.dimensions[0] - 1,
                self.dimensions[1] // 2,
                self.dimensions[2] // 2,
            )
        elif face == Face.TOP:
            return (
                self.dimensions[0] // 2,
                self.dimensions[1] - 1,
                self.dimensions[2] // 2,
            )
        elif face == Face.BOTTOM:
            return (self.dimensions[0] // 2, 0, self.dimensions[2] // 2)


# =============================================================================
# КОНФИГУРАЦИЯ РЕШЕТКИ
# =============================================================================


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

    # Топология
    neighbors: int = 6
    validate_connections: bool = True
    cache_neighbors: bool = True

    # Интерфейсы ввода/вывода
    input_face: Face = Face.FRONT
    output_face: Face = Face.BACK
    embedding_mapping: str = "linear"

    # НОВОЕ: Стратегия размещения I/O точек
    placement_strategy: PlacementStrategy = PlacementStrategy.PROPORTIONAL
    io_strategy_config: Optional[Dict[str, Any]] = None

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
    cell_config: Optional[Dict[str, Any]] = None
    auto_sync_cell_config: bool = True

    def __post_init__(self):
        """Валидация и нормализация конфигурации после создания"""
        self._validate_dimensions()
        self._normalize_boundary_conditions()
        self._setup_logging()

    def _validate_dimensions(self):
        """Проверка корректности размеров решетки"""
        if len(self.dimensions) != 3:
            raise ValueError("Dimensions must be a 3-tuple (X, Y, Z)")

        if any(dim < 1 for dim in self.dimensions):
            raise ValueError("All dimensions must be positive")

        if any(dim > 1000 for dim in self.dimensions):
            logging.warning(
                f"Large dimensions {self.dimensions} may cause performance issues"
            )

    def _normalize_boundary_conditions(self):
        """Нормализация граничных условий"""
        if isinstance(self.boundary_conditions, str):
            try:
                self.boundary_conditions = BoundaryCondition(self.boundary_conditions)
            except ValueError:
                raise ValueError(
                    f"Unknown boundary condition: {self.boundary_conditions}"
                )

    def _setup_logging(self):
        """Настройка логирования для модуля"""
        if self.enable_logging:
            logging.basicConfig(level=getattr(logging, self.log_level.upper()))

    @property
    def total_cells(self) -> int:
        """Общее количество клеток в решетке"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    @property
    def device(self) -> torch.device:
        """Устройство для вычислений (CPU или GPU)"""
        if self.gpu_enabled and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def load_lattice_config(config_path: Optional[str] = None) -> LatticeConfig:
    """
    Загрузка конфигурации решетки через ConfigManager или из файла.

    Args:
        config_path: Путь к файлу конфигурации. Если None, пробуем ConfigManager, затем default.yaml

    Returns:
        LatticeConfig: Загруженная конфигурация
    """
    try:
        # Сначала пробуем загрузить через ConfigManager
        if config_path is None:
            try:
                from utils.config_manager import get_global_config_manager

                config_manager = get_global_config_manager()

                if config_manager:
                    # Пробуем получить конфигурацию lattice_3d
                    lattice_data = config_manager.get_config("lattice_3d")
                    if lattice_data:
                        logging.info("[OK] Loaded lattice config from ConfigManager")
                    else:
                        # Пробуем секцию lattice
                        lattice_data = config_manager.get_config("lattice")
                        if lattice_data:
                            logging.info(
                                "[OK] Loaded lattice config from ConfigManager (lattice section)"
                            )

                    if lattice_data:
                        # Преобразуем конфигурацию через ConfigManager
                        return _build_lattice_config_from_data(lattice_data)

            except ImportError:
                logging.warning(
                    "ConfigManager not available, falling back to file loading"
                )
            except Exception as e:
                logging.warning(
                    f"ConfigManager error: {e}, falling back to file loading"
                )

        # Fallback: загружаем из файла
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "default.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # Извлекаем секцию lattice_3d
        lattice_data = yaml_data.get("lattice_3d", {})

        # Преобразуем в параметры LatticeConfig
        config_params = {
            "dimensions": tuple(lattice_data.get("dimensions", [8, 8, 8])),
            "boundary_conditions": lattice_data.get("boundary_conditions", "walls"),
            "parallel_processing": lattice_data.get("parallel_processing", True),
            "gpu_enabled": lattice_data.get("gpu_enabled", True),
            "batch_size": lattice_data.get("batch_size", 1),
        }

        # Добавляем вложенные параметры
        if "initialization" in lattice_data:
            init_data = lattice_data["initialization"]
            config_params.update(
                {
                    "initialization_method": init_data.get("method", "normal"),
                    "initialization_std": init_data.get("std", 0.1),
                    "initialization_mean": init_data.get("mean", 0.0),
                }
            )

        if "topology" in lattice_data:
            topo_data = lattice_data["topology"]
            config_params.update(
                {
                    "neighbors": topo_data.get("neighbors", 6),
                    "validate_connections": topo_data.get("validate_connections", True),
                    "cache_neighbors": topo_data.get("cache_neighbors", True),
                }
            )

        if "io_interfaces" in lattice_data:
            io_data = lattice_data["io_interfaces"]
            config_params.update(
                {
                    "input_face": Face(io_data.get("input_face", "front")),
                    "output_face": Face(io_data.get("output_face", "back")),
                    "embedding_mapping": io_data.get("embedding_mapping", "linear"),
                }
            )

        # НОВОЕ: Загрузка конфигурации I/O стратегии
        if "io_strategy" in lattice_data:
            io_strategy_data = lattice_data["io_strategy"]
            config_params.update(
                {
                    "placement_strategy": PlacementStrategy(
                        io_strategy_data.get("placement_method", "proportional")
                    ),
                    "io_strategy_config": io_strategy_data,
                }
            )
        else:
            # Значения по умолчанию
            config_params.update(
                {
                    "placement_strategy": PlacementStrategy.PROPORTIONAL,
                    "io_strategy_config": {
                        "coverage_ratio": {
                            "min_percentage": 7.8,
                            "max_percentage": 15.6,
                        },
                        "absolute_limits": {"min_points": 5, "max_points": 0},
                        "seed": 42,
                    },
                }
            )

        if "diagnostics" in lattice_data:
            diag_data = lattice_data["diagnostics"]
            config_params.update(
                {
                    "enable_logging": diag_data.get("enable_logging", True),
                    "log_level": diag_data.get("log_level", "INFO"),
                    "track_performance": diag_data.get("track_performance", True),
                    "validate_states": diag_data.get("validate_states", True),
                }
            )

        if "optimization" in lattice_data:
            opt_data = lattice_data["optimization"]
            config_params.update(
                {
                    "memory_efficient": opt_data.get("memory_efficient", True),
                    "use_checkpointing": opt_data.get("use_checkpointing", False),
                    "mixed_precision": opt_data.get("mixed_precision", False),
                }
            )

        # Загружаем конфигурацию cell_prototype если требуется автосинхронизация
        integration_data = yaml_data.get("integration", {})
        if integration_data.get("cell_prototype", {}).get("auto_sync", True):
            try:
                config_params["cell_config"] = load_cell_config()
                config_params["auto_sync_cell_config"] = True
                logging.info("Auto-synced with cell_prototype configuration")
            except Exception as e:
                logging.warning(f"Could not auto-sync cell_prototype config: {e}")
                config_params["auto_sync_cell_config"] = False

        return _build_lattice_config_from_data(lattice_data)

    except FileNotFoundError:
        logging.warning(
            f"Config file not found: {config_path}. Using default configuration."
        )
        return LatticeConfig()
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise


def _build_lattice_config_from_data(lattice_data: Dict[str, Any]) -> LatticeConfig:
    """
    Создание LatticeConfig из словаря данных.

    Args:
        lattice_data: Словарь с конфигурацией lattice_3d

    Returns:
        LatticeConfig: Настроенная конфигурация
    """
    # Преобразуем в параметры LatticeConfig
    config_params = {
        "dimensions": tuple(lattice_data.get("dimensions", [8, 8, 8])),
        "boundary_conditions": lattice_data.get("boundary_conditions", "walls"),
        "parallel_processing": lattice_data.get("parallel_processing", True),
        "gpu_enabled": lattice_data.get("gpu_enabled", True),
        "batch_size": lattice_data.get("batch_size", 1),
    }

    # Добавляем вложенные параметры
    if "initialization" in lattice_data:
        init_data = lattice_data["initialization"]
        config_params.update(
            {
                "initialization_method": init_data.get("method", "normal"),
                "initialization_std": init_data.get("std", 0.1),
                "initialization_mean": init_data.get("mean", 0.0),
            }
        )

    if "topology" in lattice_data:
        topo_data = lattice_data["topology"]
        config_params.update(
            {
                "neighbors": topo_data.get("neighbors", 6),
                "validate_connections": topo_data.get("validate_connections", True),
                "cache_neighbors": topo_data.get("cache_neighbors", True),
            }
        )

    if "io_interfaces" in lattice_data:
        io_data = lattice_data["io_interfaces"]
        config_params.update(
            {
                "input_face": Face(io_data.get("input_face", "front")),
                "output_face": Face(io_data.get("output_face", "back")),
                "embedding_mapping": io_data.get("embedding_mapping", "linear"),
            }
        )

    # НОВОЕ: Загрузка конфигурации I/O стратегии
    if "io_strategy" in lattice_data:
        io_strategy_data = lattice_data["io_strategy"]
        config_params.update(
            {
                "placement_strategy": PlacementStrategy(
                    io_strategy_data.get("placement_method", "proportional")
                ),
                "io_strategy_config": io_strategy_data,
            }
        )
    else:
        # Значения по умолчанию
        config_params.update(
            {
                "placement_strategy": PlacementStrategy.PROPORTIONAL,
                "io_strategy_config": {
                    "coverage_ratio": {"min_percentage": 7.8, "max_percentage": 15.6},
                    "absolute_limits": {"min_points": 5, "max_points": 0},
                    "seed": 42,
                },
            }
        )

    if "diagnostics" in lattice_data:
        diag_data = lattice_data["diagnostics"]
        config_params.update(
            {
                "enable_logging": diag_data.get("enable_logging", True),
                "log_level": diag_data.get("log_level", "INFO"),
                "track_performance": diag_data.get("track_performance", True),
                "validate_states": diag_data.get("validate_states", True),
            }
        )

    if "optimization" in lattice_data:
        opt_data = lattice_data["optimization"]
        config_params.update(
            {
                "memory_efficient": opt_data.get("memory_efficient", True),
                "use_checkpointing": opt_data.get("use_checkpointing", False),
                "mixed_precision": opt_data.get("mixed_precision", False),
            }
        )

    # Загружаем конфигурацию cell_prototype если требуется автосинхронизация
    try:
        from utils.config_manager import get_global_config_manager

        config_manager = get_global_config_manager()
        if config_manager:
            cell_config = config_manager.get_config("cell_prototype")
            if cell_config:
                config_params["cell_config"] = cell_config
                config_params["auto_sync_cell_config"] = True
                logging.info(
                    "Auto-synced with cell_prototype configuration from ConfigManager"
                )
            else:
                logging.info("cell_prototype config not found in ConfigManager")
    except Exception as e:
        logging.warning(f"Could not auto-sync cell_prototype config: {e}")
        config_params["auto_sync_cell_config"] = False

    return LatticeConfig(**config_params)


# =============================================================================
# СИСТЕМА КООРДИНАТ 3D
# =============================================================================


class Position3D:
    """
    Класс для работы с трехмерными координатами в решетке.

    Предоставляет функции для:
    - Преобразования между 3D координатами и линейными индексами
    - Валидации координат
    - Вычисления расстояний и соседства
    """

    def __init__(self, dimensions: Dimensions3D):
        """
        Инициализация системы координат.

        Args:
            dimensions: Размеры решетки (X, Y, Z)
        """
        self.dimensions = dimensions
        self.x_size, self.y_size, self.z_size = dimensions
        self.total_positions = self.x_size * self.y_size * self.z_size

    def to_linear_index(self, coords: Coordinates3D) -> int:
        """
        Преобразование 3D координат в линейный индекс.

        Использует порядок: Z * (X * Y) + Y * X + X

        Args:
            coords: 3D координаты (x, y, z)

        Returns:
            int: Линейный индекс
        """
        x, y, z = coords
        self._validate_coordinates(coords)
        return z * (self.x_size * self.y_size) + y * self.x_size + x

    def to_3d_coordinates(self, linear_index: int) -> Coordinates3D:
        """
        Преобразование линейного индекса в 3D координаты.

        Args:
            linear_index: Линейный индекс

        Returns:
            Coordinates3D: 3D координаты (x, y, z)
        """
        if not 0 <= linear_index < self.total_positions:
            raise ValueError(
                f"Linear index {linear_index} out of range [0, {self.total_positions})"
            )

        z = linear_index // (self.x_size * self.y_size)
        remainder = linear_index % (self.x_size * self.y_size)
        y = remainder // self.x_size
        x = remainder % self.x_size

        return (x, y, z)

    def _validate_coordinates(self, coords: Coordinates3D) -> None:
        """Валидация 3D координат"""
        x, y, z = coords
        if not (0 <= x < self.x_size):
            raise ValueError(f"X coordinate {x} out of range [0, {self.x_size})")
        if not (0 <= y < self.y_size):
            raise ValueError(f"Y coordinate {y} out of range [0, {self.y_size})")
        if not (0 <= z < self.z_size):
            raise ValueError(f"Z coordinate {z} out of range [0, {self.z_size})")

    def is_valid_coordinates(self, coords: Coordinates3D) -> bool:
        """Проверка валидности координат без исключения"""
        try:
            self._validate_coordinates(coords)
            return True
        except ValueError:
            return False

    def get_all_coordinates(self) -> List[Coordinates3D]:
        """Получение списка всех валидных координат в решетке"""
        coordinates = []
        for z in range(self.z_size):
            for y in range(self.y_size):
                for x in range(self.x_size):
                    coordinates.append((x, y, z))
        return coordinates

    def manhattan_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> int:
        """Манхэттенское расстояние между двумя точками"""
        return (
            abs(coord1[0] - coord2[0])
            + abs(coord1[1] - coord2[1])
            + abs(coord1[2] - coord2[2])
        )

    def euclidean_distance(self, coord1: Coordinates3D, coord2: Coordinates3D) -> float:
        """Евклидово расстояние между двумя точками"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))


# =============================================================================
# ТОПОЛОГИЯ СОСЕДСТВА
# =============================================================================


class NeighborTopology:
    """
    Система управления соседством клеток в 3D решетке.

    Реализует различные типы граничных условий и предоставляет
    эффективные методы для получения соседей каждой клетки.
    """

    # Направления к 6 соседям в 3D пространстве
    NEIGHBOR_DIRECTIONS = [
        (-1, 0, 0),  # Влево
        (1, 0, 0),  # Вправо
        (0, -1, 0),  # Вниз
        (0, 1, 0),  # Вверх
        (0, 0, -1),  # Назад
        (0, 0, 1),  # Вперед
    ]

    def __init__(self, config: LatticeConfig):
        """
        Инициализация системы соседства.

        Args:
            config: Конфигурация решетки
        """
        self.config = config
        self.dimensions = config.dimensions
        self.boundary_conditions = config.boundary_conditions
        self.position_system = Position3D(self.dimensions)

        # Кэш для карты соседства
        self._neighbor_cache: Dict[int, List[int]] = {}
        self._cache_initialized = False

        if config.cache_neighbors:
            self._build_neighbor_cache()

    def get_neighbors(self, coords: Coordinates3D) -> List[Coordinates3D]:
        """
        Получение соседей для заданных координат.

        Args:
            coords: 3D координаты клетки

        Returns:
            List[Coordinates3D]: Список координат соседних клеток
        """
        x, y, z = coords
        neighbors = []

        for dx, dy, dz in self.NEIGHBOR_DIRECTIONS:
            neighbor_coords = (x + dx, y + dy, z + dz)

            # Применяем граничные условия
            processed_coords = self._apply_boundary_conditions(neighbor_coords)

            if processed_coords is not None:
                neighbors.append(processed_coords)

        return neighbors

    def get_neighbor_indices(self, linear_index: int) -> List[int]:
        """
        Получение индексов соседей для линейного индекса.

        Args:
            linear_index: Линейный индекс клетки

        Returns:
            List[int]: Список линейных индексов соседних клеток
        """
        if self.config.cache_neighbors and self._cache_initialized:
            return self._neighbor_cache.get(linear_index, [])

        coords = self.position_system.to_3d_coordinates(linear_index)
        neighbor_coords = self.get_neighbors(coords)

        return [self.position_system.to_linear_index(nc) for nc in neighbor_coords]

    def _apply_boundary_conditions(
        self, coords: Coordinates3D
    ) -> Optional[Coordinates3D]:
        """
        Применение граничных условий к координатам.

        Args:
            coords: Координаты (могут быть вне границ)

        Returns:
            Optional[Coordinates3D]: Обработанные координаты или None если недоступны
        """
        x, y, z = coords
        x_size, y_size, z_size = self.dimensions

        if self.boundary_conditions == BoundaryCondition.WALLS:
            # Стенки: координаты вне границ недоступны
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return coords

        elif self.boundary_conditions == BoundaryCondition.PERIODIC:
            # Периодические: решетка замыкается в тор
            return (x % x_size, y % y_size, z % z_size)

        elif self.boundary_conditions == BoundaryCondition.ABSORBING:
            # Поглощающие: границы поглощают сигналы (как стенки для топологии)
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return coords

        elif self.boundary_conditions == BoundaryCondition.REFLECTING:
            # Отражающие: координаты отражаются от границ
            if x < 0:
                x = -x
            elif x >= x_size:
                x = 2 * x_size - x - 1

            if y < 0:
                y = -y
            elif y >= y_size:
                y = 2 * y_size - y - 1

            if z < 0:
                z = -z
            elif z >= z_size:
                z = 2 * z_size - z - 1

            # Проверяем, что отраженные координаты валидны
            if not (0 <= x < x_size and 0 <= y < y_size and 0 <= z < z_size):
                return None
            return (x, y, z)

        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary_conditions}")

    def _build_neighbor_cache(self):
        """
        Предварительное построение кэша соседства для всех клеток.

        Повышает производительность при частых запросах соседей.
        """
        logging.info("Building neighbor cache for lattice...")

        for linear_index in range(self.position_system.total_positions):
            coords = self.position_system.to_3d_coordinates(linear_index)
            neighbor_coords = self.get_neighbors(coords)
            neighbor_indices = [
                self.position_system.to_linear_index(nc) for nc in neighbor_coords
            ]
            self._neighbor_cache[linear_index] = neighbor_indices

        self._cache_initialized = True
        logging.info(f"Neighbor cache built: {len(self._neighbor_cache)} entries")

    def validate_topology(self) -> Dict[str, Any]:
        """
        Валидация топологии решетки и возврат статистики.

        Returns:
            Dict[str, Any]: Статистика топологии
        """
        stats = {
            "total_cells": self.position_system.total_positions,
            "boundary_conditions": self.boundary_conditions.value,
            "neighbor_counts": {},
            "symmetry_check": True,
            "connectivity_check": True,
        }

        # Подсчет количества соседей для каждой клетки
        neighbor_counts = []
        for i in range(self.position_system.total_positions):
            neighbors = self.get_neighbor_indices(i)
            neighbor_counts.append(len(neighbors))

        # Статистика количества соседей
        unique_counts = list(set(neighbor_counts))
        for count in unique_counts:
            stats["neighbor_counts"][str(count)] = neighbor_counts.count(count)

        # Проверка симметрии соседства
        for i in range(self.position_system.total_positions):
            neighbors = self.get_neighbor_indices(i)
            for neighbor_idx in neighbors:
                neighbor_neighbors = self.get_neighbor_indices(neighbor_idx)
                if i not in neighbor_neighbors:
                    stats["symmetry_check"] = False
                    logging.warning(
                        f"Asymmetric neighbor relationship: {i} -> {neighbor_idx}"
                    )

        logging.info(f"Topology validation completed: {stats}")
        return stats


# =============================================================================
# ОСНОВНОЙ КЛАСС РЕШЕТКИ
# =============================================================================


class Lattice3D(nn.Module):
    """
    Основной класс трехмерной решетки клеток.

    Управляет сеткой клеток cell_prototype, их состояниями и взаимодействием.
    Реализует синхронное обновление всех клеток и интерфейсы ввода/вывода.

    Биологическая аналогия: кора головного мозга - 3D ткань из одинаковых
    нейронов, где каждый нейрон обрабатывает сигналы от соседей.
    """

    def __init__(self, config: LatticeConfig):
        """
        Инициализация трехмерной решетки клеток.

        Args:
            config: Конфигурация решетки
        """
        super().__init__()
        self.config = config

        # Основные компоненты
        self.position_system = Position3D(config.dimensions)
        self.topology = NeighborTopology(config)

        # НОВОЕ: Инициализация системы размещения I/O точек
        self.io_placer = IOPointPlacer(
            lattice_dimensions=config.dimensions,
            strategy=config.placement_strategy,
            config=config.io_strategy_config or {},
            seed=(
                config.io_strategy_config.get("seed", 42)
                if config.io_strategy_config
                else 42
            ),
        )

        # Создание прототипа клетки (один для всех позиций)
        self.cell_prototype = self._create_cell_prototype()

        # Состояния всех клеток решетки
        self.states = self._initialize_states()

        # Кэширование для производительности
        self._face_indices = self._compute_face_indices()

        # Статистика и диагностика
        self.step_count = 0
        self.performance_stats = {
            "forward_calls": 0,
            "total_time": 0.0,
            "avg_time_per_step": 0.0,
        }

        logging.info(
            f"Lattice3D initialized: {config.dimensions} = {config.total_cells} cells"
        )
        if config.track_performance:
            logging.info(f"Performance tracking enabled, device: {config.device}")

    def _create_cell_prototype(self) -> CellPrototype:
        """
        Создание прототипа клетки на основе конфигурации.

        Returns:
            CellPrototype: Настроенный прототип клетки
        """
        # Получаем конфигурацию cell_prototype
        if self.config.cell_config is not None:
            cell_config = self.config.cell_config
        else:
            # Загружаем базовую конфигурацию
            cell_config = load_cell_config()

        # Создаем экземпляр клетки
        try:
            # Check if this is NCA or gMLP cell configuration
            if isinstance(cell_config, dict) and cell_config.get("cell_type") == "nca":
                # Create NCA cell
                from core.cell_prototype.architectures.minimal_nca_cell import (
                    MinimalNCACell,
                )

                nca_params = {
                    "state_size": cell_config.get("state_size", 8),
                    "neighbor_count": cell_config.get("neighbor_count", 6),
                    "hidden_dim": cell_config.get("hidden_dim", 4),
                    "external_input_size": cell_config.get("external_input_size", 1),
                    "activation": cell_config.get("activation", "tanh"),
                    "target_params": cell_config.get("target_params", 150),
                }

                cell = MinimalNCACell(**nca_params)
                cell = cell.to(self.config.device)

                logging.info(
                    f"NCA cell prototype created: input_size={cell.external_input_size}, state_size={cell.state_size}"
                )

            elif (
                isinstance(cell_config, dict) and cell_config.get("cell_type") == "gmlp"
            ):
                # Create gMLP cell - fallback to existing logic
                cell = create_cell_from_config(cell_config)  # Передаем конфигурацию
                cell = cell.to(self.config.device)

                logging.info(
                    f"gMLP cell prototype created: input_size={cell.input_size}, state_size={cell.state_size}"
                )

            else:
                # Fallback to standard cell_prototype creation
                cell = create_cell_from_config(cell_config)  # Передаем конфигурацию
                cell = cell.to(self.config.device)

                logging.info(
                    f"Default cell prototype created: input_size={cell.input_size}, state_size={cell.state_size}"
                )

            return cell

        except Exception as e:
            logging.error(f"Failed to create cell prototype: {e}")
            raise RuntimeError(f"Cell prototype creation failed: {e}")

    def _initialize_states(self) -> torch.Tensor:
        """
        Инициализация состояний всех клеток решетки.

        Returns:
            torch.Tensor: Тензор состояний [total_cells, state_size]
        """
        total_cells = self.config.total_cells
        state_size = self.cell_prototype.state_size

        # Выбираем метод инициализации
        if self.config.initialization_method == "normal":
            states = torch.normal(
                mean=self.config.initialization_mean,
                std=self.config.initialization_std,
                size=(total_cells, state_size),
            )
        elif self.config.initialization_method == "uniform":
            states = torch.uniform(
                low=-self.config.initialization_std,
                high=self.config.initialization_std,
                size=(total_cells, state_size),
            )
        elif self.config.initialization_method == "zeros":
            states = torch.zeros(total_cells, state_size)
        else:
            raise ValueError(
                f"Unknown initialization method: {self.config.initialization_method}"
            )

        states = states.to(self.config.device)

        # Регистрируем как параметр модуля для обучения
        self.register_buffer("_states", states)

        logging.info(
            f"States initialized: {states.shape} with method '{self.config.initialization_method}'"
        )
        return states

    def _compute_face_indices(self) -> Dict[Face, List[int]]:
        """
        Предвычисление индексов клеток на каждой грани решетки.

        Returns:
            Dict[Face, List[int]]: Словарь граней и соответствующих индексов
        """
        x_size, y_size, z_size = self.config.dimensions
        face_indices = {}

        # FRONT face (Z = 0)
        face_indices[Face.FRONT] = [
            self.position_system.to_linear_index((x, y, 0))
            for x in range(x_size)
            for y in range(y_size)
        ]

        # BACK face (Z = max)
        face_indices[Face.BACK] = [
            self.position_system.to_linear_index((x, y, z_size - 1))
            for x in range(x_size)
            for y in range(y_size)
        ]

        # LEFT face (X = 0)
        face_indices[Face.LEFT] = [
            self.position_system.to_linear_index((0, y, z))
            for y in range(y_size)
            for z in range(z_size)
        ]

        # RIGHT face (X = max)
        face_indices[Face.RIGHT] = [
            self.position_system.to_linear_index((x_size - 1, y, z))
            for y in range(y_size)
            for z in range(z_size)
        ]

        # TOP face (Y = max)
        face_indices[Face.TOP] = [
            self.position_system.to_linear_index((x, y_size - 1, z))
            for x in range(x_size)
            for z in range(z_size)
        ]

        # BOTTOM face (Y = 0)
        face_indices[Face.BOTTOM] = [
            self.position_system.to_linear_index((x, 0, z))
            for x in range(x_size)
            for z in range(z_size)
        ]

        logging.info(
            f"Face indices computed: {[(face.name, len(indices)) for face, indices in face_indices.items()]}"
        )
        return face_indices

    def forward(self, external_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Один шаг обновления всех клеток решетки.

        Args:
            external_inputs: Внешние входы [batch_size, input_size] или None

        Returns:
            torch.Tensor: Новые состояния всех клеток [total_cells, state_size]
        """
        import time

        start_time = time.time()

        # Применяем cell_prototype ко всем клеткам параллельно
        if self.config.parallel_processing:
            new_states = self._parallel_forward(external_inputs)
        else:
            new_states = self._sequential_forward(external_inputs)

        # Обновляем состояния (синхронно)
        self._states = new_states
        self.step_count += 1

        # Обновляем статистику производительности
        if self.config.track_performance:
            self._update_performance_stats(time.time() - start_time)

        return new_states

    def _parallel_forward(
        self, external_inputs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Параллельное обновление всех клеток.

        Args:
            external_inputs: Внешние входы или None

        Returns:
            torch.Tensor: Новые состояния всех клеток
        """
        total_cells = self.config.total_cells
        batch_size = total_cells

        # Собираем данные для всех клеток
        all_neighbor_states = []
        all_own_states = []
        all_external_inputs = []

        for cell_idx in range(total_cells):
            # Получаем соседей для каждой клетки
            neighbor_indices = self.topology.get_neighbor_indices(cell_idx)

            # Собираем состояния соседей
            neighbor_states = []
            for neighbor_idx in neighbor_indices:
                neighbor_states.append(self._states[neighbor_idx])

            # Если соседей меньше 6, дополняем нулями
            while len(neighbor_states) < 6:
                neighbor_states.append(torch.zeros_like(self._states[0]))

            # Формируем тензор состояний соседей [1, 6, state_size]
            neighbor_tensor = torch.stack(neighbor_states[:6]).unsqueeze(0)
            all_neighbor_states.append(neighbor_tensor)

            # Собственное состояние [1, state_size]
            own_state = self._states[cell_idx].unsqueeze(0)
            all_own_states.append(own_state)

            # Внешний вход для этой клетки
            if external_inputs is not None:
                ext_input = self._get_external_input_for_cell(cell_idx, external_inputs)
            else:
                ext_input = torch.zeros(
                    1, self.cell_prototype.input_size, device=self.config.device
                )
            all_external_inputs.append(ext_input)

        # Объединяем все данные в батчи
        batch_neighbor_states = torch.cat(
            all_neighbor_states, dim=0
        )  # [total_cells, 6, state_size]
        batch_own_states = torch.cat(all_own_states, dim=0)  # [total_cells, state_size]
        batch_external_inputs = torch.cat(
            all_external_inputs, dim=0
        )  # [total_cells, input_size]

        # Применяем cell_prototype ко всем клеткам одновременно
        new_states = self.cell_prototype(
            batch_neighbor_states, batch_own_states, batch_external_inputs
        )

        return new_states

    def _sequential_forward(
        self, external_inputs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Последовательное обновление клеток (для отладки).

        Args:
            external_inputs: Внешние входы или None

        Returns:
            torch.Tensor: Новые состояния всех клеток
        """
        new_states = torch.zeros_like(self._states)

        for cell_idx in range(self.config.total_cells):
            # Получаем соседей
            neighbor_indices = self.topology.get_neighbor_indices(cell_idx)

            # Собираем состояния соседей
            neighbor_states = []
            for neighbor_idx in neighbor_indices:
                neighbor_states.append(self._states[neighbor_idx])

            # Дополняем до 6 соседей
            while len(neighbor_states) < 6:
                neighbor_states.append(torch.zeros_like(self._states[0]))

            # Формируем входные данные
            neighbor_tensor = torch.stack(neighbor_states[:6]).unsqueeze(
                0
            )  # [1, 6, state_size]
            own_state = self._states[cell_idx].unsqueeze(0)  # [1, state_size]

            # Внешний вход
            if external_inputs is not None:
                ext_input = self._get_external_input_for_cell(cell_idx, external_inputs)
            else:
                ext_input = torch.zeros(
                    1, self.cell_prototype.input_size, device=self.config.device
                )

            # Применяем cell_prototype
            new_state = self.cell_prototype(neighbor_tensor, own_state, ext_input)
            new_states[cell_idx] = new_state.squeeze(0)

        return new_states

    def _get_external_input_for_cell(
        self, cell_idx: int, external_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Получение внешнего входа для конкретной клетки.

        Args:
            cell_idx: Индекс клетки
            external_inputs: Тензор всех внешних входов

        Returns:
            torch.Tensor: Внешний вход для данной клетки [1, input_size]
        """
        # Determine input size based on cell type
        if hasattr(self.cell_prototype, "external_input_size"):
            # NCA cell
            input_size = self.cell_prototype.external_input_size
        else:
            # gMLP or other cell
            input_size = self.cell_prototype.input_size

        # Проверяем, является ли клетка точкой ввода
        input_points_3d = self.io_placer.get_input_points(self.config.input_face)
        input_point_indices = [
            self.position_system.to_linear_index(point_3d)
            for point_3d in input_points_3d
        ]

        if cell_idx in input_point_indices:
            # Находим индекс этой точки в списке точек ввода
            point_index = input_point_indices.index(cell_idx)

            if point_index < external_inputs.shape[0]:
                # Проверяем соответствие размеров
                if external_inputs.shape[1] == input_size:
                    return external_inputs[point_index].unsqueeze(0)
                else:
                    # Адаптируем размер входа (например, обрезаем или дополняем нулями)
                    ext_input = external_inputs[point_index]
                    if ext_input.size(0) > input_size:
                        ext_input = ext_input[:input_size]
                    elif ext_input.size(0) < input_size:
                        padding = torch.zeros(
                            input_size - ext_input.size(0), device=self.config.device
                        )
                        ext_input = torch.cat([ext_input, padding])

                    return ext_input.unsqueeze(0)

        # Если клетка не является точкой ввода или нет соответствующего входа
        return torch.zeros(1, input_size, device=self.config.device)

    def _update_performance_stats(self, step_time: float):
        """Обновление статистики производительности"""
        self.performance_stats["forward_calls"] += 1
        self.performance_stats["total_time"] += step_time
        self.performance_stats["avg_time_per_step"] = (
            self.performance_stats["total_time"]
            / self.performance_stats["forward_calls"]
        )

    # Дополнительные методы для управления решеткой (будут добавлены в следующих шагах)

    def get_states(self) -> torch.Tensor:
        """Получение текущих состояний всех клеток"""
        return self._states.clone()

    def set_states(self, new_states: torch.Tensor):
        """Установка новых состояний всех клеток"""
        if new_states.shape != self._states.shape:
            raise ValueError(
                f"States shape mismatch: expected {self._states.shape}, got {new_states.shape}"
            )
        self._states = new_states.to(self.config.device)

    def get_face_states(self, face: Face) -> torch.Tensor:
        """Получение состояний клеток на указанной грани"""
        face_indices = self._face_indices[face]
        return self._states[face_indices]

    def get_output_states(self) -> torch.Tensor:
        """
        НОВОЕ: Получение состояний только из выходных точек с использованием I/O стратегии.

        Returns:
            torch.Tensor: Состояния клеток в выходных точках [num_output_points, state_size]
        """
        # Используем IOPointPlacer для определения точек вывода
        output_points_3d = self.io_placer.get_output_points(self.config.output_face)

        # Конвертируем 3D координаты в линейные индексы
        output_point_indices = []
        for point_3d in output_points_3d:
            linear_idx = self.position_system.to_linear_index(point_3d)
            output_point_indices.append(linear_idx)

        # Возвращаем состояния только выходных точек
        return self._states[output_point_indices]

    def get_io_point_info(self) -> Dict[str, Any]:
        """
        НОВОЕ: Получение информации о размещении I/O точек.

        Returns:
            Dict[str, Any]: Информация о точках ввода/вывода
        """
        input_points_3d = self.io_placer.get_input_points(self.config.input_face)
        output_points_3d = self.io_placer.get_output_points(self.config.output_face)

        # Подсчитываем площади граней
        face_area_input = len(
            self.io_placer._get_face_coordinates(self.config.input_face)
        )
        face_area_output = len(
            self.io_placer._get_face_coordinates(self.config.output_face)
        )

        return {
            "strategy": self.config.placement_strategy.value,
            "input_face": self.config.input_face.value,
            "output_face": self.config.output_face.value,
            "input_points": {
                "count": len(input_points_3d),
                "coordinates": input_points_3d,
                "coverage_percentage": (len(input_points_3d) / face_area_input) * 100,
            },
            "output_points": {
                "count": len(output_points_3d),
                "coordinates": output_points_3d,
                "coverage_percentage": (len(output_points_3d) / face_area_output) * 100,
            },
            "face_areas": {"input": face_area_input, "output": face_area_output},
        }

    def reset_states(self):
        """Сброс состояний к начальным значениям"""
        self._states = self._initialize_states()
        self.step_count = 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return self.performance_stats.copy()


# =============================================================================
# УТИЛИТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================


def create_lattice_from_config(config_path: Optional[str] = None) -> Lattice3D:
    """
    Создание решетки на основе конфигурационного файла.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Lattice3D: Созданная и настроенная решетка
    """
    config = load_lattice_config(config_path)
    return Lattice3D(config)


def validate_lattice_config(config: LatticeConfig) -> Dict[str, Any]:
    """
    Валидация конфигурации решетки.

    Args:
        config: Конфигурация для проверки

    Returns:
        Dict[str, Any]: Результаты валидации
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # Проверка размеров
    if config.total_cells > 100000:
        validation_results["warnings"].append(
            f"Large lattice size ({config.total_cells} cells) may impact performance"
        )

    # Проверка GPU
    if config.gpu_enabled and not torch.cuda.is_available():
        validation_results["warnings"].append(
            "GPU enabled but CUDA not available, falling back to CPU"
        )

    # Проверка конфигурации клеток
    if config.cell_config is None and config.auto_sync_cell_config:
        validation_results["recommendations"].append(
            "Consider loading cell_prototype configuration for better integration"
        )

    return validation_results


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СОВМЕСТИМОСТИ
# =============================================================================


def load_cell_config() -> Dict[str, Any]:
    """
    Загрузка конфигурации cell_prototype для совместимости.

    Returns:
        Dict[str, Any]: Базовая конфигурация cell_prototype
    """
    return {
        "cell_prototype": {
            "input_size": 12,
            "state_size": 8,
            "architecture": {"hidden_size": 16, "activation": "tanh", "use_bias": True},
        }
    }


# =============================================================================
# ЭКСПОРТЫ МОДУЛЯ
# =============================================================================

__all__ = [
    # Основные классы
    "Lattice3D",
    "LatticeConfig",
    "Position3D",
    "NeighborTopology",
    "IOPointPlacer",  # НОВОЕ
    # Енумы
    "BoundaryCondition",
    "Face",
    "PlacementStrategy",  # НОВОЕ
    # Функции
    "load_lattice_config",
    "create_lattice_from_config",
    "validate_lattice_config",
    # Типы
    "Coordinates3D",
    "Dimensions3D",
]
