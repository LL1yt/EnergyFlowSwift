"""
Модуль 3D Решетка клеток
========================

Основной класс для создания и управления трёхмерной решеткой клеток.
Содержит базовую функциональность forward pass'а и управления состояниями.

Пластичность (STDP, конкурентное обучение) вынесена в отдельный модуль.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
import logging
import time
import collections
from datetime import datetime
import json

# Импорты из других модулей проекта
from core.cell_prototype import CellPrototype, create_cell_from_config

# Локальные импорты из этого же модуля
from .config import LatticeConfig, load_lattice_config
from .enums import Face
from .io import IOPointPlacer
from .position import Position3D
from .topology import NeighborTopology
from .plasticity import PlasticityMixin
from .clustering import ClusteringMixin
from ..log_utils import _get_caller_info

# Добавляем импорт для коллекций
from collections import deque


class Lattice3D(nn.Module, PlasticityMixin, ClusteringMixin):
    """
    Трёхмерная решётка клеток (нейронов).

    Наследует от:
    - PlasticityMixin: механизмы пластичности (STDP, конкурентное обучение, BCM)
    - ClusteringMixin: функциональная кластеризация с координацией
    """

    def __init__(self, config: LatticeConfig):
        """
        Инициализация решетки.

        Args:
            config: Объект конфигурации LatticeConfig.
        """
        super().__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if config.gpu_enabled and torch.cuda.is_available() else "cpu"
        )
        self.pos_helper = Position3D(config.dimensions)
        self.logger = logging.getLogger(__name__)

        # --- Enhanced Initialization Logging ---
        caller_info = _get_caller_info()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = {"error": "Failed to serialize config"}

        self.logger.info(
            f"[START] INIT Lattice3D @ {timestamp}\n"
            f"     FROM: {caller_info}\n"
            f"     WITH_CONFIG: {json.dumps(config_dict, indent=2, default=str)}"
        )
        # --- End of Logging ---

        if config.enable_logging:
            self.logger.info(f"Initializing Lattice3D on device: {self.device}")
            self.logger.info(
                f"Dimensions: {config.dimensions}, Total cells: {config.total_cells}"
            )
            self.logger.info(
                f"Neighbor strategy: {getattr(config, 'neighbor_finding_strategy', 'tiered')}"
            )

        # Создаем прототип клетки
        self.cell_prototype = self._create_cell_prototype()
        self.state_size = self.cell_prototype.state_size

        # Размещение I/O точек
        io_seed = config.seed
        if config.io_strategy_config and "seed" in config.io_strategy_config:
            io_seed = config.io_strategy_config["seed"]

        self.io_placer = IOPointPlacer(
            config.dimensions,
            config.placement_strategy,
            config.io_strategy_config or {},
            seed=io_seed,
        )
        self.input_points = self.io_placer.get_input_points(config.input_face)
        self.output_points = self.io_placer.get_output_points(config.output_face)
        self.input_indices = [
            self.pos_helper.to_linear_index(p) for p in self.input_points
        ]
        self.output_indices = [
            self.pos_helper.to_linear_index(p) for p in self.output_points
        ]

        # Инициализация топологии соседства
        all_coords = self.pos_helper.get_all_coordinates()
        self.topology = NeighborTopology(config, all_coords=all_coords)

        # Веса соединений
        self.connection_weights = torch.ones(
            self.config.total_cells,
            self.config.neighbors,
            dtype=torch.float32,
            device=self.device,
        )
        self.logger.info(
            f"Connection weights tensor created with shape: {self.connection_weights.shape}"
        )

        # === Инициализация пластичности (через mixin) ===
        self._init_plasticity()

        # === Инициализация кластеризации (через mixin) ===
        self._init_clustering(config.to_dict())

        # Инициализация состояний клеток
        self.states = self._initialize_states()

        # Кэш для индексов граней
        self._face_indices_cache = self._compute_face_indices()

        # Отслеживание производительности
        self.perf_stats = {
            "total_steps": 0,
            "total_time": 0.0,
            "avg_time_per_step": 0.0,
        }

        if self.config.validate_states:
            self._validate_initial_setup()

    def _create_cell_prototype(self) -> CellPrototype:
        """
        Создает экземпляр прототипа клетки на основе конфигурации.
        """
        if self.config.auto_sync_cell_config:
            # Новая структура конфигурации с cell_prototype
            if self.config.cell_config.get("cell_prototype"):
                prototype_config = self.config.cell_config["cell_prototype"]
                if prototype_config.get("minimal_nca_cell"):
                    prototype_config["minimal_nca_cell"][
                        "neighbor_count"
                    ] = self.config.neighbors
                elif prototype_config.get("gmlp_opt_connections"):
                    self.logger.info(
                        f"ERROR: тут проверить для чего используется gmlp_opt_connections @ \n"
                    )
                    prototype_config["gmlp_opt_connections"][
                        "neighbor_count"
                    ] = self.config.neighbors

            # Старая структура для совместимости
            elif self.config.cell_config.get("gmlp_opt_connections"):
                self.logger.info(
                    f"ERROR: тут проверить для чего используется gmlp_opt_connections @ \n"
                )
                self.config.cell_config["gmlp_opt_connections"][
                    "neighbor_count"
                ] = self.config.neighbors
            elif self.config.cell_config.get("minimal_nca_cell"):
                self.config.cell_config["minimal_nca_cell"][
                    "neighbor_count"
                ] = self.config.neighbors

        # Извлекаем правильную конфигурацию для create_cell_from_config
        if "cell_prototype" in self.config.cell_config:
            # Новая структура: передаем содержимое cell_prototype
            config_for_cell = self.config.cell_config["cell_prototype"]
        else:
            # Старая структура: передаем как есть
            config_for_cell = self.config.cell_config

        cell_prototype = create_cell_from_config(config_for_cell)
        # Перемещаем прототип на нужное устройство
        return cell_prototype.to(self.device)

    def _initialize_states(self) -> torch.Tensor:
        """
        Инициализирует тензор состояний клеток.
        """
        init_method = self.config.initialization_method
        dims = (self.config.total_cells, self.state_size)

        # Устанавливаем seed для воспроизводимости инициализации
        torch.manual_seed(self.config.seed)

        if init_method == "zeros":
            states = torch.zeros(dims, device=self.device)
        elif init_method == "ones":
            states = torch.ones(dims, device=self.device)
        elif init_method == "uniform":
            states = torch.rand(dims, device=self.device)
        elif init_method == "normal":
            states = (
                torch.randn(dims, device=self.device) * self.config.initialization_std
                + self.config.initialization_mean
            )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        return states

    def _compute_face_indices(self) -> Dict[Face, List[int]]:
        """
        Предварительно вычисляет линейные индексы клеток для каждой грани.
        """
        x_size, y_size, z_size = self.config.dimensions
        face_indices: Dict[Face, List[int]] = {face: [] for face in Face}

        for x in range(x_size):
            for y in range(y_size):
                face_indices[Face.FRONT].append(
                    self.pos_helper.to_linear_index((x, y, 0))
                )
                face_indices[Face.BACK].append(
                    self.pos_helper.to_linear_index((x, y, z_size - 1))
                )

        for y in range(y_size):
            for z in range(z_size):
                face_indices[Face.LEFT].append(
                    self.pos_helper.to_linear_index((0, y, z))
                )
                face_indices[Face.RIGHT].append(
                    self.pos_helper.to_linear_index((x_size - 1, y, z))
                )

        for x in range(x_size):
            for z in range(z_size):
                face_indices[Face.TOP].append(
                    self.pos_helper.to_linear_index((x, y_size - 1, z))
                )
                face_indices[Face.BOTTOM].append(
                    self.pos_helper.to_linear_index((x, 0, z))
                )

        return face_indices

    def forward(self, external_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Выполняет один шаг обновления состояния решетки.
        """
        start_time = time.time()

        if self.config.parallel_processing:
            new_states = self._parallel_forward(external_inputs)
        else:
            raise NotImplementedError("Sequential processing is currently disabled.")

        self.states = new_states

        step_time = time.time() - start_time
        self._update_performance_stats(step_time)

        return self.states

    def _parallel_forward(
        self, external_inputs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Параллельный forward pass для всех клеток.
        """
        # 1. Сбор состояний соседей
        neighbor_indices = self.topology.get_all_neighbors_flat().to(self.device)
        neighbor_states = self.states[neighbor_indices]

        # 2. Подготовка внешних входов
        # Транслируем external_inputs на все клетки
        ext_input_expanded = torch.zeros(
            (self.config.total_cells, self.cell_prototype.external_input_size),
            device=self.device,
        )
        if external_inputs is not None and len(self.input_indices) > 0:
            # Используем mean() для агрегации если входов несколько, а точка одна
            if external_inputs.shape[0] > len(self.input_indices):
                aggregated_input = external_inputs.mean(dim=0, keepdim=True)
                ext_input_expanded[self.input_indices] = aggregated_input
            else:
                ext_input_expanded[self.input_indices] = external_inputs

        # 3. Прямой проход через все клетки
        new_states = self.cell_prototype(
            neighbor_states, self.states, ext_input_expanded
        )

        # 4. Применение маски замороженных клеток
        if self.config.frozen_cells_mask is not None:
            frozen_mask = self.config.frozen_cells_mask.to(self.device)
            new_states = torch.where(frozen_mask, self.states, new_states)

        return new_states

    def _get_external_input_for_cell(
        self, cell_idx: int, external_inputs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        (Больше не используется в параллельной версии)
        Возвращает внешний вход для конкретной клетки.
        """
        if external_inputs is None:
            return None
        try:
            input_pos_index = self.input_indices.index(cell_idx)
            return external_inputs[input_pos_index]
        except ValueError:
            return None

    def _update_performance_stats(self, step_time: float):
        """Обновляет статистику производительности."""
        self.perf_stats["total_steps"] += 1
        self.perf_stats["total_time"] += step_time
        self.perf_stats["avg_time_per_step"] = (
            self.perf_stats["total_time"] / self.perf_stats["total_steps"]
        )

    def get_states(self) -> torch.Tensor:
        return self.states.detach().cpu()

    def set_states(self, new_states: torch.Tensor):
        if new_states.shape != self.states.shape:
            raise ValueError("Shape of new_states must match the lattice states.")
        self.states = new_states.to(self.device)

    def get_face_states(self, face: Face) -> torch.Tensor:
        indices = self._face_indices_cache[face]
        return self.states[indices].detach().cpu()

    def get_output_states(self) -> torch.Tensor:
        return self.states[self.output_indices].detach().cpu()

    def get_io_point_info(self) -> Dict[str, Any]:
        """Возвращает информацию о точках ввода/вывода."""
        return {
            "input_points": [p.to_tuple() for p in self.input_points],
            "output_points": [p.to_tuple() for p in self.output_points],
            "input_indices": self.input_indices,
            "output_indices": self.output_indices,
            "input_face": self.config.input_face.name,
            "output_face": self.config.output_face.name,
        }

    def reset_states(self):
        """Сбрасывает состояние решетки к начальному."""
        self.logger.info("Resetting lattice states.")
        self.states = self._initialize_states()
        self.cell_prototype.reset_memory()
        self.perf_stats = {
            "total_steps": 0,
            "total_time": 0.0,
            "avg_time_per_step": 0.0,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.perf_stats

    def _validate_initial_setup(self):
        """Проверяет корректность начальных состояний и конфигурации."""
        assert (
            self.states.shape[0] == self.config.total_cells
        ), "Mismatch in state count"
        assert self.states.shape[1] == self.state_size, "Mismatch in state size"
        assert not torch.isnan(self.states).any(), "NaNs in initial states"
        self.logger.info("Initial setup validation passed.")


def create_lattice_from_config(
    config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None
) -> Lattice3D:
    """
    Фабричная функция для создания Lattice3D из файла конфигурации или словаря.
    """
    if config_path:
        config = load_lattice_config(config_path)
    elif config_dict:
        config = LatticeConfig.from_dict(config_dict)
    else:
        raise ValueError("Either config_path or config_dict must be provided.")
    return Lattice3D(config)
