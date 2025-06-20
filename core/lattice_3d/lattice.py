"""
Основной модуль 3D Решетки
==========================

Содержит главный класс `Lattice3D`, который объединяет все компоненты:
конфигурацию, топологию, I/O, состояния клеток и логику их обновления.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
import logging
import time

# Импорты из других модулей проекта
from core.cell_prototype import CellPrototype, create_cell_from_config

# Локальные импорты из этого же модуля
from .config import LatticeConfig, load_lattice_config
from .enums import Face
from .io import IOPointPlacer
from .position import Position3D
from .topology import NeighborTopology

# Добавляем импорт для коллекций
from collections import deque


class Lattice3D(nn.Module):
    """
    Главный класс, реализующий 3D решетку клеток.
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

        if config.enable_logging:
            self.logger.info(f"Initializing Lattice3D on device: {self.device}")
            self.logger.info(
                f"Dimensions: {config.dimensions}, Total cells: {config.total_cells}"
            )
            self.logger.info(
                f"Neighbor strategy: {getattr(config, 'neighbor_finding_strategy', 'local')}"
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
        self.register_buffer(
            "connection_weights",
            torch.ones(
                self.config.total_cells, self.config.neighbors, device=self.device
            ),
        )
        self.logger.info(
            f"Connection weights tensor created with shape: {self.connection_weights.shape}"
        )

        # === НОВОЕ: STDP механизм ===
        # Отслеживание активности для STDP
        self.enable_stdp = getattr(config, "enable_plasticity", False)
        if self.enable_stdp:

            # Circular buffer для истории активности (memory efficient)
            self.activity_history_size = getattr(config, "activity_history_size", 10)
            self.activity_history = deque(maxlen=self.activity_history_size)

            # STDP параметры
            self.stdp_config = getattr(config, "stdp_config", {})
            self.activity_threshold = self.stdp_config.get("activity_threshold", 0.1)
            self.learning_rate = self.stdp_config.get("learning_rate", 0.01)
            self.A_plus = self.stdp_config.get("A_plus", 0.01)  # LTP амплитуда
            self.A_minus = self.stdp_config.get("A_minus", 0.01)  # LTD амплитуда
            self.tau_plus = self.stdp_config.get("tau_plus", 20)  # LTP time constant
            self.tau_minus = self.stdp_config.get("tau_minus", 20)  # LTD time constant
            self.weight_bounds = self.stdp_config.get("weight_bounds", [0.1, 2.0])

            self.logger.info("STDP mechanism enabled with parameters:")
            self.logger.info(f"  Activity threshold: {self.activity_threshold}")
            self.logger.info(f"  Learning rate: {self.learning_rate}")
            self.logger.info(f"  Weight bounds: {self.weight_bounds}")
        else:
            self.previous_states = None
            self.activity_history = None

        # Инициализация состояний клеток
        self.states = self._initialize_states()

        # Инициализируем previous_states после создания self.states
        if self.enable_stdp:
            self.previous_states = torch.zeros_like(self.states)

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
                elif prototype_config.get("gmlp_cell"):
                    prototype_config["gmlp_cell"][
                        "neighbor_count"
                    ] = self.config.neighbors

            # Старая структура для совместимости
            elif self.config.cell_config.get("gmlp_cell"):
                self.config.cell_config["gmlp_cell"][
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

        # === НОВОЕ: Сохраняем предыдущие состояния для STDP ===
        if self.enable_stdp:
            self.previous_states = self.states.clone()

        # Принудительно используем векторизованный метод
        new_states = self._parallel_forward(external_inputs)

        self.states = new_states

        # === НОВОЕ: Отслеживание активности для STDP ===
        if self.enable_stdp:
            # Вычисляем активность как норму изменения состояния
            state_change = torch.norm(new_states - self.previous_states, dim=1)
            active_cells = (
                (state_change > self.activity_threshold).detach().cpu().numpy()
            )

            # Добавляем в circular buffer
            self.activity_history.append(
                {
                    "step": self.perf_stats["total_steps"],
                    "active_cells": active_cells,
                    "state_change": state_change.detach().cpu().numpy(),
                }
            )

        if self.config.track_performance:
            self._update_performance_stats(time.time() - start_time)

        self.perf_stats["total_steps"] += 1
        return self.states

    def _parallel_forward(
        self, external_inputs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Параллельное (векторизованное) обновление всех клеток (быстро, для GPU)."""
        # 1. Получаем индексы всех соседей для всех клеток одним батчем
        neighbor_indices = self.topology.get_all_neighbor_indices_batched()

        # 2. Собираем состояния соседей
        neighbor_states = self.states[neighbor_indices]

        # 3. Получаем веса соединений
        neighbor_weights = self.connection_weights

        # 4. Подготавливаем external_input для всех клеток
        if external_inputs is not None:
            # ОТЛАДКА: логируем размеры
            self.logger.debug(f"[DEBUG] external_inputs shape: {external_inputs.shape}")
            self.logger.debug(f"[DEBUG] total_cells: {self.config.total_cells}")
            self.logger.debug(f"[DEBUG] input_indices count: {len(self.input_indices)}")

            # Создаем тензор для всех клеток
            all_external_inputs = torch.zeros(
                self.config.total_cells,
                self.cell_prototype.external_input_size,
                device=self.device,
            )

            # Заполняем только input клетки
            for i, input_idx in enumerate(self.input_indices):
                if i < external_inputs.shape[0]:
                    all_external_inputs[input_idx] = external_inputs[i]

            external_inputs_for_cells = all_external_inputs
        else:
            external_inputs_for_cells = None

        # 5. Вызываем прототип клетки для всех клеток сразу
        # Проверяем, поддерживает ли клетка connection_weights (gMLP)
        if (
            hasattr(self.cell_prototype, "forward")
            and "connection_weights" in self.cell_prototype.forward.__code__.co_varnames
        ):
            new_states = self.cell_prototype(
                neighbor_states,
                self.states,
                neighbor_weights,
                external_inputs_for_cells,
            )
        else:
            # Fallback для простых клеток без поддержки весов
            new_states = self.cell_prototype(
                neighbor_states, self.states, external_inputs_for_cells
            )

        return new_states

    def _get_external_input_for_cell(
        self, cell_idx: int, external_inputs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if external_inputs is None:
            return None

        try:
            input_position = self.input_indices.index(cell_idx)
            return external_inputs[input_position]
        except ValueError:
            return None

    def _update_performance_stats(self, step_time: float):
        self.perf_stats["total_time"] += step_time
        self.perf_stats["avg_time_per_step"] = self.perf_stats["total_time"] / (
            self.perf_stats["total_steps"] + 1
        )

    def get_states(self) -> torch.Tensor:
        return self.states

    def set_states(self, new_states: torch.Tensor):
        if new_states.shape != self.states.shape:
            raise ValueError("Shape of new states must match existing states.")
        self.states = new_states

    def get_face_states(self, face: Face) -> torch.Tensor:
        face_indices = self._face_indices_cache.get(face, [])
        return self.states[face_indices]

    def get_output_states(self) -> torch.Tensor:
        return self.states[self.output_indices]

    def get_io_point_info(self) -> Dict[str, Any]:
        return {
            "input_points_count": len(self.input_points),
            "output_points_count": len(self.output_points),
            "input_points_coords": self.input_points,
            "output_points_coords": self.output_points,
        }

    def reset_states(self):
        self.states = self._initialize_states()
        self.perf_stats["total_steps"] = 0
        self.perf_stats["total_time"] = 0.0
        self.perf_stats["avg_time_per_step"] = 0.0

    def apply_stdp_update(self) -> Dict[str, Any]:
        """
        Применяет STDP правило для обновления весов связей.

        Основано на биологическом правиле:
        - LTP (Long Term Potentiation): если сосед активен ДО текущей клетки → вес++
        - LTD (Long Term Depression): если сосед активен ПОСЛЕ текущей клетки → вес--

        Returns:
            Dict с статистикой STDP обновления
        """
        if not self.enable_stdp or len(self.activity_history) < 2:
            return {"message": "STDP not enabled or insufficient history"}

        # Получаем два последних временных шага
        current_activity = self.activity_history[-1]
        previous_activity = self.activity_history[-2]

        current_active = current_activity["active_cells"]
        previous_active = previous_activity["active_cells"]

        # Статистика для отчета
        ltp_updates = 0
        ltd_updates = 0
        total_weight_change = 0.0

        # Batch processing для эффективности
        with torch.no_grad():
            # Получаем индексы всех соседей
            neighbor_indices = self.topology.get_all_neighbor_indices_batched()

            # Vectorized STDP update для всех клеток
            for cell_idx in range(self.config.total_cells):
                if not current_active[cell_idx]:
                    continue  # Клетка не активна - не обновляем веса

                # Получаем соседей этой клетки
                cell_neighbors = neighbor_indices[cell_idx]

                for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
                    if neighbor_cell_idx == cell_idx:
                        continue  # Пропускаем self-connections

                    # STDP правило
                    delta_w = 0.0

                    if previous_active[neighbor_cell_idx] and current_active[cell_idx]:
                        # LTP: сосед был активен на предыдущем шаге, текущая клетка активна сейчас
                        delta_w = self.A_plus * self.learning_rate
                        ltp_updates += 1
                    elif (
                        current_active[cell_idx]
                        and not previous_active[neighbor_cell_idx]
                    ):
                        # LTD: текущая клетка активна, но сосед НЕ был активен ранее
                        delta_w = -self.A_minus * self.learning_rate
                        ltd_updates += 1

                    if delta_w != 0.0:
                        # Обновляем вес связи
                        old_weight = self.connection_weights[
                            cell_idx, neighbor_idx
                        ].item()
                        new_weight = old_weight + delta_w

                        # Применяем bounds checking
                        new_weight = max(
                            self.weight_bounds[0],
                            min(self.weight_bounds[1], new_weight),
                        )

                        self.connection_weights[cell_idx, neighbor_idx] = new_weight
                        total_weight_change += abs(new_weight - old_weight)

        # Статистика для мониторинга
        active_cells_count = int(current_active.sum())

        return {
            "active_cells": active_cells_count,
            "ltp_updates": ltp_updates,
            "ltd_updates": ltd_updates,
            "total_weight_change": total_weight_change,
            "avg_weight_change": total_weight_change
            / max(1, ltp_updates + ltd_updates),
            "connection_weights_stats": {
                "min": float(self.connection_weights.min().item()),
                "max": float(self.connection_weights.max().item()),
                "mean": float(self.connection_weights.mean().item()),
                "std": float(self.connection_weights.std().item()),
            },
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        return self.perf_stats.copy()

    def _validate_initial_setup(self):
        self.logger.debug("Validating initial lattice setup...")
        assert self.states.shape == (self.config.total_cells, self.state_size)
        self.logger.debug("Initial states validation passed.")


def create_lattice_from_config(
    config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None
) -> Lattice3D:
    """
    Фабричная функция для создания экземпляра Lattice3D из файла или словаря.
    """
    if config_path:
        config = load_lattice_config(config_path)
    elif config_dict:
        config = LatticeConfig.from_dict(config_dict)
    else:
        raise ValueError("Either config_path or config_dict must be provided.")
    return Lattice3D(config)
