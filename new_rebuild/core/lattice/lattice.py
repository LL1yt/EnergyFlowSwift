"""
Модуль 3D Решетка клеток
========================

Основной класс для создания и управления трёхмерной решеткой клеток.
Содержит базовую функциональность forward pass'а и управления состояниями.

Адаптирован для работы с ProjectConfig и новыми клетками из new_rebuild.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Union
import logging
import time
from datetime import datetime
import json

# Импорты из new_rebuild
from ...config import get_project_config
from ..cells import CellFactory, BaseCell

# Локальные импорты из lattice модуля
from .enums import Face
from .io import IOPointPlacer
from .position import Position3D
from .topology import NeighborTopology


class Lattice3D(nn.Module):
    """
    Трёхмерная решётка клеток (нейронов).

    Упрощенная версия для clean архитектуры, работающая с:
    - ProjectConfig вместо LatticeConfig
    - NCACell и GMLPCell из new_rebuild
    - Централизованным логированием
    """

    def __init__(self):
        """
        Инициализация решетки на основе ProjectConfig.
        """
        super().__init__()
        self.config = get_project_config()
        self.device = torch.device(self.config.device)
        self.pos_helper = Position3D(self.config.lattice_dimensions)
        self.logger = logging.getLogger(__name__)

        # Логирование инициализации с централизованной системой
        from ...utils.logging import log_init

        if self.config.debug_mode:
            log_init(
                "Lattice3D",
                dimensions=self.config.lattice_dimensions,
                total_cells=self.pos_helper.total_positions,
                architecture=self.config.architecture_type,
                device=str(self.device),
            )

        # Создаем клетки
        self.cell_factory = CellFactory()
        self.cells = self._create_cells()

        # Инициализация топологии соседства
        all_coords = self.pos_helper.get_all_coordinates()
        self.topology = NeighborTopology(all_coords=all_coords)

        # Размещение I/O точек (упрощенная версия)
        from .enums import PlacementStrategy

        self.io_placer = IOPointPlacer(
            lattice_dimensions=self.config.lattice_dimensions,
            strategy=PlacementStrategy.FULL_FACE,  # Используем полное покрытие граней
            config={},  # Пустая конфигурация для простоты
            seed=42,
        )

        # I/O точки (по умолчанию используем FRONT и BACK)
        self.input_points = self.io_placer.get_input_points(Face.FRONT)
        self.output_points = self.io_placer.get_output_points(Face.BACK)
        self.input_indices = [
            self.pos_helper.to_linear_index(p) for p in self.input_points
        ]
        self.output_indices = [
            self.pos_helper.to_linear_index(p) for p in self.output_points
        ]

        # Инициализация состояний клеток
        self.states = self._initialize_states()

        # Отслеживание производительности
        self.perf_stats = {
            "total_steps": 0,
            "total_time": 0.0,
            "avg_time_per_step": 0.0,
        }

        if self.config.debug_mode:
            from ...utils.logging import get_logger

            logger = get_logger(__name__)
            logger.info(
                f"✅ Lattice3D initialized successfully:\n"
                f"     INPUT_POINTS: {len(self.input_points)}\n"
                f"     OUTPUT_POINTS: {len(self.output_points)}\n"
                f"     CELL_TYPE: {type(self.cells).__name__}"
            )

    def _create_cells(self) -> BaseCell:
        """
        Создает клетки на основе конфигурации архитектуры.
        """
        # Преобразуем ProjectConfig в словарь для NCA/gMLP клеток
        if self.config.architecture_type == "nca":
            # NCA DEPRECATED: используем GNN параметры для совместимости
            nca_config = {
                "state_size": self.config.gnn_state_size,  # Используем GNN параметры
                "hidden_dim": 3,  # Жестко задаем для NCA
                "neighbor_count": self.config.gnn_neighbor_count,  # Синхронизация
                "activation": "tanh",  # Стандартная NCA активация
                "target_params": 69,  # Стандартные NCA параметры
                "device": self.config.device,
                "debug_mode": self.config.debug_mode,
            }
            cell = self.cell_factory.create_cell("nca", nca_config)
        elif self.config.architecture_type == "gnn":
            # GNN клетка (заменяет gMLP)
            gnn_config = {
                "state_size": self.config.gnn_state_size,
                "message_dim": self.config.gnn_message_dim,
                "hidden_dim": self.config.gnn_hidden_dim,
                "neighbor_count": self.config.gnn_neighbor_count,
                "external_input_size": self.config.gnn_external_input_size,
                "activation": self.config.gnn_activation,
                "target_params": self.config.gnn_target_params,
                "use_attention": self.config.gnn_use_attention,
                "device": self.config.device,
                "debug_mode": self.config.debug_mode,
            }
            cell = self.cell_factory.create_cell("gnn", gnn_config)
        elif self.config.architecture_type == "gmlp":
            # Legacy совместимость: gMLP → GNN
            gnn_config = {
                "state_size": self.config.gnn_state_size,
                "message_dim": self.config.gnn_message_dim,
                "hidden_dim": self.config.gnn_hidden_dim,
                "neighbor_count": self.config.gnn_neighbor_count,
                "external_input_size": self.config.gnn_external_input_size,
                "activation": self.config.gnn_activation,
                "target_params": self.config.gnn_target_params,
                "use_attention": self.config.gnn_use_attention,
                "device": self.config.device,
                "debug_mode": self.config.debug_mode,
            }
            cell = self.cell_factory.create_cell(
                "gmlp", gnn_config
            )  # CellFactory сделает маппинг
        elif self.config.architecture_type == "hybrid":
            # Для hybrid используем GNN (заменил gMLP)
            gnn_config = {
                "state_size": self.config.gnn_state_size,
                "message_dim": self.config.gnn_message_dim,
                "hidden_dim": self.config.gnn_hidden_dim,
                "neighbor_count": self.config.gnn_neighbor_count,
                "external_input_size": self.config.gnn_external_input_size,
                "activation": self.config.gnn_activation,
                "target_params": self.config.gnn_target_params,
                "use_attention": self.config.gnn_use_attention,
                "device": self.config.device,
                "debug_mode": self.config.debug_mode,
            }
            cell = self.cell_factory.create_cell("gnn", gnn_config)
        elif self.config.architecture_type == "moe":
            # MoE архитектура: используем GNN как базовую клетку
            # MoE процессор будет создаваться отдельно на уровне выше
            gnn_config = {
                "state_size": self.config.gnn_state_size,
                "message_dim": self.config.gnn_message_dim,
                "hidden_dim": self.config.gnn_hidden_dim,
                "neighbor_count": self.config.gnn_neighbor_count,
                "external_input_size": self.config.gnn_external_input_size,
                "activation": self.config.gnn_activation,
                "target_params": self.config.gnn_target_params,
                "use_attention": self.config.gnn_use_attention,
                "device": self.config.device,
                "debug_mode": self.config.debug_mode,
            }
            cell = self.cell_factory.create_cell("gnn", gnn_config)
        else:
            raise ValueError(
                f"Неподдерживаемый тип архитектуры: {self.config.architecture_type}"
            )

        return cell.to(self.device)

    def _initialize_states(self) -> torch.Tensor:
        """
        Инициализирует тензор состояний клеток.
        """
        # Получаем размер состояния из клетки
        if hasattr(self.cells, "state_size"):
            state_size = self.cells.state_size
        else:
            # Fallback для определения размера состояния
            if self.config.architecture_type == "nca":
                state_size = self.config.nca_state_size
            elif self.config.architecture_type == "gnn":
                state_size = self.config.gnn_state_size
            elif self.config.architecture_type == "gmlp":
                # Legacy: gMLP теперь использует GNN параметры
                state_size = self.config.gnn_state_size
            elif self.config.architecture_type == "hybrid":
                # Hybrid использует GNN параметры (NCA deprecated)
                state_size = self.config.gnn_state_size
            elif self.config.architecture_type == "moe":
                # MoE использует GNN параметры как базовые
                state_size = self.config.gnn_state_size
            else:
                # Fallback
                state_size = self.config.gnn_state_size

        dims = (self.pos_helper.total_positions, state_size)

        # Инициализация малыми случайными значениями
        torch.manual_seed(42)  # Для воспроизводимости
        states = torch.randn(*dims, device=self.device, dtype=torch.float32) * 0.1

        if self.config.debug_mode:
            from ...utils.logging import get_logger

            logger = get_logger(__name__)
            logger.info(f"States initialized with shape: {states.shape}")

        return states

    def forward(self, external_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Выполняет один шаг forward pass через решетку.

        Args:
            external_inputs: Внешние входы для клеток (опционально)

        Returns:
            torch.Tensor: Новые состояния всех клеток
        """
        start_time = time.time()

        # Получаем соседей для всех клеток
        neighbor_indices = self.topology.get_all_neighbor_indices_batched()

        # Собираем состояния соседей
        neighbor_states = self._gather_neighbor_states(neighbor_indices)

        # Подготавливаем внешние входы
        processed_external_inputs = self._prepare_external_inputs(external_inputs)

        # Применяем клетки ко всем позициям
        new_states = self.cells(
            neighbor_states=neighbor_states,
            own_state=self.states,
            external_input=processed_external_inputs,
        )

        # Обновляем состояния
        self.states = new_states

        # Обновляем статистику
        step_time = time.time() - start_time
        self._update_performance_stats(step_time)

        return self.states

    def _gather_neighbor_states(self, neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Собирает состояния соседей для всех клеток.

        Args:
            neighbor_indices: Тензор индексов соседей [total_cells, max_neighbors]

        Returns:
            torch.Tensor: Состояния соседей [total_cells, max_neighbors, state_size]
        """
        batch_size, max_neighbors = neighbor_indices.shape
        state_size = self.states.shape[1]

        # Создаем тензор для состояний соседей
        neighbor_states = torch.zeros(
            batch_size,
            max_neighbors,
            state_size,
            device=self.device,
            dtype=self.states.dtype,
        )

        # Заполняем состояния соседей
        for i in range(batch_size):
            for j in range(max_neighbors):
                neighbor_idx = neighbor_indices[i, j]
                if neighbor_idx >= 0:  # -1 означает отсутствие соседа
                    neighbor_states[i, j] = self.states[neighbor_idx]

        return neighbor_states

    def _prepare_external_inputs(
        self, external_inputs: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Подготавливает внешние входы для клеток.
        """
        if external_inputs is None:
            return None

        # Убеждаемся, что внешние входы на правильном устройстве
        if external_inputs.device != self.device:
            external_inputs = external_inputs.to(self.device)

        # Проверяем размерность
        if external_inputs.shape[0] != self.pos_helper.total_positions:
            raise ValueError(
                f"External inputs batch size {external_inputs.shape[0]} "
                f"doesn't match total cells {self.pos_helper.total_positions}"
            )

        return external_inputs

    def _update_performance_stats(self, step_time: float):
        """Обновляет статистику производительности."""
        self.perf_stats["total_steps"] += 1
        self.perf_stats["total_time"] += step_time
        self.perf_stats["avg_time_per_step"] = (
            self.perf_stats["total_time"] / self.perf_stats["total_steps"]
        )

    def get_states(self) -> torch.Tensor:
        """Возвращает текущие состояния клеток."""
        return self.states

    def set_states(self, new_states: torch.Tensor):
        """Устанавливает новые состояния клеток."""
        if new_states.shape != self.states.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.states.shape}, got {new_states.shape}"
            )
        self.states = new_states.to(self.device)

    def get_face_states(self, face: Face) -> torch.Tensor:
        """Возвращает состояния клеток на указанной грани."""
        face_coords = self.io_placer._get_face_coordinates(face)
        face_indices = [self.pos_helper.to_linear_index(coord) for coord in face_coords]
        return self.states[face_indices]

    def get_output_states(self) -> torch.Tensor:
        """Возвращает состояния выходных клеток."""
        return self.states[self.output_indices]

    def get_input_states(self) -> torch.Tensor:
        """Возвращает состояния входных клеток."""
        return self.states[self.input_indices]

    def set_input_states(self, input_states: torch.Tensor):
        """Устанавливает состояния входных клеток."""
        if input_states.shape[0] != len(self.input_indices):
            raise ValueError(
                f"Input states count {input_states.shape[0]} "
                f"doesn't match input points {len(self.input_indices)}"
            )
        self.states[self.input_indices] = input_states.to(self.device)

    def reset_states(self):
        """Сбрасывает состояния клеток к начальным значениям."""
        self.states = self._initialize_states()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности."""
        return dict(self.perf_stats)

    def get_io_point_info(self) -> Dict[str, Any]:
        """Возвращает информацию о точках ввода/вывода."""
        return {
            "input_points": len(self.input_points),
            "output_points": len(self.output_points),
            "input_indices": self.input_indices,
            "output_indices": self.output_indices,
            "input_coordinates": self.input_points,
            "output_coordinates": self.output_points,
        }

    def validate_lattice(self) -> Dict[str, Any]:
        """Валидирует решетку и возвращает статистику."""
        stats = {
            "dimensions": self.config.lattice_dimensions,
            "total_cells": self.pos_helper.total_positions,
            "architecture_type": self.config.architecture_type,
            "device": str(self.device),
            "state_shape": list(self.states.shape),
            "input_points": len(self.input_points),
            "output_points": len(self.output_points),
        }

        # Добавляем статистику топологии
        topology_stats = self.topology.validate_topology()
        stats["topology"] = topology_stats

        # Добавляем информацию о клетках
        if hasattr(self.cells, "get_info"):
            stats["cell_info"] = self.cells.get_info()

        return stats


def create_lattice() -> Lattice3D:
    """
    Фабричная функция для создания решетки на основе ProjectConfig.
    """
    return Lattice3D()
