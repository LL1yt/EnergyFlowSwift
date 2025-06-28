"""
Модуль 3D Решетка клеток для MoE архитектуры
==========================================

Упрощенная версия Lattice3D, работающая только с MoE архитектурой.
Интегрирована с spatial optimization для эффективной обработки больших решеток.

АРХИТЕКТУРА: ТОЛЬКО MoE
- Базовые клетки: GNN Cell
- Обработка связей: MoE Connection Processor с spatial optimization
- Убрана legacy совместимость и другие архитектуры
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
import logging
import time
from datetime import datetime

# Импорты из new_rebuild
from ...config import get_project_config
from ..cells import create_cell

# Локальные импорты из lattice модуля
from .enums import Face
from .io import IOPointPlacer
from .position import Position3D

# Unified Spatial optimization для MoE архитектуры
from .spatial_optimization.unified_spatial_optimizer import (
    create_unified_spatial_optimizer,
    OptimizationConfig,
    OptimizationMode,
)


class Lattice3D(nn.Module):
    """
    Трёхмерная решётка клеток для MoE архитектуры.

    Упрощенная версия, работающая только с:
    - MoE архитектурой и MoE Connection Processor
    - GNN базовыми клетками
    - Spatial optimization для эффективности
    - Централизованным логированием
    """

    def __init__(self):
        """
        Инициализация решетки для MoE архитектуры.
        """
        super().__init__()
        self.config = get_project_config()
        self.device = torch.device(self.config.current_device)
        self.pos_helper = Position3D(self.config.lattice.dimensions)
        self.logger = logging.getLogger(__name__)

        # Проверяем что используется MoE архитектура
        if not self.config.expert.enabled:
            raise ValueError(
                f"Lattice3D теперь поддерживает только MoE архитектуру. "
                f"Включите 'expert.enabled' в конфигурации."
            )

        # Логирование инициализации
        from ...utils.logging import log_init

        if self.config.logging.debug_mode:
            log_init(
                "Lattice3D_MoE",
                dimensions=self.config.lattice.dimensions,
                total_cells=self.pos_helper.total_positions,
                architecture="moe",
                device=str(self.device),
            )

        # Создаем GNN клетки для MoE
        self.cells = self._create_gnn_cells()

        # Unified Spatial Optimizer с MoE поддержкой
        lattice3d_cfg = self.config.lattice3d
        spatial_config = OptimizationConfig(
            enable_moe=lattice3d_cfg.enable_moe,
            enable_morton_encoding=lattice3d_cfg.enable_morton_encoding,
            target_performance_ms=lattice3d_cfg.target_performance_ms,
        )

        # Создаем унифицированный оптимизатор (MoE processor будет добавлен позже в forward)
        self.spatial_optimizer = create_unified_spatial_optimizer(
            dimensions=self.config.lattice.dimensions, config=spatial_config
        )

        # Размещение I/O точек
        from .enums import PlacementStrategy

        self.io_placer = IOPointPlacer(
            lattice_dimensions=self.config.lattice.dimensions,
            strategy=PlacementStrategy.FULL_FACE,
            config={},
            seed=self.config.init.seed,
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

        if self.config.logging.debug_mode:
            from ...utils.logging import get_logger

            logger = get_logger(__name__)
            logger.info(
                f"✅ Lattice3D MoE initialized successfully:\n"
                f"     INPUT_POINTS: {len(self.input_points)}\n"
                f"     OUTPUT_POINTS: {len(self.output_points)}\n"
                f"     CELL_TYPE: {type(self.cells).__name__}\n"
                f"     SPATIAL_OPTIMIZER: {type(self.spatial_optimizer).__name__}"
            )

    def _create_gnn_cells(self):
        """
        Создает GNN клетки для MoE архитектуры.
        """
        gnn_config = {
            "state_size": self.config.model.state_size,
            "message_dim": self.config.model.message_dim,
            "hidden_dim": self.config.model.hidden_dim,
            "neighbor_count": self.config.model.neighbor_count,
            "external_input_size": self.config.model.external_input_size,
            "activation": self.config.model.activation,
            "target_params": self.config.model.target_params,
            "use_attention": self.config.model.use_attention,
            "device": self.config.current_device,
            "debug_mode": self.config.logging.debug_mode,
        }
        cell = create_cell("vectorized_gnn", **gnn_config)
        return cell.to(self.device)

    def _create_moe_processor(self):
        """Создаёт MoE processor для MoE архитектуры"""
        from ..moe import create_moe_connection_processor

        return create_moe_connection_processor(
            dimensions=self.config.lattice.dimensions,
            state_size=self.config.model.state_size,
            device=self.device,
        )

    def _initialize_states(self) -> torch.Tensor:
        """
        Инициализирует тензор состояний клеток.
        """
        state_size = (
            self.cells.state_size
            if hasattr(self.cells, "state_size")
            else self.config.model.state_size
        )

        dims = (self.pos_helper.total_positions, state_size)

        # Инициализация малыми случайными значениями
        torch.manual_seed(self.config.init.seed)  # Для воспроизводимости
        states = torch.randn(*dims, device=self.device, dtype=torch.float32) * 0.1

        if self.config.logging.debug_mode:
            from ...utils.logging import get_logger

            logger = get_logger(__name__)
            logger.info(f"States initialized with shape: {states.shape}")

        return states

    def forward(self, external_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Выполняет один шаг forward pass через решетку с MoE архитектурой.

        Args:
            external_inputs: Внешние входы для клеток (опционально)

        Returns:
            torch.Tensor: Новые состояния всех клеток
        """
        start_time = time.time()

        # Создаем MoE processor
        moe_processor = self._create_moe_processor()

        # Устанавливаем MoE processor в унифицированный оптимизатор
        self.spatial_optimizer.moe_processor = moe_processor

        # Unified Spatial Optimizer автоматически выберет лучший режим обработки
        optimization_result = self.spatial_optimizer.optimize_lattice_forward(
            self.states
        )

        # Извлекаем новые состояния и дополнительную информацию
        new_states = optimization_result.new_states

        # Логируем производительность если включен debug режим
        if self.config.logging.debug_mode:
            self.logger.info(
                f"Spatial optimization: {optimization_result.processing_time_ms:.1f}ms, "
                f"режим: {optimization_result.mode_used.value}, "
                f"память: {optimization_result.memory_usage_mb:.1f}MB"
            )

        # Обновляем состояния
        self.states = new_states

        # Обновляем статистику
        step_time = time.time() - start_time
        self._update_performance_stats(step_time, optimization_result)

        return self.states

    def _update_performance_stats(self, step_time: float, optimization_result=None):
        """Обновляет статистику производительности."""
        self.perf_stats["total_steps"] += 1
        self.perf_stats["total_time"] += step_time
        self.perf_stats["avg_time_per_step"] = (
            self.perf_stats["total_time"] / self.perf_stats["total_steps"]
        )

        # Добавляем статистику от унифицированного оптимизатора
        if optimization_result:
            self.perf_stats["spatial_optimization"] = {
                "processing_time_ms": optimization_result.processing_time_ms,
                "mode_used": optimization_result.mode_used.value,
                "memory_usage_mb": optimization_result.memory_usage_mb,
                "gpu_utilization": optimization_result.gpu_utilization,
                "neighbors_found": optimization_result.neighbors_found,
                "cache_hit_rate": optimization_result.cache_hit_rate,
            }

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
            "dimensions": self.config.lattice.dimensions,
            "total_cells": self.pos_helper.total_positions,
            "architecture_type": "moe",
            "device": str(self.device),
            "state_shape": list(self.states.shape),
            "input_points": len(self.input_points),
            "output_points": len(self.output_points),
        }

        # Добавляем расширенную статистику от унифицированного оптимизатора
        if hasattr(self.spatial_optimizer, "get_comprehensive_stats"):
            stats["spatial_optimizer"] = (
                self.spatial_optimizer.get_comprehensive_stats()
            )

        # Добавляем информацию о клетках
        if hasattr(self.cells, "get_info"):
            stats["cell_info"] = self.cells.get_info()

        return stats

    def cleanup(self):
        """Освобождает ресурсы унифицированного оптимизатора."""
        if hasattr(self.spatial_optimizer, "cleanup"):
            self.spatial_optimizer.cleanup()
            self.logger.info("🧹 Unified Spatial Optimizer ресурсы освобождены")

    def __del__(self):
        """Деструктор для автоматической очистки ресурсов."""
        try:
            self.cleanup()
        except:
            pass  # Игнорируем ошибки при деструкции


def create_lattice() -> Lattice3D:
    """
    Фабричная функция для создания MoE решетки на основе ProjectConfig.
    """
    return Lattice3D()
