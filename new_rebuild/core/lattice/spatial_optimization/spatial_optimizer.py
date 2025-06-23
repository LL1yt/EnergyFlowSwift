#!/usr/bin/env python3
"""
Базовый Spatial Optimizer
=========================

Основа для пространственной оптимизации 3D решеток.
Обеспечивает эффективный поиск соседей и базовую инфраструктуру
для оптимизированной обработки больших решеток.
"""

import torch
from typing import Dict, List, Optional
import time

from ....config.project_config import create_spatial_config_for_lattice
from ....config.project_config import get_project_config
from .hierarchical_index import HierarchicalSpatialIndex
from ..spatial_hashing import Coordinates3D, SpatialHashGrid
from ..position import Position3D
from ....utils.logging import get_logger

logger = get_logger(__name__)


class SpatialOptimizer:
    """
    Базовый класс для пространственной оптимизации

    Обеспечивает эффективный поиск соседей и основные операции
    для оптимизированной обработки 3D решеток.
    """

    def __init__(self, dimensions: Coordinates3D, config: Optional[dict] = None):
        self.dimensions = dimensions
        self.config = config or get_project_config().get_spatial_optim_config()

        # Базовые компоненты
        self.pos_helper = Position3D(dimensions)
        self.spatial_index: Optional[HierarchicalSpatialIndex] = None
        self.spatial_grid: Optional[SpatialHashGrid] = None

        # Статистика производительности
        self.performance_stats = {
            "total_queries": 0,
            "total_time_ms": 0.0,
            "avg_neighbors_found": 0.0,
        }

        # Строим spatial index
        self._build_spatial_index()

        logger.info(f"🗂️ SpatialOptimizer готов для {dimensions}")

    def _build_spatial_index(self):
        """Строит пространственный индекс для эффективного поиска"""
        # Создаем иерархический индекс
        self.spatial_index = HierarchicalSpatialIndex(self.dimensions, self.config)

        # Также создаем базовый spatial grid для fallback
        max_dim = max(self.dimensions)
        cell_size = max(1, max_dim // 32)  # Разумный размер ячейки
        self.spatial_grid = SpatialHashGrid(self.dimensions, cell_size)

        # Заполняем индексы всеми координатами
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

        coords_list = []
        indices_list = []

        for idx in range(total_cells):
            coords = self.pos_helper.index_to_coords(idx)
            coords_list.append(coords)
            indices_list.append(idx)

            # Добавляем в базовый grid
            self.spatial_grid.insert(coords, idx)

        # Batch insert в иерархический индекс
        self.spatial_index.insert_batch(coords_list, indices_list)

        logger.info(f"   📊 Индексированы {total_cells:,} клеток")

    def find_neighbors_optimized(
        self, coords: Coordinates3D, radius: float
    ) -> List[int]:
        """
        Оптимизированный поиск соседей в радиусе

        Args:
            coords: координаты центральной точки
            radius: радиус поиска

        Returns:
            list индексов соседних клеток
        """
        start_time = time.time()

        try:
            # Используем иерархический индекс для больших радиусов
            if radius > 10.0 and self.spatial_index:
                neighbors = list(self.spatial_index.query_hierarchical(coords, radius))
            else:
                # Используем базовый spatial grid для малых радиусов
                neighbors = list(self.spatial_grid.query_radius(coords, radius))

            # Убираем саму точку из результатов если она там есть
            center_idx = self.pos_helper.coords_to_index(coords)
            if center_idx in neighbors:
                neighbors.remove(center_idx)

        except Exception as e:
            logger.warning(f"⚠️ Ошибка в поиске соседей: {e}")
            neighbors = []

        # Обновляем статистику
        query_time = (time.time() - start_time) * 1000  # в миллисекундах
        self.performance_stats["total_queries"] += 1
        self.performance_stats["total_time_ms"] += query_time
        self.performance_stats["avg_neighbors_found"] = (
            self.performance_stats["avg_neighbors_found"] * 0.9 + len(neighbors) * 0.1
        )

        return neighbors

    def optimize_lattice_forward(
        self, states: torch.Tensor, neighbor_processor_fn
    ) -> torch.Tensor:
        """
        Базовая оптимизированная обработка решетки

        Args:
            states: [num_cells, state_size] - состояния клеток
            neighbor_processor_fn: функция обработки соседей

        Returns:
            new_states: [num_cells, state_size] - новые состояния
        """
        logger.info(f"🚀 Запуск базовой spatial optimization")

        start_time = time.time()
        num_cells = states.shape[0]

        # Простая обработка без chunking для базового класса
        output_states = states.clone()

        for cell_idx in range(num_cells):
            coords = self.pos_helper.index_to_coords(cell_idx)
            neighbors = self.find_neighbors_optimized(
                coords, self.config["max_search_radius"]
            )

            if neighbors:
                # Применяем функцию обработки соседей
                new_state = neighbor_processor_fn(
                    states[cell_idx], states[neighbors], cell_idx, neighbors
                )
                output_states[cell_idx] = new_state

        processing_time = time.time() - start_time
        logger.info(f"✅ Базовая optimization завершена за {processing_time:.3f}s")

        return output_states

    def get_performance_stats(self) -> Dict[str, float]:
        """Получить статистику производительности"""
        stats = self.performance_stats.copy()

        if stats["total_queries"] > 0:
            stats["avg_query_time_ms"] = stats["total_time_ms"] / stats["total_queries"]
        else:
            stats["avg_query_time_ms"] = 0.0

        return stats

    def cleanup(self):
        """Освобождение ресурсов"""
        if hasattr(self, "spatial_index"):
            self.spatial_index = None
        if hasattr(self, "spatial_grid"):
            self.spatial_grid = None

        logger.info("🧹 SpatialOptimizer ресурсы освобождены")


def create_spatial_optimizer(dimensions: Coordinates3D) -> SpatialOptimizer:
    """
    Фабричная функция для создания базового Spatial Optimizer

    Args:
        dimensions: размеры решетки (x, y, z)

    Returns:
        SpatialOptimizer настроенный для данной решетки
    """
    config = create_spatial_config_for_lattice(dimensions)

    logger.info(f"🏭 Создание базового Spatial Optimizer для {dimensions}")

    return SpatialOptimizer(dimensions=dimensions, config=config)


def estimate_memory_requirements(dimensions: Coordinates3D) -> Dict[str, float]:
    """
    Оценка требований к памяти для Spatial Optimization

    Args:
        dimensions: размеры решетки (x, y, z)

    Returns:
        dict с оценками памяти в GB
    """
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]

    # Базовые требования
    cell_states_gb = total_cells * 32 * 4 / (1024**3)  # float32 состояния
    spatial_index_gb = total_cells * 16 / (1024**3)  # индекс координат
    neighbor_cache_gb = total_cells * 26 * 4 / (1024**3)  # кэш соседей
    processing_overhead_gb = 0.5  # временные buffers

    total_memory_gb = (
        cell_states_gb + spatial_index_gb + neighbor_cache_gb + processing_overhead_gb
    )

    return {
        "cell_states_gb": cell_states_gb,
        "spatial_index_gb": spatial_index_gb,
        "neighbor_cache_gb": neighbor_cache_gb,
        "processing_overhead_gb": processing_overhead_gb,
        "total_memory_gb": total_memory_gb,
        "recommended_gpu_memory_gb": total_memory_gb * 1.2,  # 20% запас
    }
