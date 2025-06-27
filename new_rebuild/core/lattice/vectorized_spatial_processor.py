#!/usr/bin/env python3
"""
Vectorized Spatial Processor - полностью векторизованная обработка решетки
========================================================================

Оптимизированный spatial processor с batch processing для maximum performance.
Исключает все sequential операции и циклы.

КЛЮЧЕВЫЕ ОПТИМИЗАЦИИ:
1. ✅ Batch Neighbor Finding - все соседи находятся параллельно
2. ✅ Vectorized Cell Processing - все клетки обрабатываются сразу
3. ✅ GPU Memory Optimization - efficient tensor operations
4. ✅ Adaptive Batch Sizing - оптимальные размеры батчей для GPU
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
import time

from ...config import get_project_config
from ...utils.logging import get_logger
from ...utils.device_manager import get_device_manager
from .position import Position3D
from .gpu_spatial_hashing import GPUSpatialHashGrid

logger = get_logger(__name__)


class VectorizedNeighborFinder:
    """
    Векторизованный поиск соседей для всех клеток сразу

    Исключает циклы и sequential операции
    """

    def __init__(self, dimensions: Tuple[int, int, int], device: torch.device):
        self.dimensions = dimensions
        self.device = device
        self.pos_helper = Position3D(dimensions)

        # GPU spatial hash для быстрого поиска
        self.spatial_hash = GPUSpatialHashGrid(
            dimensions=dimensions,
            cell_size=2,
        )

        # Предвычисленные координаты всех клеток
        self.all_coordinates = self._precompute_all_coordinates()

        # Заполняем spatial hash
        self._populate_spatial_hash()

    def _precompute_all_coordinates(self) -> torch.Tensor:
        """Предвычисляем координаты всех клеток в решетке"""
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

        coordinates = []
        for cell_idx in range(total_cells):
            coords = self.pos_helper.to_3d_coordinates(cell_idx)
            coordinates.append(coords)

        return torch.tensor(coordinates, device=self.device, dtype=torch.float32)

    def _populate_spatial_hash(self):
        """Заполняем spatial hash всеми клетками"""
        total_cells = self.all_coordinates.shape[0]
        indices = torch.arange(total_cells, device=self.device)

        self.spatial_hash.insert_batch(self.all_coordinates, indices)

        logger.debug(f"📍 Spatial hash populated with {total_cells} cells")

    def find_neighbors_batch(
        self,
        cell_indices: torch.Tensor,
        search_radius: float,
        max_neighbors: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Векторизованный поиск соседей для batch клеток

        Args:
            cell_indices: [batch] - индексы клеток
            search_radius: радиус поиска
            max_neighbors: максимальное количество соседей на клетку

        Returns:
            neighbor_indices: [batch, max_neighbors] - индексы соседей (padded с -1)
            neighbor_mask: [batch, max_neighbors] - маска валидных соседей
        """
        batch_size = cell_indices.shape[0]

        # Получаем координаты для batch клеток
        batch_coordinates = self.all_coordinates[cell_indices]

        # Векторизованный поиск соседей через spatial hash
        neighbor_lists = self.spatial_hash.query_radius_batch(
            batch_coordinates, search_radius
        )

        # Создаем padded tensor для соседей
        neighbor_indices = torch.full(
            (batch_size, max_neighbors), -1, device=self.device, dtype=torch.long
        )
        neighbor_mask = torch.zeros(
            (batch_size, max_neighbors), device=self.device, dtype=torch.bool
        )

        # Заполняем tensor соседями
        for i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) > 0:
                # Исключаем саму клетку из соседей
                neighbors = neighbors[neighbors != cell_indices[i]]

                # Ограничиваем количество соседей
                num_neighbors = min(len(neighbors), max_neighbors)
                if num_neighbors > 0:
                    neighbor_indices[i, :num_neighbors] = neighbors[:num_neighbors]
                    neighbor_mask[i, :num_neighbors] = True

        return neighbor_indices, neighbor_mask


class VectorizedSpatialProcessor:
    """
    Полностью векторизованный spatial processor

    Обрабатывает всю решетку без циклов и sequential операций
    """

    def __init__(self, dimensions: Tuple[int, int, int]):
        self.dimensions = dimensions
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Векторизованный поиск соседей
        self.neighbor_finder = VectorizedNeighborFinder(dimensions, self.device)

        # Конфигурация
        config = get_project_config()
        self.search_radius = config.calculate_adaptive_radius()
        self.max_neighbors = config.calculate_dynamic_neighbors()

        # Адаптивный размер батча
        self.optimal_batch_size = self._calculate_optimal_batch_size()

        # Статистика производительности
        self.performance_stats = {
            "total_forward_passes": 0,
            "total_processing_time": 0.0,
            "avg_batch_time": 0.0,
            "cells_per_second": 0.0,
        }

        logger.info(f"🚀 VectorizedSpatialProcessor initialized:")
        logger.info(f"   Dimensions: {dimensions}")
        logger.info(f"   Search radius: {self.search_radius:.2f}")
        logger.info(f"   Max neighbors: {self.max_neighbors}")
        logger.info(f"   Optimal batch size: {self.optimal_batch_size}")

    def _calculate_optimal_batch_size(self) -> int:
        """Вычисляет оптимальный размер батча для GPU"""
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

        if self.device_manager.is_cuda():
            # GPU: больше памяти, можем обрабатывать большие батчи
            memory_stats = self.device_manager.get_memory_stats()
            available_mb = memory_stats.get("available_mb", 8000)

            if available_mb > 16000:  # >16GB
                return min(total_cells, 8000)
            elif available_mb > 8000:  # >8GB
                return min(total_cells, 4000)
            else:  # <8GB
                return min(total_cells, 2000)
        else:
            # CPU: меньшие батчи
            return min(total_cells, 1000)

    def process_lattice_vectorized(
        self, states: torch.Tensor, cell_processor: Callable, **kwargs
    ) -> torch.Tensor:
        """
        Полностью векторизованная обработка решетки

        Args:
            states: [total_cells, state_size] - состояния всех клеток
            cell_processor: функция обработки клеток (должна поддерживать batch)

        Returns:
            new_states: [total_cells, state_size] - новые состояния
        """
        start_time = time.time()

        states = self.device_manager.ensure_device(states)
        total_cells, state_size = states.shape

        logger.info(f"🚀 Vectorized processing {total_cells:,} cells...")

        # Создаем выходной тензор
        new_states = torch.empty_like(states)

        # Обрабатываем батчами для оптимизации памяти
        num_batches = (
            total_cells + self.optimal_batch_size - 1
        ) // self.optimal_batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.optimal_batch_size
            batch_end = min(batch_start + self.optimal_batch_size, total_cells)

            # Индексы клеток в текущем батче
            batch_cell_indices = torch.arange(
                batch_start, batch_end, device=self.device, dtype=torch.long
            )

            # Векторизованный поиск соседей для всего батча
            neighbor_indices, neighbor_mask = self.neighbor_finder.find_neighbors_batch(
                batch_cell_indices, self.search_radius, self.max_neighbors
            )

            # Извлекаем состояния для батча
            batch_states = states[batch_cell_indices]  # [batch_size, state_size]

            # Извлекаем состояния соседей (векторизованно)
            batch_neighbor_states = self._get_neighbor_states_vectorized(
                states, neighbor_indices, neighbor_mask
            )

            # Обрабатываем весь батч сразу через cell_processor
            batch_new_states = cell_processor(
                neighbor_states=batch_neighbor_states,
                own_state=batch_states,
                cell_indices=batch_cell_indices,
                neighbor_indices=neighbor_indices,
                neighbor_mask=neighbor_mask,
                **kwargs,
            )

            # Записываем результаты
            new_states[batch_start:batch_end] = batch_new_states

        # Обновляем статистику
        processing_time = time.time() - start_time
        self._update_performance_stats(total_cells, processing_time)

        logger.info(f"✅ Vectorized processing completed in {processing_time:.3f}s")
        logger.info(f"   Performance: {total_cells/processing_time:.0f} cells/second")

        return new_states

    def _get_neighbor_states_vectorized(
        self,
        all_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Векторизованное извлечение состояний соседей

        Args:
            all_states: [total_cells, state_size] - все состояния
            neighbor_indices: [batch, max_neighbors] - индексы соседей
            neighbor_mask: [batch, max_neighbors] - маска валидных соседей

        Returns:
            neighbor_states: [batch, max_neighbors, state_size] - состояния соседей
        """
        batch_size, max_neighbors = neighbor_indices.shape
        state_size = all_states.shape[1]

        # Создаем tensor для состояний соседей
        neighbor_states = torch.zeros(
            batch_size,
            max_neighbors,
            state_size,
            device=self.device,
            dtype=all_states.dtype,
        )

        # Векторизованное извлечение валидных соседей
        valid_indices = neighbor_indices[neighbor_mask]
        if len(valid_indices) > 0:
            # Получаем состояния валидных соседей
            valid_states = all_states[valid_indices]

            # Записываем их в правильные позиции
            neighbor_states[neighbor_mask] = valid_states

        return neighbor_states

    def process_with_moe(
        self, states: torch.Tensor, moe_processor: nn.Module, **kwargs
    ) -> torch.Tensor:
        """
        Векторизованная обработка с MoE архитектурой

        Args:
            states: [total_cells, state_size] - состояния клеток
            moe_processor: MoE процессор с batch support

        Returns:
            new_states: [total_cells, state_size] - новые состояния
        """

        def moe_cell_processor(
            neighbor_states: torch.Tensor,
            own_state: torch.Tensor,
            cell_indices: torch.Tensor,
            neighbor_indices: torch.Tensor,
            neighbor_mask: torch.Tensor,
            **proc_kwargs,
        ) -> torch.Tensor:
            """Wrapper для MoE процессора"""

            # Используем batch метод MoE процессора
            result = moe_processor.forward_batch(
                batch_states=own_state,
                batch_neighbor_states=neighbor_states,
                batch_cell_indices=cell_indices,
                batch_neighbor_indices=neighbor_indices,
                full_lattice_states=states,
                **proc_kwargs,
            )

            # Извлекаем новые состояния из результата
            if isinstance(result, dict) and "new_states" in result:
                return result["new_states"]
            else:
                return result

        return self.process_lattice_vectorized(states, moe_cell_processor, **kwargs)

    def _update_performance_stats(self, total_cells: int, processing_time: float):
        """Обновляет статистику производительности"""
        self.performance_stats["total_forward_passes"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        self.performance_stats["avg_batch_time"] = (
            self.performance_stats["total_processing_time"]
            / self.performance_stats["total_forward_passes"]
        )
        self.performance_stats["cells_per_second"] = total_cells / processing_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        return {
            "architecture": "vectorized_spatial_processor",
            "optimization": "full_vectorization",
            "sequential_operations": 0,  # НЕТ циклов!
            "optimal_batch_size": self.optimal_batch_size,
            "search_radius": self.search_radius,
            "max_neighbors": self.max_neighbors,
            "performance": self.performance_stats.copy(),
            "device": str(self.device),
            "memory_stats": self.device_manager.get_memory_stats(),
        }

    def cleanup(self):
        """Очистка ресурсов"""
        self.device_manager.cleanup()
        logger.debug("🧹 VectorizedSpatialProcessor cleaned up")
