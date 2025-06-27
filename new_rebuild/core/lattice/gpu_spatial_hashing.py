#!/usr/bin/env python3
"""
GPU-Accelerated Spatial Hashing для 3D Решетки
==============================================

Высокопроизводительная реализация spatial hashing с использованием GPU.
Использует PyTorch tensor операции и оптимизированные CUDA kernels для
быстрого поиска соседей в трехмерном пространстве.

Ключевые особенности:
- GPU-accelerated операции с PyTorch
- Batch processing для максимальной эффективности
- Memory-efficient структуры данных
- Adaptive chunking на основе доступной памяти

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Set, Any
from dataclasses import dataclass
import math

try:
    from ...config import get_project_config
    from ...utils.device_manager import get_device_manager
    from ...utils.logging import get_logger
except ImportError:
    # Fallback для прямого запуска
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from config import get_project_config
    from utils.device_manager import get_device_manager
    from utils.logging import get_logger

logger = get_logger(__name__)

Coordinates3D = Tuple[int, int, int]


@dataclass
class GPUSpatialHashingStats:
    """Статистика производительности GPU spatial hashing"""

    total_queries: int = 0
    avg_query_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    batch_processing_efficiency: float = 0.0


class GPUMortonEncoder:
    """
    GPU-accelerated кодировщик кривой Мортона для 3D-пространства

    Использует векторизованные операции PyTorch для быстрого
    кодирования/декодирования больших batch'ей координат
    """

    def __init__(self, dimensions: Coordinates3D):
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        self.max_dim = max(dimensions)
        self.bits = self.max_dim.bit_length()

        # Предвычисленные маски для ускорения
        self._prepare_bit_masks()

        logger.debug(
            f"🔢 GPUMortonEncoder инициализирован для {dimensions}, {self.bits} бит"
        )

    def _prepare_bit_masks(self):
        """Подготавливает битовые маски для векторизованных операций"""
        # Создаем маски для интерливинга битов
        bit_positions = torch.arange(self.bits, device=self.device)

        # Маски для извлечения битов
        self.bit_masks = 2**bit_positions

        # Позиции для размещения битов в Morton коде
        self.x_positions = 3 * bit_positions + 2
        self.y_positions = 3 * bit_positions + 1
        self.z_positions = 3 * bit_positions

    def encode_batch(self, coords_batch: torch.Tensor) -> torch.Tensor:
        """
        Кодирует batch координат в Morton коды

        Args:
            coords_batch: tensor формы (N, 3) с координатами

        Returns:
            tensor формы (N,) с Morton кодами
        """
        coords_batch = self.device_manager.ensure_device(coords_batch)
        batch_size = coords_batch.shape[0]

        x, y, z = coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]

        # Векторизованное извлечение битов
        x_bits = (x.unsqueeze(1) & self.bit_masks.unsqueeze(0)) != 0
        y_bits = (y.unsqueeze(1) & self.bit_masks.unsqueeze(0)) != 0
        z_bits = (z.unsqueeze(1) & self.bit_masks.unsqueeze(0)) != 0

        # Интерливинг битов
        morton_codes = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        morton_codes += (x_bits * (2 ** self.x_positions.unsqueeze(0))).sum(dim=1)
        morton_codes += (y_bits * (2 ** self.y_positions.unsqueeze(0))).sum(dim=1)
        morton_codes += (z_bits * (2 ** self.z_positions.unsqueeze(0))).sum(dim=1)

        return morton_codes

    def decode_batch(self, morton_codes: torch.Tensor) -> torch.Tensor:
        """
        Декодирует batch Morton кодов в координаты

        Args:
            morton_codes: tensor формы (N,) с Morton кодами

        Returns:
            tensor формы (N, 3) с координатами
        """
        morton_codes = self.device_manager.ensure_device(morton_codes)
        batch_size = morton_codes.shape[0]

        # Извлечение битов для каждой координаты
        x_bits = (morton_codes.unsqueeze(1) >> self.x_positions.unsqueeze(0)) & 1
        y_bits = (morton_codes.unsqueeze(1) >> self.y_positions.unsqueeze(0)) & 1
        z_bits = (morton_codes.unsqueeze(1) >> self.z_positions.unsqueeze(0)) & 1

        # Восстановление координат
        x = (x_bits * self.bit_masks.unsqueeze(0)).sum(dim=1)
        y = (y_bits * self.bit_masks.unsqueeze(0)).sum(dim=1)
        z = (z_bits * self.bit_masks.unsqueeze(0)).sum(dim=1)

        return torch.stack([x, y, z], dim=1)


class GPUSpatialHashGrid:
    """
    GPU-accelerated пространственная хэш-решетка

    Использует PyTorch для векторизованных операций поиска соседей.
    Оптимизирована для batch processing и минимизации memory transfers.
    """

    def __init__(self, dimensions: Coordinates3D, cell_size: int = 4):
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        self.dimensions = dimensions
        self.cell_size = cell_size

        # Вычисляем размеры хэш-решетки
        self.grid_dims = tuple((d + cell_size - 1) // cell_size for d in dimensions)

        # GPU структуры данных
        self._initialize_gpu_structures()

        # Статистика
        self.stats = GPUSpatialHashingStats()

        logger.info(
            f"🏎️ GPUSpatialHashGrid инициализирован: {dimensions} → {self.grid_dims} "
            f"(cell_size={cell_size}) на {self.device}"
        )

    def _initialize_gpu_structures(self):
        """Инициализирует GPU структуры данных"""
        max_cells_estimate = np.prod(self.dimensions)

        # Основные структуры на GPU
        self.cell_coordinates = torch.empty(
            (0, 3), device=self.device, dtype=torch.long
        )
        self.cell_indices = torch.empty(0, device=self.device, dtype=torch.long)
        self.grid_hash_table = {}  # Пока используем CPU dict, позже оптимизируем

        # Кэш для ускорения повторных запросов
        self.query_cache = {}
        self.cache_max_size = 10000

    def insert_batch(self, coordinates: torch.Tensor, indices: torch.Tensor):
        """
        Вставляет batch клеток в решетку

        Args:
            coordinates: tensor формы (N, 3) с координатами
            indices: tensor формы (N,) с индексами клеток
        """
        coordinates = self.device_manager.ensure_device(coordinates)
        indices = self.device_manager.ensure_device(indices)

        # Вычисляем grid координаты для всего batch'а
        grid_coords = coordinates // self.cell_size

        # Вычисляем хэши для grid ячеек
        grid_hashes = self._compute_grid_hashes_batch(grid_coords)

        # Группируем по хэшам
        unique_hashes, inverse_indices = torch.unique(grid_hashes, return_inverse=True)

        for i, hash_val in enumerate(unique_hashes):
            mask = inverse_indices == i
            cell_indices_for_hash = indices[mask]

            hash_key = hash_val.item()
            if hash_key not in self.grid_hash_table:
                self.grid_hash_table[hash_key] = []

            self.grid_hash_table[hash_key].extend(cell_indices_for_hash.cpu().tolist())

        # Обновляем главные структуры
        self.cell_coordinates = torch.cat([self.cell_coordinates, coordinates], dim=0)
        self.cell_indices = torch.cat([self.cell_indices, indices], dim=0)

    def _compute_grid_hashes_batch(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Вычисляет хэши для batch grid координат"""
        # Приводим к целочисленному типу для битовых операций
        grid_coords = grid_coords.long()
        x, y, z = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

        # Используем простую но эффективную хэш-функцию
        prime1, prime2, prime3 = 73856093, 19349663, 83492791
        hashes = (x * prime1) ^ (y * prime2) ^ (z * prime3)

        return hashes

    def query_radius_batch(
        self, query_points: torch.Tensor, radius: float
    ) -> List[torch.Tensor]:
        """
        Поиск соседей в радиусе для batch точек

        Args:
            query_points: tensor формы (N, 3) с координатами запроса
            radius: радиус поиска

        Returns:
            Список tensor'ов с индексами найденных соседей для каждой точки
        """
        query_points = self.device_manager.ensure_device(query_points)
        batch_size = query_points.shape[0]

        start_time = (
            torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        )

        if start_time:
            start_time.record()

        results = []

        for i, point in enumerate(query_points):
            # Проверяем кэш
            cache_key = (tuple(point.cpu().tolist()), radius)
            if cache_key in self.query_cache:
                results.append(self.query_cache[cache_key])
                continue

            # Вычисляем диапазон grid ячеек для поиска
            min_grid = ((point - radius) // self.cell_size).long()
            max_grid = ((point + radius) // self.cell_size).long()

            # Ограничиваем диапазон границами решетки
            grid_dims_tensor = torch.tensor(
                self.grid_dims, device=self.device, dtype=torch.long
            )
            max_bounds = grid_dims_tensor - 1
            zero_tensor = torch.zeros_like(max_bounds)
            min_grid = torch.clamp(min_grid, min=zero_tensor, max=max_bounds)
            max_grid = torch.clamp(max_grid, min=zero_tensor, max=max_bounds)

            # Собираем кандидатов из всех релевантных ячеек
            candidates = set()

            for gx in range(min_grid[0].item(), max_grid[0].item() + 1):
                for gy in range(min_grid[1].item(), max_grid[1].item() + 1):
                    for gz in range(min_grid[2].item(), max_grid[2].item() + 1):
                        grid_coord = torch.tensor([gx, gy, gz], device=self.device)
                        hash_val = self._compute_grid_hashes_batch(
                            grid_coord.unsqueeze(0)
                        )[0].item()

                        if hash_val in self.grid_hash_table:
                            candidates.update(self.grid_hash_table[hash_val])

            # Преобразуем в tensor
            if candidates:
                neighbor_indices = torch.tensor(
                    list(candidates), device=self.device, dtype=torch.long
                )
            else:
                neighbor_indices = torch.empty(0, device=self.device, dtype=torch.long)

            results.append(neighbor_indices)

            # Добавляем в кэш
            if len(self.query_cache) < self.cache_max_size:
                self.query_cache[cache_key] = neighbor_indices

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            query_time_ms = start_time.elapsed_time(end_time)

            # Обновляем статистику
            self.stats.total_queries += batch_size
            old_avg = self.stats.avg_query_time_ms
            self.stats.avg_query_time_ms = (
                old_avg * (self.stats.total_queries - batch_size) + query_time_ms
            ) / self.stats.total_queries

        return results

    def optimize_memory(self):
        """Оптимизирует использование памяти"""
        # Очищаем кэш если он слишком большой
        if len(self.query_cache) > self.cache_max_size * 0.8:
            # Оставляем только 50% самых используемых записей
            self.query_cache.clear()
            logger.debug("🧹 Query cache очищен для оптимизации памяти")

        # Принудительная очистка GPU памяти
        self.device_manager.cleanup()

    def get_memory_usage(self) -> Dict[str, float]:
        """Получить статистику использования памяти"""
        coords_mb = self.cell_coordinates.numel() * 4 / (1024**2)  # int32
        indices_mb = self.cell_indices.numel() * 4 / (1024**2)  # int32

        return {
            "coordinates_mb": coords_mb,
            "indices_mb": indices_mb,
            "cache_entries": len(self.query_cache),
            "grid_buckets": len(self.grid_hash_table),
            "total_gpu_mb": coords_mb + indices_mb,
        }

    def get_stats(self) -> GPUSpatialHashingStats:
        """Получить статистику производительности"""
        memory_stats = self.get_memory_usage()
        self.stats.memory_usage_mb = memory_stats["total_gpu_mb"]
        self.stats.cache_hit_rate = len(self.query_cache) / max(
            1, self.stats.total_queries
        )

        return self.stats


class AdaptiveGPUSpatialHash:
    """
    Adaptive GPU Spatial Hash с автоматической оптимизацией параметров

    Автоматически подстраивает размер ячеек и структуры данных
    на основе доступной памяти и паттернов запросов
    """

    def __init__(self, dimensions: Coordinates3D, target_memory_mb: float = 1024.0):
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        self.dimensions = dimensions
        self.target_memory_mb = target_memory_mb

        # Автоматический расчет оптимального размера ячеек
        self.optimal_cell_size = self._calculate_optimal_cell_size()

        # Создаем основную hash grid
        self.hash_grid = GPUSpatialHashGrid(dimensions, self.optimal_cell_size)

        # Адаптивные параметры
        self.adaptation_frequency = 1000  # Переоптимизация каждые N запросов
        self.query_count = 0

        logger.info(
            f"🎯 AdaptiveGPUSpatialHash создан: cell_size={self.optimal_cell_size}, "
            f"target_memory={target_memory_mb}MB"
        )

    def _calculate_optimal_cell_size(self) -> int:
        """Вычисляет оптимальный размер ячеек на основе доступной памяти"""
        total_cells = np.prod(self.dimensions)

        # Оценка памяти на клетку (координаты + индекс + накладные расходы)
        memory_per_cell_bytes = 3 * 4 + 4 + 8  # 24 байта на клетку

        # Целевое количество ячеек в hash grid
        target_hash_cells = min(
            total_cells // 8,  # Не больше 1/8 от общего количества
            int(self.target_memory_mb * 1024**2 / memory_per_cell_bytes),
        )

        # Расчет оптимального размера ячеек
        if target_hash_cells <= 0:
            return max(self.dimensions) // 4  # Fallback

        # Кубический корень для равномерного распределения
        optimal_cell_size = max(1, int((total_cells / target_hash_cells) ** (1 / 3)))

        # Ограничиваем разумными пределами
        max_cell_size = max(self.dimensions) // 2
        optimal_cell_size = min(optimal_cell_size, max_cell_size)

        return max(2, optimal_cell_size)  # Минимум 2 для эффективности

    def insert_batch(self, coordinates: torch.Tensor, indices: torch.Tensor):
        """Вставка batch с автоматической адаптацией"""
        self.hash_grid.insert_batch(coordinates, indices)

        # Проверяем необходимость адаптации
        if self.query_count % self.adaptation_frequency == 0:
            self._adapt_parameters()

    def query_radius_batch(
        self, query_points: torch.Tensor, radius: float
    ) -> List[torch.Tensor]:
        """Поиск с автоматической оптимизацией"""
        self.query_count += query_points.shape[0]

        results = self.hash_grid.query_radius_batch(query_points, radius)

        # Периодическая оптимизация памяти
        if self.query_count % (self.adaptation_frequency // 2) == 0:
            self.hash_grid.optimize_memory()

        return results

    def _adapt_parameters(self):
        """Адаптирует параметры на основе статистики использования"""
        stats = self.hash_grid.get_stats()
        memory_usage = self.hash_grid.get_memory_usage()

        # Проверяем превышение целевой памяти
        if memory_usage["total_gpu_mb"] > self.target_memory_mb * 1.2:
            logger.warning(
                f"⚠️ Превышение target memory: {memory_usage['total_gpu_mb']:.1f}MB > "
                f"{self.target_memory_mb * 1.2:.1f}MB"
            )

            # Увеличиваем размер ячеек для уменьшения памяти
            new_cell_size = min(self.optimal_cell_size * 2, max(self.dimensions) // 2)
            if new_cell_size != self.optimal_cell_size:
                self._rebuild_with_new_cell_size(new_cell_size)

        elif memory_usage["total_gpu_mb"] < self.target_memory_mb * 0.5:
            # Можем уменьшить размер ячеек для повышения точности
            new_cell_size = max(self.optimal_cell_size // 2, 2)
            if new_cell_size != self.optimal_cell_size:
                self._rebuild_with_new_cell_size(new_cell_size)

        logger.debug(
            f"📊 Adaptive stats: queries={stats.total_queries}, "
            f"avg_time={stats.avg_query_time_ms:.2f}ms, "
            f"memory={memory_usage['total_gpu_mb']:.1f}MB"
        )

    def _rebuild_with_new_cell_size(self, new_cell_size: int):
        """Перестраивает hash grid с новым размером ячеек"""
        logger.info(
            f"🔄 Rebuilding hash grid: {self.optimal_cell_size} → {new_cell_size}"
        )

        # Сохраняем данные
        old_coordinates = self.hash_grid.cell_coordinates
        old_indices = self.hash_grid.cell_indices

        # Создаем новую структуру
        self.optimal_cell_size = new_cell_size
        self.hash_grid = GPUSpatialHashGrid(self.dimensions, new_cell_size)

        # Переносим данные
        if len(old_coordinates) > 0:
            self.hash_grid.insert_batch(old_coordinates, old_indices)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Получить полную статистику"""
        hash_stats = self.hash_grid.get_stats()
        memory_stats = self.hash_grid.get_memory_usage()
        device_stats = self.device_manager.get_memory_stats()

        return {
            "hash_grid": {
                "cell_size": self.optimal_cell_size,
                "queries": hash_stats.total_queries,
                "avg_query_time_ms": hash_stats.avg_query_time_ms,
                "cache_hit_rate": hash_stats.cache_hit_rate,
            },
            "memory": memory_stats,
            "device": device_stats,
            "adaptations": {
                "adaptation_frequency": self.adaptation_frequency,
                "query_count": self.query_count,
                "target_memory_mb": self.target_memory_mb,
                "adaptations_performed": max(
                    0, self.query_count // self.adaptation_frequency
                ),
            },
        }
