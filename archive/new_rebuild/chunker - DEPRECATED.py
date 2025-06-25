#!/usr/bin/env python3
"""
Lattice Chunker - Разбивка решеток на управляемые части
======================================================

LatticeChunker автоматически разбивает большие решетки на части,
которые помещаются в GPU память, с учетом соседства между chunk'ами.
"""

from typing import List
from ....config.project_config import ChunkInfo
from ....config.project_config import get_project_config
from ..spatial_hashing import Coordinates3D
from ..position import Position3D
from ....utils.logging import get_logger

logger = get_logger(__name__)


class LatticeChunker:
    """
    ⚠️ DEPRECATED: Разбивка больших решеток на управляемые chunk'и
    ============================================================

    🚨 УСТАРЕЛ: Используйте AdaptiveGPUChunker для лучшей производительности!

    Автоматически разбивает решетку на части, которые помещаются в GPU память,
    с учетом соседства между chunk'ами.

    ЗАМЕНЕН НА: new_rebuild.core.lattice.spatial_optimization.adaptive_chunker.AdaptiveGPUChunker
    ПРИЧИНА: GPU acceleration, adaptive memory management, better performance

    DEPRECATED с 28 декабря 2025. Будет удален в версии 2.0.
    """

    def __init__(self, dimensions: Coordinates3D, config: dict = None):
        self.dimensions = dimensions
        self.config = config or get_project_config().get_spatial_optim_config()
        self.pos_helper = Position3D(dimensions)

        self.chunks: List[ChunkInfo] = []
        self._create_chunks()

        logger.info(f"🧩 LatticeChunker создал {len(self.chunks)} chunk'ов")

    def _create_chunks(self):
        """Создает chunk'и с оптимальным разбиением"""
        x_dim, y_dim, z_dim = self.dimensions
        chunk_size = self.config["chunk_size"]

        # Вычисляем количество chunk'ов по каждой оси
        x_chunks = max(1, (x_dim + chunk_size - 1) // chunk_size)
        y_chunks = max(1, (y_dim + chunk_size - 1) // chunk_size)
        z_chunks = max(1, (z_dim + chunk_size - 1) // chunk_size)

        chunk_id = 0

        for z_idx in range(z_chunks):
            for y_idx in range(y_chunks):
                for x_idx in range(x_chunks):
                    # Координаты chunk'а
                    start_x = x_idx * chunk_size
                    start_y = y_idx * chunk_size
                    start_z = z_idx * chunk_size

                    end_x = min(start_x + chunk_size, x_dim)
                    end_y = min(start_y + chunk_size, y_dim)
                    end_z = min(start_z + chunk_size, z_dim)

                    # Создаем chunk info
                    chunk_info = self._create_chunk_info(
                        chunk_id, (start_x, start_y, start_z), (end_x, end_y, end_z)
                    )

                    self.chunks.append(chunk_info)
                    chunk_id += 1

        # Вычисляем соседние chunk'и
        self._compute_neighbor_chunks()

    def _create_chunk_info(
        self, chunk_id: int, start: Coordinates3D, end: Coordinates3D
    ) -> ChunkInfo:
        """Создает информацию о chunk'е"""
        # Вычисляем клетки в chunk'е
        cell_indices = []
        for z in range(start[2], end[2]):
            for y in range(start[1], end[1]):
                for x in range(start[0], end[0]):
                    cell_idx = self.pos_helper.to_linear_index((x, y, z))
                    cell_indices.append(cell_idx)

        # Оценка размера памяти
        num_cells = len(cell_indices)
        memory_size_mb = num_cells * 32 * 4 / (1024**2)  # float32 состояния

        return ChunkInfo(
            chunk_id=chunk_id,
            start_coords=start,
            end_coords=end,
            cell_indices=cell_indices,
            neighbor_chunks=[],  # Заполним позже
            memory_size_mb=memory_size_mb,
        )

    def _compute_neighbor_chunks(self):
        """Вычисляет соседние chunk'и для каждого chunk'а"""
        for chunk in self.chunks:
            neighbor_chunk_ids = []

            for other_chunk in self.chunks:
                if chunk.chunk_id != other_chunk.chunk_id:
                    if self._are_chunks_neighbors(chunk, other_chunk):
                        neighbor_chunk_ids.append(other_chunk.chunk_id)

            chunk.neighbor_chunks = neighbor_chunk_ids

    def _are_chunks_neighbors(self, chunk1: ChunkInfo, chunk2: ChunkInfo) -> bool:
        """Проверяет, являются ли chunk'и соседними"""
        overlap = self.config["chunk_overlap"]

        # Расширяем границы chunk1 на overlap
        start1 = (
            max(0, chunk1.start_coords[0] - overlap),
            max(0, chunk1.start_coords[1] - overlap),
            max(0, chunk1.start_coords[2] - overlap),
        )
        end1 = (
            min(self.dimensions[0], chunk1.end_coords[0] + overlap),
            min(self.dimensions[1], chunk1.end_coords[1] + overlap),
            min(self.dimensions[2], chunk1.end_coords[2] + overlap),
        )

        # Проверяем пересечение с chunk2
        return (
            start1[0] < chunk2.end_coords[0]
            and end1[0] > chunk2.start_coords[0]
            and start1[1] < chunk2.end_coords[1]
            and end1[1] > chunk2.start_coords[1]
            and start1[2] < chunk2.end_coords[2]
            and end1[2] > chunk2.start_coords[2]
        )

    def get_chunk_by_coords(self, coords: Coordinates3D) -> ChunkInfo:
        """Находит chunk по координатам"""
        for chunk in self.chunks:
            if (
                chunk.start_coords[0] <= coords[0] < chunk.end_coords[0]
                and chunk.start_coords[1] <= coords[1] < chunk.end_coords[1]
                and chunk.start_coords[2] <= coords[2] < chunk.end_coords[2]
            ):
                return chunk

        raise ValueError(f"Chunk не найден для координат {coords}")

    def get_processing_schedule(self) -> List[List[int]]:
        """
        Возвращает оптимальное расписание обработки chunk'ов

        Группирует chunk'и так, чтобы минимизировать конфликты памяти
        и максимизировать параллельную обработку.
        """
        remaining_chunks = set(range(len(self.chunks)))
        schedule = []

        while remaining_chunks:
            # Выбираем batch chunk'ов для параллельной обработки
            current_batch = []
            used_neighbors = set()

            for chunk_id in list(remaining_chunks):
                chunk = self.chunks[chunk_id]

                # Проверяем, можем ли добавить этот chunk в текущий batch
                conflicts = set(chunk.neighbor_chunks) & used_neighbors

                if (
                    not conflicts
                    and len(current_batch) < self.config["max_chunks_in_memory"]
                ):
                    current_batch.append(chunk_id)
                    used_neighbors.add(chunk_id)
                    used_neighbors.update(chunk.neighbor_chunks)

            # Удаляем обработанные chunk'и
            for chunk_id in current_batch:
                remaining_chunks.remove(chunk_id)

            schedule.append(current_batch)

        logger.debug(f"   📅 Создано расписание: {len(schedule)} batch'ей")
        return schedule
