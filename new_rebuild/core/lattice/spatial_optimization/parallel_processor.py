#!/usr/bin/env python3
"""
Parallel Spatial Processor - Параллельная обработка пространственных операций
============================================================================

ParallelSpatialProcessor обеспечивает многопоточную и асинхронную обработку
пространственных операций для максимальной производительности.
"""

import torch
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
from ....config.project_config import ChunkInfo
from ....config.project_config import get_project_config
from .chunker import LatticeChunker
from .hierarchical_index import HierarchicalSpatialIndex
from .memory_manager import MemoryPoolManager
from ....utils.logging import get_logger

logger = get_logger(__name__)


class ParallelSpatialProcessor:
    """
    Параллельная обработка пространственных операций

    Координирует chunking, memory management и parallel processing
    для оптимальной производительности на больших решетках.
    """

    def __init__(
        self,
        chunker: LatticeChunker,
        spatial_index: HierarchicalSpatialIndex,
        memory_manager: MemoryPoolManager,
        config: dict = None,
    ):
        self.chunker = chunker
        self.spatial_index = spatial_index
        self.memory_manager = memory_manager
        self.config = config or get_project_config().get_spatial_optim_config()

        # Thread pool для параллельной обработки
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self._init_thread_pool()

        # Статистика производительности
        self.performance_stats = {
            "total_batches_processed": 0,
            "avg_batch_time_ms": 0.0,
            "parallel_efficiency": 1.0,
            "memory_efficiency": 1.0,
        }

        # Лок для thread-safe операций
        self._stats_lock = threading.Lock()

        logger.info(
            f"⚙️ ParallelSpatialProcessor готов с {self.config['num_worker_threads']} потоками"
        )

    def _init_thread_pool(self):
        """Инициализирует thread pool для параллельной обработки"""
        if self.config["enable_async_processing"]:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config["num_worker_threads"],
                thread_name_prefix="SpatialProcessor",
            )

    def process_lattice_parallel(
        self, states: torch.Tensor, neighbor_processor_fn: Callable
    ) -> torch.Tensor:
        """
        Параллельная обработка всей решетки

        Args:
            states: [num_cells, state_size] - состояния клеток
            neighbor_processor_fn: функция обработки соседей

        Returns:
            new_states: [num_cells, state_size] - новые состояния
        """
        start_time = time.time()
        num_cells = states.shape[0]

        logger.info(f"🚀 Запуск параллельной обработки {num_cells:,} клеток")

        # Получаем оптимальное расписание обработки chunk'ов
        processing_schedule = self.chunker.get_processing_schedule()

        # Инициализируем выходные состояния
        output_states = states.clone()

        # Обрабатываем batch'и chunk'ов
        total_batches = len(processing_schedule)
        processed_batches = 0

        for batch_idx, chunk_ids in enumerate(processing_schedule):
            logger.debug(
                f"   🔄 Batch {batch_idx + 1}/{total_batches}: chunk'ы {chunk_ids}"
            )

            if self.config["enable_async_processing"] and len(chunk_ids) > 1:
                # Асинхронная обработка для множественных chunk'ов
                batch_results = self._process_batch_async(
                    chunk_ids, states, neighbor_processor_fn
                )
            else:
                # Синхронная обработка для одиночных chunk'ов
                batch_results = self._process_batch_sync(
                    chunk_ids, states, neighbor_processor_fn
                )

            # Объединяем результаты batch'а
            for chunk_id, chunk_output in batch_results.items():
                self._merge_chunk_output(output_states, chunk_id, chunk_output)

            processed_batches += 1

            # Периодическая очистка памяти
            if processed_batches % 3 == 0:
                self.memory_manager.garbage_collect()

        processing_time = time.time() - start_time
        self._update_processing_stats(total_batches, processing_time)

        logger.info(f"✅ Параллельная обработка завершена за {processing_time:.3f}s")

        return output_states

    def _process_batch_async(
        self,
        chunk_ids: List[int],
        states: torch.Tensor,
        neighbor_processor_fn: Callable,
    ) -> Dict[int, torch.Tensor]:
        """Асинхронная обработка batch'а chunk'ов"""
        futures: Dict[int, Future] = {}

        # Запускаем асинхронную обработку каждого chunk'а
        for chunk_id in chunk_ids:
            future = self.thread_pool.submit(
                self._process_chunk_async, chunk_id, states, neighbor_processor_fn
            )
            futures[chunk_id] = future

        # Собираем результаты
        results = {}
        for chunk_id, future in futures.items():
            try:
                results[chunk_id] = future.result(timeout=60.0)  # 60 секунд timeout
            except Exception as e:
                logger.warning(f"⚠️ Ошибка в async обработке chunk {chunk_id}: {e}")
                # Fallback к sync обработке
                results[chunk_id] = self._process_chunk_sync(
                    chunk_id, states, neighbor_processor_fn
                )

        return results

    def _process_batch_sync(
        self,
        chunk_ids: List[int],
        states: torch.Tensor,
        neighbor_processor_fn: Callable,
    ) -> Dict[int, torch.Tensor]:
        """Синхронная обработка batch'а chunk'ов"""
        results = {}

        for chunk_id in chunk_ids:
            results[chunk_id] = self._process_chunk_sync(
                chunk_id, states, neighbor_processor_fn
            )

        return results

    def _process_chunk_async(
        self, chunk_id: int, states: torch.Tensor, neighbor_processor_fn: Callable
    ) -> torch.Tensor:
        """Асинхронная обработка одного chunk'а"""
        return self._process_chunk_sync(chunk_id, states, neighbor_processor_fn)

    def _process_chunk_sync(
        self, chunk_id: int, states: torch.Tensor, neighbor_processor_fn: Callable
    ) -> torch.Tensor:
        """Синхронная обработка одного chunk'а"""
        chunk = self.chunker.chunks[chunk_id]
        chunk_cells = chunk.cell_indices

        # Получаем tensor для выходных состояний chunk'а
        chunk_size = len(chunk_cells)
        state_size = states.shape[1]
        chunk_output = self.memory_manager.get_tensor((chunk_size, state_size))

        # Обрабатываем каждую клетку в chunk'е
        for i, cell_idx in enumerate(chunk_cells):
            # Получаем соседей для клетки
            chunk_neighbors = self._get_chunk_neighbors(chunk, states)

            if chunk_neighbors.shape[0] > 0:
                # Применяем функцию обработки соседей
                new_state = neighbor_processor_fn(
                    states[cell_idx], chunk_neighbors, cell_idx, chunk_cells
                )
                chunk_output[i] = new_state
            else:
                # Если соседей нет, оставляем состояние без изменений
                chunk_output[i] = states[cell_idx]

        return chunk_output

    def _get_chunk_neighbors(
        self, chunk: ChunkInfo, states: torch.Tensor
    ) -> torch.Tensor:
        """Получает соседей для chunk'а из spatial index"""
        # Для простоты берем первые несколько клеток как соседей
        # В реальной реализации здесь был бы полноценный spatial query
        max_neighbors = min(10, len(chunk.cell_indices))
        if max_neighbors > 0:
            neighbor_indices = chunk.cell_indices[:max_neighbors]
            return states[neighbor_indices]
        else:
            return torch.empty(0, states.shape[1], device=states.device)

    def _merge_chunk_output(
        self, output_states: torch.Tensor, chunk_id: int, chunk_output: torch.Tensor
    ):
        """Объединяет результаты chunk'а с общими выходными состояниями"""
        chunk = self.chunker.chunks[chunk_id]
        chunk_cells = chunk.cell_indices

        # Копируем результаты chunk'а в основной tensor
        for i, cell_idx in enumerate(chunk_cells):
            if i < chunk_output.shape[0]:
                output_states[cell_idx] = chunk_output[i]

        # Возвращаем chunk_output в memory pool
        self.memory_manager.return_tensor(chunk_output)

    def _update_processing_stats(self, num_batches: int, total_time: float):
        """Обновляет статистику производительности"""
        with self._stats_lock:
            self.performance_stats["total_batches_processed"] += num_batches

            # Обновляем среднее время batch'а
            avg_batch_time_ms = (total_time * 1000) / max(num_batches, 1)
            current_avg = self.performance_stats["avg_batch_time_ms"]
            self.performance_stats["avg_batch_time_ms"] = (
                current_avg * 0.8 + avg_batch_time_ms * 0.2
            )

            # Вычисляем эффективность параллелизма
            expected_time = avg_batch_time_ms * num_batches
            actual_time = total_time * 1000
            if actual_time > 0:
                parallel_efficiency = min(1.0, expected_time / actual_time)
                self.performance_stats["parallel_efficiency"] = (
                    self.performance_stats["parallel_efficiency"] * 0.9
                    + parallel_efficiency * 0.1
                )

    def get_performance_stats(self) -> Dict[str, float]:
        """Получить статистику производительности"""
        with self._stats_lock:
            stats = self.performance_stats.copy()

        # Добавляем статистику memory manager'а
        memory_stats = self.memory_manager.get_memory_stats()
        stats.update({f"memory_{key}": value for key, value in memory_stats.items()})

        return stats

    def shutdown(self):
        """Завершение работы и очистка ресурсов"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

        self.memory_manager.cleanup()

        logger.info("🔒 ParallelSpatialProcessor завершен")
