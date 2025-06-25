#!/usr/bin/env python3
"""
Adaptive GPU Chunker - Интеллектуальная разбивка решеток с GPU поддержкой
========================================================================

AdaptiveGPUChunker автоматически разбивает большие решетки на оптимальные части
с учетом доступной GPU памяти, паттернов доступа и требований производительности.

Ключевые особенности:
- Dynamic chunk sizing на основе доступной памяти
- GPU memory monitoring в реальном времени
- Adaptive load balancing между chunk'ами
- Intelligent prefetching для оптимизации производительности

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future

from ....config import get_project_config, AdaptiveChunkerConfig
from ..position import Position3D
from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager
from .memory_manager import get_memory_pool_manager

logger = get_logger(__name__)

Coordinates3D = Tuple[int, int, int]


@dataclass
class AdaptiveChunkInfo:
    """Расширенная информация о chunk'е с adaptive характеристиками"""

    chunk_id: int
    start_coords: Coordinates3D
    end_coords: Coordinates3D
    cell_indices: List[int]
    neighbor_chunks: List[int] = field(default_factory=list)
    memory_size_mb: float = 0.0
    # GPU специфичные поля
    gpu_memory_usage_mb: float = 0.0
    last_access_time: float = field(default_factory=time.time)
    access_frequency: int = 0
    processing_priority: int = 0
    # Adaptive характеристики
    optimal_batch_size: int = 1000
    preferred_device: str = "cuda"
    memory_pressure_level: float = 0.0  # 0.0 = низкое, 1.0 = критическое
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    neighbor_access_pattern: Dict[int, int] = field(default_factory=dict)
    prefetched_data: Optional[torch.Tensor] = None


@dataclass
class ChunkProcessingTask:
    """Задача обработки chunk'а"""

    chunk_id: int
    operation_type: str  # "load", "process", "unload", "prefetch"
    priority: int = 0
    estimated_memory_mb: float = 0.0
    dependencies: List[int] = field(default_factory=list)
    callback: Optional[callable] = None


class AdaptiveMemoryPredictor:
    """Предсказатель использования памяти для оптимизации chunking'а"""

    def __init__(self):
        self.device_manager = get_device_manager()
        self.historical_usage = []
        cfg = get_project_config().adaptive_chunker
        self.max_history = cfg.max_history
        self.memory_per_cell_base = cfg.memory_per_cell_base
        self.memory_overhead_factor = cfg.memory_overhead_factor
        self.min_available_memory_mb = cfg.min_available_memory_mb
        self.cuda_fallback_available_mb = cfg.cuda_fallback_available_mb
        self.cpu_fallback_available_mb = cfg.cpu_fallback_available_mb
        self.safe_memory_buffer = cfg.safe_memory_buffer

    def predict_chunk_memory(self, chunk_info: AdaptiveChunkInfo) -> float:
        """
        Предсказывает использование памяти для chunk'а

        Args:
            chunk_info: Информация о chunk'е

        Returns:
            Ожидаемое использование памяти в MB
        """
        num_cells = len(chunk_info.cell_indices)

        # Базовое использование памяти
        base_memory = num_cells * self.memory_per_cell_base

        # Накладные расходы на структуры данных
        overhead_memory = base_memory * self.memory_overhead_factor

        # Учитываем историческое использование
        if self.historical_usage:
            avg_historical = np.mean(
                self.historical_usage[-10:]
            )  # последние 10 записей
            predicted_memory = 0.7 * overhead_memory + 0.3 * avg_historical
        else:
            predicted_memory = overhead_memory

        # Конвертируем в MB
        predicted_memory_mb = predicted_memory / (1024**2)

        return predicted_memory_mb

    def update_actual_usage(self, chunk_id: int, actual_memory_mb: float):
        """Обновляет фактическое использование памяти"""
        self.historical_usage.append(actual_memory_mb)

        # Ограничиваем историю
        if len(self.historical_usage) > self.max_history:
            self.historical_usage = self.historical_usage[-self.max_history :]

    def get_available_memory_mb(self) -> float:
        """Получает доступную GPU память"""
        device_stats = self.device_manager.get_memory_stats()
        cfg = get_project_config().adaptive_chunker
        if self.device_manager.is_cuda():
            available_mb = device_stats.get(
                "available_mb", cfg.cuda_fallback_available_mb
            )
        else:
            available_mb = device_stats.get(
                "available_mb", cfg.cpu_fallback_available_mb
            )
        safe_available_mb = available_mb * cfg.safe_memory_buffer
        return max(cfg.min_available_memory_mb, safe_available_mb)


class ChunkScheduler:
    """Планировщик обработки chunk'ов с учетом памяти и зависимостей"""

    def __init__(self, max_concurrent_chunks: int = None):
        cfg = get_project_config().adaptive_chunker
        self.max_concurrent_chunks = max_concurrent_chunks or cfg.max_concurrent_chunks
        self.task_queue = Queue()
        self.active_chunks: Set[int] = set()
        self.chunk_locks = {}
        self.memory_predictor = AdaptiveMemoryPredictor()

        self.device_manager = get_device_manager()
        if self.device_manager.is_cuda():
            self.main_stream = torch.cuda.Stream()
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.main_stream = None
            self.prefetch_stream = None

        # Thread pool для асинхронной обработки
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_chunks)
        self.running = True

        # Запускаем scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        logger.info(
            f"📅 ChunkScheduler запущен: max_concurrent={max_concurrent_chunks}"
        )

    def schedule_task(self, task: ChunkProcessingTask) -> Future:
        """Планирует задачу обработки chunk'а"""
        # Предсказываем память
        if hasattr(task, "chunk_info"):
            task.estimated_memory_mb = self.memory_predictor.predict_chunk_memory(
                task.chunk_info
            )

        # Добавляем в очередь
        future = Future()
        self.task_queue.put((task, future))

        return future

    def _scheduler_loop(self):
        """Основной цикл планировщика"""
        while self.running:
            try:
                # Проверяем доступную память
                available_memory = self.memory_predictor.get_available_memory_mb()

                # Обрабатываем задачи если есть место
                if len(self.active_chunks) < self.max_concurrent_chunks:
                    try:
                        task, future = self.task_queue.get(timeout=1.0)

                        # Проверяем, можем ли выполнить задачу
                        if task.estimated_memory_mb <= available_memory:
                            self._execute_task(task, future)
                        else:
                            # Возвращаем задачу в очередь с более низким приоритетом
                            task.priority -= 1
                            self.task_queue.put((task, future))

                    except Empty:
                        continue

                time.sleep(0.1)  # Небольшая пауза

            except Exception as e:
                logger.error(f"❌ Ошибка в scheduler loop: {e}")

    def _execute_task(self, task: ChunkProcessingTask, future: Future):
        """Выполняет задачу обработки chunk'а"""
        self.active_chunks.add(task.chunk_id)

        def task_wrapper():
            try:
                start_time = time.time()

                # Выполняем задачу
                if task.callback:
                    result = task.callback(task)
                else:
                    result = f"Processed chunk {task.chunk_id}"

                processing_time = (time.time() - start_time) * 1000  # ms

                # Обновляем статистику
                logger.debug(
                    f"✅ Chunk {task.chunk_id} обработан за {processing_time:.1f}ms"
                )

                future.set_result(result)

            except Exception as e:
                logger.error(f"❌ Ошибка обработки chunk {task.chunk_id}: {e}")
                future.set_exception(e)
            finally:
                self.active_chunks.discard(task.chunk_id)

        # Запускаем асинхронно
        self.executor.submit(task_wrapper)

    def shutdown(self):
        """Завершает работу планировщика"""
        self.running = False
        self.executor.shutdown(wait=True)
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()


class AdaptiveGPUChunker:
    """
    Адаптивная разбивка решетки на chunk'и
    """

    def __init__(self, dimensions: Coordinates3D, config: dict = None):
        self.config = config or get_project_config().adaptive_chunker
        self.dimensions = dimensions
        self.pos_helper = Position3D(dimensions)
        self.device_manager = get_device_manager()
        self.memory_predictor = AdaptiveMemoryPredictor()
        self.scheduler = ChunkScheduler(self.config.get("max_concurrent_chunks"))

        if self.device_manager.is_cuda():
            self.prefetch_events: Dict[int, torch.cuda.Event] = {}

        self._chunks: List[AdaptiveChunkInfo] = self._create_adaptive_chunks()
        self._neighbor_map: Dict[int, List[int]] = {}
        self._compute_neighbor_chunks()

        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Memory management
        self.memory_manager = get_memory_pool_manager(self.config)

        # Performance monitoring
        self.performance_stats = {
            "total_chunks": 0,
            "memory_efficiency": 0.0,
            "avg_chunk_processing_time_ms": 0.0,
            "memory_pressure_events": 0,
            "adaptive_rebalancing_events": 0,
        }

        logger.info(
            f"🎯 AdaptiveGPUChunker создан: {len(self._chunks)} chunks на {self.device}"
        )

    @property
    def chunks(self) -> List[AdaptiveChunkInfo]:
        """Совместимость: возвращает adaptive_chunks как chunks"""
        return self._chunks

    def _create_adaptive_chunks(self) -> List[AdaptiveChunkInfo]:
        """Создает adaptive chunk'и с оптимальным размером"""
        available_memory_mb = self.memory_predictor.get_available_memory_mb()
        optimal_chunk_size = self._calculate_optimal_chunk_size(available_memory_mb)
        x_dim, y_dim, z_dim = self.dimensions

        # Вычисляем количество chunk'ов по каждой оси
        x_chunks = max(1, (x_dim + optimal_chunk_size - 1) // optimal_chunk_size)
        y_chunks = max(1, (y_dim + optimal_chunk_size - 1) // optimal_chunk_size)
        z_chunks = max(1, (z_dim + optimal_chunk_size - 1) // optimal_chunk_size)

        chunk_id = 0
        chunks = []

        for z_idx in range(z_chunks):
            for y_idx in range(y_chunks):
                for x_idx in range(x_chunks):
                    # Координаты chunk'а
                    start_x = x_idx * optimal_chunk_size
                    start_y = y_idx * optimal_chunk_size
                    start_z = z_idx * optimal_chunk_size

                    end_x = min(start_x + optimal_chunk_size, x_dim)
                    end_y = min(start_y + optimal_chunk_size, y_dim)
                    end_z = min(start_z + optimal_chunk_size, z_dim)

                    # Создаем adaptive chunk info
                    chunk_info = self._create_adaptive_chunk_info(
                        chunk_id,
                        (start_x, start_y, start_z),
                        (end_x, end_y, end_z),
                        available_memory_mb,
                    )

                    chunks.append(chunk_info)
                    chunk_id += 1

        # Вычисляем соседние chunk'и и оптимизируем
        self._compute_neighbor_chunks()
        self._optimize_chunk_parameters()

        self.performance_stats["total_chunks"] = len(chunks)

        return chunks

    def _calculate_optimal_chunk_size(self, available_memory_mb: float) -> int:
        cfg = get_project_config().adaptive_chunker
        total_cells = np.prod(self.dimensions)
        target_memory_per_chunk_mb = (
            available_memory_mb * 0.75 / self.config.get("max_chunks_in_memory", 4)
        )
        memory_per_cell_bytes = cfg.memory_per_cell_base
        cells_per_chunk = int(
            target_memory_per_chunk_mb * 1024**2 / memory_per_cell_bytes
        )
        if cells_per_chunk <= 0:
            chunk_size = max(self.dimensions) // cfg.chunk_size_fallback_div
        else:
            chunk_size = max(cfg.min_chunk_size, int(cells_per_chunk ** (1 / 3)))
        max_chunk_size = max(self.dimensions) // 2
        min_chunk_size = cfg.min_chunk_size
        optimal_size = max(min_chunk_size, min(chunk_size, max_chunk_size))

        logger.debug(
            f"📏 Optimal chunk size: {optimal_size} "
            f"(available_memory={available_memory_mb:.1f}MB, cells_per_chunk={cells_per_chunk})"
        )

        return optimal_size

    def _create_adaptive_chunk_info(
        self,
        chunk_id: int,
        start: Coordinates3D,
        end: Coordinates3D,
        available_memory_mb: float,
    ) -> AdaptiveChunkInfo:
        cfg = get_project_config().adaptive_chunker

        # Вычисляем клетки в chunk'е
        cell_indices = []
        for z in range(start[2], end[2]):
            for y in range(start[1], end[1]):
                for x in range(start[0], end[0]):
                    cell_idx = self.pos_helper.to_linear_index((x, y, z))
                    cell_indices.append(cell_idx)

        # Базовая информация
        num_cells = len(cell_indices)
        memory_size_mb = num_cells * 64 / (1024**2)  # приблизительная оценка

        # Создаем adaptive chunk info
        chunk_info = AdaptiveChunkInfo(
            chunk_id=chunk_id,
            start_coords=start,
            end_coords=end,
            cell_indices=cell_indices,
            neighbor_chunks=[],  # Заполним позже
            memory_size_mb=memory_size_mb,
            gpu_memory_usage_mb=0.0,
            last_access_time=time.time(),
            access_frequency=0,
            processing_priority=self._calculate_initial_priority(start, end),
            optimal_batch_size=min(cfg.optimal_batch_size, num_cells),
            preferred_device=cfg.preferred_device,
            memory_pressure_level=min(1.0, memory_size_mb / available_memory_mb),
        )

        # После создания chunk_info, если мы на GPU, создаем prefetch event
        if self.device_manager.is_cuda():
            self.prefetch_events[chunk_id] = torch.cuda.Event()

        return chunk_info

    def _calculate_initial_priority(
        self, start: Coordinates3D, end: Coordinates3D
    ) -> int:
        """Вычисляет начальный приоритет chunk'а"""
        # Центральные chunk'и имеют более высокий приоритет
        center = tuple(d // 2 for d in self.dimensions)
        chunk_center = tuple((s + e) // 2 for s, e in zip(start, end))

        # Расстояние от центра решетки
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(center, chunk_center)) ** 0.5
        max_distance = sum(d**2 for d in self.dimensions) ** 0.5

        # Приоритет от 1 до 100 (центральные - выше)
        priority = int(100 * (1 - distance / max_distance))

        return max(1, priority)

    def _compute_neighbor_chunks(self):
        """Вычисляет соседние chunk'и для каждого chunk'а"""
        overlap = self.config.get("chunk_overlap", 8)

        for chunk in self._chunks:
            neighbor_chunk_ids = []

            for other_chunk in self._chunks:
                if chunk.chunk_id != other_chunk.chunk_id:
                    if self._are_chunks_neighbors(chunk, other_chunk, overlap):
                        neighbor_chunk_ids.append(other_chunk.chunk_id)

            chunk.neighbor_chunks = neighbor_chunk_ids
            self._neighbor_map[chunk.chunk_id] = neighbor_chunk_ids

    def _are_chunks_neighbors(
        self, chunk1: AdaptiveChunkInfo, chunk2: AdaptiveChunkInfo, overlap: int
    ) -> bool:
        """Проверяет, являются ли chunk'и соседними"""
        # Расширяем границы chunk1 на overlap
        start1 = tuple(max(0, s - overlap) for s in chunk1.start_coords)
        end1 = tuple(
            min(d, e + overlap) for d, e in zip(self.dimensions, chunk1.end_coords)
        )

        # Проверяем пересечение с chunk2
        return all(
            s1 < chunk2.end_coords[i] and e1 > chunk2.start_coords[i]
            for i, (s1, e1) in enumerate(zip(start1, end1))
        )

    def _optimize_chunk_parameters(self):
        cfg = get_project_config().adaptive_chunker
        for chunk in self._chunks:
            # Оптимизируем batch size на основе размера chunk'а
            num_cells = len(chunk.cell_indices)

            if num_cells < cfg.optimal_batch_size_small:
                chunk.optimal_batch_size = num_cells
            elif num_cells < cfg.optimal_batch_size_medium:
                chunk.optimal_batch_size = num_cells // 4
            else:
                chunk.optimal_batch_size = cfg.optimal_batch_size_large

            # Устанавливаем приоритет на основе memory pressure
            if chunk.memory_pressure_level > cfg.memory_pressure_high:
                chunk.processing_priority = max(
                    1, chunk.processing_priority - cfg.processing_priority_low_delta
                )
            elif chunk.memory_pressure_level < cfg.memory_pressure_low:
                chunk.processing_priority = min(
                    100, chunk.processing_priority + cfg.processing_priority_high_delta
                )

    def get_chunk_by_coords(self, coords: Coordinates3D) -> AdaptiveChunkInfo:
        """Находит chunk по координатам с обновлением статистики доступа"""
        for chunk in self._chunks:
            if all(
                chunk.start_coords[i] <= coords[i] < chunk.end_coords[i]
                for i in range(3)
            ):
                # Обновляем статистику доступа
                chunk.last_access_time = time.time()
                chunk.access_frequency += 1

                return chunk

        raise ValueError(f"Chunk не найден для координат {coords}")

    def get_adaptive_processing_schedule(self) -> List[List[int]]:
        """
        Возвращает adaptive расписание обработки chunk'ов

        Учитывает приоритеты, использование памяти и зависимости
        """
        available_memory = self.memory_predictor.get_available_memory_mb()
        max_concurrent = self.config.get("max_chunks_in_memory", 4)

        # Сортируем chunk'и по приоритету и memory pressure
        sorted_chunks = sorted(
            self._chunks,
            key=lambda c: (c.processing_priority, -c.memory_pressure_level),
            reverse=True,
        )

        schedule = []
        remaining_chunks = set(c.chunk_id for c in sorted_chunks)

        while remaining_chunks:
            current_batch = []
            current_memory_usage = 0.0
            used_neighbors = set()

            for chunk in sorted_chunks:
                if chunk.chunk_id not in remaining_chunks:
                    continue

                # Проверяем ограничения
                estimated_memory = chunk.memory_size_mb
                conflicts = set(chunk.neighbor_chunks) & used_neighbors

                can_add = (
                    not conflicts
                    and len(current_batch) < max_concurrent
                    and current_memory_usage + estimated_memory
                    <= available_memory * 0.9
                )

                if can_add:
                    current_batch.append(chunk.chunk_id)
                    current_memory_usage += estimated_memory
                    used_neighbors.add(chunk.chunk_id)
                    used_neighbors.update(chunk.neighbor_chunks)

            # Удаляем обработанные chunk'и
            for chunk_id in current_batch:
                remaining_chunks.remove(chunk_id)

            if current_batch:
                schedule.append(current_batch)
            else:
                # Если не можем добавить ни одного chunk'а, добавляем первый доступный
                if remaining_chunks:
                    first_chunk = next(iter(remaining_chunks))
                    schedule.append([first_chunk])
                    remaining_chunks.remove(first_chunk)

        logger.debug(
            f"📅 Adaptive schedule создано: {len(schedule)} batches, "
            f"avg_batch_size={np.mean([len(b) for b in schedule]):.1f}"
        )

        return schedule

    def _prefetch_chunk_data(self, chunk_id: int, all_states: torch.Tensor):
        """Асинхронно предзагружает данные для chunk'а."""
        if not self.device_manager.is_cuda():
            return

        chunk_info = self._chunks[chunk_id]

        with torch.cuda.stream(self.scheduler.prefetch_stream):
            indices = torch.tensor(
                chunk_info.cell_indices, device="cpu", dtype=torch.long
            )
            chunk_info.prefetched_data = all_states[indices].to(
                self.device_manager.get_device(), non_blocking=True
            )
            self.prefetch_events[chunk_id].record()

    def process_chunk_async(
        self,
        chunk_id: int,
        operation: str,
        callback: Optional[callable] = None,
        all_states: Optional[torch.Tensor] = None,
    ) -> Future:
        if (
            operation == "process"
            and all_states is not None
            and self.device_manager.is_cuda()
        ):
            self._prefetch_chunk_data(chunk_id, all_states)

            for neighbor_id in self._neighbor_map.get(chunk_id, []):
                self._prefetch_chunk_data(neighbor_id, all_states)

        def processing_wrapper(task):
            if self.device_manager.is_cuda():
                self.scheduler.main_stream.wait_event(self.prefetch_events[chunk_id])

            with torch.cuda.stream(self.scheduler.main_stream):
                if callback:
                    return callback(chunk_id, self._chunks[chunk_id])
            return None

        task = ChunkProcessingTask(
            chunk_id=chunk_id,
            operation_type=operation,
            callback=processing_wrapper,
        )
        return self.scheduler.schedule_task(task)

    def rebalance_chunks(self):
        """Перебалансировка chunk'ов на основе текущей статистики"""
        current_memory = self.memory_predictor.get_available_memory_mb()

        # Находим chunk'и с высоким memory pressure
        high_pressure_chunks = [
            c for c in self._chunks if c.memory_pressure_level > 0.8
        ]

        if high_pressure_chunks:
            logger.info(
                f"🔄 Rebalancing {len(high_pressure_chunks)} high-pressure chunks"
            )

            # Понижаем приоритет chunk'ов с высоким давлением
            for chunk in high_pressure_chunks:
                chunk.processing_priority = max(1, chunk.processing_priority - 10)
                chunk.optimal_batch_size = max(100, chunk.optimal_batch_size // 2)

            self.performance_stats["adaptive_rebalancing_events"] += 1

    def get_memory_stats(self) -> Dict[str, float]:
        """Получить статистику использования памяти"""
        total_chunks_memory = sum(c.memory_size_mb for c in self._chunks)
        active_chunks_memory = sum(
            c.gpu_memory_usage_mb for c in self._chunks if c.gpu_memory_usage_mb > 0
        )

        device_stats = self.device_manager.get_memory_stats()

        return {
            "total_chunks_memory_mb": total_chunks_memory,
            "active_chunks_memory_mb": active_chunks_memory,
            "memory_efficiency": active_chunks_memory / max(1, total_chunks_memory),
            "available_memory_mb": self.memory_predictor.get_available_memory_mb(),
            "device_stats": device_stats,
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Получить полную статистику adaptive chunker'а"""
        memory_stats = self.get_memory_stats()

        # Статистика по chunk'ам
        chunk_stats = {
            "total_chunks": len(self._chunks),
            "avg_chunk_size": np.mean([len(c.cell_indices) for c in self._chunks]),
            "avg_memory_usage_mb": np.mean([c.memory_size_mb for c in self._chunks]),
            "avg_access_frequency": np.mean([c.access_frequency for c in self._chunks]),
            "high_pressure_chunks": len(
                [c for c in self._chunks if c.memory_pressure_level > 0.8]
            ),
        }

        return {
            "performance": self.performance_stats,
            "memory": memory_stats,
            "chunks": chunk_stats,
            "scheduler": {
                "active_chunks": len(self.scheduler.active_chunks),
                "queue_size": self.scheduler.task_queue.qsize(),
            },
        }

    def cleanup(self):
        """Очистка ресурсов"""
        self.scheduler.shutdown()
        self.memory_manager.cleanup()

        logger.info(" AdaptiveGPUChunker очищен")
