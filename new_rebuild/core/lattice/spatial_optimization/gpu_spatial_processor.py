#!/usr/bin/env python3
"""
GPU Spatial Processor - Интегрированный процессор пространственных операций
==========================================================================

Объединяет GPU-accelerated spatial hashing и adaptive chunking в единую
высокопроизводительную систему для обработки пространственных запросов
в больших 3D решетках.

Ключевые особенности:
- Unified API для spatial hashing и chunking
- Automatic memory management и optimization
- Real-time performance monitoring
- Intelligent prefetching и caching
- Seamless GPU/CPU fallback

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union, Any, Callable
from dataclasses import dataclass
import time
import threading
from concurrent.futures import Future, as_completed
import asyncio


try:
    # Относительные импорты для использования в качестве модуля
    from ....config.project_config import get_project_config
    from ....utils.logging import get_logger
    from ....utils.device_manager import get_device_manager
    from ..position import Position3D
    from ..gpu_spatial_hashing import (
        AdaptiveGPUSpatialHash,
        GPUSpatialHashGrid,
        GPUSpatialHashingStats,
        GPUMortonEncoder,
    )
    from .adaptive_chunker import AdaptiveGPUChunker, ChunkProcessingTask

except ImportError:
    # Абсолютные импорты для обратной совместимости или прямого запуска
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from config.project_config import get_project_config
    from utils.logging import get_logger
    from utils.device_manager import get_device_manager
    from core.lattice.position import Position3D
    from core.lattice.gpu_spatial_hashing import (
        AdaptiveGPUSpatialHash,
        GPUSpatialHashGrid,
        GPUSpatialHashingStats,
        GPUMortonEncoder,
    )
    from new_rebuild.core.lattice.spatial_optimization.adaptive_chunker import (
        AdaptiveGPUChunker,
        ChunkProcessingTask,
    )

logger = get_logger(__name__)

Coordinates3D = Tuple[int, int, int]


@dataclass
class SpatialQuery:
    """Запрос пространственного поиска"""

    query_id: str
    coordinates: torch.Tensor  # (N, 3) tensor с координатами
    radius: float
    chunk_ids: Optional[Set[int]] = None
    priority: int = 0
    callback: Optional[callable] = None


@dataclass
class SpatialQueryResult:
    """Результат пространственного запроса"""

    query_id: str
    neighbor_lists: List[torch.Tensor]  # Список neighbors для каждой точки
    processing_time_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    chunks_accessed: Set[int]


class GPUSpatialProcessor:
    """
    Интегрированный GPU Spatial Processor

    Объединяет spatial hashing и adaptive chunking для эффективной
    обработки пространственных запросов в больших 3D решетках.
    """

    def __init__(self, dimensions: Coordinates3D, config: dict = None):
        self.dimensions = dimensions
        self.config = config or get_project_config().get_spatial_optim_config()

        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Инициализируем компоненты
        self._initialize_components()

        # Query management
        from queue import Queue

        self.query_queue = Queue()
        self.active_queries: Dict[str, SpatialQuery] = {}
        self.query_results: Dict[str, SpatialQueryResult] = {}

        # Performance monitoring
        self.performance_metrics = {
            "total_queries": 0,
            "avg_query_time_ms": 0.0,
            "memory_efficiency": 0.0,
            "gpu_utilization": 0.0,
            "cache_hit_rate": 0.0,
            "chunk_rebalancing_events": 0,
        }

        # Background task для обработки запросов
        self._start_background_processing()

        logger.info(
            f"🚀 GPUSpatialProcessor инициализирован: {dimensions} на {self.device}"
        )

    def _initialize_components(self):
        """Инициализация внутренних компонентов: Morton Encoder, Adaptive Hash и Chunker."""
        project_cfg = get_project_config()

        # GPU-специфичные компоненты
        self.morton_encoder = GPUMortonEncoder(self.dimensions)
        self.adaptive_hash = AdaptiveGPUSpatialHash(
            self.dimensions,
            project_cfg.spatial.memory_pool_size_gb * 1024 * 0.6,
        )

        # Chunker инициализируется со своей собственной конфигурацией
        self.chunker = AdaptiveGPUChunker(self.dimensions)

        if self.config.get("log_memory_usage", False):
            logger.info("Компоненты GPUSpatialProcessor инициализированы.")

        # Интеграционный layer для координации между компонентами
        self._setup_integration_layer()

    def _setup_integration_layer(self):
        """Настраивает интеграцию между chunker и spatial hash"""
        # Маппинг chunk'ов к spatial hash regions
        self.chunk_to_hash_mapping: Dict[int, Set[int]] = {}
        self.hash_to_chunk_mapping: Dict[int, Set[int]] = {}

        # Кэш для ускорения межкомпонентных операций
        self.integration_cache = {}
        self.cache_max_size = 5000

        # Thread-safe locks для concurrent access
        self.mapping_lock = threading.RLock()

    def _start_background_processing(self):
        """Запускает фоновую обработку запросов"""
        self.processing_active = True

        # Запускаем async event loop в отдельном потоке
        self.processing_thread = threading.Thread(target=self._run_async_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _run_async_processing(self):
        """Запускает async обработку в отдельном потоке"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._async_processing_loop())
        except Exception as e:
            logger.error(f"❌ Ошибка в async processing loop: {e}")
        finally:
            loop.close()

    async def _async_processing_loop(self):
        """Основной цикл асинхронной обработки запросов"""
        from queue import Empty

        while self.processing_active:
            try:
                # Ждем новый запрос (с таймаутом)
                try:
                    query = self.query_queue.get(timeout=1.0)
                    await self._process_spatial_query(query)
                except Empty:
                    # Периодические задачи обслуживания
                    await self._perform_maintenance_tasks()

            except Exception as e:
                logger.error(f"❌ Ошибка обработки spatial query: {e}")

    async def _process_spatial_query(self, query: SpatialQuery):
        """Обрабатывает пространственный запрос"""
        start_time = time.time()

        try:
            # Определяем затронутые chunk'и
            affected_chunks = self._identify_affected_chunks(query)

            # Загружаем необходимые chunk'и в память
            await self._ensure_chunks_loaded(affected_chunks)

            # Выполняем spatial hash поиск
            neighbor_lists = self.spatial_hash.query_radius_batch(
                query.coordinates, query.radius
            )

            # Фильтруем результаты по chunk boundaries если необходимо
            if query.chunk_ids:
                neighbor_lists = self._filter_by_chunks(neighbor_lists, query.chunk_ids)

            processing_time_ms = (time.time() - start_time) * 1000

            # Создаем результат
            result = SpatialQueryResult(
                query_id=query.query_id,
                neighbor_lists=neighbor_lists,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=self._estimate_query_memory_usage(query),
                cache_hit_rate=self._calculate_cache_hit_rate(query),
                chunks_accessed=affected_chunks,
            )

            # Сохраняем результат и уведомляем callback
            self.query_results[query.query_id] = result

            if query.callback:
                query.callback(result)

            # Обновляем метрики
            self._update_performance_metrics(result)

            logger.debug(
                f"✅ Query {query.query_id} обработан за {processing_time_ms:.1f}ms, "
                f"chunks: {len(affected_chunks)}"
            )

        except Exception as e:
            logger.error(f"❌ Ошибка обработки query {query.query_id}: {e}")
        finally:
            # Очищаем активный запрос
            self.active_queries.pop(query.query_id, None)

    def _identify_affected_chunks(self, query: SpatialQuery) -> Set[int]:
        """Определяет chunk'и, затронутые запросом"""
        affected_chunks = set()

        # Анализируем каждую координату запроса
        for coord_idx in range(query.coordinates.shape[0]):
            coord = tuple(query.coordinates[coord_idx].cpu().numpy().astype(int))

            try:
                # Находим chunk для координаты
                chunk_info = self.chunker.get_chunk_by_coords(coord)
                affected_chunks.add(chunk_info.chunk_id)

                # Добавляем соседние chunk'и если radius большой
                if query.radius > chunk_info.memory_size_mb * 0.1:  # эвристика
                    affected_chunks.update(chunk_info.neighbor_chunks)

            except ValueError:
                logger.warning(f"⚠️ Координата {coord} вне boundaries решетки")

        return affected_chunks

    async def _ensure_chunks_loaded(self, chunk_ids: Set[int]):
        """Гарантирует загрузку необходимых chunk'ов в память"""
        # Проверяем какие chunk'и уже загружены
        chunks_to_load = []

        for chunk_id in chunk_ids:
            chunk_info = self.chunker.adaptive_chunks[chunk_id]
            if chunk_info.gpu_memory_usage_mb == 0:  # не загружен
                chunks_to_load.append(chunk_id)

        if chunks_to_load:
            # Загружаем chunk'и асинхронно
            load_tasks = []
            for chunk_id in chunks_to_load:
                future = self.chunker.process_chunk_async(
                    chunk_id, "load", self._chunk_load_callback
                )
                load_tasks.append(future)

            # Ждем завершения загрузки
            if load_tasks:
                # Конвертируем в awaitable
                await asyncio.get_event_loop().run_in_executor(
                    None, self._wait_for_chunk_loading, load_tasks
                )

    def _wait_for_chunk_loading(self, futures: List[Future]):
        """Ждет завершения загрузки chunk'ов"""
        for future in as_completed(futures):
            try:
                result = future.result(timeout=10.0)  # 10 секунд таймаут
                logger.debug(f"✅ Chunk загружен: {result}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки chunk: {e}")

    def _chunk_load_callback(self, task: ChunkProcessingTask):
        """Callback для загрузки chunk'а"""
        try:
            chunk_info = self.chunker.adaptive_chunks[task.chunk_id]
        except (IndexError, KeyError):
            logger.error(f"❌ Chunk {task.chunk_id} не найден")
            return f"Chunk {task.chunk_id} not found"

        # Создаем координаты и индексы для spatial hash
        coordinates = []
        indices = []

        for cell_idx in chunk_info.cell_indices:
            coord = self.chunker.pos_helper.to_3d_coordinates(cell_idx)
            coordinates.append(coord)
            indices.append(cell_idx)

        if coordinates:
            # Конвертируем в tensors
            coords_tensor = torch.tensor(
                coordinates, device=self.device, dtype=torch.float32
            )
            indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)

            # Добавляем в spatial hash
            self.spatial_hash.insert_batch(coords_tensor, indices_tensor)

            # Обновляем статистику chunk'а
            chunk_info.gpu_memory_usage_mb = self._estimate_chunk_gpu_memory(chunk_info)
            chunk_info.last_access_time = time.time()

            logger.debug(
                f"📦 Chunk {task.chunk_id} загружен: {len(coordinates)} клеток"
            )

        return f"Chunk {task.chunk_id} loaded successfully"

    def _filter_by_chunks(
        self, neighbor_lists: List[torch.Tensor], allowed_chunk_ids: Set[int]
    ) -> List[torch.Tensor]:
        """Фильтрует результаты по разрешенным chunk'ам"""
        filtered_lists = []

        for neighbors in neighbor_lists:
            if len(neighbors) == 0:
                filtered_lists.append(neighbors)
                continue

            # Определяем chunk'и для каждого neighbor'а
            valid_neighbors = []

            for neighbor_idx in neighbors:
                neighbor_coord = self.chunker.pos_helper.to_3d_coordinates(
                    neighbor_idx.item()
                )
                try:
                    neighbor_chunk = self.chunker.get_chunk_by_coords(neighbor_coord)
                    if neighbor_chunk.chunk_id in allowed_chunk_ids:
                        valid_neighbors.append(neighbor_idx)
                except ValueError:
                    pass  # neighbor вне boundaries

            if valid_neighbors:
                filtered_neighbors = torch.stack(valid_neighbors)
            else:
                filtered_neighbors = torch.empty(
                    0, device=self.device, dtype=torch.long
                )

            filtered_lists.append(filtered_neighbors)

        return filtered_lists

    def _estimate_query_memory_usage(self, query: SpatialQuery) -> float:
        """Оценивает использование памяти запросом"""
        # Базовое использование памяти
        query_memory = query.coordinates.numel() * 4 / (1024**2)  # float32 в MB

        # Добавляем оценку для результатов
        avg_neighbors_per_point = 50  # эвристическая оценка
        result_memory = (
            query.coordinates.shape[0] * avg_neighbors_per_point * 4 / (1024**2)
        )

        return query_memory + result_memory

    def _calculate_cache_hit_rate(self, query: SpatialQuery) -> float:
        """Вычисляет cache hit rate для запроса"""
        # Получаем статистику из spatial hash
        hash_stats = self.spatial_hash.get_comprehensive_stats()
        return hash_stats.get("spatial_hash", {}).get("cache_hit_rate", 0.0)

    def _estimate_chunk_gpu_memory(self, chunk_info: AdaptiveChunkInfo) -> float:
        """Оценивает GPU память, используемую chunk'ом"""
        num_cells = len(chunk_info.cell_indices)

        # Координаты: 3 * 4 bytes per cell
        coordinates_memory = num_cells * 3 * 4

        # Индексы: 4 bytes per cell
        indices_memory = num_cells * 4

        # Накладные расходы: примерно 20%
        overhead = (coordinates_memory + indices_memory) * 0.2

        total_memory_bytes = coordinates_memory + indices_memory + overhead
        return total_memory_bytes / (1024**2)  # конвертируем в MB

    async def _perform_maintenance_tasks(self):
        """Выполняет периодические задачи обслуживания"""
        # Оптимизация памяти
        self.spatial_hash.hash_grid.optimize_memory()

        # Перебалансировка chunk'ов
        self.chunker.rebalance_chunks()

        # Очистка старых результатов запросов
        self._cleanup_old_query_results()

        # Обновление метрик производительности
        self._update_integration_metrics()

    def _cleanup_old_query_results(self):
        """Очищает старые результаты запросов"""
        if len(self.query_results) > 1000:  # максимум 1000 результатов
            # Оставляем только последние 500
            sorted_results = sorted(
                self.query_results.items(), key=lambda x: x[1].processing_time_ms
            )

            # Удаляем старые результаты
            for query_id, _ in sorted_results[:-500]:
                del self.query_results[query_id]

    def _update_performance_metrics(self, result: SpatialQueryResult):
        """Обновляет метрики производительности"""
        self.performance_metrics["total_queries"] += 1

        # Обновляем среднее время обработки
        old_avg = self.performance_metrics["avg_query_time_ms"]
        total_queries = self.performance_metrics["total_queries"]

        new_avg = (
            old_avg * (total_queries - 1) + result.processing_time_ms
        ) / total_queries
        self.performance_metrics["avg_query_time_ms"] = new_avg

        # Обновляем cache hit rate
        self.performance_metrics["cache_hit_rate"] = result.cache_hit_rate

    def _update_integration_metrics(self):
        """Обновляет метрики интеграции компонентов"""
        # Получаем статистику от компонентов
        chunker_stats = self.chunker.get_comprehensive_stats()
        hash_stats = self.spatial_hash.get_comprehensive_stats()

        # Вычисляем общую эффективность памяти
        chunker_efficiency = chunker_stats.get("memory", {}).get(
            "memory_efficiency", 0.0
        )
        hash_memory = hash_stats.get("memory", {}).get("total_gpu_mb", 0.0)
        total_memory = chunker_stats.get("memory", {}).get(
            "total_chunks_memory_mb", 0.0
        )

        if total_memory > 0:
            self.performance_metrics["memory_efficiency"] = (
                chunker_efficiency * 0.7 + (hash_memory / total_memory) * 0.3
            )

        # GPU utilization (приблизительная оценка)
        device_stats = self.device_manager.get_memory_stats()
        if self.device_manager.is_cuda() and "allocated_mb" in device_stats:
            allocated_mb = device_stats["allocated_mb"]
            reserved_mb = device_stats.get("reserved_mb", allocated_mb)

            if reserved_mb > 0:
                self.performance_metrics["gpu_utilization"] = allocated_mb / reserved_mb

    # === PUBLIC API ===

    def query_neighbors_async(
        self,
        coordinates: Union[torch.Tensor, np.ndarray, List],
        radius: float,
        chunk_ids: Optional[Set[int]] = None,
        priority: int = 0,
        callback: Optional[callable] = None,
    ) -> str:
        """
        Асинхронный поиск соседей

        Args:
            coordinates: Координаты для поиска (N, 3)
            radius: Радиус поиска
            chunk_ids: Ограничить поиск определенными chunk'ами
            priority: Приоритет запроса (0-100)
            callback: Callback функция для результата

        Returns:
            query_id для отслеживания запроса
        """
        # Конвертируем координаты в tensor
        if isinstance(coordinates, (np.ndarray, list)):
            coords_tensor = torch.tensor(
                coordinates, device=self.device, dtype=torch.float32
            )
        else:
            coords_tensor = self.device_manager.ensure_device(coordinates)

        # Генерируем уникальный query_id
        query_id = f"query_{int(time.time() * 1000000)}_{len(self.active_queries)}"

        # Создаем запрос
        query = SpatialQuery(
            query_id=query_id,
            coordinates=coords_tensor,
            radius=radius,
            chunk_ids=chunk_ids,
            priority=priority,
            callback=callback,
        )

        # Добавляем в активные запросы и очередь
        self.active_queries[query_id] = query

        # Отправляем в queue (thread-safe)
        self.query_queue.put(query)

        return query_id

    def query_neighbors_sync(
        self,
        coordinates: Union[torch.Tensor, np.ndarray, List],
        radius: float,
        chunk_ids: Optional[Set[int]] = None,
        timeout: float = 30.0,
    ) -> SpatialQueryResult:
        """
        Синхронный поиск соседей

        Args:
            coordinates: Координаты для поиска
            radius: Радиус поиска
            chunk_ids: Ограничить поиск определенными chunk'ами
            timeout: Таймаут в секундах

        Returns:
            Результат поиска
        """
        result_ready = threading.Event()
        result_container = {}

        def sync_callback(result: SpatialQueryResult):
            result_container["result"] = result
            result_ready.set()

        # Запускаем асинхронный запрос
        query_id = self.query_neighbors_async(
            coordinates,
            radius,
            chunk_ids,
            priority=100,  # высокий приоритет для sync запросов
            callback=sync_callback,
        )

        # Ждем результат
        if result_ready.wait(timeout):
            return result_container["result"]
        else:
            raise TimeoutError(f"Query {query_id} не завершился за {timeout}s")

    def get_query_result(self, query_id: str) -> Optional[SpatialQueryResult]:
        """Получить результат запроса по ID"""
        return self.query_results.get(query_id)

    def is_query_complete(self, query_id: str) -> bool:
        """Проверить, завершен ли запрос"""
        return query_id in self.query_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить полную статистику производительности"""
        # Собираем статистику от всех компонентов
        chunker_stats = self.chunker.get_comprehensive_stats()
        hash_stats = self.spatial_hash.get_comprehensive_stats()
        device_stats = self.device_manager.get_memory_stats()

        return {
            "processor": self.performance_metrics,
            "chunker": chunker_stats,
            "spatial_hash": hash_stats,
            "device": device_stats,
            "integration": {
                "active_queries": len(self.active_queries),
                "cached_results": len(self.query_results),
                "cache_size": len(self.integration_cache),
            },
        }

    def optimize_performance(self):
        """Принудительная оптимизация производительности"""
        logger.info("🔧 Запущена принудительная оптимизация производительности")

        # Оптимизация spatial hash
        self.spatial_hash.hash_grid.optimize_memory()

        # Перебалансировка chunk'ов
        self.chunker.rebalance_chunks()

        # Очистка кэшей
        self.integration_cache.clear()
        self._cleanup_old_query_results()

        # Принудительная очистка GPU памяти
        self.device_manager.cleanup()

        logger.info("✅ Оптимизация производительности завершена")

    def shutdown(self):
        """Завершение работы processor'а"""
        logger.info("🛑 Завершение работы GPUSpatialProcessor")

        # Останавливаем background processing
        self.processing_active = False

        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        # Завершаем работу компонентов
        self.chunker.cleanup()

        # Финальная очистка памяти
        self.device_manager.cleanup()

        logger.info("✅ GPUSpatialProcessor завершен")
