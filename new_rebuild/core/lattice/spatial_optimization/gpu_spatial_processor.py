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
    from ....config import get_project_config
    from ....utils.logging import get_logger
    from ....utils.device_manager import get_device_manager
    from ..position import Position3D
    from ..gpu_spatial_hashing import (
        AdaptiveGPUSpatialHash,
        GPUSpatialHashGrid,
        GPUSpatialHashingStats,
        GPUMortonEncoder,
    )
    from .adaptive_chunker import AdaptiveGPUChunker, ChunkProcessingTask, AdaptiveChunkInfo

except ImportError:
    # Абсолютные импорты для обратной совместимости или прямого запуска
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from config import get_project_config
    from utils.logging import get_logger
    from utils.device_manager import get_device_manager
    from core.lattice.position import Position3D
    from core.lattice.gpu_spatial_hashing import (
        AdaptiveGPUSpatialHash,
        GPUSpatialHashGrid,
        GPUSpatialHashingStats,
        GPUMortonEncoder,
    )
    from core.lattice.spatial_optimization.adaptive_chunker import AdaptiveGPUChunker, ChunkProcessingTask, AdaptiveChunkInfo
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
        
        # Флаг инициализации spatial hash (инициализируется только один раз)
        self._spatial_hash_initialized = False

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

        if getattr(self.config, "log_memory_usage", False):
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
            neighbor_lists = self.adaptive_hash.query_radius_batch(
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
            self.adaptive_hash.insert_batch(coords_tensor, indices_tensor)

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
        hash_stats = self.adaptive_hash.get_comprehensive_stats()
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
        self.adaptive_hash.hash_grid.optimize_memory()

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
        hash_stats = self.adaptive_hash.get_comprehensive_stats()

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
        hash_stats = self.adaptive_hash.get_comprehensive_stats()
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
        self.adaptive_hash.hash_grid.optimize_memory()

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

    # === PUBLIC API ===
    
    def process_lattice(
        self,
        states: torch.Tensor,
        processor_fn: callable,
        chunker=None
    ) -> torch.Tensor:
        """
        Обрабатывает решетку с использованием spatial optimization
        
        Args:
            states: Состояния клеток решетки
            processor_fn: Функция обработки
            chunker: Chunker для разбивки (опционально)
            
        Returns:
            Обработанные состояния
        """
        # Используем наш внутренний chunker если не передан внешний
        active_chunker = chunker or self.chunker
        
        # Инициализируем spatial hash всеми координатами клеток
        self._populate_spatial_hash(states)
        
        # Получаем расписание обработки chunk'ов
        schedule = active_chunker.get_adaptive_processing_schedule()
        
        # Обрабатываем chunk'и согласно расписанию
        processed_states = states.clone()
        
        # Store updates to apply later (to avoid in-place operations during autograd)
        updates = {}
        
        for batch in schedule:
            batch_futures = []
            
            for chunk_id in batch:
                # Планируем асинхронную обработку chunk'а
                future = active_chunker.process_chunk_async(
                    chunk_id,
                    "process",
                    callback=lambda cid, chunk_info: self._process_chunk_with_function(
                        chunk_info, processed_states, processor_fn, updates
                    ),
                    all_states=processed_states
                )
                batch_futures.append(future)
            
            # Ждем завершения обработки всех chunk'ов в batch'е
            for future in batch_futures:
                try:
                    future.result(timeout=30.0)  # 30 секунд таймаут
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки chunk'а: {e}")
        
        # Apply all updates at once to create new tensor
        if updates:
            if processed_states.dim() == 3:
                new_states = processed_states.clone()
                for indices, chunk_states in updates.items():
                    new_states[:, list(indices), :] = chunk_states
                processed_states = new_states
            else:
                new_states = processed_states.clone()
                for indices, chunk_states in updates.items():
                    new_states[list(indices)] = chunk_states
                processed_states = new_states
        
        return processed_states
    
    def _process_chunk_with_function(
        self,
        chunk_info,
        all_states: torch.Tensor,
        processor_fn: callable,
        updates: dict = None
    ) -> str:
        """Обрабатывает один chunk с заданной функцией"""
        try:
            # DEBUG: Логируем размерности
            logger.debug(f"🔧 CHUNK PROCESSING: all_states shape {all_states.shape}")
            logger.debug(f"🔧 CHUNK INDICES count: {len(chunk_info.cell_indices)}")
            logger.debug(f"🔧 CHUNK INDICES range: {min(chunk_info.cell_indices)} - {max(chunk_info.cell_indices)}")
            
            # Получаем индексы клеток chunk'а
            indices = torch.tensor(
                chunk_info.cell_indices,
                device=self.device,
                dtype=torch.long
            )
            
            # ИСПРАВЛЯЕМ индексирование: all_states имеет shape [batch, cells, features]
            # Индексы chunk_info.cell_indices относятся к cells dimension (второй размерности)
            logger.debug(f"🔍 INDEXING DEBUG: all_states.shape={all_states.shape}, indices.shape={indices.shape}")
            logger.debug(f"🔍 INDICES SAMPLE: {indices[:5].tolist()} ... {indices[-5:].tolist()}")
            
            if all_states.dim() == 3:  # [batch, cells, features]
                batch_size, num_cells, features = all_states.shape
                max_cell_index = num_cells - 1
                
                logger.debug(f"🔍 BATCH INDEXING: batch_size={batch_size}, num_cells={num_cells}, max_index={max_cell_index}")
                
                if torch.any(indices > max_cell_index):
                    invalid_indices = indices[indices > max_cell_index]
                    logger.error(f"❌ INVALID CELL INDICES: {invalid_indices.tolist()} > {max_cell_index}")
                    logger.error(f"❌ All states shape: {all_states.shape}")
                    raise RuntimeError(f"Cell index out of bounds: max valid cell index is {max_cell_index}")
                
                logger.debug(f"🔍 BEFORE INDEXING: about to do all_states[:, indices, :]")
                # Извлекаем состояния для chunk'а: [:, indices, :] - все батчи, выбранные клетки, все фичи
                chunk_states = all_states[:, indices, :]  # [batch, chunk_cells, features]
                logger.debug(f"🔍 AFTER INDEXING: chunk_states.shape={chunk_states.shape}")
                
            else:  # Fallback для других форматов
                max_index = all_states.shape[0] - 1
                if torch.any(indices > max_index):
                    invalid_indices = indices[indices > max_index]
                    logger.error(f"❌ INVALID INDICES: {invalid_indices.tolist()} > {max_index}")
                    raise RuntimeError(f"Index out of bounds: max valid index is {max_index}")
                
                chunk_states = all_states[indices]
            
            # Обрабатываем каждую клетку в chunk'е
            # Create list to collect processed states instead of in-place modification
            processed_states_list = []
            
            for i, cell_idx in enumerate(indices):
                if all_states.dim() == 3:  # [batch, cells, features]
                    cell_state = chunk_states[:, i:i+1, :]  # [batch, 1, features] - состояние одной клетки для всех батчей
                else:
                    cell_state = chunk_states[i:i+1]  # Fallback
                
                # Получаем координаты клетки
                cell_coords = self.chunker.pos_helper.to_3d_coordinates(cell_idx.item())
                
                # Используем spatial hash для поиска соседей
                try:
                    # Конвертируем координаты в tensor для spatial hash
                    coords_tensor = torch.tensor(
                        [cell_coords], device=self.device, dtype=torch.float32
                    )
                    
                    # Ищем соседей в радиусе (адаптивный радиус)
                    config = get_project_config()
                    search_radius = config.calculate_adaptive_radius()
                    
                    neighbor_lists = self.adaptive_hash.query_radius_batch(
                        coords_tensor, search_radius
                    )
                    
                    neighbor_indices = neighbor_lists[0] if neighbor_lists else torch.empty(
                        0, device=self.device, dtype=torch.long
                    )
                    
                except Exception as e:
                    logger.debug(f"⚠️ Не удалось найти соседей для клетки {cell_idx}: {e}")
                    neighbor_indices = torch.empty(0, device=self.device, dtype=torch.long)
                
                # Собираем состояния соседей
                if len(neighbor_indices) > 0:
                    # Get neighbor states from all_states
                    if all_states.dim() == 3:  # [batch, cells, features]
                        # Extract neighbor states for all batches
                        neighbor_states = all_states[:, neighbor_indices, :]  # [batch, num_neighbors, features]
                        # For now, average across batch dimension for neighbors
                        # This is a simplification - in production you might want batch-aware neighbor processing
                        neighbor_states = neighbor_states.mean(dim=0)  # [num_neighbors, features]
                    else:
                        neighbor_states = all_states[neighbor_indices]  # [num_neighbors, features]
                else:
                    neighbor_states = torch.empty(0, all_states.shape[-1], device=self.device)
                
                # Применяем функцию обработки к одной клетке
                # ИСПРАВЛЕНО: Преобразуем cell_idx в int, так как MoE processor ожидает int
                # ИСПРАВЛЕНО: Преобразуем neighbor_indices в список int'ов
                processed_state = processor_fn(
                    cell_state,
                    neighbor_states,
                    cell_idx.item() if isinstance(cell_idx, torch.Tensor) else cell_idx,
                    neighbor_indices.tolist() if isinstance(neighbor_indices, torch.Tensor) else neighbor_indices
                )
                
                if all_states.dim() == 3:  # [batch, cells, features]
                    processed_states_list.append(processed_state.squeeze(1))  # Убираем размерность клетки
                else:
                    processed_states_list.append(processed_state.squeeze(0))
            
            # Stack all processed states into a single tensor
            if all_states.dim() == 3:
                processed_chunk_states = torch.stack(processed_states_list, dim=1)  # [batch, chunk_cells, features]
            else:
                processed_chunk_states = torch.stack(processed_states_list, dim=0)  # [chunk_cells, features]
            
            # Store updates instead of applying them directly
            if updates is not None:
                # Convert indices to tuple for use as dictionary key
                indices_tuple = tuple(indices.cpu().numpy())
                updates[indices_tuple] = processed_chunk_states
            else:
                # Fallback to direct update if no updates dict provided
                if all_states.dim() == 3:  # [batch, cells, features]
                    all_states[:, indices, :] = processed_chunk_states
                else:
                    all_states[indices] = processed_chunk_states
            
            return f"Chunk {chunk_info.chunk_id} processed successfully"
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки chunk {chunk_info.chunk_id}: {e}")
            return f"Chunk {chunk_info.chunk_id} processing failed: {e}"
    
    def _populate_spatial_hash(self, states: torch.Tensor):
        """Заполняет spatial hash координатами всех клеток решетки"""
        total_cells = states.shape[0]
        
        # Генерируем все координаты и индексы
        all_coordinates = []
        all_indices = []
        
        for cell_idx in range(total_cells):
            coords = self.chunker.pos_helper.to_3d_coordinates(cell_idx)
            all_coordinates.append(coords)
            all_indices.append(cell_idx)
        
        if all_coordinates:
            # Конвертируем в tensors
            coords_tensor = torch.tensor(
                all_coordinates, device=self.device, dtype=torch.float32
            )
            indices_tensor = torch.tensor(
                all_indices, device=self.device, dtype=torch.long
            )
            
            # Добавляем все координаты в spatial hash
            self.adaptive_hash.insert_batch(coords_tensor, indices_tensor)
            
            logger.debug(f"📍 Spatial hash заполнен: {len(all_coordinates)} клеток")
    
    def find_neighbors(
        self, coords: Union[Tuple[int, int, int], List[int], torch.Tensor], radius: float
    ) -> List[int]:
        """
        Простой API для поиска соседей в радиусе
        
        Args:
            coords: координаты точки (x, y, z)
            radius: радиус поиска
            
        Returns:
            Список индексов соседних клеток
        """
        # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ для диагностики
        logger.info(f"🔍 [GPUSpatialProcessor.find_neighbors] Вызван для coords={coords}, radius={radius}")
        
        # Ленивая инициализация spatial hash, если он пустой
        self._ensure_spatial_hash_initialized()
        
        # Преобразуем координаты в тензор
        if isinstance(coords, (tuple, list)):
            coords_tensor = torch.tensor([coords], device=self.device, dtype=torch.float32)
        elif isinstance(coords, torch.Tensor):
            if coords.dim() == 1:
                coords_tensor = coords.unsqueeze(0).to(self.device)
            else:
                coords_tensor = coords.to(self.device)
        else:
            raise ValueError(f"Неподдерживаемый тип координат: {type(coords)}")
        
        # Используем существующий adaptive_hash для поиска
        neighbor_lists = self.adaptive_hash.query_radius_batch(coords_tensor, radius)
        
        # Возвращаем первый результат как список Python integers
        if neighbor_lists and len(neighbor_lists) > 0:
            neighbors_tensor = neighbor_lists[0]
            neighbors_list = neighbors_tensor.cpu().tolist()
            
            # Убираем саму точку из результатов (нужно вычислить индекс центральной точки)
            if isinstance(coords, (tuple, list)):
                # Преобразуем координаты в линейный индекс
                center_x, center_y, center_z = coords
                if hasattr(self.chunker, 'pos_helper'):
                    center_idx = self.chunker.pos_helper.to_linear_index((center_x, center_y, center_z))
                    if center_idx in neighbors_list:
                        neighbors_list.remove(center_idx)
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ результата
            logger.info(f"   [GPUSpatialProcessor.find_neighbors] Найдено {len(neighbors_list)} соседей")
            if len(neighbors_list) > 10:
                logger.info(f"   Первые 10 соседей: {neighbors_list[:10]}...")
            else:
                logger.info(f"   Все соседи: {neighbors_list}")
            
            return neighbors_list
        else:
            logger.info(f"   [GPUSpatialProcessor.find_neighbors] Соседей не найдено!")
            return []
    
    def _ensure_spatial_hash_initialized(self):
        """
        Ленивая инициализация spatial hash
        Заполняет hash координатами всех клеток решетки если он еще не инициализирован
        """
        # Инициализируем spatial hash только один раз
        if not self._spatial_hash_initialized:
            logger.debug("🔧 Инициализируем spatial hash автоматически...")
            
            # Вычисляем общее количество клеток в решетке
            total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
            
            # Создаем dummy states для инициализации
            dummy_states = torch.zeros(total_cells, 1, device=self.device)
            
            # Заполняем spatial hash
            self._populate_spatial_hash(dummy_states)
            
            # Устанавливаем флаг что инициализация завершена
            self._spatial_hash_initialized = True
            
            logger.info(f"✅ Spatial hash автоматически инициализирован для {total_cells} клеток")
