# GPU Spatial Optimization Guide

## Обзор

Этот документ описывает новые GPU-accelerated компоненты для пространственной оптимизации в 3D Cellular Neural Network проекте.

## Компоненты

### 1. GPU Spatial Hashing (`gpu_spatial_hashing.py`)

#### GPUMortonEncoder

- **Назначение**: GPU-accelerated кодирование Morton (Z-order) для 3D координат
- **Особенности**: Векторизованные операции, batch processing
- **Применение**: Улучшение пространственной локальности данных

```python
from core.lattice.gpu_spatial_hashing import GPUMortonEncoder

encoder = GPUMortonEncoder((100, 100, 100))
coordinates = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
morton_codes = encoder.encode_batch(coordinates)
```

#### GPUSpatialHashGrid

- **Назначение**: Высокопроизводительный spatial hash для поиска соседей
- **Особенности**: GPU memory management, query caching, batch operations
- **Применение**: Быстрый поиск соседей в больших решетках

```python
from core.lattice.gpu_spatial_hashing import GPUSpatialHashGrid

hash_grid = GPUSpatialHashGrid((128, 128, 128), cell_size=8)
hash_grid.insert_batch(coordinates, cell_indices)
neighbors = hash_grid.query_radius_batch(query_points, radius=5.0)
```

#### AdaptiveGPUSpatialHash

- **Назначение**: Самооптимизирующийся spatial hash
- **Особенности**: Автоматическая настройка параметров, memory management
- **Применение**: Адаптация к различным размерам решеток и паттернам данных

```python
from core.lattice.gpu_spatial_hashing import AdaptiveGPUSpatialHash

adaptive_hash = AdaptiveGPUSpatialHash((200, 200, 200), target_memory_mb=1024)
adaptive_hash.insert_batch(coordinates, indices)
results = adaptive_hash.query_radius_batch(queries, radius)
```

### 2. Adaptive Chunking (`adaptive_chunker.py`)

#### AdaptiveGPUChunker

- **Назначение**: Интеллектуальное разбиение решеток на управляемые части
- **Особенности**: Memory-aware chunking, priority scheduling, load balancing
- **Применение**: Эффективная обработка больших решеток с ограниченной памятью

```python
from core.lattice.spatial_optimization.adaptive_chunker import AdaptiveGPUChunker

chunker = AdaptiveGPUChunker((500, 500, 500))
schedule = chunker.get_adaptive_processing_schedule()

# Асинхронная обработка chunk'а
future = chunker.process_chunk_async(chunk_id=0, operation="load")
```

#### Ключевые классы данных:

- **AdaptiveChunkInfo**: Расширенная информация о chunk'е
- **ChunkProcessingTask**: Задача обработки chunk'а
- **AdaptiveMemoryPredictor**: Предсказатель использования памяти

### 3. Integrated Spatial Processor (`gpu_spatial_processor.py`)

#### GPUSpatialProcessor

- **Назначение**: Унифицированный интерфейс для всех spatial операций
- **Особенности**: Async/sync API, intelligent prefetching, performance monitoring
- **Применение**: Главный компонент для spatial запросов

```python
from core.lattice.spatial_optimization.gpu_spatial_processor import GPUSpatialProcessor

processor = GPUSpatialProcessor((300, 300, 300))

# Синхронный поиск
result = processor.query_neighbors_sync(coordinates, radius=10.0)

# Асинхронный поиск
query_id = processor.query_neighbors_async(
    coordinates, radius=10.0,
    callback=my_callback_function
)
```

## Преимущества

### Производительность

- **GPU acceleration**: Все операции оптимизированы для GPU
- **Batch processing**: Эффективная обработка множественных запросов
- **Memory efficiency**: Интеллектуальное управление памятью
- **Cache optimization**: Многоуровневые кэши для часто используемых данных

### Адаптивность

- **Dynamic sizing**: Автоматическая настройка размеров chunk'ов
- **Memory pressure detection**: Реагирует на изменения доступной памяти
- **Load balancing**: Распределение нагрузки между chunk'ами
- **Performance monitoring**: Реальное время отслеживания метрик

### Масштабируемость

- **Large lattices**: Поддержка решеток до 1000³ клеток
- **Memory constraints**: Работа с ограниченной GPU памятью
- **Concurrent processing**: Параллельная обработка множественных chunk'ов
- **Graceful degradation**: Fallback на CPU при необходимости

## Использование

### Базовый пример

```python
from new_rebuild.core.lattice.spatial_optimization.gpu_spatial_processor import GPUSpatialProcessor
import torch

# Создаем processor для решетки 100x100x100
processor = GPUSpatialProcessor((100, 100, 100))

# Подготавливаем запрос
query_coordinates = torch.tensor([
    [25, 25, 25],
    [50, 50, 50],
    [75, 75, 75]
], dtype=torch.float32)

# Выполняем поиск соседей
result = processor.query_neighbors_sync(
    coordinates=query_coordinates,
    radius=8.0,
    timeout=30.0
)

# Анализируем результаты
print(f"Найдено соседей: {[len(neighbors) for neighbors in result.neighbor_lists]}")
print(f"Время обработки: {result.processing_time_ms:.2f}ms")
print(f"Использование памяти: {result.memory_usage_mb:.2f}MB")

# Получаем статистику производительности
stats = processor.get_performance_stats()
print(f"Всего запросов: {stats['processor']['total_queries']}")
print(f"Среднее время: {stats['processor']['avg_query_time_ms']:.2f}ms")

# Завершаем работу
processor.shutdown()
```

### Продвинутый пример с async

```python
import asyncio
from concurrent.futures import as_completed

processor = GPUSpatialProcessor((200, 200, 200))

# Функция обработки результата
def process_result(result):
    print(f"Query {result.query_id} completed: {len(result.neighbor_lists)} results")

# Запускаем несколько асинхронных запросов
query_ids = []
for i in range(10):
    coords = torch.randint(0, 200, (20, 3), dtype=torch.float32)
    query_id = processor.query_neighbors_async(
        coordinates=coords,
        radius=12.0,
        priority=i * 10,
        callback=process_result
    )
    query_ids.append(query_id)

# Ждем завершения всех запросов
completed = 0
while completed < len(query_ids):
    for query_id in query_ids:
        if processor.is_query_complete(query_id):
            completed += 1
    time.sleep(0.1)

processor.shutdown()
```

## Конфигурация

### Основные параметры в `project_config.py`:

```python
# Spatial optimization settings
spatial_chunk_size: int = 64
spatial_chunk_overlap: int = 8
spatial_max_chunks_in_memory: int = 4
spatial_memory_pool_size_gb: float = 12.0
spatial_garbage_collect_frequency: int = 100
spatial_prefetch_chunks: bool = True
```

### Адаптивные параметры:

```python
# GPU spatial hashing
gpu_spatial_target_memory_mb: float = 1024.0
gpu_spatial_cell_size: int = 8
gpu_spatial_cache_size: int = 10000

# Adaptive chunking
adaptive_chunk_priority_boost: int = 10
adaptive_memory_pressure_threshold: float = 0.8
adaptive_rebalancing_frequency: int = 1000
```

## Тестирование

Для тестирования всех компонентов запустите:

```bash
cd new_rebuild
python test_gpu_spatial_optimization.py
```

Тест включает:

- GPU Morton Encoder validation
- Spatial Hash Grid performance
- Adaptive Spatial Hash adaptation
- Adaptive Chunker scheduling
- Integrated Spatial Processor functionality
- Performance benchmarks

## Мониторинг

### Получение статистики

```python
processor = GPUSpatialProcessor(dimensions)

# Полная статистика
stats = processor.get_performance_stats()

# Специфичная статистика компонентов
chunker_stats = processor.chunker.get_comprehensive_stats()
hash_stats = processor.spatial_hash.get_comprehensive_stats()
device_stats = processor.device_manager.get_memory_stats()
```

### Ключевые метрики

- **Query performance**: Время обработки запросов
- **Memory efficiency**: Эффективность использования памяти
- **Cache hit rate**: Процент попаданий в кэш
- **Chunk utilization**: Использование chunk'ов
- **GPU utilization**: Загрузка GPU

## Оптимизация

### Настройка для больших решеток (>1M клеток):

```python
config = {
    "spatial_chunk_size": 128,
    "spatial_max_chunks_in_memory": 2,
    "spatial_memory_pool_size_gb": 8.0,
    "spatial_prefetch_chunks": True
}
```

### Настройка для высокой производительности:

```python
config = {
    "spatial_chunk_size": 32,
    "spatial_max_chunks_in_memory": 8,
    "spatial_memory_pool_size_gb": 16.0,
    "spatial_garbage_collect_frequency": 50
}
```

### Настройка для ограниченной памяти:

```python
config = {
    "spatial_chunk_size": 64,
    "spatial_max_chunks_in_memory": 2,
    "spatial_memory_pool_size_gb": 4.0,
    "spatial_prefetch_chunks": False
}
```

## Интеграция

### С существующими компонентами:

```python
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.lattice.spatial_optimization import GPUSpatialProcessor

# Создаем решетку
lattice = Lattice3D((100, 100, 100))

# Создаем spatial processor
spatial_processor = GPUSpatialProcessor(lattice.dimensions)

# Интегрируем в обработку
def process_lattice_neighbors(lattice, spatial_processor):
    for cell_id in range(lattice.total_cells):
        coords = lattice.get_coordinates(cell_id)
        neighbors = spatial_processor.query_neighbors_sync(
            torch.tensor([coords]), radius=5.0
        )
        # Обработка соседей...
```

## Troubleshooting

### Общие проблемы:

1. **CUDA out of memory**:

   - Уменьшите `spatial_max_chunks_in_memory`
   - Увеличьте `spatial_chunk_size`
   - Снизьте `spatial_memory_pool_size_gb`

2. **Медленная производительность**:

   - Увеличьте `spatial_cache_size`
   - Включите `spatial_prefetch_chunks`
   - Оптимизируйте размер batch'ей

3. **Высокое использование CPU**:
   - Проверьте доступность CUDA
   - Убедитесь в корректной настройке device_manager
   - Увеличьте `spatial_garbage_collect_frequency`

### Логирование

Включите подробное логирование:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Или в конфигурации
config.debug_mode = True
config.log_level = "DEBUG"
```

## Заключение

GPU Spatial Optimization компоненты обеспечивают высокопроизводительную обработку пространственных запросов в больших 3D решетках. Интеллектуальное управление памятью и адаптивные алгоритмы позволяют эффективно работать с различными размерами решеток и ограничениями ресурсов.

Для получения максимальной производительности рекомендуется:

1. Настроить параметры под конкретное приложение
2. Мониторить метрики производительности
3. Использовать асинхронный API для высокой пропускной способности
4. Регулярно оптимизировать память через `optimize_performance()`
