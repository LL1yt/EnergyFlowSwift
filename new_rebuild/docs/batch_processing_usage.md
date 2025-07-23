# Batch Processing Usage Guide

## Быстрый старт

### 1. Простое включение batch обработки для существующей решетки

```python
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.lattice.spatial_optimization.batch_integration import upgrade_lattice_to_batch

# Создаем обычную решетку
lattice = Lattice3D(dimensions=(15, 15, 15))

# Обновляем для batch обработки
lattice = upgrade_lattice_to_batch(lattice)

# Включаем batch режим
lattice.set_batch_enabled(True)

# Используем как обычно - теперь с batch оптимизацией!
result = lattice.forward()

# Получаем статистику производительности
perf_report = lattice.get_batch_performance()
print(f"Performance: {perf_report}")
```

### 2. Создание новой решетки с batch оптимизацией

```python
from new_rebuild.core.lattice.spatial_optimization.batch_integration import create_batch_optimized_spatial_optimizer
from new_rebuild.core.lattice import Lattice3D

# Создаем решетку с batch-оптимизированным spatial optimizer
lattice = Lattice3D(
    dimensions=(15, 15, 15),
    spatial_optimizer_factory=lambda dims, moe: create_batch_optimized_spatial_optimizer(
        dimensions=dims,
        moe_processor=moe,
        enable_batch=True,
        batch_threshold=4  # Минимум 4 клетки для batch обработки
    )
)
```

### 3. Динамическое переключение режимов

```python
# Переключение между batch и per-cell режимами
lattice.set_batch_enabled(True)   # Batch режим
lattice.set_batch_enabled(False)  # Per-cell режим

# A/B тестирование
import time

# Тест per-cell
lattice.set_batch_enabled(False)
start = time.time()
lattice.forward()
percell_time = time.time() - start

# Тест batch
lattice.set_batch_enabled(True)
start = time.time()
lattice.forward()
batch_time = time.time() - start

print(f"Speedup: {percell_time/batch_time:.2f}x")
```

## Архитектура

### Модульная структура

```
new_rebuild/core/moe/batch/
├── __init__.py
├── batch_moe_processor.py      # Основной batch процессор
├── batch_neighbor_extractor.py # Извлечение соседей для batch'а
├── batch_expert_processor.py   # Параллельная обработка экспертов
└── batch_adapter.py           # Адаптер с fallback механизмом
```

### Ключевые компоненты

1. **BatchMoEProcessor** - заменяет per-cell обработку на batch
2. **BatchNeighborExtractor** - эффективно извлекает соседей для множества клеток
3. **BatchExpertProcessor** - параллельно обрабатывает экспертов через CUDA streams
4. **BatchProcessingAdapter** - автоматически выбирает оптимальный режим

## Производительность

### Ожидаемые улучшения

- **8×8×8 решетка**: 2-3x ускорение
- **15×15×15 решетка**: 5-10x ускорение
- **30×30×30 решетка**: 10-20x ускорение

### Оптимальные параметры

```python
# Рекомендуемые настройки для разных размеров решеток

# Маленькие решетки (≤10×10×10)
batch_threshold = 8  # Обрабатывать batch'ами только chunk'и ≥8 клеток

# Средние решетки (15×15×15)
batch_threshold = 4  # Более агрессивный батчинг

# Большие решетки (≥30×30×30)
batch_threshold = 2  # Максимальный батчинг
```

## Отладка и профилирование

### Детальная статистика

```python
# Получить полный отчет о производительности
report = lattice.get_batch_performance()

# Содержит:
# - Количество обработанных chunk'ов
# - Среднее время на клетку
# - Сравнение batch vs per-cell
# - Статистику по экспертам
```

### Логирование

```python
# Включить подробное логирование
import logging
logger = logging.getLogger("new_rebuild.core.moe.batch")
logger.setLevel(logging.DEBUG)
```

## Тестирование

### Проверка корректности

```python
# Запуск тестов
python -m pytest new_rebuild/tests/test_batch_processing.py -v

# Или отдельный тест корректности
python -c "from new_rebuild.tests.test_batch_processing import TestBatchProcessing; t = TestBatchProcessing(); t.setup(); t.test_batch_vs_percell_correctness()"
```

### Benchmark производительности

```python
# Полный benchmark
python new_rebuild/tests/test_batch_processing.py
```

## Troubleshooting

### Batch обработка не дает ускорения

1. Проверьте размер chunk'ов - слишком маленькие chunk'и не эффективны
2. Убедитесь что используется GPU: `torch.cuda.is_available()`
3. Проверьте batch_threshold - возможно он слишком высокий

### Результаты отличаются от per-cell

1. Проверьте что эксперты поддерживают batch размерности
2. Убедитесь что нет in-place операций в экспертах
3. Используйте тест корректности для отладки

### Out of memory при batch обработке

1. Уменьшите размер chunk'ов в конфигурации
2. Используйте gradient checkpointing (уже включен для некоторых экспертов)
3. Уменьшите batch_threshold для обработки меньшими порциями