# Гайд по миграции на Unified Spatial Optimizer

## Обзор изменений

Мы унифицировали `SpatialOptimizer` и `MoESpatialOptimizer` в единую высокопроизводительную систему `UnifiedSpatialOptimizer` с полной GPU поддержкой и интеграцией MoE архитектуры.

## Основные преимущества

### ✅ Что улучшилось:
- **Единый API** для всех типов пространственной оптимизации
- **Автоматический выбор** CPU/GPU режима на основе доступных ресурсов
- **Полная GPU-acceleration** с Morton encoding
- **Интегрированная MoE поддержка** без дополнительной настройки
- **Adaptive performance optimization** в реальном времени
- **Лучшая обработка ошибок** с fallback механизмами
- **Detailed performance monitoring** и статистика

### 🚀 Новые возможности:
- **GPUMortonEncoder** для оптимизированного spatial indexing
- **Adaptive chunking** для больших решеток
- **Hybrid режим** CPU/GPU с автоматическим переключением
- **Real-time memory management**
- **Performance-based mode switching**

## Схема миграции

### Миграция SpatialOptimizer

#### Старый код:
```python
from core.lattice.spatial_optimization.spatial_optimizer import SpatialOptimizer

# Создание оптимизатора
optimizer = SpatialOptimizer(dimensions=(10, 10, 10))

# Поиск соседей
neighbors = optimizer.find_neighbors_optimized(coords=(5, 5, 5), radius=2.0)

# Оптимизация решетки
new_states = optimizer.optimize_lattice_forward(states, processor_fn)

# Получение статистики
stats = optimizer.get_performance_stats()
```

#### Новый код:
```python
from core.lattice.spatial_optimization.unified_spatial_optimizer import (
    create_unified_spatial_optimizer,
    OptimizationConfig,
    OptimizationMode
)

# Создание оптимизатора (AUTO режим - рекомендуется)
optimizer = create_unified_spatial_optimizer(dimensions=(10, 10, 10))

# Или с кастомной конфигурацией
config = OptimizationConfig(
    mode=OptimizationMode.AUTO,  # AUTO, CPU_ONLY, GPU_ONLY, HYBRID
    enable_moe=False,
    enable_morton_encoding=True,
    target_performance_ms=50.0
)
optimizer = create_unified_spatial_optimizer(dimensions=(10, 10, 10), config=config)

# Поиск соседей (тот же API)
neighbors = optimizer.find_neighbors_optimized(coords=(5, 5, 5), radius=2.0)

# Оптимизация решетки (улучшенный API с detailed результатами)
result = optimizer.optimize_lattice_forward(states)
new_states = result.new_states

# Получение расширенной статистики
stats = optimizer.get_comprehensive_stats()

# Принудительная оптимизация производительности
optimizer.optimize_performance()

# Обязательная очистка ресурсов
optimizer.cleanup()
```

### Миграция MoESpatialOptimizer

#### Старый код:
```python
from core.lattice.spatial_optimization.moe_spatial_optimizer import MoESpatialOptimizer

# Создание MoE оптимизатора
moe_optimizer = MoESpatialOptimizer(
    dimensions=(10, 10, 10), 
    moe_processor=moe_processor
)

# MoE оптимизация
new_states = moe_optimizer.optimize_moe_forward(states, moe_processor)
```

#### Новый код:
```python
from core.lattice.spatial_optimization.unified_spatial_optimizer import (
    create_unified_spatial_optimizer,
    OptimizationConfig
)

# Создание унифицированного оптимизатора с MoE
config = OptimizationConfig(
    enable_moe=True,
    enable_morton_encoding=True,
    mode=OptimizationMode.AUTO
)

optimizer = create_unified_spatial_optimizer(
    dimensions=(10, 10, 10),
    config=config,
    moe_processor=moe_processor
)

# Единый API для MoE оптимизации
result = optimizer.optimize_lattice_forward(states)
new_states = result.new_states

# Дополнительная информация о MoE обработке
print(f"MoE режим использован: {result.mode_used}")
print(f"Время обработки: {result.processing_time_ms}ms")
print(f"Использование памяти: {result.memory_usage_mb}MB")
```

## Режимы работы

### 1. AUTO режим (рекомендуется)
```python
config = OptimizationConfig(mode=OptimizationMode.AUTO)
```
- Автоматически выбирает оптимальный режим
- Переключается между CPU/GPU на основе производительности
- Адаптируется к доступным ресурсам

### 2. CPU_ONLY режим
```python
config = OptimizationConfig(mode=OptimizationMode.CPU_ONLY)
```
- Принудительно использует только CPU
- Совместим со старыми системами
- Не требует CUDA

### 3. GPU_ONLY режим
```python
config = OptimizationConfig(mode=OptimizationMode.GPU_ONLY)
```
- Принудительно использует только GPU
- Требует CUDA и достаточно GPU памяти
- Максимальная производительность

### 4. HYBRID режим
```python
config = OptimizationConfig(mode=OptimizationMode.HYBRID)
```
- Комбинирует CPU и GPU
- Автоматический fallback при ошибках GPU
- Оптимален для переменных нагрузок

## Конфигурационные опции

```python
config = OptimizationConfig(
    mode=OptimizationMode.AUTO,           # Режим работы
    enable_moe=True,                      # Включить MoE поддержку
    enable_morton_encoding=True,          # Включить Morton encoding
    enable_adaptive_chunking=True,        # Включить adaptive chunking
    max_memory_gb=8.0,                   # Максимальная память
    target_performance_ms=10.0,          # Целевое время обработки
    fallback_enabled=True                 # Включить fallback механизмы
)
```

## Обработка результатов

```python
result = optimizer.optimize_lattice_forward(states)

# Основные результаты
new_states = result.new_states
processing_time = result.processing_time_ms
memory_usage = result.memory_usage_mb

# Дополнительная информация
mode_used = result.mode_used           # Какой режим использовался
neighbors_found = result.neighbors_found
gpu_utilization = result.gpu_utilization
cache_hit_rate = result.cache_hit_rate
chunks_processed = result.chunks_processed
```

## Оценка требований к памяти

```python
from core.lattice.spatial_optimization.unified_spatial_optimizer import (
    estimate_unified_memory_requirements
)

# Оценка для разных конфигураций
estimates = estimate_unified_memory_requirements(
    dimensions=(50, 50, 50),
    config=OptimizationConfig(enable_moe=True)
)

print(f"Общая память: {estimates['total_memory_gb']:.2f}GB")
print(f"Рекомендуемая GPU память: {estimates['recommended_gpu_memory_gb']:.2f}GB")
print(f"CPU память: {estimates['recommended_system_memory_gb']:.2f}GB")
```

## Мониторинг производительности

```python
# Получение полной статистики
stats = optimizer.get_comprehensive_stats()

# Анализ производительности
unified_stats = stats['unified_optimizer']
print(f"Активный режим: {unified_stats['active_mode']}")
print(f"MoE включен: {unified_stats['moe_enabled']}")
print(f"Morton включен: {unified_stats['morton_enabled']}")

# Статистика CPU процессора
cpu_stats = stats['cpu_processor']
print(f"CPU запросы: {cpu_stats['total_queries']}")

# Статистика GPU процессора (если доступен)
if 'gpu_processor' in stats:
    gpu_stats = stats['gpu_processor']
    print(f"GPU статистика доступна")

# Анализ производительности
if 'performance_analysis' in stats:
    perf = stats['performance_analysis']
    print(f"Среднее время: {perf['avg_time_ms']:.1f}ms")
    print(f"Распределение режимов: {perf['mode_distribution']}")
```

## Рекомендации по миграции

### Пошаговый план:

1. **Установите зависимости**
   ```bash
   # Убедитесь что PyTorch установлен с CUDA поддержкой (опционально)
   pip install torch torchvision torchaudio
   ```

2. **Обновите импорты**
   - Замените импорты старых классов на `unified_spatial_optimizer`
   - Добавьте импорты для `OptimizationConfig` и `OptimizationMode`

3. **Создайте конфигурацию**
   - Начните с `OptimizationMode.AUTO`
   - Включите нужные features (MoE, Morton encoding)
   - Настройте target performance

4. **Обновите код обработки**
   - Используйте `result = optimize_lattice_forward()` вместо прямого возврата states
   - Добавьте обработку расширенной статистики
   - Добавьте `optimizer.cleanup()` в конце

5. **Протестируйте производительность**
   - Сравните с предыдущей версией
   - Настройте конфигурацию под ваши нужды
   - Добавьте мониторинг производительности

6. **Удалите старый код**
   - Уберите импорты `SpatialOptimizer` и `MoESpatialOptimizer`
   - Обновите тесты

### Типичные проблемы и решения:

**Проблема**: CUDA недоступна
```python
# Решение: используйте CPU_ONLY режим
config = OptimizationConfig(mode=OptimizationMode.CPU_ONLY)
```

**Проблема**: Недостаточно GPU памяти
```python
# Решение: используйте HYBRID режим или уменьшите max_memory_gb
config = OptimizationConfig(
    mode=OptimizationMode.HYBRID,
    max_memory_gb=4.0
)
```

**Проблема**: Медленная производительность
```python
# Решение: настройте target_performance_ms и включите оптимизации
config = OptimizationConfig(
    target_performance_ms=10.0,
    enable_morton_encoding=True,
    enable_adaptive_chunking=True
)
```

## Обратная совместимость

Старые классы `SpatialOptimizer` и `MoESpatialOptimizer` остаются доступными для обратной совместимости, но помечены как **DEPRECATED**.

**Настоятельно рекомендуется** мигрировать на `UnifiedSpatialOptimizer` для получения всех новых преимуществ и будущих обновлений.

## Поддержка

При возникновении проблем с миграцией:

1. Проверьте логи - UnifiedSpatialOptimizer предоставляет детальное логирование
2. Используйте AUTO режим для автоматической настройки
3. Добавьте fallback_enabled=True для стабильности
4. Проверьте тесты в `test_unified_spatial_optimizer.py`

Система спроектирована для плавной миграции с максимальной обратной совместимостью.