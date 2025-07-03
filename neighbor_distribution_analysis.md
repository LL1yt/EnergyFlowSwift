# Анализ проблемы с распределением соседей

## Проблема
В логах тренировки мы видим неправильное распределение соседей:
- LOCAL: 6 connections
- FUNCTIONAL: 12 connections  
- DISTANT: 197 connections (!)

При этом ожидалось ~14 DISTANT соединений.

## Корень проблемы

### 1. Несоответствие между поиском соседей и классификацией

**Текущая логика:**
- `UnifiedSpatialOptimizer` ищет соседей в радиусе `adaptive_radius`
- `ConnectionCacheManager` также использует `adaptive_radius` для поиска всех соседей
- Но `distant_threshold = 1.0 * adaptive_radius`

Это означает, что ВСЕ найденные соседи должны попадать в одну из трех категорий.

### 2. Возможные причины большого числа DISTANT соединений

1. **Ошибка в spatial optimizer**: Возможно, `adaptive_radius` вычисляется неправильно или используется другое значение
2. **Ошибка в классификации**: Соседи в диапазоне `functional_threshold < d <= distant_threshold` правильно классифицируются как DISTANT, но их слишком много
3. **Проблема с кэшем**: Кэш может быть создан с одними параметрами, а используется с другими

## Необходимые исправления

### 1. Проверить консистентность adaptive_radius
```python
# В UnifiedSpatialOptimizer.find_neighbors_by_radius_safe
adaptive_radius = config.calculate_adaptive_radius()
# Должно совпадать с значением в ConnectionCacheManager

# В ConnectionCacheManager.__init__
self.adaptive_radius = config.calculate_adaptive_radius()
```

### 2. Исправить границы интервалов
Текущая логика имеет несоответствие:
- LOCAL: `d < local_threshold` (правая граница НЕ включена)
- FUNCTIONAL: `d <= functional_threshold` (правая граница включена)

Это означает, что точки на границе `local_threshold` попадают в FUNCTIONAL, а не в LOCAL.

### 3. Добавить логирование для отладки
Нужно добавить детальное логирование:
- Какой adaptive_radius используется в spatial optimizer
- Сколько соседей найдено
- Как они распределяются по расстояниям

## Временное решение
Убедиться, что:
1. Везде используется одинаковый `adaptive_radius`
2. `distant_threshold` действительно равен `adaptive_radius`
3. Spatial optimizer не находит соседей за пределами `adaptive_radius`