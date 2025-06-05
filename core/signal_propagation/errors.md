# Signal Propagation Module - Error Documentation

## Реальные Ошибки, Найденные и Решенные

Этот документ содержит только ошибки, которые фактически возникали во время разработки и тестирования модуля signal_propagation.

---

## ERROR-001: Tensor Dimension Mismatch in Lattice Integration

**Дата:** December 5, 2025  
**Статус:** ✅ РЕШЕНА

### Описание Ошибки

```
RuntimeError: Tensors must have same number of dimensions: got 4 and 2
```

### Контекст

Ошибка возникла при интеграции SignalPropagator с Lattice3D во время выполнения метода `propagate_step()`.

### Стек вызовов

```python
File "test_signal_propagation.py", line 199, in test_signal_propagator_basic
    new_state = propagator.propagate_step()
File "core\signal_propagation\signal_propagator.py", line 177, in propagate_step
    self.current_signals = self._apply_propagation_logic(self.current_signals)
File "core\signal_propagation\signal_propagator.py", line 198, in _apply_propagation_logic
    return self._wave_propagation(signals)
File "core\signal_propagation\signal_propagator.py", line 210, in _wave_propagation
    updated_signals = self.lattice.forward(signals)
File "core\lattice_3d\main.py", line 851, in _get_external_input_for_cell
    ext_input = torch.cat([ext_input, padding], dim=1)
```

### Причина

SignalPropagator использовал 4D тензоры формата `[x, y, z, state_size]`, а Lattice3D ожидал 2D тензоры для внешних входов. Прямой вызов `lattice.forward(signals)` приводил к несоответствию размерностей.

### Решение

Создан новый метод `_apply_cellular_dynamics()` который:

1. Преобразует 4D тензоры сигналов в формат для клеток
2. Получает состояния соседей для каждой клетки
3. Применяет клеточную динамику через `cell_prototype`
4. Возвращает обновленные сигналы в том же 4D формате

### Измененные файлы

- `core/signal_propagation/signal_propagator.py`
  - Заменен прямой вызов `lattice.forward()` на `_apply_cellular_dynamics()`
  - Добавлены методы: `_apply_cellular_dynamics()`, `_get_neighbor_states()`, `_is_valid_position()`

### Предотвращение

- Тщательное планирование интерфейсов между модулями
- Документирование ожидаемых форматов данных
- Unit тесты для проверки совместимости форматов

---

## ERROR-002: PyTorch Function Type Error with torch.sin()

**Дата:** December 5, 2025  
**Статус:** ✅ РЕШЕНА

### Описание Ошибки

```
TypeError: sin(): argument 'input' (position 1) must be Tensor, not float
```

### Контекст

Ошибка возникла в методе `_calculate_wave_effect()` при попытке применить `torch.sin()` к числовому значению вместо тензора.

### Стек вызовов

```python
File "core\signal_propagation\signal_propagator.py", line 328, in _calculate_wave_effect
    wave_effect = wave_amplitude * torch.sin(self.time_manager.current_step * 0.1) * signals
```

### Причина

`torch.sin()` требует тензор в качестве аргумента, но `self.time_manager.current_step * 0.1` возвращал обычное число типа `float`.

### Проблемный код

```python
wave_effect = wave_amplitude * torch.sin(self.time_manager.current_step * 0.1) * signals
```

### Решение

Преобразование числового значения в тензор с правильным типом данных:

```python
# Преобразуем текущий шаг в тензор для torch.sin()
time_tensor = torch.tensor(self.time_manager.current_step * 0.1, dtype=signals.dtype)
wave_effect = wave_amplitude * torch.sin(time_tensor) * signals
```

### Измененные файлы

- `core/signal_propagation/signal_propagator.py` - метод `_calculate_wave_effect()`

### Предотвращение

- Проверка типов данных для PyTorch функций
- Unit тесты для всех математических операций
- Использование типизации для ранней детекции подобных ошибок

---

## ERROR-003: CUDA GPU Compatibility Issue

**Дата:** December 5, 2025  
**Статус:** ✅ РЕШЕНА (ОБХОДНОЕ РЕШЕНИЕ)

### Описание Ошибки

```
CUDA error: no kernel image is available for execution on the device
```

### Контекст

При попытке использования GPU (RTX 5090) для вычислений PyTorch, система не могла выполнить операции из-за несовместимости архитектуры.

### Детали Системы

- GPU: RTX 5090 (compute capability sm_120)
- PyTorch: Скомпилирован для максимальной архитектуры sm_90
- Несовместимость: sm_120 > sm_90

### Причина

RTX 5090 имеет более новую архитектуру (sm_120), чем поддерживается текущей версией PyTorch (максимум sm_90).

### Решение

Принудительное отключение GPU для обеспечения стабильной работы:

```python
# В тестах добавлен параметр gpu_enabled=False
lattice_config = LatticeConfig(
    size=(3, 3, 3),
    cell_config=cell_config,
    boundary_condition="reflective",
    gpu_enabled=False  # Отключение GPU для совместимости
)
```

### Альтернативные Решения

1. **Обновление PyTorch** до версии с поддержкой sm_120
2. **Использование CPU** для текущих потребностей
3. **Компиляция PyTorch** с поддержкой новых архитектур

### Временные Ограничения

- Система работает только на CPU
- Производительность ограничена для больших решеток
- Требуется обновление для полной GPU поддержки

### Предотвращение

- Проверка совместимости GPU архитектуры при установке
- Автоматическое переключение на CPU при проблемах с GPU
- Документирование требований к системе

---

## ERROR-004: Module Import Structure Issues

**Дата:** December 5, 2025  
**Статус:** ✅ РЕШЕНА

### Описание Ошибки

```
ImportError: cannot import name 'TimeMode' from 'core.signal_propagation'
```

### Контекст

При импорте енумов из модуля signal_propagation в тестах и других модулях.

### Причина

Неполные экспорты в `__init__.py` файле модуля - не все енумы были добавлены в `__all__`.

### Проблемные импорты

- `TimeMode` - не экспортировался
- `PropagationMode` - не экспортировался
- `ConvergenceMode` - не экспортировался

### Решение

Обновлен `core/signal_propagation/__init__.py` с полными экспортами:

```python
__all__ = [
    # Енумы
    'TimeMode',
    'PropagationMode',
    'PatternType',
    'ConvergenceMode',
    # ... остальные экспорты
]
```

### Измененные файлы

- `core/signal_propagation/__init__.py`

### Предотвращение

- Автоматическая проверка экспортов при добавлении новых классов
- Import тесты для всех публичных компонентов
- Регулярная проверка полноты `__all__` списков

---

## Статистика Ошибок

### Общие Показатели

- **Всего найдено ошибок:** 4
- **Решенных ошибок:** 4 (100%)
- **Критических ошибок:** 2 (ERROR-001, ERROR-002)
- **Ошибок совместимости:** 1 (ERROR-003)
- **Ошибок конфигурации:** 1 (ERROR-004)

### Время на Решение

- **ERROR-001:** ~30 минут (требовал архитектурных изменений)
- **ERROR-002:** ~5 минут (простое исправление типа)
- **ERROR-003:** ~10 минут (обходное решение)
- **ERROR-004:** ~5 минут (обновление экспортов)

### Категории Ошибок

1. **Типы данных:** 50% (ERROR-001, ERROR-002)
2. **Совместимость:** 25% (ERROR-003)
3. **Конфигурация:** 25% (ERROR-004)

### Уроки

1. **Тестирование интеграции** критически важно
2. **Проверка типов данных** должна быть автоматической
3. **Совместимость системы** нужно проверять заранее
4. **Структура экспортов** требует внимания

---

## Рекомендации для Будущих Модулей

### Предотвращение Ошибок

1. **Ранняя интеграция** - тестировать взаимодействие модулей сразу
2. **Типизация** - использовать строгую типизацию для всех интерфейсов
3. **Проверка совместимости** - автоматически детектировать системные ограничения
4. **Полные экспорты** - тестировать импорты всех публичных компонентов

### Тестирование

1. **Unit тесты** для каждого метода
2. **Integration тесты** для взаимодействия компонентов
3. **Compatibility тесты** для разных систем
4. **Import тесты** для всех экспортируемых компонентов

**Последнее обновление:** December 5, 2025
