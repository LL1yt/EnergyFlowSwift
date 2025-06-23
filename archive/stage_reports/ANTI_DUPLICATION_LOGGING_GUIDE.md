# 🚫 Система предотвращения дублирования логов

## 🎯 Проблема

С ростом проекта появляется риск дублирования логов:

- Логи в разных модулях об одном событии
- Повторные логи при цепочке вызовов
- Логи от разных систем (legacy + new_rebuild)
- Спам в консоли из-за повторяющихся сообщений

## ✅ Решение

Архитектурная система предотвращения дублирования через:

### 1. **Временное дедублирование**

```python
# Автоматически игнорирует повторы в течение 1 секунды
logger.info("Создание клетки...")  # Логируется
logger.info("Создание клетки...")  # Игнорируется (дубликат)
```

### 2. **Контекстное логирование**

```python
# Четкая иерархия - кто что логирует
with LogContext("cell_creation", cell_type="NCA"):
    logger.info("Начинаем создание...")     # [cell_creation:cell_type=NCA] Начинаем создание...
    create_nca_cell()
    logger.info("Создание завершено...")    # [cell_creation:cell_type=NCA] Создание завершено...
```

### 3. **Нормализация сообщений**

```python
# Эти сообщения считаются одинаковыми:
logger.info("NCA: 55 params (target: 69)")
logger.info("NCA: 56 params (target: 69)")  # Игнорируется - числа нормализованы
```

## 🔧 Использование

### Базовая настройка

```python
from new_rebuild.utils import setup_logging

# С антидублированием (рекомендуется)
setup_logging(
    debug_mode=True,
    enable_deduplication=True,  # ✅ Включаем дедупликацию
    enable_context=True         # ✅ Включаем контекстное логирование
)
```

### Контекстное логирование для модулей

```python
from new_rebuild.utils import LogContext, get_logger

logger = get_logger(__name__)

def create_lattice():
    with LogContext("lattice_creation", size="6x6x6"):
        logger.info("Инициализация решетки...")    # [lattice_creation:size=6x6x6] Инициализация...

        with LogContext("memory_allocation"):
            logger.info("Выделение памяти...")     # [memory_allocation] Выделение памяти...
            allocate_memory()

        logger.info("Решетка создана!")           # [lattice_creation:size=6x6x6] Решетка создана!
```

### Специализированные функции для клеток

```python
from new_rebuild.utils import log_cell_init, log_cell_forward

# Автоматически используют контекст и предотвращают дублирование
log_cell_init("NCA", total_params=55, target_params=69, state_size=4)

# В debug режиме
log_cell_forward("NCA", {"neighbor_states": torch.Size([2, 26, 4])})
```

## 📋 Правила предотвращения дублирования

### ✅ DO - Делайте так:

1. **Используйте контексты для группировки операций**

```python
with LogContext("training", epoch=1):
    train_model()  # Все логи помечены [training:epoch=1]
```

2. **Логируйте события, не действия**

```python
logger.info("🚀 INIT ModelTrainer")        # ✅ Событие
logger.info("✅ Training completed")        # ✅ Событие
```

3. **Используйте специализированные функции**

```python
log_cell_init("NCA", ...)     # ✅ Специализированная функция
log_performance("forward", 0.05, batch_size=32)  # ✅ Специализированная
```

### ❌ DON'T - Избегайте:

1. **Дублирующих логов в цепочке вызовов**

```python
def create_cell():
    logger.info("Creating cell...")    # ❌
    cell = NCACell()                  # Внутри тоже логирует создание
    logger.info("Cell created...")    # ❌ Дублирование
```

2. **Логирования в циклах без контекста**

```python
for i in range(100):
    logger.info("Processing item...")  # ❌ Спам в логах
```

3. **Повторных логов об одном событии**

```python
logger.info("Model initialized")
setup_model()
logger.info("Model setup complete")    # ❌ То же событие
```

## 🛠️ Конфигурация дедупликации

```python
from new_rebuild.utils.logging import DuplicationManager

# Настройка времени дедупликации
dedup_manager = DuplicationManager(dedup_window_seconds=2)  # 2 секунды

# Настройка логирования с кастомным менеджером
setup_logging(
    debug_mode=True,
    enable_deduplication=True,
    dedup_window=2  # Время окна дедупликации
)
```

## 🔍 Отладка дедупликации

Если логи неожиданно исчезают:

```python
# Временно отключите дедупликацию для отладки
setup_logging(
    debug_mode=True,
    enable_deduplication=False,  # ❌ Отключили дедупликацию
    enable_context=True
)
```

## 📊 Мониторинг эффективности

```python
from new_rebuild.utils.logging import _deduplication_manager

# Статистика дедупликации
stats = _deduplication_manager.get_stats()
print(f"Всего сообщений: {stats['total']}")
print(f"Дедуплицированных: {stats['deduplicated']}")
print(f"Эффективность: {stats['efficiency']:.1%}")
```

## 🎯 Best Practices

1. **Один контекст на операцию** - каждая логическая операция в своем контексте
2. **Специализированные функции** - используйте `log_cell_*`, `log_performance` и т.д.
3. **Event-based логирование** - логируйте что произошло, не что делаете
4. **Иерархическое планирование** - планируйте кто что логирует на этапе архитектуры

---

**Результат**: Чистые, информативные логи без дублирования даже в крупном проекте! 🎉
