# Automated Training Module - Модульная архитектура

Система автоматизированного долгосрочного обучения с прогрессивным увеличением сложности для 3D Cellular Neural Network.

## Обзор рефакторинга

Оригинальный файл `automated_training.py` (629 строк) был разделен на модульную архитектуру для улучшения читаемости, поддерживаемости и тестируемости.

### Преимущества новой архитектуры

1. **Разделение ответственности** - каждый модуль отвечает за конкретную область
2. **Уменьшение размера файлов** - файлы стали более управляемыми (100-200 строк каждый)
3. **Лучшая тестируемость** - каждый компонент можно тестировать независимо
4. **Переиспользование** - компоненты могут использоваться в других частях системы
5. **Обратная совместимость** - API остался неизменным

## Структура модулей

```
training/automated_training/
├── __init__.py                 # Точка входа модуля
├── automated_trainer.py        # Главный класс-интегратор
├── progressive_config.py       # Управление конфигурациями стадий
├── stage_runner.py            # Выполнение тренировочных процессов
├── session_manager.py         # Управление сессиями и логированием
├── cli_interface.py           # CLI интерфейс
└── README.md                  # Документация (этот файл)
```

### Размеры файлов (до/после)

| Компонент               | До        | После      |
| ----------------------- | --------- | ---------- |
| Оригинальный файл       | 629 строк | -          |
| `automated_trainer.py`  | -         | ~140 строк |
| `progressive_config.py` | -         | ~180 строк |
| `stage_runner.py`       | -         | ~250 строк |
| `session_manager.py`    | -         | ~220 строк |
| `cli_interface.py`      | -         | ~280 строк |
| `__init__.py`           | -         | ~40 строк  |

**Итого**: 629→1110 строк (включая документацию и улучшенную обработку ошибок)

## Описание компонентов

### 1. AutomatedTrainer

**Файл**: `automated_trainer.py`
**Назначение**: Главный класс-интегратор

Объединяет все компоненты и предоставляет простой API для запуска автоматизированного обучения.

```python
from training.automated_training import AutomatedTrainer

trainer = AutomatedTrainer(
    mode="development",
    max_total_time_hours=8.0,
    dataset_limit_override=1000  # для тестирования
)
trainer.run_automated_training()
```

### 2. ProgressiveConfigManager

**Файл**: `progressive_config.py`
**Назначение**: Управление конфигурациями стадий

Реализует стратегию прогрессивного увеличения сложности обучения:

- Stage 1: Маленький датасет, много эпох (изучение основ)
- Stage 2: Средний датасет, средние эпохи (консолидация)
- Stage 3-5: Увеличение сложности

```python
config_manager = ProgressiveConfigManager(dataset_limit_override=500)
stage_config = config_manager.get_stage_config(1)
estimated_time = config_manager.estimate_stage_time(stage_config, "development")
```

### 3. TrainingStageRunner

**Файл**: `stage_runner.py`
**Назначение**: Выполнение тренировочных процессов

Управляет subprocess-ами, мониторит выполнение в реальном времени и обрабатывает результаты.

```python
stage_runner = TrainingStageRunner(mode="development", timeout_multiplier=2.0)
result = stage_runner.run_stage(stage_config, estimated_time)
```

### 4. SessionManager

**Файл**: `session_manager.py`
**Назначение**: Управление сессиями и логированием

Отвечает за логирование прогресса, сохранение истории результатов и генерацию сводок.

```python
session_manager = SessionManager(max_total_time_hours=8.0)
session_manager.add_stage_result(result)
summary = session_manager.get_session_summary()
```

### 5. CLIInterface

**Файл**: `cli_interface.py`
**Назначение**: Интерфейс командной строки

Предоставляет расширенный CLI интерфейс с валидацией аргументов и подробной справкой.

```python
from training.automated_training import CLIInterface

cli = CLIInterface()
cli.main()  # или cli.main(["--mode", "research", "--max-hours", "12"])
```

## Использование

### Быстрый старт

```python
# Простейший способ
from training.automated_training import AutomatedTrainer

trainer = AutomatedTrainer()
trainer.run_automated_training()
```

### Через CLI (рекомендуется)

```bash
# Стандартное обучение
python automated_training_refactored.py

# Быстрое тестирование
python automated_training_refactored.py --dataset-limit 100 --max-hours 1

# Показать конфигурации без запуска
python automated_training_refactored.py --test-config

# Продакшн режим
python automated_training_refactored.py --mode production --max-hours 24
```

### Программное использование отдельных компонентов

```python
# Только проверка конфигураций
from training.automated_training import ProgressiveConfigManager

config_manager = ProgressiveConfigManager()
stages = config_manager.get_all_stages_info()
print(f"Всего стадий: {len(stages)}")

# Только выполнение одной стадии
from training.automated_training import TrainingStageRunner

runner = TrainingStageRunner(mode="development")
result = runner.run_stage(stage_config, estimated_time)
```

## Миграция с оригинальной версии

### 1. Обратная совместимость

Новая версия полностью обратно совместима:

```python
# Старый код продолжает работать
from automated_training import AutomatedTrainer  # будет работать
trainer = AutomatedTrainer(mode="development")
trainer.run_automated_training()
```

### 2. Рекомендуемый подход

```python
# Новый подход (рекомендуется)
from training.automated_training import AutomatedTrainer
# Или используйте CLI: python automated_training_refactored.py
```

### 3. Переименование файлов

1. Переименуйте `automated_training.py` → `automated_training_legacy.py`
2. Переименуйте `automated_training_refactored.py` → `automated_training.py`
3. Обновите импорты в зависимых файлах

## Тестирование

### Тестирование конфигураций

```bash
python automated_training.py --test-config
```

### Быстрое тестирование

```bash
python automated_training.py --dataset-limit 50 --max-hours 0.5
```

### Тестирование отдельных компонентов

```python
# Тест конфигурационного менеджера
from training.automated_training import ProgressiveConfigManager

config_manager = ProgressiveConfigManager(dataset_limit_override=100)
for stage in range(1, 6):
    config = config_manager.get_stage_config(stage)
    print(f"Stage {stage}: {config.description}")
```

## Расширение функциональности

### Добавление новых стратегий конфигурации

Модифицируйте `ProgressiveConfigManager._base_configs`:

```python
# В progressive_config.py
self._base_configs = {
    # ... существующие стадии ...
    6: {
        "dataset_limit": 100000,
        "epochs": 3,
        "batch_size": 256,
        "description": "Ultra-scale (massive data, minimal epochs)",
    }
}
```

### Добавление новых типов метрик

Модифицируйте `TrainingStageRunner._extract_similarity_from_output()`:

```python
def _extract_metrics_from_output(self, output: str) -> Dict[str, float]:
    metrics = {}
    for line in output.split("\n"):
        if "final_similarity:" in line:
            metrics["similarity"] = float(line.split(":")[-1].strip())
        elif "final_loss:" in line:
            metrics["loss"] = float(line.split(":")[-1].strip())
    return metrics
```

## Мониторинг и отладка

### Логи

- **Консольные логи**: Детальная информация о прогрессе
- **Файловые логи**: `logs/automated_training.log`
- **Сессионные логи**: `logs/automated_training/automated_session_YYYYMMDD_HHMMSS.json`

### Структура сессионного лога

```json
{
  "mode": "development",
  "start_time": "2025-01-10T15:30:00",
  "elapsed_hours": 2.5,
  "training_history": [
    {
      "stage": 1,
      "success": true,
      "actual_time_minutes": 15.2,
      "final_similarity": 0.8456
    }
  ],
  "summary": {
    "total_stages": 3,
    "best_similarity": 0.8956,
    "similarity_trend": [0.8456, 0.8721, 0.8956]
  }
}
```

## Производительность

### Оптимизации в новой архитектуре

1. **Ленивая инициализация** - компоненты создаются только при необходимости
2. **Оптимизированное логирование** - буферизация вывода subprocess-ов
3. **Валидация входных данных** - раннее обнаружение ошибок
4. **Улучшенная обработка таймаутов** - более точные оценки времени

### Рекомендации по использованию ресурсов

- **Development режим**: 2-8 часов, подходит для отладки
- **Research режим**: 8-24 часа, для экспериментов
- **Production режим**: 24+ часов, для финального обучения

## Устранение неполадок

### Частые проблемы

1. **ImportError при запуске**

   ```bash
   # Проверьте структуру файлов
   ls -la training/automated_training/
   ```

2. **Таймауты процессов**

   ```bash
   # Увеличьте таймаут
   python automated_training.py --timeout-multiplier 3.0
   ```

3. **Нехватка времени для стадий**
   ```bash
   # Увеличьте лимит времени или уменьшите размер данных
   python automated_training.py --max-hours 12 --dataset-limit 1000
   ```

### Отладочная информация

```bash
# Включите подробные логи
python automated_training.py --verbose

# Или только показать конфигурацию
python automated_training.py --test-config
```

## Вклад в развитие

При добавлении новых функций следуйте принципам модульной архитектуры:

1. **Одна ответственность** - каждый класс решает одну задачу
2. **Слабая связанность** - минимальные зависимости между модулями
3. **Высокая когезия** - связанная функциональность в одном модуле
4. **Тестируемость** - каждый компонент можно тестировать отдельно

---

**Версия**: v2.0.0
**Дата**: Январь 2025
**Автор**: 3D Cellular Neural Network Project
