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
├── cli_argument_parser.py      # Логика парсинга CLI аргументов
├── cli_handler.py              # Обработчики команд CLI
├── cli_interface.py            # CLI интерфейс
├── logging/                    # Модуль логирования
│   ├── __init__.py
│   ├── core.py
│   ├── formatters.py
│   ├── helpers.py
│   └── metrics_logger.py
├── logging_config.py           # Фасад для старого API логирования
├── process_runner.py           # Утилита для запуска внешних процессов
├── progressive_config.py       # Управление конфигурациями стадий
├── session_manager.py          # Управление состоянием сессии
├── session_persistence.py      # Сохранение и загрузка сессий
├── stage_runner.py             # Выполнение тренировочных процессов
├── types.py                    # Централизованные типы данных (dataclasses)
└── README.md                   # Документация (этот файл)
```

### Размеры файлов (после рефакторинга)

| Компонент                | Строк | Описание                                             |
| ------------------------ | ----- | ---------------------------------------------------- |
| `automated_trainer.py`   | ~170  | Главный класс-интегратор                             |
| `cli_interface.py`       | ~100  | Структура CLI, валидация, запуск                     |
| `cli_argument_parser.py` | ~90   | Определение аргументов CLI                           |
| `cli_handler.py`         | ~80   | Логика выполнения команд CLI                         |
| `progressive_config.py`  | ~180  | Управление конфигурациями стадий                     |
| `stage_runner.py`        | ~150  | Выполнение одной стадии обучения                     |
| `process_runner.py`      | ~70   | Утилита для запуска subprocess-ов                    |
| `session_manager.py`     | ~180  | Управление состоянием сессии                         |
| `session_persistence.py` | ~70   | Сохранение/загрузка сессии в JSON                    |
| `types.py`               | ~50   | Общие типы данных (dataclasses)                      |
| `logging/` (пакет)       | ~250  | Модульная система логирования                        |
| **Итого**                | ~1390 | (было ~1110) - больше файлов, но меньше код в каждом |

## Описание компонентов

### 1. AutomatedTrainer

**Файл**: `automated_trainer.py`
**Назначение**: Главный класс-интегратор. Связывает все компоненты вместе и содержит главный цикл обучения.

### 2. CLI (Интерфейс командной строки)

- **`cli_interface.py`**: Определяет `CLIInterface`, который парсит аргументы, настраивает логирование и вызывает обработчики.
- **`cli_argument_parser.py`**: Содержит функции для создания и конфигурации `ArgumentParser`.
- **`cli_handler.py`**: Содержит функции, которые выполняют основную логику, вызываемую из CLI (например, запуск обучения или показ конфигурации).

### 3. ProgressiveConfigManager

**Файл**: `progressive_config.py`
**Назначение**: Управление конфигурациями стадий. Реализует стратегию прогрессивного увеличения сложности обучения.

### 4. StageRunner & ProcessRunner

- **`stage_runner.py`**: Отвечает за выполнение одной стадии обучения. Формирует команду и обрабатывает результат.
- **`process_runner.py`**: Абстракция для запуска внешних процессов с таймаутами и захватом вывода.

### 5. SessionManager & SessionPersistence

- **`session_manager.py`**: Управляет состоянием сессии: отслеживает время, историю результатов, генерирует сводки.
- **`session_persistence.py`**: Отвечает за сериализацию, сохранение и загрузку состояния сессии в JSON-файлы.

### 6. Logging (пакет)

**Каталог**: `logging/`
**Назначение**: Централизованная и модульная система логирования.

- **`core.py`**: Основная логика настройки логгеров.
- **`formatters.py`**: Пользовательские форматтеры (`StructuredFormatter`).
- **`helpers.py`**: Вспомогательные функции (`get_logger`, `log_stage_start`).
- **`metrics_logger.py`**: Специализированный логгер для метрик.

### 7. Types

**Файл**: `types.py`
**Назначение**: Определяет общие структуры данных (`dataclasses`), используемые в разных модулях (`StageConfig`, `StageResult`, `SessionSummary`), для избежания циклических зависимостей.

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
