# 🚀 Training Commands Help - 3D Cellular Neural Network

## 📊 Основные команды для обучения

### 1. 🔧 Подготовка данных (SNLI Dataset)

```bash
# Создать полный SNLI датасет (1/5 от полного = ~110K пар)
python generate_snli_embedding_dataset.py --fraction 0.2 --batch-size 32

# Создать тестовый SNLI датасет (1% = ~5K пар)
python generate_snli_embedding_dataset.py --fraction 0.01 --batch-size 16

# Проверить доступные датасеты
python precomputed_embedding_loader.py
```

---

### 2. 🎯 Обычное обучение (run_dynamic_training.py)

#### Быстрые тесты:

```bash
# Минимальный тест (100 примеров, 5 эпох)
python run_dynamic_training.py --mode development --dataset-limit 100 --epochs 5 --batch-size 16

# Средний тест (2K примеров, 10 эпох)
python run_dynamic_training.py --mode development --dataset-limit 2000 --epochs 10 --batch-size 32

# Большой тест (10K примеров, 20 эпох)
python run_dynamic_training.py --mode development --dataset-limit 10000 --epochs 20 --batch-size 64
```

#### Разные режимы:

```bash
# Development mode (маленькая модель, scale=0.01)
python run_dynamic_training.py --mode development --dataset-limit 5000 --epochs 15

# Research mode (средняя модель, scale=0.1)
python run_dynamic_training.py --mode research --dataset-limit 20000 --epochs 25

# Production mode (большая модель, scale=1.0)
python run_dynamic_training.py --mode production --dataset-limit 50000 --epochs 30
```

#### Custom scale:

```bash
# Очень маленькая модель
python run_dynamic_training.py --mode development --scale 0.005 --dataset-limit 1000 --epochs 10

# Средняя модель
python run_dynamic_training.py --mode development --scale 0.05 --dataset-limit 15000 --epochs 20

# Большая модель
python run_dynamic_training.py --mode development --scale 0.2 --dataset-limit 40000 --epochs 25
```

---

### 3. 🧠 Smart Resume (автоматический поиск чекпоинтов)

#### Проверка доступных чекпоинтов:

```bash
# Показать совместимые чекпоинты (без обучения)
python smart_resume_training.py --list-only --mode development

# Показать для других режимов
python smart_resume_training.py --list-only --mode research
python smart_resume_training.py --list-only --mode production
```

#### Автоматическое продолжение:

```bash
# Продолжить с лучшего чекпоинта (5 доп. эпох)
python smart_resume_training.py --mode development --additional-epochs 5

# Продолжить с лимитом данных (для быстрого теста)
python smart_resume_training.py --mode development --dataset-limit 3000 --additional-epochs 10 --batch-size 32

# Длительное продолжение
python smart_resume_training.py --mode development --dataset-limit 20000 --additional-epochs 30 --batch-size 64
```

---

### 4. 📊 Параметры командной строки

#### Общие параметры:

```
--mode              : development, research, validation, production
--dataset-limit     : Ограничить количество примеров (например: 1000, 5000, 20000)
--epochs            : Количество эпох обучения
--batch-size        : Размер батча (16, 32, 64, 128, 256)
--scale             : Custom scale factor (0.001 - 1.0)
--debug             : Включить debug режим
```

#### Специфичные для smart_resume:

```
--additional-epochs : Сколько дополнительных эпох обучать
--list-only         : Только показать чекпоинты, не обучать
```

---

### 5. 🎯 Рекомендуемые сценарии

#### 🚀 Быстрый старт (первый запуск):

```bash
# 1. Создать данные
python generate_snli_embedding_dataset.py --fraction 0.01

# 2. Первое обучение
python run_dynamic_training.py --mode development --dataset-limit 1000 --epochs 5 --batch-size 32

# 3. Продолжить
python smart_resume_training.py --mode development --additional-epochs 10 --batch-size 32
```

#### 🔬 Эксперименты с архитектурой:

```bash
# Маленькая модель
python run_dynamic_training.py --mode development --scale 0.01 --dataset-limit 5000 --epochs 15

# Средняя модель
python run_dynamic_training.py --mode development --scale 0.05 --dataset-limit 10000 --epochs 20

# Большая модель
python run_dynamic_training.py --mode development --scale 0.1 --dataset-limit 20000 --epochs 25
```

#### 🎯 Серьезное обучение:

```bash
# 1. Создать полные данные
python generate_snli_embedding_dataset.py --fraction 0.2

# 2. Обучение на полных данных
python run_dynamic_training.py --mode research --epochs 50 --batch-size 64

# 3. Продолжение
python smart_resume_training.py --mode research --additional-epochs 25 --batch-size 64
```

---

### 6. 📁 Полезные команды

#### Мониторинг:

```bash
# Проверить размер датасетов
ls -lh data/embeddings/

# Проверить чекпоинты
ls -lh checkpoints/versioned/

# Посмотреть логи
ls -lh logs/
cat logs/dynamic_training_*.json | jq '.training_info'
```

#### Очистка:

```bash
# Очистить старые чекпоинты
rm -rf checkpoints/versioned/*

# Очистить логи
rm -f logs/dynamic_training_*.json

# Освободить место (оставить только последний датасет)
cd data/embeddings && ls -t *.pt | tail -n +2 | xargs rm -f
```

---

### 7. 🔍 Примеры логов и результатов

#### Хорошие результаты:

```
Final similarity: 0.67+ (очень хорошо)
Final similarity: 0.50+ (хорошо)
Final similarity: 0.30+ (нормально)
Final similarity: 0.10+ (слабо)
```

#### Время обучения:

```
Development mode (~1K neurons): ~30 секунд/эпоха
Research mode (~10K neurons): ~2 минуты/эпоха
Production mode (~100K neurons): ~10 минут/эпоха
```

---

### 8. ⚠️ Решение проблем

#### Если чекпоинт не загружается:

```bash
# Начать с чистого листа
rm -rf checkpoints/*
python run_dynamic_training.py --mode development --dataset-limit 2000 --epochs 10
```

#### Если память кончается:

```bash
# Уменьшить batch-size
--batch-size 16  # вместо 64

# Уменьшить dataset-limit
--dataset-limit 1000  # вместо 10000

# Использовать меньший scale
--scale 0.005  # вместо 0.01
```

#### Если слишком медленно:

```bash
# Увеличить batch-size
--batch-size 128  # вместо 32

# Уменьшить dataset-limit для тестов
--dataset-limit 5000  # вместо полного датасета

# Использовать development mode
--mode development  # самая быстрая конфигурация
```

---

**🎯 Быстрый старт для нового пользователя:**

1. `python generate_snli_embedding_dataset.py --fraction 0.01`
2. `python run_dynamic_training.py --mode development --dataset-limit 1000 --epochs 5`
3. `python smart_resume_training.py --mode development --additional-epochs 10`

# 🎯 Training Commands Help

Полное руководство по обучению 3D клеточной нейронной сети

## 📖 Оглавление

- [Базовые команды](#базовые-команды)
- [Smart Resume обучение](#smart-resume-обучение)
- [Dynamic обучение](#dynamic-обучение)
- [Автоматизированное обучение](#автоматизированное-обучение)
- [Warm-up Learning Rate](#warm-up-learning-rate)
- [Стратегии долгого обучения](#стратегии-долгого-обучения)
- [Параметры](#параметры)
- [Примеры использования](#примеры-использования)
- [Устранение проблем](#устранение-проблем)

---

## 🚀 Базовые команды

### Quick Start (рекомендуется для начинающих)

```bash
# Быстрое обучение для тестирования
python smart_resume_training.py --dataset-limit 1000 --additional-epochs 5

# Среднее обучение
python smart_resume_training.py --dataset-limit 5000 --additional-epochs 10

# Долгое обучение
python smart_resume_training.py --dataset-limit 20000 --additional-epochs 20
```

---

## 🔄 Smart Resume обучение

**Автоматически находит лучший checkpoint и продолжает обучение**

### Основные команды:

```bash
# Базовая команда
python smart_resume_training.py

# С ограничением датасета и дополнительными эпохами
python smart_resume_training.py --dataset-limit 5000 --additional-epochs 10

# Разные режимы
python smart_resume_training.py --mode development --dataset-limit 2000
python smart_resume_training.py --mode research --dataset-limit 10000
python smart_resume_training.py --mode production --dataset-limit 50000

# Custom scale
python smart_resume_training.py --scale 0.05 --dataset-limit 5000
```

### 🎯 Фиксированная выборка:

```bash
# Случайная выборка (по умолчанию, воспроизводимая с seed=42)
python smart_resume_training.py --dataset-limit 5000

# Фиксированная выборка (первые N примеров)
python smart_resume_training.py --dataset-limit 5000 --fixed-sampling
```

**Когда использовать фиксированную выборку:**

- ✅ При отладке (один и тот же датасет)
- ✅ При сравнении методов
- ✅ При resume обучения с точно теми же данными
- ❌ При обычном обучении (лучше случайная выборка)

---

## ⚡ Dynamic обучение

**Обучение с динамической конфигурацией (разные размеры решетки)**

```bash
# Разные режимы (автоматически подбирают размер решетки)
python run_dynamic_training.py --mode development  # Маленькая решетка
python run_dynamic_training.py --mode research     # Средняя решетка
python run_dynamic_training.py --mode validation   # Большая решетка

# Custom scale
python run_dynamic_training.py --scale 0.02 --dataset-limit 3000 --epochs 15

# С фиксированной выборкой
python run_dynamic_training.py --dataset-limit 5000 --fixed-sampling
```

---

## 🤖 Автоматизированное обучение

**Новый скрипт для долгого автоматического обучения**

### Основные команды:

```bash
# Ночное обучение (8 часов)
python automated_training.py --max-hours 8

# Короткая сессия (2 часа)
python automated_training.py --max-hours 2

# С указанием режима
python automated_training.py --mode research --max-hours 12

# Посмотреть план без запуска
python automated_training.py --test-config
```

### 📊 Стратегия автоматического обучения:

**Stage 1:** Foundation Learning

- 2,000 примеров, 20 эпох, batch=32
- Изучение основных паттернов

**Stage 2:** Consolidation

- 5,000 примеров, 15 эпох, batch=64
- Консолидация знаний

**Stage 3:** Refinement

- 10,000 примеров, 12 эпох, batch=64
- Тонкая настройка

**Stage 4:** Mastery

- 20,000 примеров, 8 эпох, batch=128
- Мастерство на больших данных

**Stage 5:** Perfection

- 50,000 примеров, 5 эпох, batch=128
- Финальная полировка

---

## 🌡️ Warm-up Learning Rate

**Постепенное увеличение learning rate при resume**

### Как работает:

```
Эпохи:  1     2     3     4     5+
LR:   0.2x  0.6x  0.8x  1.0x  1.0x
```

### Когда применяется:

- ✅ Автоматически при smart resume
- ✅ Первые 3 эпохи после загрузки checkpoint
- ✅ Помогает избежать "забывания" весов

### Тестирование:

```bash
# Посмотреть как работает warm-up
python warmup_scheduler.py
```

---

## 📈 Стратегии долгого обучения

### 1. 🎯 Автоматическая стратегия (рекомендуется):

```bash
# Запустить на ночь (8 часов)
python automated_training.py --max-hours 8

# Выходные (24 часа)
python automated_training.py --max-hours 24 --mode research
```

### 2. 🔧 Ручная стратегия:

```bash
# Этап 1: Основы (малый датасет, много эпох)
python smart_resume_training.py --dataset-limit 2000 --additional-epochs 20

# Этап 2: Расширение (средний датасет, средние эпохи)
python smart_resume_training.py --dataset-limit 5000 --additional-epochs 15

# Этап 3: Мастерство (большой датасет, мало эпох)
python smart_resume_training.py --dataset-limit 20000 --additional-epochs 10
```

### 3. 💪 Интенсивная стратегия:

```bash
# Сразу большой датасет (если есть время и ресурсы)
python smart_resume_training.py --dataset-limit 50000 --additional-epochs 30 --batch-size 128
```

---

## ⚙️ Параметры

### Общие параметры:

| Параметр          | Описание            | Значения                                      | По умолчанию    |
| ----------------- | ------------------- | --------------------------------------------- | --------------- |
| `--mode`          | Режим конфигурации  | development, research, validation, production | development     |
| `--dataset-limit` | Лимит датасета      | Число (1000-100000)                           | Весь датасет    |
| `--batch-size`    | Размер батча        | 16, 32, 64, 128, 256                          | Из конфига      |
| `--scale`         | Custom scale factor | 0.01-1.0                                      | Зависит от mode |
| `--debug`         | Подробные логи      | Флаг                                          | False           |

### Smart Resume параметры:

| Параметр              | Описание              | Значения |
| --------------------- | --------------------- | -------- |
| `--additional-epochs` | Дополнительные эпохи  | 1-100    |
| `--fixed-sampling`    | Фиксированная выборка | Флаг     |

### Dynamic Training параметры:

| Параметр           | Описание              | Значения |
| ------------------ | --------------------- | -------- |
| `--epochs`         | Общее количество эпох | 1-100    |
| `--fixed-sampling` | Фиксированная выборка | Флаг     |

### Automated Training параметры:

| Параметр        | Описание           | Значения |
| --------------- | ------------------ | -------- |
| `--max-hours`   | Максимальное время | 1.0-24.0 |
| `--test-config` | Показать план      | Флаг     |

---

## 💡 Примеры использования

### Сценарий 1: Первое обучение

```bash
# Начинаем с малого
python smart_resume_training.py --dataset-limit 1000 --additional-epochs 5
```

### Сценарий 2: Продолжение обучения

```bash
# Система автоматически найдет лучший checkpoint
python smart_resume_training.py --dataset-limit 5000 --additional-epochs 10
```

### Сценарий 3: Эксперименты с разными режимами

```bash
# Маленькая решетка
python run_dynamic_training.py --mode development --dataset-limit 3000

# Большая решетка
python run_dynamic_training.py --mode research --dataset-limit 10000
```

### Сценарий 4: Ночное обучение

```bash
# Автоматическое долгое обучение
python automated_training.py --max-hours 8 --mode research

# Посмотреть план перед запуском
python automated_training.py --test-config
```

### Сценарий 5: Отладка

```bash
# Фиксированная выборка для воспроизводимости
python smart_resume_training.py --dataset-limit 500 --additional-epochs 3 --fixed-sampling --debug
```

### Сценарий 6: Максимальная производительность

```bash
# Большая решетка, большой датасет, автоматизация
python automated_training.py --mode production --max-hours 12
```

---

## 🔧 Устранение проблем

### Проблема: "No compatible checkpoints found"

**Решение:**

```bash
# Принудительно начать новое обучение
python smart_resume_training.py --dataset-limit 1000 --additional-epochs 5
```

### Проблема: Out of memory

**Решение:**

```bash
# Уменьшить batch size
python smart_resume_training.py --batch-size 16 --dataset-limit 2000

# Или использовать меньшую решетку
python run_dynamic_training.py --mode development --batch-size 32
```

### Проблема: Очень медленное обучение

**Решение:**

```bash
# Уменьшить размер данных
python smart_resume_training.py --dataset-limit 1000 --additional-epochs 3

# Или использовать автоматизацию с тайм-лимитом
python automated_training.py --max-hours 2
```

### Проблема: Хочу точно те же данные при resume

**Решение:**

```bash
# Использовать фиксированную выборку
python smart_resume_training.py --dataset-limit 5000 --fixed-sampling
```

### Проблема: Модель "забывает" после resume

**Решение:**

- ✅ Warm-up автоматически включается в smart_resume_training.py
- ✅ Начинаем с меньшего learning rate
- ✅ Постепенно увеличиваем до нормального

---

## 📊 Мониторинг результатов

### Файлы логов:

```
logs/main.log                           # Основные логи
logs/automated_training/session_*.json  # Логи автоматического обучения
logs/dynamic_training_*.json           # Логи dynamic обучения
checkpoints/                           # Сохраненные модели
```

### Посмотреть последние результаты:

```bash
# Последние 50 строк основного лога
tail -50 logs/main.log

# Все checkpoints
python -c "from model_weights_manager import ModelWeightsManager; mgr = ModelWeightsManager(); mgr.list_checkpoints()"
```

### Команды для мониторинга:

```bash
# Во время обучения (в другом терминале)
watch -n 30 "tail -10 logs/main.log"

# Посмотреть использование GPU (если есть)
watch -n 5 nvidia-smi
```

---

## 🎯 Рекомендации

### Для быстрого тестирования:

```bash
python smart_resume_training.py --dataset-limit 1000 --additional-epochs 5
```

### Для серьезного обучения:

```bash
python automated_training.py --max-hours 8 --mode research
```

### Для экспериментов:

```bash
python run_dynamic_training.py --mode development --dataset-limit 3000 --fixed-sampling
```

### Для продакшена:

```bash
python automated_training.py --mode production --max-hours 24
```

---

**Успешного обучения! 🚀**
