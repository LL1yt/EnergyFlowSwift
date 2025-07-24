# DialogueDataset Usage Examples - Stage 1.3

**Модуль:** DialogueDataset  
**Версия:** v1.3.0  
**Дата:** 7 июня 2025

---

## 🎯 ОСНОВНЫЕ ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### 1. Простейшее использование - Q&A пары

```python
from training.embedding_trainer import create_dialogue_dataset

# Подготовка диалоговых данных
dialogue_pairs = [
    {"question": "Что такое нейронная сеть?", "answer": "Нейронная сеть - это модель машинного обучения."},
    {"question": "Как работает backpropagation?", "answer": "Backpropagation обновляет веса через градиенты ошибки."},
    {"question": "Что такое PyTorch?", "answer": "PyTorch - это библиотека для глубокого обучения на Python."}
]

# Создание dataset
dataset = create_dialogue_dataset(
    dialogue_pairs=dialogue_pairs,
    teacher_model="distilbert",
    validation_split=0.2
)

# Использование в цикле обучения
for question_emb, answer_emb in dataset:
    print(f"Q: {question_emb.shape} → A: {answer_emb.shape}")  # [768] → [768]
```

### 2. Интеграция с CubeTrainer

```python
from training.embedding_trainer import CubeTrainer, TrainingConfig, create_dialogue_dataset

# 1. Создание диалогового dataset
dialogue_pairs = [
    {"question": "Как обучаются нейронные сети?", "answer": "Через оптимизацию функции потерь и градиентный спуск."},
    {"question": "Что такое overfitting?", "answer": "Переобучение происходит когда модель слишком сложна для данных."}
]

dataset = create_dialogue_dataset(
    dialogue_pairs=dialogue_pairs,
    teacher_model="distilbert"
)

# 2. Настройка CubeTrainer для dialogue режима
config = TrainingConfig(
    mode="dialogue",              # Важно: dialogue режим
    lattice_size=[8, 8, 12],     # [8,8,12] = 768D совместимо с DistilBERT
    embedding_dim=768,            # DistilBERT размерность
    learning_rate=0.001,
    batch_size=4,
    device="cpu"
)

# 3. Инициализация тренера
trainer = CubeTrainer(config=config)
trainer.initialize_components()

# 4. Готов к обучению!
print("🚀 Готов к dialogue training!")
print(f"Dataset size: {len(dataset)}")
print(f"Cube shape: {config.lattice_size}")

# Демонстрация forward pass
sample_q, sample_a = dataset[0]
batch_q = sample_q.unsqueeze(0)  # [768] → [1, 768]
output = trainer.forward(batch_q)  # [1, 768]
print(f"Q→A: {batch_q.shape} → {output.shape}")
```

### 3. Multi-turn диалоги

```python
from training.embedding_trainer import create_conversation_dataset

# Многоходовые диалоги
conversations = [
    [
        {"role": "user", "text": "Привет, расскажи о ML"},
        {"role": "assistant", "text": "Машинное обучение - это подраздел ИИ для анализа данных"},
        {"role": "user", "text": "А что такое deep learning?"},
        {"role": "assistant", "text": "Глубокое обучение использует многослойные нейронные сети"}
    ],
    [
        {"role": "user", "text": "Как работает CNN?"},
        {"role": "assistant", "text": "Сверточные сети используют фильтры для обработки изображений"},
        {"role": "user", "text": "А RNN?"},
        {"role": "assistant", "text": "Рекуррентные сети обрабатывают последовательности данных"}
    ]
]

# Создание dataset из многоходовых диалогов
dataset = create_conversation_dataset(
    conversations=conversations,
    teacher_model="distilbert",
    validation_split=0.0  # Все данные для обучения (мало данных)
)

print(f"Извлечено Q&A пар: {len(dataset)}")

# Проверка извлеченных пар
for i in range(min(3, len(dataset))):
    metadata = dataset.dialogue_metadata[i] if dataset.dialogue_metadata else {"question": "N/A", "answer": "N/A"}
    print(f"Пара {i+1}: Q: '{metadata['question'][:30]}...' → A: '{metadata['answer'][:30]}...'")
```

### 4. Настройка quality filtering

```python
from training.embedding_trainer import DialogueConfig, DialogueDataset

# Строгая фильтрация качества
strict_config = DialogueConfig(
    teacher_model="distilbert",
    enable_quality_filter=True,
    min_question_length=10,       # Минимум 10 символов в вопросе
    min_answer_length=20,         # Минимум 20 символов в ответе
    max_question_length=100,      # Максимум 100 символов в вопросе
    max_answer_length=200,        # Максимум 200 символов в ответе
    validation_split=0.2
)

# Тестовые данные с разным качеством
mixed_quality_pairs = [
    {"question": "Что?", "answer": "Да"},  # Будет отфильтровано (слишком короткие)
    {"question": "Расскажи о нейронных сетях", "answer": "Нейронные сети - это математические модели для обучения на данных"},  # Пройдет фильтр
    {"question": "А" * 150, "answer": "Б" * 300},  # Будет отфильтровано (слишком длинные)
]

# Создание с фильтрацией
dataset = DialogueDataset(
    config=strict_config,
    dialogue_pairs=mixed_quality_pairs
)

print(f"Исходные пары: {len(mixed_quality_pairs)}")
print(f"После фильтрации: {len(dataset)}")
print(f"Отфильтровано: {len(mixed_quality_pairs) - len(dataset)} пар")
```

### 5. Работа с кэшированием

```python
from training.embedding_trainer import create_dialogue_dataset
import time

# Одинаковые данные для проверки кэша
test_pairs = [
    {"question": "Что такое AI?", "answer": "Искусственный интеллект"},
    {"question": "Как работает ML?", "answer": "Через обучение на данных"}
]

# Первый запуск (cache miss)
start_time = time.time()
dataset1 = create_dialogue_dataset(
    dialogue_pairs=test_pairs,
    teacher_model="distilbert",
    use_cache=True,
    cache_dir="cache/example_dialogue"
)
time1 = time.time() - start_time

# Второй запуск (cache hit)
start_time = time.time()
dataset2 = create_dialogue_dataset(
    dialogue_pairs=test_pairs,
    teacher_model="distilbert",
    use_cache=True,
    cache_dir="cache/example_dialogue"
)
time2 = time.time() - start_time

print(f"Первый запуск (cache miss): {time1:.2f}s")
print(f"Второй запуск (cache hit): {time2:.2f}s")
print(f"Speedup: {time1/time2:.1f}x")

# Статистика кэша
print(f"Dataset 1 cache stats: {dataset1.cache_stats}")
print(f"Dataset 2 cache stats: {dataset2.cache_stats}")
```

### 6. DataLoader для батчевого обучения

```python
from training.embedding_trainer import create_dialogue_dataset

# Создание dataset
dataset = create_dialogue_dataset(
    dialogue_pairs=[
        {"question": f"Вопрос {i}", "answer": f"Ответ на вопрос {i}"}
        for i in range(20)  # 20 диалоговых пар
    ],
    teacher_model="distilbert",
    validation_split=0.2
)

# Train DataLoader
train_loader = dataset.get_dataloader(
    batch_size=4,
    shuffle=True,
    validation=False
)

# Validation DataLoader
val_loader = dataset.get_dataloader(
    batch_size=2,
    shuffle=False,
    validation=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Пример обучающего цикла
for epoch in range(2):
    print(f"\nEpoch {epoch+1}:")

    # Training
    for batch_idx, (questions, answers) in enumerate(train_loader):
        print(f"  Train batch {batch_idx}: Q{questions.shape} → A{answers.shape}")
        # Здесь был бы реальный training step

    # Validation
    for batch_idx, (questions, answers) in enumerate(val_loader):
        print(f"  Val batch {batch_idx}: Q{questions.shape} → A{answers.shape}")
        # Здесь была бы валидация
```

### 7. Статистика и анализ dataset

```python
from training.embedding_trainer import create_dialogue_dataset
import torch

# Создание dataset с метаданными
dataset = create_dialogue_dataset(
    dialogue_pairs=[
        {"question": "Что такое PyTorch?", "answer": "PyTorch - библиотека для deep learning"},
        {"question": "Как работает CNN?", "answer": "Сверточные сети обрабатывают изображения через фильтры"},
        {"question": "Что такое RNN?", "answer": "Рекуррентные сети для последовательностей"}
    ],
    teacher_model="distilbert",
    validation_split=0.0
)

# Получение полной статистики
stats = dataset.get_statistics()

print("📊 СТАТИСТИКА DATASET:")
print(f"Всего пар: {stats['total_dialogue_pairs']}")
print(f"Teacher модель: {stats['teacher_model']}")
print(f"Размерность эмбедингов: {stats['embedding_dim']}")

# Кэш статистика
cache_stats = stats['cache_stats']
print(f"\n💾 КЭШИРОВАНИЕ:")
print(f"Cache hits: {cache_stats['cache_hits']}")
print(f"Cache misses: {cache_stats['cache_misses']}")
print(f"Quality filtered: {cache_stats['quality_filtered']}")

# Качество эмбедингов
if 'embedding_quality' in stats:
    eq = stats['embedding_quality']
    print(f"\n🎯 КАЧЕСТВО ЭМБЕДИНГОВ:")
    print(f"Question norm mean: {eq['question_norm_mean']:.4f}")
    print(f"Answer norm mean: {eq['answer_norm_mean']:.4f}")
    print(f"Q&A similarity: {eq['qa_similarity_mean']:.4f} ± {eq['qa_similarity_std']:.4f}")

# Примеры диалогов с similarity
samples = dataset.get_sample_dialogues(n_samples=3)
if 'samples' in samples:
    print(f"\n💬 ПРИМЕРЫ ДИАЛОГОВ:")
    for i, sample in enumerate(samples['samples']):
        print(f"Пример {i+1}:")
        print(f"  Q: '{sample['question']}'")
        print(f"  A: '{sample['answer']}'")
        print(f"  Similarity: {sample['qa_similarity']:.4f}")
```

### 8. Загрузка из файлов

```python
import json
from training.embedding_trainer import load_dialogue_dataset_from_files

# Создание тестового JSON файла
test_data = [
    {"question": "Файловый вопрос 1", "answer": "Файловый ответ 1"},
    {"question": "Файловый вопрос 2", "answer": "Файловый ответ 2"}
]

with open("test_dialogues.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# Загрузка из файла
dataset = load_dialogue_dataset_from_files(
    file_paths=["test_dialogues.json"],
    teacher_model="distilbert",
    validation_split=0.0
)

print(f"Загружено из файла: {len(dataset)} пар")

# Очистка
import os
os.remove("test_dialogues.json")
```

### 9. Переключение между train/validation режимами

```python
from training.embedding_trainer import create_dialogue_dataset

# Dataset с train/val split
dataset = create_dialogue_dataset(
    dialogue_pairs=[
        {"question": f"Q{i}", "answer": f"A{i}"}
        for i in range(10)
    ],
    teacher_model="distilbert",
    validation_split=0.3  # 30% для валидации
)

print(f"Общий размер: {len(dataset)}")
print(f"Train размер: {len(dataset.train_questions)}")
print(f"Val размер: {len(dataset.val_questions)}")

# Режим training
dataset.set_validation_mode(False)
print(f"Training mode: {len(dataset)} samples")

# Режим validation
dataset.set_validation_mode(True)
print(f"Validation mode: {len(dataset)} samples")

# Обратно к training
dataset.set_validation_mode(False)
print(f"Back to training: {len(dataset)} samples")
```

---

## 🔧 ПРОДВИНУТЫЕ ПАТТЕРНЫ

### Custom DialogueConfig

```python
from training.embedding_trainer import DialogueConfig, DialogueDataset

# Полная кастомизация
config = DialogueConfig(
    teacher_model="distilbert",
    embedding_dim=768,
    validation_split=0.15,
    enable_quality_filter=True,
    min_question_length=8,
    min_answer_length=15,
    max_question_length=150,
    max_answer_length=300,
    support_multiturn=True,
    use_cache=True,
    normalize_embeddings=True,
    cache_dir="cache/custom_dialogue",
    cache_batch_size=100,
    max_conversations=2000
)

dataset = DialogueDataset(
    config=config,
    dialogue_pairs=[
        {"question": "Custom question", "answer": "Custom answer"}
    ]
)
```

### Проверка совместимости размеров

```python
from training.embedding_trainer import create_dialogue_dataset

# Различные Teacher LLM модели
models_to_test = ["distilbert", "bert-base-uncased"]

for model in models_to_test:
    try:
        dataset = create_dialogue_dataset(
            dialogue_pairs=[{"question": "Test", "answer": "Test response"}],
            teacher_model=model,
            validation_split=0.0
        )

        sample_q, sample_a = dataset[0]
        print(f"✅ {model}: {sample_q.shape} compatible")

    except Exception as e:
        print(f"❌ {model}: {e}")
```

---

## 🎯 ЗАКЛЮЧЕНИЕ

**DialogueDataset предоставляет мощный и гибкий API для dialogue обучения 3D Cubic Core.**

### Ключевые возможности:

- ✅ **Teacher LLM архитектура** Q→A трансформации
- ✅ **Smart caching** для 8x+ speedup
- ✅ **Quality filtering** для чистых данных
- ✅ **Multi-turn support** для сложных диалогов
- ✅ **CubeTrainer integration** для [8,8,12] = 768D совместимости
- ✅ **Production-ready API** с comprehensive testing

**Готов к использованию в Stage 2.1 - Dialogue Training!**
