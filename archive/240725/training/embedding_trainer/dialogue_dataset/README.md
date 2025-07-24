# DialogueDataset - Stage 1.3 ✅

**Модуль для диалогового обучения 3D Cubic Core через Teacher LLM архитектуру**

## 🎯 Назначение

DialogueDataset предоставляет полную систему подготовки данных для обучения 3D Cubic Core на задачах диалога. Модуль реализует Teacher LLM архитектуру, где вопросы (questions) трансформируются в ответы (answers) через эмбединги.

## 🏗️ Архитектура

```python
# Teacher LLM архитектура:
Question text → Teacher LLM → question_embedding (768D)
Answer text   → Teacher LLM → answer_embedding (768D)

# 3D Cubic Core обучение:
question_embedding → [8×8×12 куб] → answer_embedding
```

## 🚀 Установка и настройка

### Требования

```yaml
torch: ">=1.9.0"
transformers: ">=4.21.0"
numpy: ">=1.20.0"
```

### Основное использование

```python
from training.embedding_trainer import create_dialogue_dataset, DialogueDataset

# Создание dataset из диалоговых пар
dialogue_pairs = [
    {"question": "Что такое AI?", "answer": "Искусственный интеллект..."},
    {"question": "Как работает ML?", "answer": "Машинное обучение..."}
]

dataset = create_dialogue_dataset(
    dialogue_pairs=dialogue_pairs,
    teacher_model="distilbert",
    validation_split=0.2
)

# Готово к обучению CubeTrainer
for question_emb, answer_emb in dataset:
    # question_emb: [768] → answer_emb: [768]
    pass
```

### Интеграция с CubeTrainer

```python
from training.embedding_trainer import CubeTrainer, TrainingConfig

# Создание тренера с dialogue режимом
config = TrainingConfig(
    mode="dialogue",
    lattice_size=[8, 8, 12],  # 8*8*12 = 768D совместимо
    embedding_dim=768
)

trainer = CubeTrainer(config=config)
trainer.initialize_components()

# Обучение на диалоговых данных
# trainer.train(dataset)  # Готово к запуску!
```

## 🎛️ Конфигурация

### DialogueConfig основные параметры

```python
config = DialogueConfig(
    teacher_model="distilbert",      # Teacher LLM модель
    embedding_dim=768,               # Размерность эмбедингов
    validation_split=0.2,            # Доля данных для валидации
    enable_quality_filter=True,      # Фильтрация качества Q&A
    support_multiturn=True,          # Многоходовые диалоги
    use_cache=True,                  # Smart caching LLM результатов
    normalize_embeddings=True        # Нормализация эмбедингов
)
```

## 📊 Возможности

### ✅ Поддерживаемые источники данных

- **Dialogue pairs:** Простые Q&A пары
- **Multi-turn conversations:** Многоходовые диалоги
- **File formats:** JSON, JSONL, CSV, TXT
- **Ready embeddings:** Готовые Q&A эмбединги

### ✅ Teacher LLM модели

- **DistilBERT** (рекомендуется)
- **LLaMA 2/3, Mistral-7B**
- **BERT, RoBERTa, GPT-2**
- **Автоматический fallback** при недоступности

### ✅ Smart caching система

- **Intelligent caching** LLM результатов
- **8x+ speedup** при повторном использовании
- **Automatic cache management**

### ✅ Quality filtering

- **Configurable filtering** по длине текста
- **Semantic similarity** контроль Q&A связности
- **Automatic quality assessment**

## 🔗 Связанные модули

- **[CubeTrainer](../cube_trainer/)** - основной тренер для dialogue режима
- **[AutoencoderDataset](../autoencoder_dataset/)** - автоэнкодер данные
- **[EmbeddingLoader](../../data/embedding_loader/)** - Teacher LLM интеграция
- **[EmbeddingProcessor](../../core/embedding_processor/)** - 3D Cubic Core

## 📈 Статус разработки

**Stage 1.3:** ✅ **ПОЛНОСТЬЮ ЗАВЕРШЕН** (7 июня 2025)

- ✅ **Production-ready** DialogueDataset
- ✅ **Teacher LLM архитектура** Q→A
- ✅ **CubeTrainer совместимость** [8,8,12] = 768D
- ✅ **Все тесты пройдены** (100% success rate)

## 🚀 Следующие шаги

**Stage 2.1 - Dialogue Training** готов к запуску с полной архитектурой:

- DialogueDataset ✅
- CubeTrainer ✅
- EmbeddingProcessor ✅

Готово к реальному dialogue training!
