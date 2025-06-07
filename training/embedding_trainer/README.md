# Embedding Trainer ✅ Stage 2.1 DIALOGUE TRAINING ЗАВЕРШЕН!

**Назначение:** Модуль для обучения 3D Cubic Core (Модуль 2) на эмбединг→эмбединг трансформациях

## 🎉 Breakthrough Milestone: DIALOGUE TRAINING FUNCTIONAL!

**Stage 2.1 успешно завершен (7 июня 2025)** - полный dialogue training pipeline работает!
3D Cubic Core научился обрабатывать Q→A трансформации через Teacher LLM Knowledge Distillation.

**Завершенные этапы:**

- ✅ **Stage 1.1** - CubeTrainer (8/8 тестов)
- ✅ **Stage 1.2** - AutoencoderDataset (10/10 тестов)
- ✅ **Stage 1.3** - DialogueDataset (ALL тестов пройдено) ⭐
- ✅ **Stage 2.1** - Dialogue Training Execution (FUNCTIONAL) ⭐ NEW!

## Обзор

EmbeddingTrainer специализируется на обучении центрального процессора системы - 3D Cubic Core. Модуль поддерживает два основных режима обучения:

1. **Autoencoder Mode** - обучение на восстановление исходных эмбедингов
2. **Dialogue Mode** - обучение на диалоговых парах (вопрос→ответ)

## Архитектура

```
Входной эмбединг → EmbeddingReshaper → 3D Cubic Core → EmbeddingReshaper → Выходной эмбединг
     (768D)              (8×8×12)         (процессор)        (8×8×12)           (768D)
```

## Основные компоненты

- ✅ **`CubeTrainer`** - основной класс для обучения куба (ЗАВЕРШЕН!)
- ✅ **`TrainingConfig`** - система конфигурации (ЗАВЕРШЕНА!)
- ✅ **`EmbeddingMetrics`** - метрики качества обучения (ЗАВЕРШЕНЫ!)
- ✅ **`AutoencoderDataset`** - датасет для autoencoder задач (ЗАВЕРШЕН!)
- ✅ **`DatasetConfig`** - конфигурация dataset'ов (ЗАВЕРШЕНА!)
- ✅ **`create_text_dataset`** - удобная функция для создания из текстов (ГОТОВА!)
- ✅ **`create_file_dataset`** - удобная функция для создания из файлов (ГОТОВА!)
- ✅ **`DialogueDataset`** - датасет для диалоговых задач (ЗАВЕРШЕН!) ⭐
- ✅ **`create_dialogue_dataset`** - функция для Q&A данных (ГОТОВА!) ⭐
- ✅ **`run_dialogue_training.py`** - полный dialogue training pipeline (FUNCTIONAL!) ⭐ NEW!

## Быстрый старт

### 1. AutoencoderDataset ✅ Готов к использованию!

```python
from training.embedding_trainer import (
    AutoencoderDataset,
    create_text_dataset,
    create_file_dataset
)

# Создание dataset из текстов ⭐ NEW!
texts = [
    "Machine learning is transforming the world",
    "Neural networks can learn complex patterns",
    "Deep learning enables amazing applications"
]

dataset = create_text_dataset(
    texts=texts,
    llm_model="distilbert",  # Поддержка 8+ LLM моделей
    validation_split=0.2,
    use_cache=True,          # Smart caching
    normalize_embeddings=True
)

# Создание DataLoaders для обучения
train_loader = dataset.get_dataloader(batch_size=32, validation=False)
val_loader = dataset.get_dataloader(batch_size=32, validation=True)

print(f"Dataset готов: {dataset}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Тестирование autoencoder format
for input_emb, target_emb in train_loader:
    print(f"Batch shapes: {input_emb.shape} -> {target_emb.shape}")
    break  # input_emb == target_emb для autoencoder режима
```

### 2. CubeTrainer ✅ Готов к обучению!

```python
from training.embedding_trainer import CubeTrainer, TrainingConfig

# Создание конфигурации
config = TrainingConfig(
    mode="autoencoder",  # autoencoder | dialogue | mixed
    lattice_size=[8, 8, 8],
    learning_rate=0.001,
    epochs=50,
    batch_size=32
)

# Создание тренера ✅ РАБОТАЕТ!
trainer = CubeTrainer(config=config)

# Инициализация компонентов
trainer.initialize_components()

# Получение информации о тренере
info = trainer.get_info()
print(f"Режим: {info['mode']}")
print(f"Компоненты готовы: {info['components_initialized']}")

# Forward pass (готов после инициализации)
# output = trainer.forward(input_embedding)
```

## Требования

- Модуль 1 (Teacher LLM Encoder) должен быть готов
- Модуль 2 (EmbeddingReshaper + EmbeddingProcessor) должен быть настроен
- Обучающие данные в формате эмбедингов

## Конфигурация

Основные настройки в `config/cube_training.yaml`:

```yaml
cube_training:
  mode: "autoencoder" # autoencoder | dialogue
  lattice_size: [8, 8, 8]
  learning_rate: 0.001
  epochs: 50
  batch_size: 32
  convergence_threshold: 0.001
  target_similarity: 0.90
```

## Связанные модули

- `core/embedding_processor/` - основной процессор для обучения
- `data/embedding_loader/` - источник обучающих эмбедингов
- `data/embedding_reshaper/` - конвертация форматов
- `evaluation/embedding_metrics/` - детальная оценка качества

## Документация

- `plan.md` - детальный план реализации
- `meta.md` - технические характеристики
- `examples.md` - примеры использования
- `diagram.mmd` - архитектурная диаграмма
