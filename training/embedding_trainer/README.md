# Embedding Trainer ✅ Stage 1.1 ЗАВЕРШЕН!

**Назначение:** Модуль для обучения 3D Cubic Core (Модуль 2) на эмбединг→эмбединг трансформациях

## 🎉 Major Milestone: CubeTrainer ГОТОВ!

**Stage 1.1 успешно завершен (6 июня 2025)** - все 8 тестов пройдены!
CubeTrainer полностью функционален и готов к обучению 3D Cubic Core.

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
- 🚀 **`AutoencoderDataset`** - датасет для autoencoder задач (Stage 1.2)
- 💬 **`DialogueDataset`** - датасет для диалоговых задач (Stage 1.3)

## Быстрый старт

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
