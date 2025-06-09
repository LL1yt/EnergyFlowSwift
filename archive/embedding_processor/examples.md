# Примеры Использования: EmbeddingProcessor

**Модуль:** `core/embedding_processor/`  
**Версия:** 2.5.0  
**Статус:** ✅ Production Ready

---

## 🔧 Базовая Установка

### Импорты

```python
import torch
import numpy as np
from core.embedding_processor import (
    EmbeddingProcessor,
    EmbeddingConfig,
    ProcessingMode,
    ProcessingMetrics,
    create_autoencoder_config,
    create_generator_config,
    create_dialogue_config,
    validate_embedding_input,
    benchmark_processor,
    generate_quality_report
)
```

### Создание тестового эмбединга

```python
def create_test_embedding():
    """Создает тестовый эмбединг 768 размерности"""
    return torch.randn(768, dtype=torch.float32)

def create_test_batch(batch_size=4):
    """Создает batch тестовых эмбедингов"""
    return torch.randn(batch_size, 768, dtype=torch.float32)
```

---

## 🔄 Пример 1: Автоэнкодер Режим

**Цель:** Точное воспроизведение входного эмбединга

```python
# Создание конфигурации автоэнкодера
config = create_autoencoder_config()
processor = EmbeddingProcessor(config)

# Обработка одного эмбединга
input_embedding = create_test_embedding()
output_embedding = processor.process(input_embedding)

# Проверка качества
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(
    input_embedding.unsqueeze(0),
    output_embedding.unsqueeze(0)
)[0, 0]

print(f"Cosine Similarity: {similarity:.4f}")
print(f"Target: ≥{config.similarity_targets[ProcessingMode.AUTOENCODER]:.2f}")
print(f"✅ Успех!" if similarity >= 0.95 else "❌ Ниже цели")

# Ожидаемый результат: Cosine Similarity: 0.9990+
```

---

## 🎲 Пример 2: Генераторный Режим

**Цель:** Семантическая генерация новых эмбедингов

```python
# Создание конфигурации генератора
config = create_generator_config()
processor = EmbeddingProcessor(config)

# Генерация от базового эмбединга
seed_embedding = create_test_embedding()
generated_embedding = processor.process(seed_embedding)

# Проверка семантической близости (но не идентичности)
similarity = cosine_similarity(
    seed_embedding.unsqueeze(0),
    generated_embedding.unsqueeze(0)
)[0, 0]

print(f"Semantic Similarity: {similarity:.4f}")
print(f"Target Range: 0.80-0.90 (семантически связан, но уникален)")
print(f"✅ В диапазоне!" if 0.80 <= similarity <= 0.90 else "⚠️ Вне диапазона")

# Проверка на неидентичность
if not torch.equal(seed_embedding, generated_embedding):
    print("✅ Генерация уникальна (не копия)")
else:
    print("❌ Идентичен входу (не генерация)")
```

---

## 💬 Пример 3: Диалоговый Режим

**Цель:** Преобразование вопрос→ответ

```python
# Создание диалоговой конфигурации
config = create_dialogue_config()
processor = EmbeddingProcessor(config)

# Симуляция диалогового эмбединга
question_embedding = create_test_embedding()
answer_embedding = processor.process(question_embedding)

# Анализ релевантности
semantic_similarity = cosine_similarity(
    question_embedding.unsqueeze(0),
    answer_embedding.unsqueeze(0)
)[0, 0]

print(f"Semantic Similarity: {semantic_similarity:.4f}")
print(f"Target: ≥{config.similarity_targets[ProcessingMode.DIALOGUE]:.2f}")

# Эксперимент с различными типами вопросов
question_types = ["technical", "creative", "analytical"]
for q_type in question_types:
    test_q = create_test_embedding()  # В реальности загружаем по типу
    test_a = processor.process(test_q)
    sim = cosine_similarity(test_q.unsqueeze(0), test_a.unsqueeze(0))[0, 0]
    print(f"{q_type.capitalize()} Q→A similarity: {sim:.3f}")
```

---

## 📦 Пример 4: Batch Processing

**Цель:** Эффективная обработка множественных эмбедингов

```python
# Создание batch конфигурации
config = create_autoencoder_config()
processor = EmbeddingProcessor(config)

# Batch данные
batch_size = 8
input_batch = create_test_batch(batch_size)
print(f"Input shape: {input_batch.shape}")

# Batch обработка
output_batch = processor.process_batch(input_batch)
print(f"Output shape: {output_batch.shape}")

# Проверка качества для каждого элемента
batch_similarities = []
for i in range(batch_size):
    sim = cosine_similarity(
        input_batch[i:i+1],
        output_batch[i:i+1]
    )[0, 0]
    batch_similarities.append(sim)

avg_similarity = np.mean(batch_similarities)
print(f"Average Batch Similarity: {avg_similarity:.4f}")
print(f"Min Similarity: {min(batch_similarities):.4f}")
print(f"Max Similarity: {max(batch_similarities):.4f}")

# Ожидаемый результат: Все similarities >0.995
```

---

## 📊 Пример 5: Мониторинг Метрик

**Цель:** Сбор и анализ метрик производительности

```python
# Создание процессора с метриками
config = create_autoencoder_config()
processor = EmbeddingProcessor(config)

# Несколько обработок для сбора статистики
test_embeddings = [create_test_embedding() for _ in range(10)]

processing_times = []
similarities = []

for embedding in test_embeddings:
    start_time = time.time()
    output = processor.process(embedding)
    processing_time = time.time() - start_time

    similarity = cosine_similarity(
        embedding.unsqueeze(0),
        output.unsqueeze(0)
    )[0, 0]

    processing_times.append(processing_time)
    similarities.append(similarity)

# Получение метрик
metrics = processor.get_metrics()
print("📊 Performance Metrics:")
print(f"Average Processing Time: {np.mean(processing_times)*1000:.2f}ms")
print(f"Average Similarity: {np.mean(similarities):.4f}")
print(f"Processing Count: {metrics.processing_count}")
print(f"Total Time: {metrics.total_processing_time:.2f}s")

# Генерация отчета о качестве
report = generate_quality_report(processor, test_embeddings)
print("\n📋 Quality Report:")
print(report)
```

---

## 🔧 Пример 6: Кастомная Конфигурация

**Цель:** Создание собственной конфигурации процессора

```python
# Создание кастомной конфигурации
custom_config = EmbeddingConfig(
    input_dim=768,
    cube_shape=(8, 8, 8),
    output_dim=768,
    processing_mode=ProcessingMode.GENERATOR,

    # Параметры решетки
    lattice_propagation_steps=15,  # Больше шагов для лучшего качества
    lattice_convergence_threshold=0.0005,  # Строже сходимость

    # Параметры EmbeddingReshaper
    reshaping_method="adaptive",
    preserve_semantics=True,
    semantic_threshold=0.98,  # Выше порог семантики

    # Кастомные цели
    similarity_targets={
        ProcessingMode.AUTOENCODER: 0.99,
        ProcessingMode.GENERATOR: 0.87,
        ProcessingMode.DIALOGUE: 0.82
    }
)

# Использование кастомной конфигурации
processor = EmbeddingProcessor(custom_config)
embedding = create_test_embedding()
result = processor.process(embedding)

print(f"Custom config processing completed")
print(f"Propagation steps used: {custom_config.lattice_propagation_steps}")
print(f"Semantic threshold: {custom_config.semantic_threshold}")
```

---

## ⚡ Пример 7: Бенчмарк Производительности

**Цель:** Измерение производительности и сравнение конфигураций

```python
# Конфигурации для сравнения
configs = {
    "autoencoder": create_autoencoder_config(),
    "generator": create_generator_config(),
    "dialogue": create_dialogue_config()
}

# Тестовые данные
test_data = [create_test_embedding() for _ in range(20)]

# Бенчмарк каждой конфигурации
results = {}
for name, config in configs.items():
    print(f"\n🔄 Benchmarking {name} configuration...")

    processor = EmbeddingProcessor(config)
    result = benchmark_processor(processor, test_data)
    results[name] = result

    print(f"Average time: {result['avg_time']*1000:.2f}ms")
    print(f"Average similarity: {result['avg_similarity']:.4f}")
    print(f"Throughput: {result['throughput']:.1f} embeddings/sec")

# Сравнение результатов
print("\n📊 Performance Comparison:")
for name, result in results.items():
    print(f"{name:12}: {result['avg_time']*1000:6.2f}ms | "
          f"Similarity: {result['avg_similarity']:.4f} | "
          f"Throughput: {result['throughput']:5.1f}/sec")
```

---

## 🧪 Пример 8: Валидация и Error Handling

**Цель:** Правильная обработка ошибок и валидация данных

```python
# Тест валидации входных данных
def test_validation():
    config = create_autoencoder_config()
    processor = EmbeddingProcessor(config)

    # Правильный эмбединг
    valid_embedding = create_test_embedding()
    is_valid, message = validate_embedding_input(valid_embedding, config)
    print(f"Valid embedding: {is_valid} - {message}")

    # Неправильная размерность
    invalid_embedding = torch.randn(512)  # Неправильный размер
    is_valid, message = validate_embedding_input(invalid_embedding, config)
    print(f"Invalid embedding: {is_valid} - {message}")

    # Неправильный тип
    try:
        result = processor.process("not_a_tensor")
    except Exception as e:
        print(f"Type error handled: {type(e).__name__}: {e}")

    # NaN значения
    nan_embedding = torch.full((768,), float('nan'))
    is_valid, message = validate_embedding_input(nan_embedding, config)
    print(f"NaN embedding: {is_valid} - {message}")

test_validation()
```

---

## 🔄 Пример 4: Surface-Only Режим (Universal Adapter)

**Цель:** Обработка surface embeddings без full cube reshaping

```python
# Создание surface-only конфигурации
from core.embedding_processor import create_surface_only_config

config = create_surface_only_config(
    surface_size=225,      # 15×15 surface
    surface_dims=(15, 15)  # Surface dimensions
)
processor = EmbeddingProcessor(config)

# Симуляция surface embeddings от Universal Adapter
surface_embedding = torch.randn(225, dtype=torch.float32)  # 15×15 = 225D

# Обработка через emergent surface processing
processed_surface = processor.forward(surface_embedding)

print(f"Input surface shape: {surface_embedding.shape}")
print(f"Output surface shape: {processed_surface.shape}")

# Проверка качества обработки
similarity = torch.cosine_similarity(
    surface_embedding,
    processed_surface,
    dim=0
).item()

print(f"Surface preservation: {similarity:.4f}")
print(f"Target: ≥{config.target_similarity:.2f}")

# Batch processing тест
batch_surfaces = torch.randn(4, 225, dtype=torch.float32)
batch_processed = processor.forward(batch_surfaces)

print(f"\nBatch processing:")
print(f"Input batch: {batch_surfaces.shape}")
print(f"Output batch: {batch_processed.shape}")

# Quality analysis для каждого примера в batch
for i in range(4):
    sim = torch.cosine_similarity(
        batch_surfaces[i],
        batch_processed[i],
        dim=0
    ).item()
    print(f"Batch item {i}: similarity = {sim:.3f}")
```

---

## 💡 Советы по Использованию

### Оптимизация Производительности

```python
# 1. Используйте batch processing для множественных эмбедингов
batch_results = processor.process_batch(embeddings_batch)

# 2. Кэшируйте процессор для повторного использования
global_processor = EmbeddingProcessor(create_autoencoder_config())

# 3. Валидируйте данные заранее
if validate_embedding_input(embedding, config)[0]:
    result = processor.process(embedding)
```

### Выбор Режима

```python
# AUTOENCODER: для точного воспроизведения
# - Семантические эмбединги для поиска
# - Компрессия с восстановлением

# GENERATOR: для креативной генерации
# - Создание вариаций контента
# - Augmentation данных

# DIALOGUE: для диалоговых систем
# - Q&A боты
# - Контекстные ответы
```

**Модуль готов к production использованию! ✅**
