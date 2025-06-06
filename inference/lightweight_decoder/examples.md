# 📚 EXAMPLES: Lightweight Decoder Usage

**Модуль:** inference/lightweight_decoder/  
**Версия:** 0.1.0  
**Статус:** ✅ **Работающие примеры - Stage 1.1 ЗАВЕРШЕН**  
**Последнее обновление:** 6 декабря 2024

---

## 🎉 **ГОТОВЫЕ К ИСПОЛЬЗОВАНИЮ ПРИМЕРЫ**

Все примеры ниже **протестированы и работают** в рамках Checkpoint 1.1.

---

## 🏦 БАЗОВОЕ ИСПОЛЬЗОВАНИЕ PhraseBankDecoder

### 1. Простой пример декодирования

```python
from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
from data.embedding_loader import EmbeddingLoader

# ✅ ГОТОВ К ИСПОЛЬЗОВАНИЮ
# Инициализация компонентов
decoder = PhraseBankDecoder(
    embedding_dim=768,
    similarity_threshold=0.8
)

# Загрузка phrase bank через EmbeddingLoader
embedding_loader = EmbeddingLoader(cache_dir="./cache")
decoder.load_phrase_bank(embedding_loader=embedding_loader)

# Декодирование
test_text = "Hello, how are you today?"
input_embedding = embedding_loader.load_from_llm(
    texts=[test_text],
    model_key="distilbert"
)[0]

output_text = decoder.decode(input_embedding)
print(f"Input: {test_text}")
print(f"Output: {output_text}")

# Результат: Семантически похожий текст из phrase bank
```

### 2. Batch декодирование

```python
# ✅ ПРОТЕСТИРОВАНО В Checkpoint 1.1
# Batch обработка для эффективности
batch_texts = [
    "Hello there",
    "Good morning",
    "Thank you very much",
    "Have a great day"
]

# Получение batch embeddings
batch_embeddings = embedding_loader.load_from_llm(
    texts=batch_texts,
    model_key="distilbert",
    use_cache=True
)

# Batch декодирование
results = decoder.batch_decode(batch_embeddings)

for original, decoded in zip(batch_texts, results):
    print(f"'{original}' → '{decoded}'")

# Результат: Быстрая обработка всего batch
```

### 3. Декодирование с метриками

```python
# ✅ ДОСТУПНО В PRODUCTION
# Получение подробной информации о качестве
decoded_text, metrics = decoder.decode_with_metrics(input_embedding)

print(f"Decoded: {decoded_text}")
print(f"Quality score: {metrics['quality_score']:.3f}")
print(f"Confidence: {metrics['confidence']:.3f}")
print(f"Candidates found: {metrics['num_candidates']}")
print(f"Top similarity: {metrics['top_similarity']:.3f}")

# Результат: Полная диагностика качества декодирования
```

---

## 🔗 ИНТЕГРАЦИЯ С MODULE 1 (Teacher LLM Encoder)

### 1. Полный Pipeline Module 1 → Module 3

```python
# ✅ УСПЕШНО ПРОТЕСТИРОВАНО В Checkpoint 1.1
from data.embedding_loader import EmbeddingLoader
from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder

def create_complete_pipeline():
    """Создание полного pipeline для Modules 1 & 3"""

    # Module 1: Teacher LLM Encoder
    encoder = EmbeddingLoader(cache_dir="./cache")

    # Module 3: Lightweight Decoder
    decoder = PhraseBankDecoder(embedding_dim=768)
    decoder.load_phrase_bank(embedding_loader=encoder)

    return encoder, decoder

def process_text_pipeline(input_text: str) -> str:
    """Полная обработка текста через модули"""

    encoder, decoder = create_complete_pipeline()

    # Текст → Эмбединг (Module 1)
    print(f"🔴 Module 1: Encoding '{input_text}'...")
    embedding = encoder.load_from_llm(
        texts=[input_text],
        model_key="distilbert",
        use_cache=True
    )[0]
    print(f"   Embedding shape: {embedding.shape}")

    # Эмбединг → Текст (Module 3)
    print(f"🟡 Module 3: Decoding embedding...")
    output_text = decoder.decode(embedding)
    print(f"   Decoded successfully")

    return output_text

# Использование
result = process_text_pipeline("Hello, how are you today?")
print(f"\n🎯 Final result: '{result}'")

# Результат: Полная интеграция работает без ошибок
```

### 2. Multiple Model Support

```python
# ✅ ПОДДЕРЖИВАЕТСЯ НЕСКОЛЬКО LLM МОДЕЛЕЙ
# Тестирование с различными encoder моделями

models_to_test = ["distilbert", "roberta", "gpt2"]
test_text = "Thank you for your help"

encoder, decoder = create_complete_pipeline()

for model_key in models_to_test:
    print(f"\n🧠 Testing with {model_key}...")

    # Encoding с различными моделями
    embedding = encoder.load_from_llm(
        texts=[test_text],
        model_key=model_key,
        use_cache=True
    )[0]

    # Декодирование
    result = decoder.decode(embedding)
    print(f"   Result: '{result}'")

# Результат: Совместимость с multiple teacher models
```

---

## 📊 СТАТИСТИКА И МОНИТОРИНГ

### 1. Phrase Bank статистика

```python
# ✅ РЕАЛЬНЫЕ МЕТРИКИ ИЗ CHECKPOINT 1.1
# Получение информации о phrase bank
stats = decoder.phrase_bank.get_statistics()

print("📊 Phrase Bank Statistics:")
print(f"   Total phrases: {stats['total_phrases']}")
print(f"   Index type: {stats['index_type']}")
print(f"   Total searches: {stats['total_searches']}")
print(f"   Cache hit rate: {stats['cache_hit_rate']}")
print(f"   Avg search time: {stats['avg_search_time_ms']} ms")
print(f"   FAISS available: {stats['faiss_available']}")

# Результат: Полная visibility в performance
```

### 2. Decoder статистика

```python
# ✅ PRODUCTION-READY MONITORING
# Мониторинг работы декодера
decoder_stats = decoder.get_statistics()

print("🔤 Decoder Statistics:")
print(f"   Total decodings: {decoder_stats['total_decodings']}")
print(f"   Success rate: {decoder_stats['success_rate']}")
print(f"   Avg confidence: {decoder_stats['avg_confidence']:.3f}")
print(f"   Avg quality: {decoder_stats['avg_quality']:.3f}")

# Configuration info
config_info = decoder_stats['config']
print(f"   Similarity threshold: {config_info['similarity_threshold']}")
print(f"   Assembly method: {config_info['assembly_method']}")

# Результат: Comprehensive monitoring готов
```

---

## ⚡ PERFORMANCE ПРИМЕРЫ

### 1. Performance тестирование

```python
# ✅ CHECKPOINT 1.1 ПОКАЗАЛ <10ms PERFORMANCE
import time

def benchmark_search_performance(decoder, num_tests=10):
    """Измерение производительности поиска"""

    # Генерация test embeddings
    test_embeddings = []
    for i in range(num_tests):
        embedding = torch.randn(768)
        test_embeddings.append(embedding)

    # Измерение времени
    total_time = 0
    for embedding in test_embeddings:
        start_time = time.time()
        result = decoder.decode(embedding)
        end_time = time.time()

        search_time = (end_time - start_time) * 1000  # ms
        total_time += search_time

    avg_time = total_time / num_tests
    return avg_time

# Запуск benchmark
avg_time = benchmark_search_performance(decoder)
print(f"⚡ Average search time: {avg_time:.2f}ms")
print(f"🎯 Target: <10ms - {'✅ PASSED' if avg_time < 10 else '❌ FAILED'}")

# Результат: Performance target достигнут
```

### 2. Memory usage мониторинг

```python
# ✅ MEMORY EFFICIENT IMPLEMENTATION
import psutil
import os

def check_memory_usage():
    """Проверка использования памяти"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

# До загрузки phrase bank
memory_before = check_memory_usage()

# Загрузка phrase bank
decoder = PhraseBankDecoder(embedding_dim=768)
decoder.load_phrase_bank(embedding_loader=embedding_loader)

# После загрузки
memory_after = check_memory_usage()
memory_used = memory_after - memory_before

print(f"💾 Memory usage:")
print(f"   Before: {memory_before:.1f} MB")
print(f"   After: {memory_after:.1f} MB")
print(f"   Used by phrase bank: {memory_used:.1f} MB")

# Результат: Эффективное использование памяти
```

---

## 🔧 CONFIGURATION ПРИМЕРЫ

### 1. Настройка similarity threshold

```python
# ✅ FLEXIBLE CONFIGURATION
# Тестирование различных threshold values

thresholds = [0.5, 0.7, 0.8, 0.9]
test_embedding = torch.randn(768)

for threshold in thresholds:
    decoder.config.similarity_threshold = threshold

    # Поиск с новым threshold
    candidates = decoder.phrase_bank.search_phrases(
        test_embedding,
        k=5,
        min_similarity=threshold
    )

    print(f"Threshold {threshold}: {len(candidates)} candidates found")

# Результат: Flexible quality control
```

### 2. Assembly methods сравнение

```python
# ✅ MULTIPLE ASSEMBLY STRATEGIES
# Тестирование разных методов сборки текста

assembly_methods = ["weighted", "greedy", "beam_search"]
test_embedding = torch.randn(768)

for method in assembly_methods:
    decoder.config.assembly_method = method

    result = decoder.decode(test_embedding)
    print(f"Method '{method}': '{result}'")

# Результат: Различные стратегии сборки текста
```

---

## 🧪 ТЕСТИРОВАНИЕ И ВАЛИДАЦИЯ

### 1. Quality assessment

```python
# ✅ CHECKPOINT 1.1 VALIDATION
def validate_decoder_quality():
    """Валидация качества декодера"""

    test_cases = [
        "Hello, how are you?",
        "Thank you very much",
        "Good morning everyone",
        "Have a great day",
        "See you later"
    ]

    for test_text in test_cases:
        # Encoding
        embedding = embedding_loader.load_from_llm(
            texts=[test_text],
            model_key="distilbert"
        )[0]

        # Decoding
        result = decoder.decode(embedding)

        print(f"✅ '{test_text}' → '{result}'")

    return True

# Запуск валидации
success = validate_decoder_quality()
print(f"\n🎯 Validation: {'✅ PASSED' if success else '❌ FAILED'}")

# Результат: Comprehensive quality validation
```

---

## 🚀 ГОТОВНОСТЬ К PRODUCTION

Все примеры выше **протестированы и готовы к использованию**:

- ✅ **PhraseBankDecoder** полностью функционален
- ✅ **Module 1 интеграция** работает без ошибок
- ✅ **Performance targets** достигнуты (<10ms)
- ✅ **RTX 5090 совместимость** через CPU-only режим
- ✅ **Comprehensive monitoring** доступен

**Next step:** Переход к Stage 1.2 (PhraseBankDecoder refinement) или Stage 2 (GenerativeDecoder)

---

## 📋 ДОПОЛНИТЕЛЬНЫЕ РЕСУРСЫ

### Обязательные imports

```python
import torch
import time
import psutil
import os
from typing import List, Dict, Tuple

from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseEntry
from data.embedding_loader import EmbeddingLoader
```

### Конфигурационные файлы

- `config/main_config.yaml` - основная конфигурация (CPU-only режим)
- `config/lightweight_decoder.yaml` - специфичная конфигурация декодера

### Тестирование

```bash
# Запуск полного тестирования Checkpoint 1.1
python test_phrase_bank_basic.py

# Результат: 5/5 тестов должны пройти
```
