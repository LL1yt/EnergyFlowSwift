# 📚 EXAMPLES: Lightweight Decoder Usage

**Модуль:** inference/lightweight_decoder/  
**Версия:** 0.1.0  
**Статус:** 🆕 Примеры для Phase 2.7 реализации

---

## 🎯 БАЗОВЫЕ ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### 1. Simple PhraseBankDecoder

```python
from inference.lightweight_decoder import PhraseBankDecoder
import torch

# Инициализация
decoder = PhraseBankDecoder(
    embedding_dim=768,
    phrase_bank_size=50000,
    similarity_threshold=0.8
)

# Загрузка phrase bank
decoder.load_phrase_bank("data/phrase_banks/common_phrases.pkl")

# Декодирование
input_embedding = torch.randn(768)  # От EmbeddingProcessor
generated_text = decoder.decode(input_embedding)

print(f"Generated: {generated_text}")
# Output: "The quick brown fox jumps over the lazy dog"
```

### 2. GenerativeDecoder с настройками

```python
from inference.lightweight_decoder import GenerativeDecoder

# Создание компактной модели
decoder = GenerativeDecoder(
    embedding_dim=768,
    vocab_size=32000,
    hidden_size=1024,
    num_layers=4,
    num_heads=8
)

# Настройка temperature для creativity control
decoder.set_temperature(0.7)  # Более консервативная генерация

# Генерация с ограничением длины
input_embedding = torch.randn(768)
generated_text = decoder.generate(
    input_embedding,
    max_length=100
)

print(f"Generated: {generated_text}")
# Output: "Artificial intelligence represents a fascinating frontier..."
```

### 3. HybridDecoder - лучшее из двух подходов

```python
from inference.lightweight_decoder import HybridDecoder, PhraseBankDecoder, GenerativeDecoder

# Создание компонентов
phrase_decoder = PhraseBankDecoder(768, 50000, 0.8)
generative_decoder = GenerativeDecoder(768, 32000, 1024, 4)

# Гибридный подход
hybrid_decoder = HybridDecoder(
    phrase_decoder=phrase_decoder,
    generative_decoder=generative_decoder,
    confidence_threshold=0.75
)

# Автоматический выбор лучшего подхода
input_embedding = torch.randn(768)
result = hybrid_decoder.decode(input_embedding)

print(f"Generated: {result}")
print(f"Confidence: {hybrid_decoder.get_confidence(input_embedding)}")
```

---

## 🔧 КОНФИГУРАЦИОННЫЕ ПРИМЕРЫ

### 1. DecoderFactory - управление через конфигурацию

```python
from inference.lightweight_decoder import DecoderFactory

# Загрузка из конфигурации
config = {
    "type": "hybrid",
    "phrase_bank": {
        "size": 50000,
        "threshold": 0.8
    },
    "generative": {
        "hidden_size": 1024,
        "num_layers": 4,
        "temperature": 0.7
    }
}

decoder = DecoderFactory.create_decoder("hybrid", config)

# Использование
input_embedding = torch.randn(768)
output = decoder.decode(input_embedding)
```

### 2. Switching между стратегиями

```python
# Runtime переключение стратегий
hybrid_decoder.set_strategy("phrase")      # Только phrase bank
result1 = hybrid_decoder.decode(embedding)

hybrid_decoder.set_strategy("generative")  # Только генеративная модель
result2 = hybrid_decoder.decode(embedding)

hybrid_decoder.set_strategy("hybrid")      # Автоматический выбор
result3 = hybrid_decoder.decode(embedding)

print(f"Phrase only: {result1}")
print(f"Generative only: {result2}")
print(f"Hybrid approach: {result3}")
```

---

## 🌊 ИНТЕГРАЦИЯ С ПОЛНОЙ СИСТЕМОЙ

### 1. End-to-End Pipeline

```python
from data.embedding_loader import EmbeddingLoader
from core.embedding_processor import EmbeddingProcessor
from inference.lightweight_decoder import HybridDecoder

# Полная система Module 1 + 2 + 3
class CompleteCognitiveSystem:
    def __init__(self):
        # Module 1: Teacher LLM Encoder
        self.encoder = EmbeddingLoader(
            model_name="llama3-8b",
            cache_enabled=True
        )

        # Module 2: 3D Cubic Core
        self.processor = EmbeddingProcessor(
            lattice_size=(8, 8, 8),
            propagation_steps=10
        )

        # Module 3: Lightweight Decoder
        self.decoder = HybridDecoder.from_config("config/decoder.yaml")

    def process_text(self, input_text: str) -> str:
        """Полная обработка текста через все три модуля"""

        # Текст → Эмбединг (Module 1)
        embedding = self.encoder.encode_text(input_text)
        print(f"Embedding shape: {embedding.shape}")

        # Эмбединг → Обработанный эмбединг (Module 2)
        processed = self.processor.process(embedding)
        print(f"Processing complete, similarity: {self.processor.last_similarity}")

        # Обработанный эмбединг → Текст (Module 3)
        output_text = self.decoder.decode(processed)
        print(f"Decoding complete")

        return output_text

# Использование полной системы
system = CompleteCognitiveSystem()

# Autoencoder режим
input_text = "Hello, how are you today?"
output = system.process_text(input_text)
print(f"Input: {input_text}")
print(f"Output: {output}")

# Dialogue режим
input_text = "What is artificial intelligence?"
output = system.process_text(input_text)
print(f"Question: {input_text}")
print(f"Answer: {output}")
```

### 2. Batch Processing для эффективности

```python
import torch

# Batch обработка для множественных inputs
def batch_decode(decoder, embeddings_batch):
    """Эффективная batch обработка"""

    results = []
    batch_size = embeddings_batch.shape[0]

    print(f"Processing batch of {batch_size} embeddings...")

    for i, embedding in enumerate(embeddings_batch):
        result = decoder.decode(embedding)
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{batch_size}")

    return results

# Пример использования
embeddings_batch = torch.randn(50, 768)  # 50 эмбедингов
decoder = HybridDecoder.from_config("config/decoder.yaml")

decoded_texts = batch_decode(decoder, embeddings_batch)

for i, text in enumerate(decoded_texts[:5]):  # Первые 5 результатов
    print(f"Text {i+1}: {text}")
```

---

## 🧪 EVALUATION И TESTING

### 1. Quality Assessment

```python
from inference.lightweight_decoder.utils import calculate_bleu, semantic_similarity

# Оценка качества генерации
def evaluate_decoder_quality(decoder, test_embeddings, reference_texts):
    """Comprehensive quality evaluation"""

    results = {
        'bleu_scores': [],
        'semantic_similarities': [],
        'generation_times': []
    }

    for embedding, reference in zip(test_embeddings, reference_texts):
        # Генерация с замером времени
        import time
        start_time = time.time()
        generated = decoder.decode(embedding)
        generation_time = time.time() - start_time

        # BLEU score
        bleu = calculate_bleu([reference], generated)

        # Semantic similarity
        similarity = semantic_similarity(reference, generated)

        results['bleu_scores'].append(bleu)
        results['semantic_similarities'].append(similarity)
        results['generation_times'].append(generation_time)

    # Средние метрики
    avg_bleu = sum(results['bleu_scores']) / len(results['bleu_scores'])
    avg_similarity = sum(results['semantic_similarities']) / len(results['semantic_similarities'])
    avg_time = sum(results['generation_times']) / len(results['generation_times'])

    print(f"Average BLEU: {avg_bleu:.3f}")
    print(f"Average Semantic Similarity: {avg_similarity:.3f}")
    print(f"Average Generation Time: {avg_time:.3f}s")

    return results

# Использование
test_embeddings = torch.randn(100, 768)
reference_texts = ["Reference text " + str(i) for i in range(100)]

results = evaluate_decoder_quality(decoder, test_embeddings, reference_texts)
```

### 2. Performance Benchmarking

```python
import torch
import time

def benchmark_decoders():
    """Сравнение производительности всех трех подходов"""

    # Создание тестовых данных
    test_embeddings = torch.randn(100, 768)

    # Инициализация всех декодеров
    phrase_decoder = PhraseBankDecoder(768, 50000, 0.8)
    generative_decoder = GenerativeDecoder(768, 32000, 1024, 4)
    hybrid_decoder = HybridDecoder(phrase_decoder, generative_decoder, 0.75)

    decoders = {
        'Phrase Bank': phrase_decoder,
        'Generative': generative_decoder,
        'Hybrid': hybrid_decoder
    }

    results = {}

    for name, decoder in decoders.items():
        print(f"\nBenchmarking {name} Decoder...")

        start_time = time.time()
        outputs = []

        for embedding in test_embeddings:
            output = decoder.decode(embedding)
            outputs.append(output)

        total_time = time.time() - start_time
        avg_time_per_decode = total_time / len(test_embeddings)

        results[name] = {
            'total_time': total_time,
            'avg_time_per_decode': avg_time_per_decode,
            'throughput': len(test_embeddings) / total_time
        }

        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per decode: {avg_time_per_decode:.4f}s")
        print(f"Throughput: {results[name]['throughput']:.2f} decodes/sec")

    return results

# Запуск benchmark
benchmark_results = benchmark_decoders()
```

---

## 🔧 DEBUGGING И TROUBLESHOOTING

### 1. Debugging Helpers

```python
def debug_decoder_pipeline(decoder, embedding, verbose=True):
    """Детальная отладка процесса декодирования"""

    print(f"Input embedding shape: {embedding.shape}")
    print(f"Input embedding norm: {torch.norm(embedding):.4f}")

    if hasattr(decoder, 'get_confidence'):
        confidence = decoder.get_confidence(embedding)
        print(f"Confidence score: {confidence:.4f}")

    if hasattr(decoder, 'phrase_decoder'):
        print("Using hybrid decoder with phrase bank fallback")

    # Декодирование с детальным логированием
    result = decoder.decode(embedding)

    print(f"Generated text length: {len(result)}")
    print(f"Generated text: {result}")

    return result

# Использование для отладки
debug_result = debug_decoder_pipeline(hybrid_decoder, torch.randn(768))
```

### 2. Error Recovery

```python
def robust_decode_with_fallback(decoder, embedding, max_retries=3):
    """Декодирование с обработкой ошибок и fallback"""

    for attempt in range(max_retries):
        try:
            result = decoder.decode(embedding)

            # Валидация результата
            if len(result.strip()) == 0:
                raise ValueError("Empty generation result")

            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            if attempt == max_retries - 1:
                # Финальный fallback
                return "Unable to generate text from embedding"

            # Попробовать с другой стратегией
            if hasattr(decoder, 'set_strategy'):
                strategies = ['phrase', 'generative', 'hybrid']
                current_strategy = strategies[attempt % len(strategies)]
                decoder.set_strategy(current_strategy)
                print(f"Retrying with strategy: {current_strategy}")

# Использование
safe_result = robust_decode_with_fallback(hybrid_decoder, torch.randn(768))
```

---

## 🎯 EXPECTED OUTPUTS

### Качественные примеры выходов для различных типов входов:

```python
# Пример 1: Factual Query
input_text = "What is machine learning?"
# Expected output: "Machine learning is a subset of artificial intelligence..."

# Пример 2: Creative Request
input_text = "Write a short poem about nature"
# Expected output: "Trees whisper secrets in the gentle breeze..."

# Пример 3: Technical Question
input_text = "Explain neural networks"
# Expected output: "Neural networks are computational models inspired by..."

# Пример 4: Conversation
input_text = "How are you feeling today?"
# Expected output: "I'm functioning well and ready to help with your questions..."
```

---

**🎯 РЕЗУЛЬТАТ:** Comprehensive examples готовы для Phase 2.7 implementation и testing!
