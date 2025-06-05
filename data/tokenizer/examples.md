# Examples: Tokenizer Module - 3D Cellular Neural Network

**Модуль:** `data/tokenizer/`  
**Версия:** 1.0.0  
**Статус:** ✅ Готов к использованию  
**Дата:** 5 июнь 2025

---

## 🎯 Основные примеры использования

### 1. Базовое использование TokenizerManager

```python
from data.tokenizer import TokenizerManager

# Создание токенайзера (по умолчанию basic)
tokenizer = TokenizerManager(tokenizer_type='basic')

# Простая токенизация
text = "Hello world! This is a test."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['hello', 'world', 'this', 'is', 'a', 'test']

# Кодирование в ID
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
# Output: [2, 7179, 9692, 9477, 58, 838, 8449, 3]

# Декодирование обратно
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
```

### 2. Использование с конфигурацией

```python
# Кастомная конфигурация
config = {
    'tokenizer': {
        'type': 'basic',
        'max_length': 50,
        'padding': True,
        'add_special_tokens': True
    },
    'text_processing': {
        'lowercase': True,
        'remove_punctuation': False,
        'remove_urls': True
    },
    'caching': {
        'enabled': True,
        'max_size': 5000
    }
}

tokenizer = TokenizerManager(tokenizer_type='basic', config=config)

# Токенизация с ограничением длины
text = "This is a longer text that needs to be limited in length"
tokens = tokenizer.encode(text, max_length=10, padding='max_length')
print(f"Limited tokens: {tokens}")
```

### 3. Batch обработка

```python
# Обработка нескольких текстов сразу
texts = [
    "First example text",
    "Second text is longer than the first one",
    "Third and final text"
]

# Batch кодирование
batch_encoded = tokenizer.batch_encode(texts, max_length=15)
print(f"Batch encoded: {batch_encoded}")

# Batch декодирование
batch_decoded = tokenizer.batch_decode(batch_encoded)
print(f"Batch decoded: {batch_decoded}")
```

### 4. Интеграция с 3D решеткой

```python
# Подготовка входных данных для 3D клеточной нейронной сети
text = "Input data for neural network processing"
lattice_size = (8, 8, 8)  # Размер 3D решетки

# Подготовка тензора для входной грани решетки
lattice_input = tokenizer.prepare_for_lattice(text, lattice_size)
print(f"Lattice input shape: {lattice_input.shape}")  # torch.Size([8, 8])
print(f"Ready for lattice: {lattice_input}")

# Теперь можно подать на решетку
# from core.lattice_3d import Lattice3D
# lattice = Lattice3D(size=lattice_size)
# lattice.set_input_face(lattice_input)
```

---

## 🔧 Работа с TextProcessor

### 1. Базовая предобработка

```python
from data.tokenizer.text_processor import TextProcessor

# Создание процессора с настройками по умолчанию
processor = TextProcessor()

# Обработка текста с проблемами
messy_text = "  HELLO    World!!!   This is   MESSY text.  "
clean_text = processor.preprocess(messy_text)
print(f"Original: '{messy_text}'")
print(f"Cleaned:  '{clean_text}'")
# Output: 'hello world!!! this is messy text.'
```

### 2. Кастомная конфигурация предобработки

```python
# Конфигурация для агрессивной очистки
aggressive_config = {
    'lowercase': True,
    'remove_punctuation': True,
    'remove_numbers': True,
    'remove_urls': True,
    'remove_emails': True,
    'normalize_unicode': True
}

processor = TextProcessor(aggressive_config)

text_with_issues = "Check out https://example.com! Email: test@example.com. Number: 123."
processed = processor.preprocess(text_with_issues)
print(f"Aggressively cleaned: '{processed}'")
# Output: 'check out email number'
```

### 3. Статистика обработки

```python
original = "This is the ORIGINAL text with CAPS and numbers 123!"
processed = processor.preprocess(original)

# Получение статистики
stats = processor.get_processing_stats(original, processed)
print(f"Processing stats: {stats}")
# Output: {
#   'original_length': 52,
#   'processed_length': 45,
#   'original_words': 9,
#   'processed_words': 8,
#   'reduction_ratio': 0.134,
#   'config': {...}
# }
```

---

## 🏗️ Работа с адаптерами (Advanced)

### 1. Использование BERT токенайзера (если доступен)

```python
try:
    # Попытка использовать BERT (требует transformers)
    bert_tokenizer = TokenizerManager(tokenizer_type='bert-base-uncased')

    text = "Hello world! This is BERT tokenization."
    bert_tokens = bert_tokenizer.encode(text)
    print(f"BERT tokens: {bert_tokens}")

    # Специальные токены BERT
    special_tokens = bert_tokenizer.get_special_tokens()
    print(f"BERT special tokens: {special_tokens}")

except ImportError:
    print("BERT tokenizer not available (transformers not installed)")
    # Fallback к базовому токенайзеру
    tokenizer = TokenizerManager(tokenizer_type='basic')
```

### 2. Проверка доступности токенайзеров

```python
# Проверка доступности различных токенайзеров
tokenizer_types = ['bert-base-uncased', 'gpt2', 'basic']

for tokenizer_type in tokenizer_types:
    try:
        test_tokenizer = TokenizerManager(tokenizer_type=tokenizer_type)
        if test_tokenizer.is_available():
            vocab_size = test_tokenizer.get_vocab_size()
            print(f"✅ {tokenizer_type}: Available (vocab size: {vocab_size})")
        else:
            print(f"❌ {tokenizer_type}: Not available")
    except Exception as e:
        print(f"❌ {tokenizer_type}: Failed to load ({str(e)})")
```

---

## 🔗 Интеграция с другими модулями

### 1. Интеграция с EmbeddingLoader

```python
from data.tokenizer import TokenizerManager
from data.embedding_loader import EmbeddingLoader

# Создание pipeline
tokenizer = TokenizerManager(tokenizer_type='basic')
embedding_loader = EmbeddingLoader()

# Обработка текста через полный pipeline
text = "Sample text for embedding processing"

# 1. Токенизация
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# 2. Получение эмбедингов (если integration готова)
# embeddings = embedding_loader.get_embeddings_for_tokens(tokens)
# print(f"Embeddings shape: {embeddings.shape}")
```

### 2. Полный pipeline для 3D CNN

```python
import torch
from data.tokenizer import TokenizerManager

def text_to_3d_cnn_input(text: str, lattice_size: tuple) -> torch.Tensor:
    """
    Полная функция преобразования текста в вход для 3D CNN.

    Args:
        text: Входной текст
        lattice_size: Размер 3D решетки (x, y, z)

    Returns:
        Тензор готовый для подачи на 3D решетку
    """
    # Инициализация токенайзера
    tokenizer = TokenizerManager(tokenizer_type='basic')

    # Токенизация и подготовка
    lattice_input = tokenizer.prepare_for_lattice(text, lattice_size)

    return lattice_input

# Использование
text = "Input for 3D cellular neural network"
lattice_size = (10, 10, 10)
cnn_input = text_to_3d_cnn_input(text, lattice_size)
print(f"Ready for 3D CNN: {cnn_input.shape}")
```

---

## 📊 Мониторинг и отладка

### 1. Мониторинг производительности

```python
# Создание токенайзера с включенными метриками
tokenizer = TokenizerManager(tokenizer_type='basic')

# Обработка нескольких текстов
test_texts = [
    "First test text",
    "Second longer test text with more words",
    "Third text for testing performance"
]

for text in test_texts:
    tokens = tokenizer.encode(text)

# Получение метрик
metrics = tokenizer.get_metrics()
print("📊 Performance Metrics:")
print(f"  Total tokenizations: {metrics['total_tokenizations']}")
print(f"  Average tokens per text: {metrics['avg_tokens_per_text']:.2f}")
print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
print(f"  Cache size: {metrics['cache_size']}")
```

### 2. Отладка и логирование

```python
import logging

# Настройка детального логирования
logging.basicConfig(level=logging.DEBUG)

# Создание токенайзера с отладкой
tokenizer = TokenizerManager(tokenizer_type='basic')

# Обработка с подробными логами
text = "Debug tokenization process"
tokens = tokenizer.encode(text)

# Проверка состояния
print(f"Tokenizer type: {tokenizer.tokenizer_type}")
print(f"Config: {tokenizer.config}")
print(f"Cache enabled: {tokenizer._cache_enabled}")
```

### 3. Тестирование различных конфигураций

```python
# Тестирование различных настроек
configs = [
    {'text_processing': {'lowercase': True, 'remove_punctuation': False}},
    {'text_processing': {'lowercase': False, 'remove_punctuation': True}},
    {'caching': {'enabled': False}},
    {'tokenizer': {'max_length': 20, 'padding': True}}
]

test_text = "Testing DIFFERENT configurations! With punctuation."

for i, config in enumerate(configs):
    print(f"\n🧪 Configuration {i+1}:")
    tokenizer = TokenizerManager(tokenizer_type='basic', config=config)

    tokens = tokenizer.encode(test_text, max_length=15)
    print(f"Result: {tokens}")

    metrics = tokenizer.get_metrics()
    print(f"Cache enabled: {metrics.get('cache_enabled', 'N/A')}")
```

---

## 🎯 Практические сценарии

### 1. Обработка файла с текстами

```python
def process_text_file(file_path: str, output_path: str):
    """Обработка файла с текстами и сохранение токенизированных данных."""

    tokenizer = TokenizerManager(tokenizer_type='basic')

    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # Batch обработка
    processed_texts = []
    for text in texts:
        tokens = tokenizer.encode(text.strip())
        processed_texts.append(tokens)

    # Сохранение результата
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_texts, f, indent=2)

    print(f"Processed {len(texts)} texts to {output_path}")

# Использование
# process_text_file('input_texts.txt', 'tokenized_output.json')
```

### 2. Создание датасета для обучения

```python
def create_training_dataset(texts: list, lattice_size: tuple) -> list:
    """Создание датасета токенизированных данных для обучения 3D CNN."""

    tokenizer = TokenizerManager(tokenizer_type='basic')
    dataset = []

    for text in texts:
        # Подготовка входных данных для решетки
        lattice_input = tokenizer.prepare_for_lattice(text, lattice_size)
        dataset.append({
            'original_text': text,
            'lattice_input': lattice_input.tolist(),  # Для сериализации
            'shape': lattice_input.shape
        })

    return dataset

# Пример использования
sample_texts = [
    "First training example",
    "Second training sample with more content",
    "Third example for neural network"
]

dataset = create_training_dataset(sample_texts, (5, 5, 5))
print(f"Created dataset with {len(dataset)} samples")
```

---

**🎯 Итоги примеров:**

- ✅ Базовое использование токенайзера готово к работе
- ✅ Интеграция с 3D решеткой функциональна
- ✅ Batch обработка эффективна
- ✅ Мониторинг и отладка доступны
- ✅ Практические сценарии реализуемы

**📚 Дополнительная документация:**

- `README.md` - общий обзор модуля
- `meta.md` - детальные метаданные и API
- `plan.md` - план разработки с прогрессом
- `diagram.mmd` - архитектурная диаграмма
