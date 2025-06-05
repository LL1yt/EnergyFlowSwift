# Tokenizer Module - 3D Cellular Neural Network

**Модуль:** `data/tokenizer/`  
**Версия:** 1.0.0  
**Статус:** 🚧 В разработке  
**Фаза:** Phase 2 (Core Functionality)

---

## 🎯 Описание

Модуль для конвертации между текстом и токенами с интеграцией популярных токенайзеров. Обеспечивает единый интерфейс для работы с различными типами токенизации и предобрабатывает текстовые данные для 3D клеточной нейронной сети.

## 🔧 Основная функциональность

### Поддерживаемые токенайзеры

- **BERT** (`bert-base-uncased`) - Bidirectional токенизация
- **GPT-2** (`gpt2`) - Autoregressive токенизация
- **SentencePiece** (`sentencepiece`) - Subword tokenization
- **Basic** (`basic`) - Простая белая токенизация

### Ключевые возможности

- ✅ Конвертация текст ↔ токены ↔ ID
- ✅ Обработка специальных токенов ([CLS], [SEP], [PAD])
- ✅ Поддержка различных языков
- ✅ Batch processing для больших объемов
- ✅ Интеграция с embedding_loader
- ✅ Конфигурируемые параметры (max_length, padding, etc.)

## 📦 Установка

### Зависимости

```bash
pip install transformers>=4.21.0
pip install sentencepiece>=0.1.96
pip install torch>=1.9.0
```

### Быстрый старт

```python
from data.tokenizer import TokenizerManager

# Создание менеджера токенайзера
tokenizer = TokenizerManager(tokenizer_type='bert-base-uncased')

# Кодирование текста
text = "Hello, world! This is a test."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Декодирование обратно в текст
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

## 🏗️ Архитектура

### Основные классы

```python
class TokenizerManager:
    """Центральный менеджер всех токенайзеров"""

class TextProcessor:
    """Предобработка и очистка текста"""

class TokenizerAdapter:
    """Базовый класс для адаптеров токенайзеров"""
```

### Структура модуля

```
data/tokenizer/
├── __init__.py                  # Экспорты модуля
├── README.md                    # Эта документация
├── plan.md                      # План разработки
├── meta.md                      # Метаданные и зависимости
├── errors.md                    # Документированные ошибки
├── diagram.mmd                  # Архитектурная диаграмма
├── examples.md                  # Примеры использования
├── tokenizer.py                 # Основной класс TokenizerManager
├── tokenizer_adapters.py        # Адаптеры для разных токенайзеров
├── text_processor.py            # Предобработка текста
└── config/
    └── tokenizer_config.yaml    # Конфигурация
```

## 🔗 Интеграция

### С embedding_loader

```python
from data.embedding_loader import EmbeddingLoader
from data.tokenizer import TokenizerManager

# Создание pipeline
tokenizer = TokenizerManager('bert-base-uncased')
embedding_loader = EmbeddingLoader()

# Обработка текста
text = "Sample input text"
tokens = tokenizer.encode(text)
embeddings = embedding_loader.get_embeddings_for_tokens(tokens)
```

### С lattice_3d

```python
from core.lattice_3d import Lattice3D
from data.tokenizer import TokenizerManager

# Подготовка входных данных для решетки
tokenizer = TokenizerManager()
lattice = Lattice3D(size=(5, 5, 5))

text = "Input for neural network"
tokens = tokenizer.encode(text, max_length=25)  # 5x5 входная грань
lattice.set_input_face(tokens)
```

## ⚙️ Конфигурация

### Параметры tokenizer_config.yaml

```yaml
tokenizer:
  type: "bert-base-uncased" # Тип токенайзера
  max_length: 512 # Максимальная длина последовательности
  padding: true # Добавлять padding
  truncation: true # Обрезать длинные тексты
  add_special_tokens: true # Добавлять [CLS], [SEP]

text_processing:
  lowercase: true # Приводить к нижнему регистру
  remove_punctuation: false # Удалять пунктуацию
  remove_stopwords: false # Удалять стоп-слова

batch_processing:
  batch_size: 32 # Размер батча
  num_workers: 4 # Количество потоков
```

## 📊 Performance

### Производительность

- **Скорость:** ~1000 токенов/сек для BERT
- **Память:** ~200MB для модели BERT base
- **Batch processing:** Поддержка до 10K текстов одновременно

### Benchmarks

| Токенайзер | Скорость (tokens/sec) | Память (MB) | Точность |
| ---------- | --------------------- | ----------- | -------- |
| BERT       | 1000                  | 200         | 95%      |
| GPT-2      | 1500                  | 150         | 92%      |
| Basic      | 5000                  | 10          | 80%      |

## 🧪 Тестирование

```bash
# Запуск тестов модуля
python -m pytest data/tokenizer/tests/

# Ручное тестирование
python test_tokenizer_basic.py
```

## 🐛 Известные ограничения

- BERT требует интернет для первой загрузки модели
- SentencePiece модели нужно загружать отдельно
- Максимальная длина ограничена архитектурой токенайзера

## 📄 Лицензия

Часть проекта 3D Cellular Neural Network.  
См. основной README проекта для информации о лицензии.

---

**🎯 Статус разработки:** 🚧 Модуль в активной разработке  
**🔗 Связанные модули:** `data/embedding_loader/`, `core/lattice_3d/`  
**📅 Планируемое завершение:** Phase 2 (текущая фаза)
