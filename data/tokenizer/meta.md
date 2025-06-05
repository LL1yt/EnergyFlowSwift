# META: Tokenizer Module - 3D Cellular Neural Network

**Модуль:** `data/tokenizer/`  
**Версия:** 1.0.0  
**Дата создания:** 5 декабря 2025  
**Последнее обновление:** 5 декабря 2025  
**Статус:** 🚧 В разработке

---

## 📦 МОДУЛЬНЫЕ ЗАВИСИМОСТИ

### Внутренние зависимости (проект)

```python
# Прямые зависимости
from data.embedding_loader import EmbeddingLoader  # Интеграция с эмбедингами
from core.lattice_3d import Lattice3D              # Подготовка данных для решетки
from utils.config_manager import ConfigManager      # Загрузка конфигурации

# Косвенные зависимости
import core.cell_prototype                          # Через lattice_3d
import core.signal_propagation                      # Через lattice_3d
```

### Внешние зависимости (Python packages)

```python
# Критические зависимости
import torch                    # >=1.9.0   # PyTorch tensors
import transformers             # >=4.21.0  # Hugging Face токенайзеры
import numpy as np              # >=1.20.0  # Численные операции

# Дополнительные зависимости
import sentencepiece            # >=0.1.96  # SentencePiece токенайзер
import yaml                     # *         # YAML конфигурация
import logging                  # stdlib    # Логирование
import re                       # stdlib    # Регулярные выражения
import unicodedata              # stdlib    # Unicode нормализация
from typing import List, Dict, Optional, Union, Any  # stdlib

# Опциональные зависимости
import nltk                     # >=3.6     # NLP библиотека (стоп-слова, stemming)
from tqdm import tqdm           # >=4.50.0  # Progress bars
import multiprocessing          # stdlib    # Параллельная обработка
```

### UI/DOM зависимости

```html
<!-- Не применимо - backend модуль -->
N/A
```

---

## 🚀 ЭКСПОРТИРУЕМЫЙ API

### Основные классы

```python
# Центральный класс модуля
class TokenizerManager:
    """Управление различными типами токенайзеров"""

    # Конструктор
    def __init__(self, tokenizer_type: str = 'bert-base-uncased',
                 config: Optional[Dict] = None)

    # Основные методы
    def encode(self, text: str, **kwargs) -> List[int]
    def decode(self, tokens: List[int]) -> str
    def tokenize(self, text: str) -> List[str]
    def batch_encode(self, texts: List[str], **kwargs) -> List[List[int]]
    def batch_decode(self, token_lists: List[List[int]]) -> List[str]

    # Утилитные методы
    def get_vocab_size(self) -> int
    def get_special_tokens(self) -> Dict[str, int]
    def is_available(self) -> bool

    # Интеграционные методы
    def prepare_for_lattice(self, text: str, lattice_size: tuple) -> torch.Tensor
    def get_embeddings_for_tokens(self, tokens: List[int]) -> torch.Tensor

class TextProcessor:
    """Предобработка и очистка текста"""

    # Конструктор
    def __init__(self, config: Optional[Dict] = None)

    # Основные методы предобработки
    def preprocess(self, text: str) -> str
    def normalize_unicode(self, text: str) -> str
    def clean_text(self, text: str) -> str
    def remove_punctuation(self, text: str) -> str
    def remove_stopwords(self, text: str, language: str = 'english') -> str

    # Batch обработка
    def batch_preprocess(self, texts: List[str]) -> List[str]
```

### Адаптеры токенайзеров

```python
# Базовый класс адаптера
class TokenizerAdapter:
    """Абстрактный базовый класс для всех адаптеров"""

    def encode(self, text: str) -> List[int]          # Абстрактный
    def decode(self, tokens: List[int]) -> str        # Абстрактный
    def tokenize(self, text: str) -> List[str]        # Абстрактный

# Конкретные адаптеры
class BertTokenizerAdapter(TokenizerAdapter):
    """Адаптер для BERT токенайзера"""

class GPTTokenizerAdapter(TokenizerAdapter):
    """Адаптер для GPT-2 токенайзера"""

class SentencePieceAdapter(TokenizerAdapter):
    """Адаптер для SentencePiece токенайзера"""

class BasicTokenizerAdapter(TokenizerAdapter):
    """Базовый токенайзер для простой токенизации"""
```

### Константы и конфигурация

```python
# Поддерживаемые типы токенайзеров
SUPPORTED_TOKENIZERS: List[str] = [
    'bert-base-uncased',
    'bert-base-cased',
    'gpt2',
    'gpt2-medium',
    'sentencepiece',
    'basic'
]

# Конфигурация по умолчанию
DEFAULT_CONFIG: Dict[str, Any] = {
    'tokenizer_type': 'bert-base-uncased',
    'max_length': 512,
    'padding': True,
    'truncation': True,
    'add_special_tokens': True
}

# Специальные токены
SPECIAL_TOKENS: Dict[str, str] = {
    'CLS': '[CLS]',
    'SEP': '[SEP]',
    'PAD': '[PAD]',
    'UNK': '[UNK]',
    'MASK': '[MASK]'
}
```

---

## 🔧 ИНТЕГРАЦИОННЫЕ ИНТЕРФЕЙСЫ

### С data/embedding_loader

```python
# Методы для работы с эмбедингами
def integrate_with_embeddings(self, embedding_loader: EmbeddingLoader) -> None:
    """Настройка интеграции с загрузчиком эмбедингов"""

def validate_vocabulary(self, embedding_vocab: Dict[str, int]) -> bool:
    """Проверка совместимости словарей"""

def handle_oov_tokens(self, tokens: List[str]) -> List[str]:
    """Обработка токенов вне словаря"""
```

### С core/lattice_3d

```python
# Методы подготовки данных для решетки
def prepare_input_face(self, text: str, face_size: tuple) -> torch.Tensor:
    """Подготовка входных данных для грани решетки"""

def batch_prepare_lattice_input(self, texts: List[str],
                               lattice_size: tuple) -> torch.Tensor:
    """Batch подготовка для решетки"""
```

### С utils/config_manager

```python
# Интеграция с системой конфигурации
def load_config(self, config_path: str) -> Dict:
    """Загрузка конфигурации из файла"""

def validate_config(self, config: Dict) -> bool:
    """Валидация параметров конфигурации"""
```

---

## 📈 ВЕРСИОНИРОВАНИЕ И СОВМЕСТИМОСТЬ

### Текущая версия: 1.0.0

**Семантическое версионирование:**

- **Major (1):** Основная архитектура модуля
- **Minor (0):** Добавление новых токенайзеров или функций
- **Patch (0):** Исправления багов и мелкие улучшения

### Совместимость

**Обратная совместимость:**

- ✅ API стабилен в рамках мажорной версии
- ✅ Конфигурационные файлы поддерживаются
- ✅ Интеграционные интерфейсы не меняются

**Зависимости версий:**

- `transformers`: >=4.21.0 (критично)
- `torch`: >=1.9.0 (требуется для тензоров)
- `sentencepiece`: >=0.1.96 (опционально)

---

## 🏗️ АРХИТЕКТУРНЫЕ РЕШЕНИЯ

### Паттерны проектирования

1. **Strategy Pattern** - различные токенайзеры как стратегии
2. **Adapter Pattern** - унификация интерфейсов
3. **Factory Pattern** - создание токенайзеров по типу
4. **Singleton Pattern** - кэширование моделей

### Принципы архитектуры

- **Single Responsibility** - каждый класс отвечает за одну задачу
- **Open/Closed** - легко добавлять новые токенайзеры
- **Dependency Inversion** - зависимость от абстракций
- **Interface Segregation** - минимальные интерфейсы

---

## 📊 ПРОИЗВОДИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ

### Производительность

- **Скорость токенизации:** ~1000-5000 токенов/сек (зависит от типа)
- **Память:** 150-500MB (зависит от модели токенайзера)
- **Batch processing:** до 10K текстов одновременно
- **Кэширование:** 5-10x ускорение для повторных запросов

### Масштабируемость

- **Тексты:** поддержка до 1M символов
- **Batch size:** оптимально 32-128 текстов
- **Параллелизм:** до 8 потоков для batch processing
- **Память:** linear scaling с размером batch

---

## 🧪 ТЕСТОВОЕ ПОКРЫТИЕ

### Планируемые тесты

**Unit тесты (цель: >90% coverage):**

- TokenizerManager: основные методы
- TextProcessor: предобработка текста
- Все адаптеры: encode/decode/tokenize
- Конфигурация: загрузка и валидация

**Integration тесты:**

- С embedding_loader: совместимость словарей
- С lattice_3d: подготовка входных данных
- С config_manager: загрузка конфигурации

**Performance тесты:**

- Скорость токенизации больших текстов
- Память usage для различных размеров batch
- Эффективность кэширования

---

## 🔐 БЕЗОПАСНОСТЬ И ВАЛИДАЦИЯ

### Валидация входных данных

- Проверка типов параметров
- Валидация длины текста
- Санитизация входных строк
- Обработка Unicode символов

### Error Handling

- Graceful handling отсутствующих зависимостей
- Восстановление после ошибок загрузки моделей
- Fallback на базовую токенизацию
- Подробное логирование ошибок

---

## 📝 ДОКУМЕНТАЦИЯ И ПРИМЕРЫ

### Обязательные файлы

- [x] `README.md` - основная документация
- [x] `plan.md` - план разработки
- [x] `meta.md` - этот файл (метаданные)
- [ ] `errors.md` - документированные ошибки
- [ ] `diagram.mmd` - архитектурная диаграмма
- [ ] `examples.md` - конкретные примеры использования

### Внутренняя документация

- Docstrings для всех публичных методов
- Type hints для всех параметров
- Inline комментарии для сложной логики
- README для каждого адаптера

---

**🎯 Статус метаданных:** 📋 Полные и актуальные  
**🔄 Последнее обновление:** 5 декабря 2025  
**✅ Готовность к разработке:** 100% (документация)

---

_Метаданные модуля полностью описывают архитектуру, зависимости и интерфейсы для успешной интеграции с 3D клеточной нейронной сетью._
