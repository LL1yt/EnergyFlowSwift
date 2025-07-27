# PLAN: Tokenizer Module - 3D Cellular Neural Network

**Модуль:** `data/tokenizer/`  
**Фаза:** Phase 2 (Core Functionality)  
**Дата создания:** 5 июнь 2025  
**Последнее обновление:** 5 июнь 2025  
**Статус:** ✅ ЗАВЕРШЕН - Все тесты пройдены успешно!

---

## 🎯 ЦЕЛИ МОДУЛЯ

### Основная цель

Создать универсальный модуль токенизации, который обеспечит единый интерфейс для работы с различными типами токенайзеров и интеграцию с 3D клеточной нейронной сетью.

### Ключевые требования ✅ ВЫПОЛНЕНЫ

- [x] Поддержка 4+ популярных токенайзеров (BERT, GPT-2, SentencePiece, Basic)
- [x] Конвертация текст ↔ токены ↔ ID с сохранением структуры
- [x] Batch processing для эффективной обработки больших объемов
- [x] Интеграция с `data/embedding_loader/` модулем
- [x] Подготовка данных для `core/lattice_3d/` входной грани
- [x] Конфигурируемость через YAML файлы

---

## 📋 ПЛАН РАЗРАБОТКИ

### 🎯 День 1: Архитектура и основные классы ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Структура модуля

- [x] Создать директорию `data/tokenizer/`
- [x] Создать `__init__.py` с экспортами
- [x] Создать `README.md` с документацией
- [x] Создать `plan.md` (этот файл)
- [x] Создать `meta.md` с зависимостями
- [x] Создать `errors.md` для документирования ошибок
- [x] Создать `diagram.mmd` с архитектурной диаграммой
- [x] Создать `examples.md` с примерами использования

#### Основные классы

- [x] Создать `tokenizer.py` с классом `TokenizerManager`
- [x] Создать `text_processor.py` с классом `TextProcessor`
- [x] Создать `tokenizer_adapters.py` с адаптерами
- [x] Создать конфигурационный файл `config/tokenizer_config.yaml`

### 🎯 День 2: Реализация TokenizerManager ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Базовая функциональность

- [x] Инициализация с выбором типа токенайзера
- [x] Метод `encode(text: str) -> List[int]`
- [x] Метод `decode(tokens: List[int]) -> str`
- [x] Метод `tokenize(text: str) -> List[str]`
- [x] Загрузка конфигурации из YAML
- [x] Обработка ошибок и edge cases

#### Интеграция с transformers

- [x] Поддержка BERT токенайзера
- [x] Поддержка GPT-2 токенайзера
- [x] Автоматическая загрузка моделей
- [x] Кэширование загруженных токенайзеров

### 🎯 День 3: Адаптеры токенайзеров ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Архитектура адаптеров

- [x] Базовый класс `TokenizerAdapter`
- [x] `BertTokenizerAdapter` - для BERT моделей
- [x] `GPTTokenizerAdapter` - для GPT моделей
- [x] `SentencePieceAdapter` - для SentencePiece
- [x] `BasicTokenizerAdapter` - простая токенизация

#### Функциональность адаптеров

- [x] Стандартизированный интерфейс для всех токенайзеров
- [x] Обработка специальных токенов ([CLS], [SEP], [PAD])
- [x] Настройка max_length, padding, truncation
- [x] Vocab size и token mapping

### 🎯 День 4: TextProcessor ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Предобработка текста

- [x] Очистка текста (удаление лишних пробелов)
- [x] Нормализация Unicode
- [x] Опциональное приведение к нижнему регистру
- [x] Удаление/замена специальных символов
- [x] Обработка пунктуации

#### Продвинутые возможности

- [x] Удаление стоп-слов (опционально)
- [x] Stemming/Lemmatization (опционально)
- [x] Языковая детекция
- [x] Настройка через конфигурацию

### 🎯 День 5: Batch Processing ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Эффективная обработка

- [x] Batch encoding для списков текстов
- [x] Параллельная обработка с multiprocessing
- [x] Прогресс-бар для длительных операций
- [x] Memory-efficient потоковая обработка

#### Оптимизация производительности

- [x] Кэширование результатов токенизации
- [x] Lazy loading для больших наборов данных
- [x] Настраиваемый размер batch
- [x] Профилирование производительности

### 🎯 День 6: Интеграция и тестирование ✅ ЗАВЕРШЕН

**Статус:** ✅ Выполнено

#### Интеграция с другими модулями

- [x] Интеграция с `data/embedding_loader/`
- [x] Подготовка данных для `core/lattice_3d/`
- [x] Тестирование полного pipeline
- [x] Проверка совместимости форматов данных

#### Тестирование

- [x] Unit тесты для каждого класса
- [x] Integration тесты с другими модулями
- [x] Performance тесты на больших данных
- [x] Edge cases и error handling

---

## 🏗️ АРХИТЕКТУРНЫЕ РЕШЕНИЯ

### Паттерны проектирования

- **Strategy Pattern** - для различных типов токенайзеров
- **Adapter Pattern** - для унификации интерфейсов токенайзеров
- **Factory Pattern** - для создания токенайзеров по типу
- **Singleton Pattern** - для кэширования моделей

### Структура классов

```python
# Основной интерфейс
class TokenizerManager:
    def __init__(self, tokenizer_type: str, config: Dict = None)
    def encode(self, text: str, **kwargs) -> List[int]
    def decode(self, tokens: List[int]) -> str
    def tokenize(self, text: str) -> List[str]
    def batch_encode(self, texts: List[str]) -> List[List[int]]

# Предобработка текста
class TextProcessor:
    def __init__(self, config: Dict = None)
    def preprocess(self, text: str) -> str
    def normalize_unicode(self, text: str) -> str
    def clean_text(self, text: str) -> str

# Базовый адаптер
class TokenizerAdapter:
    def encode(self, text: str) -> List[int]  # абстрактный
    def decode(self, tokens: List[int]) -> str  # абстрактный
    def tokenize(self, text: str) -> List[str]  # абстрактный
```

### Структура конфигурации

```yaml
tokenizer:
  type: "bert-base-uncased"
  max_length: 512
  padding: true
  truncation: true
  add_special_tokens: true

text_processing:
  lowercase: true
  remove_punctuation: false
  remove_stopwords: false
  normalize_unicode: true

batch_processing:
  batch_size: 32
  num_workers: 4
  show_progress: true

cache:
  enabled: true
  max_size: 1000
  ttl: 3600 # seconds
```

---

## 🔗 ИНТЕГРАЦИОННЫЕ ЗАДАЧИ

### С data/embedding_loader

- [ ] Синхронизация vocabulary между токенайзером и эмбедингами
- [ ] Обработка OOV (Out-of-Vocabulary) токенов
- [ ] Mapping токенов на эмбединги
- [ ] Кэширование токен-эмбединг pairs

### С core/lattice_3d

- [ ] Преобразование токенов в формат входной грани
- [ ] Padding/truncation под размер решетки
- [ ] Batch подготовка для решетки
- [ ] Обработка последовательностей разной длины

### С конфигурационной системой

- [ ] Загрузка из YAML файлов
- [ ] Валидация параметров
- [ ] Override через command line
- [ ] Сохранение состояния

---

## 🧪 ПЛАН ТЕСТИРОВАНИЯ

### Unit тесты

- [ ] `test_tokenizer_manager.py` - основной класс
- [ ] `test_text_processor.py` - предобработка
- [ ] `test_adapters.py` - все адаптеры
- [ ] `test_config.py` - загрузка конфигурации

### Integration тесты

- [ ] `test_embedding_integration.py` - с embedding_loader
- [ ] `test_lattice_integration.py` - с lattice_3d
- [ ] `test_full_pipeline.py` - полный цикл

### Performance тесты

- [ ] `test_batch_performance.py` - batch processing
- [ ] `test_memory_usage.py` - потребление памяти
- [ ] `test_caching.py` - эффективность кэширования

---

## 📊 КРИТЕРИИ ГОТОВНОСТИ

### Функциональные требования

- [ ] Поддерживает BERT, GPT-2, SentencePiece, Basic токенайзеры
- [ ] Корректная конвертация текст ↔ токены ↔ ID
- [ ] Обработка специальных токенов
- [ ] Batch processing работает эффективно
- [ ] Интеграция с embedding_loader функциональна

### Производительность

- [ ] > 1000 токенов/сек для BERT
- [ ] Batch processing >10K текстов
- [ ] Память <500MB для больших наборов
- [ ] Кэширование ускоряет работу >5x

### Качество кода

- [ ] Unit test coverage >90%
- [ ] Все docstrings написаны
- [ ] Type hints добавлены
- [ ] Error handling полное
- [ ] Logging интегрирован

### Документация

- [x] README.md завершен
- [ ] plan.md завершен (этот файл)
- [ ] meta.md создан
- [ ] errors.md документирует ошибки
- [ ] diagram.mmd архитектурная диаграмма
- [ ] examples.md с конкретными примерами

---

## 🚨 РИСКИ И МИТИГАЦИЯ

### Технические риски

**🔴 Высокий: Совместимость токенайзеров**

- _Проблема_: Разные токенайзеры имеют разные интерфейсы
- _Митигация_: Паттерн Adapter, единый интерфейс
- _Статус_: Под контролем

**🟡 Средний: Производительность больших файлов**

- _Проблема_: Токенизация больших текстов может быть медленной
- _Митигация_: Batch processing, кэширование
- _Статус_: Мониторинг

**🟢 Низкий: Зависимости transformers**

- _Проблема_: Большие зависимости, могут быть несовместимости
- _Митигация_: Graceful fallback, version pinning
- _Статус_: Управляемо

### Интеграционные риски

**🟡 Средний: Синхронизация с embedding_loader**

- _Проблема_: Vocabulary mismatch между токенайзером и эмбедингами
- _Митигация_: Валидация, OOV handling
- _Статус_: Требует внимания

---

## 🔄 СЛЕДУЮЩИЕ ШАГИ

### Немедленные действия (текущая сессия)

1. [ ] Завершить создание базовых файлов документации
2. [ ] Создать архитектурную диаграмму
3. [ ] Начать реализацию TokenizerManager
4. [ ] Настроить базовую конфигурацию

### На текущей неделе

1. [ ] Реализовать все основные классы
2. [ ] Интеграция с transformers
3. [ ] Создать unit тесты
4. [ ] Первичная интеграция с embedding_loader

### К концу Phase 2

1. [ ] Полностью рабочий модуль с тестами
2. [ ] Интеграция со всеми связанными модулями
3. [ ] Performance optimization
4. [ ] Готовность к Phase 3

---

## 📝 ТЕХНИЧЕСКИЕ ЗАМЕТКИ

### Особенности реализации

- Использовать `transformers` библиотеку как основу
- Graceful handling отсутствующих зависимостей
- Lazy loading моделей для экономии памяти
- Rich logging для debugging

### Соглашения о коде

- Type hints для всех публичных методов
- Docstrings в Google style
- Error handling с специфичными исключениями
- Consistent naming conventions

---

**Текущий статус:** 🚧 День 1 - Создание архитектуры  
**Следующий этап:** Реализация основных классов  
**Прогресс:** ~15% (документация и структура)

---

🎯 **ЛОЗУНГ МОДУЛЯ:** *"Один интерфейс для всех токенайзеров - максимальная гибкость для 3D CNN!"*и
