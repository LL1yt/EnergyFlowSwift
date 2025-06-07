# Метаданные: Embedding Loader Module

**Версия:** 2.0.0 🎉 LLM INTEGRATION ГОТОВА!  
**Дата создания:** 5 июня 2025  
**Последнее обновление:** 5 июня 2025  
**Статус:** ✅ ГОТОВ К PRODUCTION - KNOWLEDGE DISTILLATION ENABLED!  
**Этап:** Phase 2 - LLM Integration Завершена

---

## 📦 ЭКСПОРТИРУЕМЫЙ API

### Основные классы

```python
# Главный класс для загрузки эмбедингов
class EmbeddingLoader:
    def __init__(cache_dir: str, max_cache_size: str)
    def load_embeddings(path: str, format_type: str, preprocess: bool) -> torch.Tensor
    def preprocess_embeddings(embeddings: Tensor, normalize: bool, center: bool) -> Tensor
    def cache_embeddings(embeddings: Tensor, cache_key: str) -> None
    def load_from_cache(cache_key: str) -> Optional[Tensor]
    def get_embedding_info(embeddings: Tensor) -> Dict
    def clear_cache() -> None
    def get_supported_formats() -> list

# Препроцессор для эмбедингов
class EmbeddingPreprocessor:
    def __init__()
    def preprocess(embeddings: Tensor, normalize: bool, center: bool, clip_outliers: bool, outlier_std: float) -> Tensor
    def standardize_embeddings(embeddings: Tensor) -> Tensor
    def whiten_embeddings(embeddings: Tensor) -> Tensor
    def reduce_dimensions(embeddings: Tensor, target_dim: int, method: str) -> Tensor
    def get_statistics() -> dict
    def reset_statistics() -> None
```

### Обработчики форматов

```python
# Абстрактный базовый класс
class FormatHandler(ABC):
    def load(path: str) -> Union[Tensor, np.ndarray]
    def get_vocabulary(path: str) -> Dict[str, int]

# Конкретные обработчики
class Word2VecHandler(TextFormatHandler):
    # Поддерживает .bin и .txt форматы

class GloVeHandler(TextFormatHandler):
    # Поддерживает .txt формат

class BertHandler(FormatHandler):
    # Поддерживает .pt и .pkl форматы
```

---

## 🔗 ЗАВИСИМОСТИ

### Модульные зависимости (внутренние)

#### Phase 1 модули:

- **core/lattice_3d**:

  - `Lattice3D.set_input_face()` - для подачи эмбедингов
  - `Lattice3D.get_face_data()` - для получения данных граней
  - **Статус**: ✅ Доступен

- **core/cell_prototype**:

  - `NeuralCell.get_embedding_dim()` - размерность эмбедингов
  - **Статус**: ✅ Доступен

- **utils/config_manager**:
  - `get_global_config_manager()` - централизованное управление конфигурацией
  - `ConfigManager.get_config()` - получение параметров конфигурации
  - **Статус**: ✅ Интегрирован с fallback поддержкой

#### Phase 2 модули:

- **data/tokenizer**:
  - `TokenizerManager.get_vocabulary()` - синхронизация словарей
  - **Статус**: 🔄 Будет создан (Phase 2.2)

### Внешние библиотеки

#### Обязательные:

```python
torch>=1.9.0          # Основной tensor framework
numpy>=1.21.0         # Численные операции
pyyaml>=6.0           # Конфигурационные файлы
```

#### Опциональные:

```python
gensim>=4.2.0         # Для Word2Vec .bin файлов
transformers>=4.21.0  # Для BERT токенайзеров (future)
h5py>=3.7.0          # Для HDF5 кэширования (future)
```

### UI/DOM зависимости

**Нет** - Модуль работает в backend без UI компонентов

---

## 📋 КОНФИГУРАЦИЯ

### Конфигурационные файлы

- `config/embedding_config.yaml` - основная конфигурация
- Интеграция с проектной конфигурацией через: `config/config.yaml`

### Переменные окружения

```bash
EMBEDDING_CACHE_DIR="/path/to/cache"     # Переопределение cache директории
EMBEDDING_MAX_MEMORY="4GB"              # Лимит памяти
EMBEDDING_LOG_LEVEL="INFO"              # Уровень логирования
```

### Конфигурация по умолчанию

```yaml
cache:
  cache_dir: "./data/cache/"
  max_cache_size: "2GB"

preprocessing:
  default:
    normalize: true
    center: true
    clip_outliers: false
```

---

## 🔌 ИНТЕГРАЦИОННЫЕ ТОЧКИ

### Входные интерфейсы

#### 1. File System

- **Формат**: Файлы эмбедингов (.txt, .bin, .pt, .pkl)
- **Источник**: `data/embeddings/` директория
- **Ограничения**: Максимальный размер файла ~1GB (зависит от памяти)

#### 2. Configuration System

- **Формат**: YAML конфигурация
- **Источник**: `config/embedding_config.yaml`
- **Интеграция**: Через `core.config` модуль

### Выходные интерфейсы

#### 1. К Lattice3D

```python
# Подача эмбедингов на входную грань решетки
lattice.set_input_face(embeddings: torch.Tensor)

# Batch обработка
for batch in embedding_batches:
    lattice.process_batch(batch)
```

#### 2. К Tokenizer (будущий)

```python
# Синхронизация словарей
tokenizer.sync_vocabulary(loader.get_vocabulary())

# Token-to-embedding маппинг
embedding = loader.get_token_embedding(token_id)
```

#### 3. К Visualization (будущий)

```python
# Данные для визуализации
viz_data = loader.prepare_visualization_data(embeddings)
```

---

## 📊 МЕТРИКИ И МОНИТОРИНГ

### Производительность

- **Скорость загрузки**: tokens/second, MB/second
- **Использование памяти**: peak memory, sustained memory
- **Cache hit rate**: процент попаданий в кэш
- **Throughput**: embeddings/second при обработке

### Качество данных

- **Embedding statistics**: mean, std, min, max значения
- **Dimensionality**: размерность векторов
- **Vocabulary coverage**: покрытие словаря
- **Missing values**: количество NaN/Inf значений

### Системные метрики

- **Disk usage**: размер кэша на диске
- **I/O operations**: скорость чтения файлов
- **Error rates**: частота ошибок загрузки
- **Memory leaks**: утечки памяти

---

## 🔄 LIFECYCLE MANAGEMENT

### Инициализация

```python
# Создание и настройка
loader = EmbeddingLoader(cache_dir="./cache/")
loader.configure_from_yaml("config/embedding_config.yaml")
```

### Основной цикл работы

```python
# Загрузка → Предобработка → Кэширование → Использование
embeddings = loader.load_embeddings(path, format_type)
processed = loader.preprocess_embeddings(embeddings)
loader.cache_embeddings(processed, cache_key)
```

### Завершение работы

```python
# Очистка ресурсов
loader.clear_cache()
loader.save_statistics("logs/embedding_stats.json")
```

---

## 🔧 ТЕХНИЧЕСКАЯ СПЕЦИФИКАЦИЯ

### Поддерживаемые форматы

#### Word2Vec

- **.txt**: Plain text, space-separated values
- **.bin**: Binary format (требует gensim)
- **Кодировка**: UTF-8
- **Максимальный размер**: 1GB

#### GloVe

- **.txt**: Plain text, space-separated values
- **Кодировка**: UTF-8
- **Формат**: `word value1 value2 ... valueN`

#### BERT

- **.pt**: PyTorch tensor format
- **.pkl**: Pickle serialized embeddings
- **Устройство**: CPU by default, CUDA support

### Ограничения

- **Максимальная размерность**: 4096 dimensions
- **Максимальный словарь**: 1M tokens
- **Поддерживаемые типы**: float32, float64
- **Batch size**: автоматический, зависит от памяти

---

## 📈 ВЕРСИОНИРОВАНИЕ

### Текущая версия: 1.0.0

- ✅ Базовая функциональность
- ✅ Поддержка основных форматов
- ✅ Предобработка
- 🔄 Кэширование (в разработке)
- 🔄 Интеграция с lattice_3d (в разработке)

### Планируемые версии:

- **1.1.0**: Полная интеграция с Phase 1
- **1.2.0**: Интеграция с tokenizer модулем
- **2.0.0**: Продвинутая визуализация и мониторинг

---

## 📝 CHANGELOG

### [1.0.0] - 2025-06-05

#### Added

- Базовая структура модуля
- EmbeddingLoader основной класс
- FormatHandler иерархия классов
- EmbeddingPreprocessor с полным функционалом
- YAML конфигурация
- Документация (README, plan, meta)

#### In Progress

- Unit тесты
- Интеграция с lattice_3d
- Performance optimization
- Error handling

---

**Готовность модуля**: ~15%  
**Следующий milestone**: День 2 - Format handlers тестирование  
**ETA завершения**: 7 дней (12 июня 2025)
