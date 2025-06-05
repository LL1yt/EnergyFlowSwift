# Embedding Loader Module

**Версия:** 1.0.0  
**Статус:** ✅ В разработке (Phase 2)  
**Зависимости:** torch, numpy, gensim (опционально)

## 📝 Назначение

Модуль для загрузки и предобработки векторных представлений (эмбедингов) различных типов. Обеспечивает унифицированный интерфейс для работы с популярными форматами эмбедингов в контексте 3D клеточной нейронной сети.

### Основные возможности

- **Многоформатная загрузка**: Word2Vec (.bin, .txt), GloVe (.txt), BERT (.pt, .pkl)
- **Умная предобработка**: Нормализация, центрирование, обрезка выбросов
- **Кэширование**: Автоматическое кэширование для быстрого повторного доступа
- **Интеграция**: Seamless интеграция с core/lattice_3d модулем
- **Производительность**: Оптимизация для работы с большими файлами (>100MB)

## 🚀 Быстрый старт

### Базовое использование

```python
from data.embedding_loader import EmbeddingLoader

# Инициализация загрузчика
loader = EmbeddingLoader(cache_dir="./data/cache/")

# Загрузка Word2Vec эмбедингов
embeddings = loader.load_embeddings(
    path="./data/embeddings/word2vec.bin",
    format_type="word2vec",
    preprocess=True
)

print(f"Загружены эмбединги: {embeddings.shape}")
print(f"Поддерживаемые форматы: {loader.get_supported_formats()}")
```

### Продвинутое использование

```python
from data.embedding_loader import EmbeddingLoader, EmbeddingPreprocessor

# Кастомная предобработка
loader = EmbeddingLoader()
preprocessor = EmbeddingPreprocessor()

# Загрузка без автоматической предобработки
raw_embeddings = loader.load_embeddings(
    path="./data/embeddings/glove.txt",
    format_type="glove",
    preprocess=False
)

# Кастомная предобработка
processed_embeddings = preprocessor.preprocess(
    raw_embeddings,
    normalize=True,
    center=True,
    clip_outliers=True,
    outlier_std=2.5
)

# Получение статистик
stats = preprocessor.get_statistics()
print(f"Статистики предобработки: {stats}")
```

## 📁 Структура модуля

```
data/embedding_loader/
├── __init__.py              # Экспорты модуля
├── embedding_loader.py      # Основной класс EmbeddingLoader
├── format_handlers.py       # Обработчики форматов
├── preprocessing.py         # Предобработка эмбедингов
├── config/
│   └── embedding_config.yaml # Конфигурация
├── README.md               # Этот файл
├── plan.md                 # План реализации
├── meta.md                 # Метаданные и зависимости
├── errors.md               # Документация ошибок
├── diagram.mmd             # Архитектурная диаграмма
└── examples.md             # Примеры использования
```

## 🔧 Конфигурация

Конфигурация осуществляется через `config/embedding_config.yaml`:

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

## 🎯 Интеграция с 3D Lattice

```python
from core.lattice_3d import Lattice3D
from data.embedding_loader import EmbeddingLoader

# Загрузка эмбедингов
loader = EmbeddingLoader()
embeddings = loader.load_embeddings("path/to/embeddings.txt", "glove")

# Создание решетки
lattice = Lattice3D(width=10, height=10, depth=10)

# Подача эмбедингов на входную грань
lattice.set_input_face(embeddings[:100])  # Первые 100 векторов

# Запуск обработки
output = lattice.propagate()
```

## 📋 Требования

### Системные требования

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- gensim 4.2+ (для Word2Vec .bin файлов)

### Память и производительность

- **Минимум**: 4GB RAM для файлов до 100MB
- **Рекомендуется**: 8GB RAM для файлов до 1GB
- **Кэш**: Требует дискового пространства ~2x от размера файла

## 🔍 Мониторинг и диагностика

```python
# Получение информации об эмбедингах
info = loader.get_embedding_info(embeddings)
print(f"Информация: {info}")

# Статистики предобработки
stats = loader.preprocessor.get_statistics()
print(f"Статистики: {stats}")

# Очистка кэша
loader.clear_cache()
```

## 🚨 Известные ограничения

- Gensim требуется только для бинарных Word2Vec файлов
- Максимальный размер файла ограничен доступной памятью
- BERT embeddings должны быть предварительно извлечены из модели

## 🔗 Связанные модули

- **core/lattice_3d**: Основная интеграция для подачи данных
- **data/tokenizer**: Совместная работа с токенизацией
- **data/data_visualization**: Визуализация загруженных эмбедингов

## 📞 Поддержка

При возникновении проблем обращайтесь к:

- `errors.md` - документированные ошибки и решения
- `examples.md` - детальные примеры использования
- Лог файлы в `logs/` директории
