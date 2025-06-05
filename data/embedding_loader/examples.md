# Примеры использования: Embedding Loader Module

**Дата создания:** 5 июня 2025  
**Статус:** ✅ Готов к использованию  
**Версия модуля:** 1.0.0

---

## 🚀 БАЗОВЫЕ ПРИМЕРЫ

### Пример 1: Простая загрузка Word2Vec

```python
from data.embedding_loader import EmbeddingLoader

# Инициализация загрузчика
loader = EmbeddingLoader()

# Загрузка Word2Vec текстового файла
embeddings = loader.load_embeddings(
    path="./data/embeddings/word2vec.txt",
    format_type="word2vec",
    preprocess=True
)

print(f"Загружено эмбедингов: {embeddings.shape}")
print(f"Тип данных: {embeddings.dtype}")
print(f"Устройство: {embeddings.device}")

# Получение информации об эмбедингах
info = loader.get_embedding_info(embeddings)
print(f"Статистики: {info}")
```

**Ожидаемый вывод:**

```
Загружено эмбедингов: torch.Size([400000, 300])
Тип данных: torch.float32
Устройство: cpu
Статистики: {
    'shape': torch.Size([400000, 300]),
    'memory_mb': 457.76,
    'min_value': -1.2345,
    'max_value': 1.8765,
    'mean_value': 0.0123,
    'std_value': 0.9876
}
```

### Пример 2: Загрузка GloVe с кастомной предобработкой

```python
from data.embedding_loader import EmbeddingLoader, EmbeddingPreprocessor

# Инициализация
loader = EmbeddingLoader(cache_dir="./my_cache/")
preprocessor = EmbeddingPreprocessor()

# Загрузка без автоматической предобработки
raw_embeddings = loader.load_embeddings(
    path="./data/embeddings/glove.6B.300d.txt",
    format_type="glove",
    preprocess=False  # Отключаем автоматическую предобработку
)

print(f"Сырые эмбединги: {raw_embeddings.shape}")

# Кастомная предобработка
processed_embeddings = preprocessor.preprocess(
    raw_embeddings,
    normalize=True,      # L2 нормализация
    center=True,         # Центрирование
    clip_outliers=True,  # Обрезка выбросов
    outlier_std=2.5      # Порог для выбросов
)

print(f"Обработанные эмбединги: {processed_embeddings.shape}")

# Получение статистик предобработки
stats = preprocessor.get_statistics()
print(f"До обработки - среднее: {stats['original_mean']:.4f}")
print(f"После обработки - среднее: {stats['processed_mean']:.4f}")
```

### Пример 3: Работа с BERT эмбедингами

```python
from data.embedding_loader import EmbeddingLoader
import torch

loader = EmbeddingLoader()

# Загрузка BERT эмбедингов из PyTorch файла
bert_embeddings = loader.load_embeddings(
    path="./data/embeddings/bert_embeddings.pt",
    format_type="bert",
    preprocess=True
)

print(f"BERT эмбединги: {bert_embeddings.shape}")
print(f"Размерность: {bert_embeddings.shape[1]} (должно быть 768 для BERT-base)")

# Проверка на NaN значения
has_nan = torch.isnan(bert_embeddings).any()
print(f"Содержит NaN: {has_nan}")

# Сохранение в кэш для быстрого доступа
loader.cache_embeddings(bert_embeddings, "bert_base_processed")

# Загрузка из кэша в следующий раз
cached = loader.load_from_cache("bert_base_processed")
if cached is not None:
    print("Успешно загружено из кэша!")
```

---

## 🔧 ПРОДВИНУТЫЕ ПРИМЕРЫ

### Пример 4: Интеграция с Lattice3D

```python
from data.embedding_loader import EmbeddingLoader
from core.lattice_3d import Lattice3D
import torch

# Загрузка эмбедингов
loader = EmbeddingLoader()
embeddings = loader.load_embeddings(
    path="./data/embeddings/word2vec.bin",  # Бинарный файл
    format_type="word2vec",
    preprocess=True
)

# Создание 3D решетки
lattice = Lattice3D(width=10, height=10, depth=10)

# Подготовка данных для решетки
# Берем первые 100 векторов для входной грани
input_data = embeddings[:100]

# Подача данных на входную грань решетки
lattice.set_input_face(input_data)
print(f"Данные поданы на решетку: {input_data.shape}")

# Запуск обработки
output = lattice.propagate()
print(f"Выходные данные решетки: {output.shape}")

# Анализ результатов
output_info = loader.get_embedding_info(output)
print(f"Статистики выхода: {output_info}")
```

### Пример 5: Batch обработка больших файлов

```python
from data.embedding_loader import EmbeddingLoader
import torch
from pathlib import Path

def process_large_embeddings(file_path: str, batch_size: int = 10000):
    """
    Обработка больших файлов эмбедингов по частям.
    """
    loader = EmbeddingLoader()

    # Загрузка полного файла
    print(f"Загрузка файла: {file_path}")
    embeddings = loader.load_embeddings(file_path, "glove", preprocess=True)

    total_size = embeddings.shape[0]
    print(f"Общий размер: {total_size} векторов")

    # Обработка по батчам
    processed_batches = []

    for i in range(0, total_size, batch_size):
        end_idx = min(i + batch_size, total_size)
        batch = embeddings[i:end_idx]

        # Дополнительная обработка батча
        batch_processed = loader.preprocessor.standardize_embeddings(batch)
        processed_batches.append(batch_processed)

        print(f"Обработан батч {i//batch_size + 1}: {batch.shape}")

    # Объединение результатов
    final_result = torch.cat(processed_batches, dim=0)
    print(f"Финальный результат: {final_result.shape}")

    return final_result

# Использование
result = process_large_embeddings("./data/embeddings/glove.840B.300d.txt")
print("Converted embeddings shape:", result.shape)
print("Used preprocessing:", loader.stats['preprocessing_used'])
```

### Пример 6: Работа с конфигурацией

```python
from data.embedding_loader import EmbeddingLoader
import yaml

# Кастомная конфигурация
custom_config = {
    'cache': {
        'cache_dir': './custom_cache/',
        'max_cache_size': '4GB'
    },
    'preprocessing': {
        'default': {
            'normalize': False,
            'center': True,
            'clip_outliers': True,
            'outlier_std': 2.0
        }
    }
}

# Сохранение конфигурации
with open('./temp_config.yaml', 'w') as f:
    yaml.dump(custom_config, f)

# Создание загрузчика с кастомной конфигурацией
loader = EmbeddingLoader(
    cache_dir=custom_config['cache']['cache_dir'],
    max_cache_size=custom_config['cache']['max_cache_size']
)

# Использование кастомных параметров предобработки
embeddings = loader.load_embeddings(
    path="./data/embeddings/test.txt",
    format_type="glove",
    preprocess=False  # Отключаем автоматическую
)

# Применяем кастомную предобработку
processed = loader.preprocess_embeddings(
    embeddings,
    normalize=custom_config['preprocessing']['default']['normalize'],
    center=custom_config['preprocessing']['default']['center']
)

print(f"Обработано с кастомными параметрами: {processed.shape}")
```

---

## 🧪 ПРИМЕРЫ ТЕСТИРОВАНИЯ

### Пример 7: Тестирование производительности

```python
import time
from data.embedding_loader import EmbeddingLoader
import torch

def benchmark_loading(file_path: str, format_type: str, num_runs: int = 3):
    """
    Бенчмарк скорости загрузки эмбедингов.
    """
    loader = EmbeddingLoader()

    times = []

    for run in range(num_runs):
        # Очищаем кэш для честного теста
        loader.clear_cache()

        start_time = time.time()
        embeddings = loader.load_embeddings(file_path, format_type, preprocess=True)
        end_time = time.time()

        load_time = end_time - start_time
        times.append(load_time)

        print(f"Запуск {run + 1}: {load_time:.2f} сек, "
              f"Размер: {embeddings.shape}, "
              f"Скорость: {embeddings.shape[0] / load_time:.0f} vectors/sec")

    avg_time = sum(times) / len(times)
    print(f"\nСредняя скорость загрузки: {avg_time:.2f} сек")

    return avg_time

# Тестирование
benchmark_loading("./data/embeddings/glove.6B.100d.txt", "glove")
```

### Пример 8: Тестирование кэширования

```python
from data.embedding_loader import EmbeddingLoader
import time

loader = EmbeddingLoader()

# Первая загрузка (холодный старт)
print("=== Первая загрузка (без кэша) ===")
start = time.time()
embeddings1 = loader.load_embeddings("./data/embeddings/test.txt", "glove")
first_load_time = time.time() - start
print(f"Время первой загрузки: {first_load_time:.2f} сек")

# Вторая загрузка (из кэша)
print("\n=== Вторая загрузка (из кэша) ===")
start = time.time()
embeddings2 = loader.load_embeddings("./data/embeddings/test.txt", "glove")
second_load_time = time.time() - start
print(f"Время второй загрузки: {second_load_time:.2f} сек")

# Проверка идентичности
are_identical = torch.equal(embeddings1, embeddings2)
print(f"Данные идентичны: {are_identical}")

# Ускорение от кэша
speedup = first_load_time / second_load_time
print(f"Ускорение от кэша: {speedup:.1f}x")
```

---

## 🔍 ПРИМЕРЫ ДИАГНОСТИКИ

### Пример 9: Анализ качества эмбедингов

```python
from data.embedding_loader import EmbeddingLoader, EmbeddingPreprocessor
import torch
import numpy as np

def analyze_embedding_quality(embeddings: torch.Tensor):
    """
    Анализ качества загруженных эмбедингов.
    """
    print("=== АНАЛИЗ КАЧЕСТВА ЭМБЕДИНГОВ ===")

    # Базовые статистики
    print(f"Размер: {embeddings.shape}")
    print(f"Тип данных: {embeddings.dtype}")
    print(f"Устройство: {embeddings.device}")
    print(f"Память: {embeddings.element_size() * embeddings.nelement() / 1024**2:.1f} MB")

    # Статистики значений
    print(f"\nМин значение: {embeddings.min():.4f}")
    print(f"Макс значение: {embeddings.max():.4f}")
    print(f"Среднее: {embeddings.mean():.4f}")
    print(f"Стандартное отклонение: {embeddings.std():.4f}")

    # Проверка на проблемы
    nan_count = torch.isnan(embeddings).sum().item()
    inf_count = torch.isinf(embeddings).sum().item()
    zero_vectors = (embeddings.norm(dim=1) == 0).sum().item()

    print(f"\nПроблемы:")
    print(f"NaN значений: {nan_count}")
    print(f"Inf значений: {inf_count}")
    print(f"Нулевых векторов: {zero_vectors}")

    # Анализ распределения норм
    norms = torch.norm(embeddings, dim=1)
    print(f"\nНормы векторов:")
    print(f"Средняя норма: {norms.mean():.4f}")
    print(f"Мин норма: {norms.min():.4f}")
    print(f"Макс норма: {norms.max():.4f}")

    return {
        'shape': embeddings.shape,
        'has_issues': nan_count > 0 or inf_count > 0 or zero_vectors > 0,
        'mean_norm': float(norms.mean()),
        'std_norm': float(norms.std())
    }

# Использование
loader = EmbeddingLoader()
embeddings = loader.load_embeddings("./data/embeddings/test.txt", "glove")
quality_report = analyze_embedding_quality(embeddings)

if quality_report['has_issues']:
    print("\n⚠️  Обнаружены проблемы с данными!")
else:
    print("\n✅ Эмбединги выглядят корректными")
```

### Пример 10: Сравнение различных форматов

```python
from data.embedding_loader import EmbeddingLoader
import time

def compare_formats():
    """
    Сравнение загрузки различных форматов эмбедингов.
    """
    loader = EmbeddingLoader()

    files_to_test = [
        ("./data/embeddings/word2vec.txt", "word2vec"),
        ("./data/embeddings/word2vec.bin", "word2vec"),
        ("./data/embeddings/glove.txt", "glove"),
        ("./data/embeddings/bert.pt", "bert")
    ]

    results = []

    for file_path, format_type in files_to_test:
        try:
            print(f"\n=== Тестирование {format_type}: {file_path} ===")

            start_time = time.time()
            embeddings = loader.load_embeddings(file_path, format_type)
            load_time = time.time() - start_time

            info = loader.get_embedding_info(embeddings)

            result = {
                'format': format_type,
                'file': file_path,
                'load_time': load_time,
                'shape': embeddings.shape,
                'memory_mb': info['memory_mb'],
                'speed_vectors_per_sec': embeddings.shape[0] / load_time
            }

            results.append(result)

            print(f"Время загрузки: {load_time:.2f} сек")
            print(f"Размер: {embeddings.shape}")
            print(f"Скорость: {result['speed_vectors_per_sec']:.0f} vectors/sec")

        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")

    # Сравнительная таблица
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*80)
    print(f"{'Формат':<12} {'Время (сек)':<12} {'Размер':<20} {'Скорость (v/s)':<15}")
    print("-" * 80)

    for result in results:
        print(f"{result['format']:<12} {result['load_time']:<12.2f} "
              f"{str(result['shape']):<20} {result['speed_vectors_per_sec']:<15.0f}")

# Запуск сравнения
compare_formats()
```

---

## 📝 ЗАМЕТКИ ПО ИСПОЛЬЗОВАНИЮ

### Важные моменты:

1. **Память**: Большие файлы эмбедингов могут требовать значительного объема памяти
2. **Кэширование**: Используйте кэш для ускорения повторных загрузок
3. **Предобработка**: Всегда применяйте предобработку для лучшего качества
4. **Форматы**: Бинарные Word2Vec файлы требуют установленный gensim
5. **Производительность**: Используйте batch обработку для больших датасетов

### Рекомендуемый workflow:

```python
# 1. Инициализация с кэшем
loader = EmbeddingLoader(cache_dir="./cache/")

# 2. Загрузка с предобработкой
embeddings = loader.load_embeddings(path, format_type, preprocess=True)

# 3. Проверка качества
info = loader.get_embedding_info(embeddings)
print(f"Качество: {info}")

# 4. Интеграция с решеткой
lattice.set_input_face(embeddings[:batch_size])

# 5. Очистка ресурсов
loader.clear_cache()
```

---

## 🚀 НОВЫЕ ПРИМЕРЫ: LLM & Knowledge Distillation

### Пример 11: Базовое использование LLM

```python
from data.embedding_loader import EmbeddingLoader

# Инициализация
loader = EmbeddingLoader()

# Список текстов для обработки
texts = [
    "Artificial intelligence is transforming the world",
    "Machine learning models are becoming more sophisticated",
    "Neural networks can learn complex patterns"
]

# Генерация эмбедингов через LLM
embeddings = loader.load_from_llm(
    texts=texts,
    model_key="distilbert",  # Легкая модель для тестирования
    pooling_strategy="mean"
)

print(f"Generated embeddings: {embeddings.shape}")
print(f"Model used: distilbert")
```

### Пример 12: Knowledge Distillation с LLaMA

```python
# Использование более мощной модели для production
texts = [
    "The future of AI is bright",
    "Deep learning revolutionizes computing",
    "Natural language processing advances rapidly"
]

# Генерация с LLaMA 2 (требует больше ресурсов)
try:
    embeddings = loader.load_from_llm(
        texts=texts,
        model_key="llama2-7b",  # Мощная модель
        pooling_strategy="mean",
        use_cache=True  # Кэшируем результаты
    )
    print(f"LLaMA embeddings: {embeddings.shape}")
except Exception as e:
    print(f"LLaMA не доступна: {e}")
    # Fallback на легкую модель
    embeddings = loader.load_from_llm(texts=texts, model_key="distilbert")
```

### Пример 13: Создание обучающего датасета

```python
# Большой набор текстов для обучения
training_texts = [
    "Quantum computing will change cryptography",
    "Blockchain technology ensures data integrity",
    "Cloud computing provides scalable infrastructure",
    "Internet of Things connects everyday devices",
    "Augmented reality enhances user experience",
    # ... еще тысячи текстов
]

# Создание датасета для Knowledge Distillation
dataset = loader.create_knowledge_distillation_dataset(
    texts=training_texts,
    teacher_model="llama3-8b",  # Teacher: мощная LLM
    save_path="./datasets/kd_dataset_llama3.pt"
)

print(f"Создан датасет: {dataset['num_samples']} образцов")
print(f"Размерность эмбедингов: {dataset['embedding_dim']}")
print(f"Teacher модель: {dataset['teacher_model']}")

# Этот датасет будет использован для обучения 3D CNN в Phase 3
```

### Пример 14: Батчевая обработка больших данных

```python
# Большой объем текстов
large_text_corpus = ["Text sample " + str(i) for i in range(10000)]

# Эффективная батчевая обработка
embeddings = loader.batch_load_from_llm(
    texts=large_text_corpus,
    model_key="roberta",
    batch_size=32  # Обрабатываем по 32 текста
)

print(f"Обработано {len(large_text_corpus)} текстов")
print(f"Результирующие эмбединги: {embeddings.shape}")
print(f"Статистики: {loader.stats}")
```

### Пример 15: Сравнение разных LLM моделей

```python
# Тестовый текст
test_text = ["The quick brown fox jumps over the lazy dog"]

# Список моделей для сравнения
models_to_test = ["distilbert", "roberta", "gpt2"]

results = {}
for model in models_to_test:
    try:
        embeddings = loader.load_from_llm(
            texts=test_text,
            model_key=model,
            pooling_strategy="mean"
        )

        # Получаем информацию о модели
        model_info = loader.get_llm_info(model)

        results[model] = {
            'embeddings': embeddings,
            'shape': embeddings.shape,
            'hidden_size': model_info['hidden_size'],
            'memory_req': "Unknown"  # Из конфигурации
        }

        print(f"✅ {model}: {embeddings.shape}")

    except Exception as e:
        print(f"❌ {model}: {e}")

# Сравнение результатов
for model, data in results.items():
    print(f"{model}: {data['shape']}, hidden_size: {data['hidden_size']}")
```

### Пример 16: Интеграция с 3D CNN (Готовность к Phase 3)

```python
from core.lattice_3d import Lattice3D

# Создание датасета для обучения 3D CNN
texts = ["AI will transform education", "Machine learning improves healthcare"]

# Teacher: LLM генерирует качественные эмбединги
teacher_embeddings = loader.load_from_llm(
    texts=texts,
    model_key="mistral-7b",
    pooling_strategy="mean"
)

# Student: наша 3D CNN (будет реализовано в Phase 3)
lattice = Lattice3D(size=(10, 10, 10))

# Подача эмбедингов на входную грань решетки
input_face = teacher_embeddings[:, :lattice.get_face_size()]
lattice.set_input_face(input_face)

print("✅ Интеграция с 3D CNN готова для Phase 3!")
print(f"Teacher embeddings: {teacher_embeddings.shape}")
print(f"Lattice input: {input_face.shape}")
```

### Пример 17: Продвинутые настройки pooling

```python
# Тестирование разных стратегий агрегации
text = ["Natural language processing is a branch of artificial intelligence"]

pooling_strategies = ["mean", "cls", "max"]
results = {}

for strategy in pooling_strategies:
    embeddings = loader.load_from_llm(
        texts=text,
        model_key="distilbert",
        pooling_strategy=strategy
    )
    results[strategy] = embeddings
    print(f"{strategy} pooling: {embeddings.shape}")

# Анализ различий
mean_emb = results["mean"]
cls_emb = results["cls"]
max_emb = results["max"]

print(f"Mean vs CLS similarity: {torch.cosine_similarity(mean_emb, cls_emb).item():.4f}")
print(f"Mean vs Max similarity: {torch.cosine_similarity(mean_emb, max_emb).item():.4f}")
```

### Пример 18: Кэширование и оптимизация

```python
import time

# Тест эффективности кэширования
texts = ["This is a test sentence for caching performance"]

# Первый запрос (без кэша)
start_time = time.time()
embeddings1 = loader.load_from_llm(texts=texts, model_key="gpt2", use_cache=True)
first_call_time = time.time() - start_time

# Второй запрос (из кэша)
start_time = time.time()
embeddings2 = loader.load_from_llm(texts=texts, model_key="gpt2", use_cache=True)
cached_call_time = time.time() - start_time

# Проверка идентичности результатов
identical = torch.allclose(embeddings1, embeddings2)

print(f"Первый вызов: {first_call_time:.2f}s")
print(f"Кэшированный вызов: {cached_call_time:.2f}s")
print(f"Ускорение: {first_call_time/cached_call_time:.1f}x")
print(f"Результаты идентичны: {identical}")
print(f"Статистики кэша: hits={loader.stats['cache_hits']}, misses={loader.stats['cache_misses']}")
```

### Пример 19: Мониторинг и диагностика

```python
# Получение информации о поддерживаемых моделях
supported_models = loader.list_supported_llm_models()
print("Поддерживаемые LLM модели:")
for model in supported_models:
    print(f"  - {model}")

# Детальная информация о конкретной модели
model_info = loader.get_llm_info("distilbert")
print(f"\nИнформация о модели:")
for key, value in model_info.items():
    print(f"  {key}: {value}")

# Статистики использования
print(f"\nСтатистики embedding_loader:")
for key, value in loader.stats.items():
    print(f"  {key}: {value}")
```

### Пример 20: Production Pipeline (готовность к Phase 3)

```python
# Полный production pipeline для Knowledge Distillation
class ProductionKnowledgeDistillation:
    def __init__(self):
        self.loader = EmbeddingLoader()
        self.teacher_model = "llama3-8b"  # Мощная teacher модель

    def prepare_training_data(self, text_corpus, save_dir="./kd_datasets/"):
        """Подготовка данных для обучения 3D CNN"""

        # Создание train/validation датасетов
        train_texts = text_corpus[:int(0.8 * len(text_corpus))]
        val_texts = text_corpus[int(0.8 * len(text_corpus)):]

        # Генерация train датасета
        train_dataset = self.loader.create_knowledge_distillation_dataset(
            texts=train_texts,
            teacher_model=self.teacher_model,
            save_path=f"{save_dir}/train_kd_dataset.pt"
        )

        # Генерация validation датасета
        val_dataset = self.loader.create_knowledge_distillation_dataset(
            texts=val_texts,
            teacher_model=self.teacher_model,
            save_path=f"{save_dir}/val_kd_dataset.pt"
        )

        return train_dataset, val_dataset

    def get_teacher_embeddings(self, texts):
        """Получение teacher эмбедингов для inference"""
        return self.loader.load_from_llm(
            texts=texts,
            model_key=self.teacher_model,
            pooling_strategy="mean",
            use_cache=True
        )

# Использование
pipeline = ProductionKnowledgeDistillation()

# Подготовка данных для Phase 3
sample_corpus = [f"Training sample {i}" for i in range(1000)]
train_data, val_data = pipeline.prepare_training_data(sample_corpus)

print("🚀 Production pipeline готов!")
print(f"Train dataset: {train_data['num_samples']} образцов")
print(f"Validation dataset: {val_data['num_samples']} образцов")
print("✅ Готово для Phase 3: Training Infrastructure")
```

---

## 📋 Резюме новых возможностей

### 🎯 Основные фичи

1. **LLM Integration**: Поддержка 8+ моделей (LLaMA, Mistral, GPT и др.)
2. **Knowledge Distillation**: Полный pipeline создания обучающих данных
3. **Real-time Generation**: Генерация эмбедингов в реальном времени
4. **Smart Caching**: Интеллигентное кэширование LLM результатов
5. **Batch Processing**: Эффективная обработка больших объемов
6. **Multi-device Support**: CPU и GPU поддержка
7. **Production Ready**: Готовность к Phase 3 Training Infrastructure

### 🚀 Готовность к Phase 3

- ✅ **Teacher-Student Architecture**: Готова инфраструктура для KD
- ✅ **Dataset Generation**: Автоматическое создание обучающих данных
- ✅ **Integration Points**: API для подключения к training loop
- ✅ **Quality Metrics**: Мониторинг качества эмбедингов
- ✅ **Scalability**: Поддержка production нагрузок

**🎯 Следующий шаг**: Переход к Phase 3 для реализации training infrastructure с полной поддержкой knowledge distillation!
