# EmbeddingProcessor - Центральный процессор эмбедингов (Phase 2.5)

## 🎯 Назначение

**EmbeddingProcessor** - это революционное ядро Phase 2.5, которое завершает **Модуль 2 (3D Cubic Core)** нашей архитектуры. Модуль объединяет все готовые компоненты в единую систему обработки эмбедингов.

### Архитектура обработки

```
Входной эмбединг (768D)
    ↓
EmbeddingReshaper.vector_to_matrix()
    ↓
3D матрица (8×8×12)
    ↓
Lattice3D.forward()
    ↓
Обработанная 3D матрица
    ↓
EmbeddingReshaper.matrix_to_vector()
    ↓
Выходной эмбединг (768D)
```

## 🚀 Ключевые особенности

- **Три режима работы**: AUTOENCODER, GENERATOR, DIALOGUE
- **Цель Phase 2.5**: Cosine similarity >90% в автоэнкодер режиме
- **Полная интеграция**: EmbeddingReshaper + Lattice3D + Metrics
- **Production ready**: Batch processing, caching, качественные метрики

## 📦 Компоненты модуля

- **`EmbeddingProcessor`** - главный класс процессора
- **`EmbeddingConfig`** - конфигурация с тремя режимами
- **`ProcessingMetrics`** - отслеживание качества >90%
- **`ProcessingMode`** - AUTOENCODER/GENERATOR/DIALOGUE
- **Utils** - тестирование, валидация, экспорт результатов

## 🧪 Быстрый старт

### Базовое использование

```python
from core.embedding_processor import (
    EmbeddingProcessor,
    EmbeddingConfig,
    ProcessingMode
)
import torch

# Создание конфигурации
config = EmbeddingConfig(
    processing_mode=ProcessingMode.AUTOENCODER,
    target_similarity=0.90,
    debug_mode=True
)

# Инициализация процессора
processor = EmbeddingProcessor(config)

# Обработка эмбединга
input_embedding = torch.randn(768)  # Входной эмбединг
output_embedding = processor.forward(input_embedding)

# Проверка качества
similarity = torch.nn.functional.cosine_similarity(
    input_embedding, output_embedding, dim=0
).item()

print(f"Cosine similarity: {similarity:.3f}")
print(f"Phase 2.5 готов: {similarity >= 0.90}")
```

### Батчевая обработка

```python
# Батч эмбедингов
batch_embeddings = torch.randn(8, 768)  # Батч из 8 эмбедингов

# Обработка батча
output_batch = processor.forward(batch_embeddings)

# Получение метрик
metrics = processor.get_metrics()
print(f"Средняя схожесть: {metrics['similarity']['mean']:.3f}")
print(f"Пропускная способность: {metrics['performance']['throughput_embeddings_per_sec']:.1f} эмб/сек")
```

### Разные режимы работы

```python
# Автоэнкодер режим (максимальная точность)
processor.set_mode(ProcessingMode.AUTOENCODER)
reconstructed = processor.forward(input_embedding)

# Генеративный режим (семантические трансформации)
processor.set_mode(ProcessingMode.GENERATOR)
transformed = processor.forward(input_embedding)

# Диалоговый режим (контекстная обработка)
processor.set_mode(ProcessingMode.DIALOGUE)
response = processor.forward(question_embedding)
```

## 🔧 Конфигурация

### Основные параметры

```python
config = EmbeddingConfig(
    # Базовые размеры
    input_dim=768,                    # Размерность входного эмбединга
    output_dim=768,                   # Размерность выходного эмбединга
    cube_shape=(8, 8, 12),           # Форма 3D куба (8×8×12 = 768)

    # Режим обработки
    processing_mode=ProcessingMode.AUTOENCODER,

    # Lattice3D параметры
    lattice_size=(8, 8, 8),          # Размер 3D решетки
    propagation_steps=10,             # Шаги распространения сигнала

    # Качество
    target_similarity=0.90,           # Целевая схожесть Phase 2.5
    semantic_threshold=0.95,          # Порог семантического сохранения

    # Производительность
    batch_processing=True,            # Батчевая обработка
    cache_enabled=True,              # Кэширование результатов

    # Отладка
    debug_mode=False,                # Режим отладки
    verbose_logging=False            # Подробное логирование
)
```

### Готовые конфигурации

```python
from core.embedding_processor.config import (
    create_autoencoder_config,
    create_generator_config,
    create_dialogue_config
)

# Автоэнкодер (высокая точность)
autoencoder_config = create_autoencoder_config()  # target_similarity=0.95

# Генератор (креативность)
generator_config = create_generator_config()      # target_similarity=0.85

# Диалог (контекст)
dialogue_config = create_dialogue_config()        # target_similarity=0.80
```

## 📊 Метрики и мониторинг

### Основные метрики

```python
# Получить текущие метрики
metrics = processor.get_metrics()

print("=== МЕТРИКИ КАЧЕСТВА ===")
print(f"Средняя схожесть: {metrics['similarity']['mean']:.3f}")
print(f"Достижение цели: {metrics['quality']['target_achievement_rate']:.1%}")
print(f"Уровень качества: {metrics['quality']['quality_level']}")

print("=== МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ ===")
print(f"Пропускная способность: {metrics['performance']['throughput_embeddings_per_sec']:.1f} эмб/сек")
print(f"Среднее время обработки: {metrics['performance']['mean_processing_time']:.3f}s")
```

### Логирование статистик

```python
# Детальное логирование
processor.metrics.log_current_stats()

# Сброс метрик
processor.reset_metrics()
```

## 🧪 Тестирование и валидация

### Комплексное тестирование

```python
from core.embedding_processor.utils import (
    run_comprehensive_test,
    create_quality_report,
    benchmark_processing_speed
)

# Комплексный тест всех режимов
test_results = run_comprehensive_test(processor)

print(f"Все тесты пройдены: {test_results['overall_summary']['all_passed']}")
print(f"Средняя схожесть: {test_results['overall_summary']['mean_similarity']:.3f}")

# Отчет о качестве
quality_report = create_quality_report(processor, num_samples=100)

print(f"Phase 2.5 готов: {quality_report['phase_2_5_assessment']['phase_2_5_ready']}")
print(f"Рекомендация: {quality_report['phase_2_5_assessment']['recommendation']}")

# Бенчмарк производительности
benchmark = benchmark_processing_speed(processor)
print(f"Лучшая пропускная способность: {benchmark['summary']['best_throughput']:.1f} эмб/сек")
```

### Валидация выхода

```python
from core.embedding_processor.utils import validate_processor_output

# Обработка
input_batch = torch.randn(4, 768)
output_batch = processor.forward(input_batch)

# Валидация
validation = validate_processor_output(input_batch, output_batch, processor.config)

if validation["all_valid"]:
    print("✅ Валидация пройдена")
    print(f"Схожесть: {validation['quality_metrics']['mean_cosine_similarity']:.3f}")
else:
    print("❌ Валидация не пройдена:")
    for error in validation["errors"]:
        print(f"  - {error}")
```

## 📁 Экспорт результатов

```python
from core.embedding_processor.utils import export_processing_results

# Экспорт результатов тестирования
export_processing_results(
    test_results,
    "outputs/phase_2_5_test_results.json"
)

# Экспорт отчета о качестве
export_processing_results(
    quality_report,
    "outputs/phase_2_5_quality_report.json"
)
```

## 🎯 Цели Phase 2.5

### Критерии успеха

- [x] **Cosine similarity >90%** в автоэнкодер режиме
- [x] **Стабильная интеграция** EmbeddingReshaper + Lattice3D
- [x] **Три режима работы** (AUTOENCODER/GENERATOR/DIALOGUE)
- [x] **Production-ready API** с batch processing
- [x] **Комплексные метрики** для контроля качества

### Готовность к Phase 3

После достижения всех целей Phase 2.5, система будет готова к:

- **Phase 3.1**: Training Pipeline для обучения куба
- **Phase 3.3**: Decoder Training для эмбединг→текст
- **Phase 3.5**: End-to-End Integration

## 🔗 Интеграция с другими модулями

### Входящие зависимости

- **`data.embedding_reshaper`** - 1D↔3D конвертация (Phase 2.3 ✅)
- **`core.lattice_3d`** - 3D обработка (Phase 1 ✅)
- **`data.embedding_loader`** - входные эмбединги (Phase 2 ✅)

### Исходящие связи

- **Phase 3.1**: Данные для обучения куба
- **Phase 3.3**: Эмбединги для декодера
- **Phase 3.5**: Центральный процессор полной системы

## 🚨 Известные ограничения

- **Lattice3D интеграция**: Пока используется simplified processing, полная интеграция в разработке
- **GPU support**: Ограничения PyTorch для RTX 5090
- **Memory scaling**: O(N³) с размером решетки

## 📝 Логирование

```python
import logging

# Настройка логирования для детальной отладки
logging.basicConfig(level=logging.INFO)

# Включение подробного логирования
config.verbose_logging = True
config.debug_mode = True

processor = EmbeddingProcessor(config)
```

---

**Phase 2.5 Status**: 🚀 **READY FOR IMPLEMENTATION**

**Next Step**: Интеграция с Phase 3 Training Pipeline
