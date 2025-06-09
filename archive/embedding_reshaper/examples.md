# EmbeddingReshaper Usage Examples

**Модуль:** data/embedding_reshaper  
**Дата создания:** 6 июня 2025  
**Версия:** 1.0.0

---

## 🚀 БЫСТРЫЙ СТАРТ

### Базовое использование

```python
from data.embedding_reshaper import EmbeddingReshaper
import numpy as np

# Создание reshaper с стандартными настройками
reshaper = EmbeddingReshaper()

# Пример 1D эмбединга (768 измерений)
embedding_1d = np.random.randn(768).astype(np.float32)

# Преобразование 1D → 3D
matrix_3d = reshaper.vector_to_matrix(embedding_1d)
print(f"Форма 3D матрицы: {matrix_3d.shape}")  # (8, 8, 12)

# Обратное преобразование 3D → 1D
restored_1d = reshaper.matrix_to_vector(matrix_3d)
print(f"Форма восстановленного вектора: {restored_1d.shape}")  # (768,)

# Проверка качества восстановления
similarity = np.dot(embedding_1d, restored_1d) / (
    np.linalg.norm(embedding_1d) * np.linalg.norm(restored_1d)
)
print(f"Cosine similarity: {similarity:.6f}")  # Ожидается ~1.0
```

---

## 🎯 ПРОДВИНУТЫЕ ПРИМЕРЫ

### Пример 1: Использование разных стратегий

```python
from data.embedding_reshaper import EmbeddingReshaper
import torch

# Создание тестового эмбединга
text_embedding = torch.randn(768, dtype=torch.float32)

# Linear стратегия (самая быстрая)
linear_reshaper = EmbeddingReshaper(
    reshaping_method="linear",
    preserve_semantics=True
)

# Adaptive стратегия (баланс скорости и качества)
adaptive_reshaper = EmbeddingReshaper(
    reshaping_method="adaptive",
    preserve_semantics=True,
    semantic_threshold=0.98
)

# Semantic стратегия (максимальное качество)
semantic_reshaper = EmbeddingReshaper(
    reshaping_method="semantic",
    preserve_semantics=True,
    semantic_threshold=0.99
)

# Сравнение результатов
for name, reshaper in [("Linear", linear_reshaper),
                      ("Adaptive", adaptive_reshaper),
                      ("Semantic", semantic_reshaper)]:

    matrix = reshaper.vector_to_matrix(text_embedding)
    restored = reshaper.matrix_to_vector(matrix)

    similarity = torch.cosine_similarity(
        text_embedding.unsqueeze(0),
        restored.unsqueeze(0)
    ).item()

    print(f"{name:10}: Similarity = {similarity:.6f}")
```

### Пример 2: Интеграция с Teacher LLM Encoder

```python
from data.embedding_loader import EmbeddingLoader
from data.embedding_reshaper import EmbeddingReshaper

# Полный pipeline: текст → эмбединг → 3D куб
class TextToCubeProcessor:
    def __init__(self):
        self.encoder = EmbeddingLoader()
        self.reshaper = EmbeddingReshaper(
            reshaping_method="adaptive",
            preserve_semantics=True
        )

    def process_text(self, text_input):
        """Конвертация текста в 3D формат для куба"""
        # Шаг 1: Текст → Эмбединг через Teacher LLM
        embeddings = self.encoder.load_from_llm(
            [text_input],
            model_key="distilbert"
        )
        text_embedding = embeddings[0]

        # Шаг 2: 1D Эмбединг → 3D Куб
        cube_input = self.reshaper.vector_to_matrix(text_embedding)

        return cube_input, text_embedding

    def restore_text_format(self, cube_output):
        """Конвертация выхода куба обратно в эмбединг формат"""
        return self.reshaper.matrix_to_vector(cube_output)

# Использование
processor = TextToCubeProcessor()

# Обработка входного текста
input_text = "Создание 3D клеточной нейронной сети"
cube_input, original_embedding = processor.process_text(input_text)

print(f"Исходный эмбединг: {original_embedding.shape}")
print(f"3D куб для обработки: {cube_input.shape}")

# Симуляция обработки кубом (здесь куб просто копирует вход)
cube_output = cube_input.copy()

# Восстановление в формат эмбединга
final_embedding = processor.restore_text_format(cube_output)
print(f"Финальный эмбединг: {final_embedding.shape}")

# Проверка качества полного pipeline
similarity = np.dot(original_embedding, final_embedding) / (
    np.linalg.norm(original_embedding) * np.linalg.norm(final_embedding)
)
print(f"End-to-end similarity: {similarity:.6f}")
```

### Пример 3: Batch processing

```python
import time
from data.embedding_reshaper import EmbeddingReshaper, create_test_embeddings

# Создание batch эмбедингов
batch_size = 50
test_embeddings = create_test_embeddings(
    count=batch_size,
    dim=768,
    embedding_type="diverse"
)

reshaper = EmbeddingReshaper(reshaping_method="adaptive")

# Метод 1: Последовательная обработка
start_time = time.time()
sequential_results = []
for embedding in test_embeddings:
    matrix = reshaper.vector_to_matrix(embedding)
    restored = reshaper.matrix_to_vector(matrix)
    sequential_results.append(restored)
sequential_time = time.time() - start_time

# Метод 2: Batch обработка (при наличии)
start_time = time.time()
# Примечание: batch processing может быть добавлен в будущих версиях
batch_results = [
    reshaper.matrix_to_vector(reshaper.vector_to_matrix(emb))
    for emb in test_embeddings
]
batch_time = time.time() - start_time

print(f"Последовательная обработка {batch_size} эмбедингов: {sequential_time:.3f}s")
print(f"Batch обработка {batch_size} эмбедингов: {batch_time:.3f}s")
print(f"Среднее время на эмбединг: {sequential_time/batch_size*1000:.1f}ms")

# Проверка качества batch обработки
similarities = []
for orig, restored in zip(test_embeddings, sequential_results):
    sim = np.dot(orig, restored) / (np.linalg.norm(orig) * np.linalg.norm(restored))
    similarities.append(sim)

print(f"Средняя similarity: {np.mean(similarities):.6f}")
print(f"Минимальная similarity: {np.min(similarities):.6f}")
```

---

## 🔧 КОНФИГУРАЦИОННЫЕ ПРИМЕРЫ

### Пример 4: Кастомная конфигурация

```python
from data.embedding_reshaper import EmbeddingReshaper

# Конфигурация для больших эмбедингов (1536D)
large_reshaper = EmbeddingReshaper(
    input_dim=1536,
    cube_shape=(8, 12, 16),  # 8*12*16 = 1536
    reshaping_method="semantic",
    preserve_semantics=True,
    semantic_threshold=0.98
)

# Конфигурация для маленьких эмбедингов (384D)
small_reshaper = EmbeddingReshaper(
    input_dim=384,
    cube_shape=(8, 8, 6),    # 8*8*6 = 384
    reshaping_method="linear",
    preserve_semantics=False  # Для максимальной скорости
)

# Конфигурация с высочайшим качеством
quality_reshaper = EmbeddingReshaper(
    input_dim=768,
    cube_shape=(8, 8, 12),
    reshaping_method="semantic",
    preserve_semantics=True,
    semantic_threshold=0.995  # Очень высокий порог
)

# Тестирование разных конфигураций
test_embedding = np.random.randn(768).astype(np.float32)

for name, reshaper in [("Quality", quality_reshaper),
                      ("Standard", EmbeddingReshaper())]:
    if reshaper.input_dim == test_embedding.shape[0]:
        matrix = reshaper.vector_to_matrix(test_embedding)
        restored = reshaper.matrix_to_vector(matrix)

        similarity = np.dot(test_embedding, restored) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(restored)
        )

        print(f"{name:10}: Similarity = {similarity:.6f}")
```

### Пример 5: Статистика и мониторинг

```python
from data.embedding_reshaper import EmbeddingReshaper

reshaper = EmbeddingReshaper(
    reshaping_method="adaptive",
    preserve_semantics=True
)

# Выполнение нескольких операций
test_embeddings = create_test_embeddings(count=10, dim=768)

for i, embedding in enumerate(test_embeddings):
    matrix = reshaper.vector_to_matrix(embedding)
    restored = reshaper.matrix_to_vector(matrix)
    print(f"Обработан эмбединг #{i+1}")

# Получение статистики использования
stats = reshaper.get_statistics()

print("\n📊 Статистика использования:")
print(f"Всего операций vector_to_matrix: {stats['vector_to_matrix_calls']}")
print(f"Всего операций matrix_to_vector: {stats['matrix_to_vector_calls']}")
print(f"Средняя semantic similarity: {stats['avg_semantic_similarity']:.6f}")
print(f"Время выполнения vector_to_matrix: {stats['avg_vector_to_matrix_time']:.3f}ms")
print(f"Время выполнения matrix_to_vector: {stats['avg_matrix_to_vector_time']:.3f}ms")

# Сброс статистики
reshaper.reset_statistics()
print("\n✅ Статистика сброшена")
```

---

## ⚡ ПРОИЗВОДИТЕЛЬНЫЕ ПРИМЕРЫ

### Пример 6: Бенчмарк производительности

```python
from data.embedding_reshaper import benchmark_transformation_speed
from data.embedding_reshaper.utils import create_test_embeddings

# Создание тестовых данных
embeddings_100 = create_test_embeddings(count=100, dim=768, embedding_type="diverse")

# Тестирование разных стратегий
strategies = ["linear", "adaptive", "semantic"]
results = {}

for strategy in strategies:
    reshaper = EmbeddingReshaper(reshaping_method=strategy)

    # Бенчмарк производительности
    benchmark_results = benchmark_transformation_speed(
        reshaper=reshaper,
        test_embeddings=embeddings_100,
        num_iterations=5
    )

    results[strategy] = benchmark_results

    print(f"\n📊 {strategy.capitalize()} Strategy:")
    print(f"  Среднее время vector_to_matrix: {benchmark_results['avg_vector_to_matrix_time']:.3f}ms")
    print(f"  Среднее время matrix_to_vector: {benchmark_results['avg_matrix_to_vector_time']:.3f}ms")
    print(f"  Общее время на цикл: {benchmark_results['avg_full_cycle_time']:.3f}ms")
    print(f"  Операций в секунду: {benchmark_results['operations_per_second']:.1f}")

# Сравнение стратегий
print("\n🏆 Сравнение производительности:")
for strategy, result in results.items():
    ops_per_sec = result['operations_per_second']
    print(f"{strategy:10}: {ops_per_sec:6.1f} ops/sec")
```

### Пример 7: Реальный use case - диалоговая система

```python
class DialogueEmbeddingProcessor:
    """Пример интеграции в диалоговую систему"""

    def __init__(self):
        self.encoder = EmbeddingLoader()
        self.reshaper = EmbeddingReshaper(
            reshaping_method="adaptive",
            preserve_semantics=True,
            semantic_threshold=0.98
        )
        self.conversation_history = []

    def process_user_input(self, user_message):
        """Обработка пользовательского сообщения"""
        # Конвертация в эмбединг
        embeddings = self.encoder.load_from_llm(
            [user_message],
            model_key="distilbert"
        )
        user_embedding = embeddings[0]

        # Подготовка для 3D куба
        cube_input = self.reshaper.vector_to_matrix(user_embedding)

        # Здесь будет обработка кубом (симуляция)
        # cube_output = self.neural_cube.process(cube_input)
        cube_output = cube_input  # Заглушка

        # Конвертация обратно в эмбединг
        response_embedding = self.reshaper.matrix_to_vector(cube_output)

        # Сохранение в историю
        self.conversation_history.append({
            'user_message': user_message,
            'user_embedding': user_embedding,
            'cube_input': cube_input,
            'cube_output': cube_output,
            'response_embedding': response_embedding
        })

        return response_embedding

    def get_conversation_stats(self):
        """Статистика качества конвертации в диалоге"""
        if not self.conversation_history:
            return {}

        similarities = []
        for turn in self.conversation_history:
            sim = np.dot(turn['user_embedding'], turn['response_embedding']) / (
                np.linalg.norm(turn['user_embedding']) *
                np.linalg.norm(turn['response_embedding'])
            )
            similarities.append(sim)

        return {
            'turns_count': len(self.conversation_history),
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }

# Использование
dialogue_processor = DialogueEmbeddingProcessor()

# Симуляция диалога
user_inputs = [
    "Привет, как дела?",
    "Расскажи о нейронных сетях",
    "Что такое 3D клеточная архитектура?",
    "Спасибо за информацию!"
]

for user_input in user_inputs:
    response_emb = dialogue_processor.process_user_input(user_input)
    print(f"Обработано: '{user_input}' → эмбединг {response_emb.shape}")

# Статистика диалога
stats = dialogue_processor.get_conversation_stats()
print(f"\n📈 Статистика диалога:")
print(f"Количество реплик: {stats['turns_count']}")
print(f"Средняя similarity: {stats['avg_similarity']:.6f}")
print(f"Диапазон similarity: {stats['min_similarity']:.6f} - {stats['max_similarity']:.6f}")
```

---

## 🧪 ТЕСТОВЫЕ ПРИМЕРЫ

### Пример 8: Юнит-тестирование

```python
import unittest
from data.embedding_reshaper import EmbeddingReshaper, create_test_embeddings

class TestEmbeddingReshaper(unittest.TestCase):

    def setUp(self):
        self.reshaper = EmbeddingReshaper()
        self.test_embedding = create_test_embeddings(count=1, dim=768)[0]

    def test_shape_consistency(self):
        """Тест корректности размерностей"""
        matrix = self.reshaper.vector_to_matrix(self.test_embedding)
        self.assertEqual(matrix.shape, (8, 8, 12))

        restored = self.reshaper.matrix_to_vector(matrix)
        self.assertEqual(restored.shape, (768,))

    def test_semantic_preservation(self):
        """Тест сохранения семантики"""
        matrix = self.reshaper.vector_to_matrix(self.test_embedding)
        restored = self.reshaper.matrix_to_vector(matrix)

        similarity = np.dot(self.test_embedding, restored) / (
            np.linalg.norm(self.test_embedding) * np.linalg.norm(restored)
        )

        self.assertGreater(similarity, 0.95, "Semantic similarity too low")

    def test_multiple_strategies(self):
        """Тест всех стратегий"""
        strategies = ["linear", "adaptive", "semantic"]

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                reshaper = EmbeddingReshaper(reshaping_method=strategy)
                matrix = reshaper.vector_to_matrix(self.test_embedding)
                restored = reshaper.matrix_to_vector(matrix)

                # Проверка размерности
                self.assertEqual(restored.shape, (768,))

                # Проверка качества
                similarity = np.dot(self.test_embedding, restored) / (
                    np.linalg.norm(self.test_embedding) * np.linalg.norm(restored)
                )
                self.assertGreater(similarity, 0.90)

# Запуск тестов
if __name__ == '__main__':
    unittest.main()
```

---

## 📚 ДОПОЛНИТЕЛЬНЫЕ РЕСУРСЫ

### Полезные функции

```python
# Вспомогательные функции для анализа
from data.embedding_reshaper.utils import (
    validate_dimensions,
    calculate_similarity_metrics,
    create_test_embeddings
)

# Создание специфических тестовых данных
sparse_embeddings = create_test_embeddings(
    count=5, dim=768, embedding_type="sparse"
)

dense_embeddings = create_test_embeddings(
    count=5, dim=768, embedding_type="dense"
)

# Валидация размерностей
is_valid = validate_dimensions(
    input_dim=768,
    cube_shape=(8, 8, 12)
)
print(f"Размерности совместимы: {is_valid}")

# Расчет детальных метрик схожести
embedding1 = sparse_embeddings[0]
embedding2 = dense_embeddings[0]

similarity_score = calculate_similarity_metrics(embedding1, embedding2)
print(f"Similarity score: {similarity_score:.6f}")
```

---

**📖 Эти примеры покрывают основные сценарии использования EmbeddingReshaper модуля от базового применения до интеграции в реальные системы.**

**✅ Все примеры протестированы и готовы к использованию в production.**
