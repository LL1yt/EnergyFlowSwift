"""
Вспомогательные функции для EmbeddingReshaper
============================================

Содержит утилиты для контроля качества, расчета метрик и оптимизации трансформаций.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List
import logging
from sklearn.metrics.pairwise import cosine_similarity 
import warnings

# Подавляем предупреждения sklearn для чистого вывода
warnings.filterwarnings("ignore", category=UserWarning)


def validate_semantic_preservation(
    original: Union[torch.Tensor, np.ndarray],
    transformed: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.95
) -> bool:
    """
    Проверка сохранения семантической информации при трансформации.
    
    Args:
        original: Исходный эмбединг
        transformed: Преобразованный эмбединг
        threshold: Минимальный порог cosine similarity (по умолчанию 0.95)
        
    Returns:
        True если качество сохранения выше порога
        
    Raises:
        ValueError: При несовместимых размерностях
    """
    try:
        similarity = calculate_similarity_metrics(original, transformed)
        return similarity >= threshold
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка валидации: {e}")
        return False


def calculate_similarity_metrics(
    vec1: Union[torch.Tensor, np.ndarray],
    vec2: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Расчет cosine similarity между двумя векторами.
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        Cosine similarity в диапазоне [0, 1]
        
    Raises:
        ValueError: При несовместимых размерностях или типах
    """
    # Приводим к одинаковому типу и форме
    if isinstance(vec1, torch.Tensor):
        vec1_np = vec1.detach().cpu().numpy()
    else:
        vec1_np = vec1
    
    if isinstance(vec2, torch.Tensor):
        vec2_np = vec2.detach().cpu().numpy()
    else:
        vec2_np = vec2
    
    # Приводим к 1D для сравнения
    vec1_flat = vec1_np.flatten()
    vec2_flat = vec2_np.flatten()
    
    # Проверяем размерности
    if vec1_flat.shape != vec2_flat.shape:
        raise ValueError(
            f"Несовместимые размерности: {vec1_flat.shape} vs {vec2_flat.shape}"
        )
    
    # Вычисляем cosine similarity
    vec1_reshaped = vec1_flat.reshape(1, -1)
    vec2_reshaped = vec2_flat.reshape(1, -1)
    
    similarity = cosine_similarity(vec1_reshaped, vec2_reshaped)[0][0]
    
    # Приводим к диапазону [0, 1] (cosine может быть от -1 до 1)
    similarity_normalized = (similarity + 1) / 2
    
    return float(similarity_normalized)


def optimize_shape_transformation(
    input_shape: Union[int, Tuple[int, ...]],
    target_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    Оптимизация трансформации форм для максимального сохранения информации.
    
    Args:
        input_shape: Исходная форма (например, 768 или (768,))
        target_shape: Целевая 3D форма (например, (8, 8, 12))
        
    Returns:
        Словарь с рекомендациями по оптимизации
    """
    # Приводим input_shape к единому формату
    if isinstance(input_shape, int):
        input_size = input_shape
    else:
        input_size = np.prod(input_shape)
    
    target_size = np.prod(target_shape)
    
    # Проверяем совместимость
    compatible = (input_size == target_size)
    
    # Анализируем пропорции
    if len(target_shape) == 3:
        d, h, w = target_shape
        aspect_ratios = {
            'depth_height': d / h if h != 0 else float('inf'),
            'height_width': h / w if w != 0 else float('inf'),
            'depth_width': d / w if w != 0 else float('inf')
        }
        
        # Рекомендации по балансировке
        balanced = max(aspect_ratios.values()) / min(aspect_ratios.values()) < 2.0
    else:
        aspect_ratios = {}
        balanced = True
    
    # Альтернативные формы
    alternatives = []
    if not compatible:
        # Ищем ближайшие совместимые формы
        factors = _find_factors(input_size)
        for i, f1 in enumerate(factors):
            for j, f2 in enumerate(factors[i:], i):
                for k, f3 in enumerate(factors[j:], j):
                    if f1 * f2 * f3 == input_size:
                        alternatives.append((f1, f2, f3))
        
        # Сортируем по близости к целевой форме
        alternatives.sort(key=lambda x: sum(abs(a - b) for a, b in zip(x, target_shape)))
    
    return {
        'compatible': compatible,
        'input_size': input_size,
        'target_size': target_size,
        'aspect_ratios': aspect_ratios,
        'balanced_proportions': balanced,
        'alternative_shapes': alternatives[:5],  # Топ-5 альтернатив
        'optimization_score': _calculate_optimization_score(input_size, target_shape)
    }


def _find_factors(n: int) -> List[int]:
    """Поиск всех делителей числа n."""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)


def _calculate_optimization_score(input_size: int, target_shape: Tuple[int, int, int]) -> float:
    """
    Расчет оценки оптимальности трансформации.
    
    Returns:
        Оценка от 0 до 1 (1 = оптимально)
    """
    target_size = np.prod(target_shape)
    
    # Базовая совместимость
    if input_size != target_size:
        return 0.0
    
    # Анализ баланса пропорций
    d, h, w = target_shape
    ratios = [d/h, h/w, d/w] if h != 0 and w != 0 else [1.0, 1.0, 1.0]
    
    # Оптимально когда все пропорции близки к 1 (кубическая форма)
    balance_score = 1.0 - (max(ratios) - min(ratios)) / max(ratios)
    
    # Штраф за слишком большие или маленькие измерения
    size_penalty = 0.0
    for dim in target_shape:
        if dim < 2 or dim > 32:  # Оптимальный диапазон размеров
            size_penalty += 0.1
    
    final_score = max(0.0, balance_score - size_penalty)
    return min(1.0, final_score)


def create_test_embeddings(
    count: int = 10,
    dim: int = 768,
    embedding_type: str = "random"
) -> List[np.ndarray]:
    """
    Создание тестовых эмбедингов для проверки функциональности.
    
    Args:
        count: Количество эмбедингов
        dim: Размерность каждого эмбединга
        embedding_type: Тип эмбедингов ('random', 'gaussian', 'normalized')
        
    Returns:
        Список тестовых эмбедингов
    """
    embeddings = []
    
    for i in range(count):
        if embedding_type == "random":
            emb = np.random.random(dim).astype(np.float32)
        elif embedding_type == "gaussian":
            emb = np.random.normal(0, 1, dim).astype(np.float32)
        elif embedding_type == "normalized":
            emb = np.random.random(dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Нормализуем
        else:
            raise ValueError(f"Неизвестный тип эмбединга: {embedding_type}")
        
        embeddings.append(emb)
    
    return embeddings


def benchmark_transformation_speed(
    reshaper,
    test_embeddings: List[np.ndarray],
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Бенчмарк скорости трансформации.
    
    Args:
        reshaper: Экземпляр EmbeddingReshaper
        test_embeddings: Список тестовых эмбедингов
        num_iterations: Количество итераций для усреднения
        
    Returns:
        Статистика производительности
    """
    import time
    
    if not test_embeddings:
        raise ValueError("Нужен хотя бы один тестовый эмбединг")
    
    # Время 1D → 3D трансформации
    start_time = time.time()
    for _ in range(num_iterations):
        for emb in test_embeddings:
            _ = reshaper.vector_to_matrix(emb)
    time_1d_to_3d = (time.time() - start_time) / (num_iterations * len(test_embeddings))
    
    # Время 3D → 1D трансформации
    test_3d = [reshaper.vector_to_matrix(emb) for emb in test_embeddings]
    start_time = time.time()
    for _ in range(num_iterations):
        for emb_3d in test_3d:
            _ = reshaper.matrix_to_vector(emb_3d)
    time_3d_to_1d = (time.time() - start_time) / (num_iterations * len(test_embeddings))
    
    return {
        'avg_time_1d_to_3d_ms': time_1d_to_3d * 1000,
        'avg_time_3d_to_1d_ms': time_3d_to_1d * 1000,
        'total_throughput_per_sec': 1.0 / (time_1d_to_3d + time_3d_to_1d),
        'test_embeddings_count': len(test_embeddings),
        'iterations': num_iterations
    } 