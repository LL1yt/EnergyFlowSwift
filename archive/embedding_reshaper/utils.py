"""
Вспомогательные функции для EmbeddingReshaper
============================================

Содержит утилиты для контроля качества, расчета метрик и оптимизации трансформаций.
PHASE 2.3 День 3-4: Улучшенные алгоритмы семантического сохранения >98%
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List
import logging
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr, spearmanr
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


# ==========================================
# 🚀 НОВЫЕ ФУНКЦИИ ДЛЯ СЕМАНТИЧЕСКОГО СОХРАНЕНИЯ >98%
# ==========================================

def calculate_enhanced_similarity_metrics(
    vec1: Union[torch.Tensor, np.ndarray],
    vec2: Union[torch.Tensor, np.ndarray],
    metrics_config: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Расширенный расчет метрик семантического сходства для достижения >98% точности.
    
    Использует множественные метрики с адаптивным взвешиванием:
    - Cosine similarity (основная метрика)
    - Pearson correlation (линейная зависимость)
    - Spearman correlation (монотонная зависимость)
    - Structural similarity (сохранение структуры)
    - Magnitude preservation (сохранение норм)
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор  
        metrics_config: Веса для различных метрик
        
    Returns:
        Словарь с метриками и итоговым weighted score
    """
    if metrics_config is None:
        metrics_config = {
            'cosine_weight': 0.4,        # Главная метрика
            'pearson_weight': 0.2,       # Линейная корреляция
            'spearman_weight': 0.15,     # Ранговая корреляция  
            'structural_weight': 0.15,   # Структурное сходство
            'magnitude_weight': 0.1      # Сохранение норм
        }
    
    # Приводим к одинаковому типу и форме
    if isinstance(vec1, torch.Tensor):
        vec1_np = vec1.detach().cpu().numpy()
    else:
        vec1_np = vec1
    
    if isinstance(vec2, torch.Tensor):
        vec2_np = vec2.detach().cpu().numpy()
    else:
        vec2_np = vec2
    
    # Приводим к 1D для анализа
    vec1_flat = vec1_np.flatten()
    vec2_flat = vec2_np.flatten()
    
    if vec1_flat.shape != vec2_flat.shape:
        raise ValueError(f"Несовместимые размерности: {vec1_flat.shape} vs {vec2_flat.shape}")
    
    metrics = {}
    
    # 1. Cosine Similarity (основная метрика)
    vec1_reshaped = vec1_flat.reshape(1, -1)
    vec2_reshaped = vec2_flat.reshape(1, -1)
    cosine_sim = cosine_similarity(vec1_reshaped, vec2_reshaped)[0][0]
    metrics['cosine_similarity'] = float((cosine_sim + 1) / 2)  # Нормализация к [0,1]
    
    # 2. Pearson Correlation (линейная зависимость) 
    try:
        # Проверяем на константные векторы (std = 0)
        if np.std(vec1_flat) == 0 or np.std(vec2_flat) == 0:
            # Если оба константные и равные
            if np.allclose(vec1_flat, vec2_flat):
                metrics['pearson_correlation'] = 1.0
            else:
                metrics['pearson_correlation'] = 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_corr, _ = pearsonr(vec1_flat, vec2_flat)
            metrics['pearson_correlation'] = float((pearson_corr + 1) / 2) if not np.isnan(pearson_corr) else 0.5
    except:
        metrics['pearson_correlation'] = 0.5
    
    # 3. Spearman Correlation (монотонная зависимость)
    try:
        spearman_corr, _ = spearmanr(vec1_flat, vec2_flat)
        metrics['spearman_correlation'] = float((spearman_corr + 1) / 2) if not np.isnan(spearman_corr) else 0.5
    except:
        metrics['spearman_correlation'] = 0.5
    
    # 4. Structural Similarity (сохранение относительных позиций)
    structural_sim = _calculate_structural_similarity(vec1_flat, vec2_flat)
    metrics['structural_similarity'] = structural_sim
    
    # 5. Magnitude Preservation (сохранение норм)
    norm1 = np.linalg.norm(vec1_flat)
    norm2 = np.linalg.norm(vec2_flat)
    magnitude_sim = 1.0 - abs(norm1 - norm2) / max(norm1, norm2, 1e-8)
    metrics['magnitude_preservation'] = float(max(0.0, magnitude_sim))
    
    # Взвешенная итоговая оценка
    weighted_score = (
        metrics['cosine_similarity'] * metrics_config['cosine_weight'] +
        metrics['pearson_correlation'] * metrics_config['pearson_weight'] +
        metrics['spearman_correlation'] * metrics_config['spearman_weight'] +
        metrics['structural_similarity'] * metrics_config['structural_weight'] +
        metrics['magnitude_preservation'] * metrics_config['magnitude_weight']
    )
    
    metrics['weighted_similarity'] = float(weighted_score)
    
    return metrics


def _calculate_structural_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Расчет структурного сходства - насколько хорошо сохраняются относительные позиции элементов.
    
    Использует ранговую корреляцию позиций элементов для оценки сохранения структуры.
    """
    try:
        # Получаем ранги элементов (позиции после сортировки)
        ranks1 = vec1.argsort().argsort()
        ranks2 = vec2.argsort().argsort()
        
        # Вычисляем корреляцию рангов
        rank_corr, _ = spearmanr(ranks1, ranks2)
        
        if np.isnan(rank_corr):
            return 0.5
        
        # Нормализуем к [0,1]
        return float((rank_corr + 1) / 2)
    except:
        return 0.5


def analyze_embedding_importance(
    embedding: Union[torch.Tensor, np.ndarray],
    method: str = "variance_pca"
) -> np.ndarray:
    """
    Анализ важности элементов эмбединга для семантического сохранения.
    
    Args:
        embedding: Входной эмбединг
        method: Метод анализа ('variance_pca', 'clustering', 'magnitude')
        
    Returns:
        Массив весов важности для каждого элемента [0,1]
    """
    if isinstance(embedding, torch.Tensor):
        embedding_np = embedding.detach().cpu().numpy()
    else:
        embedding_np = embedding
    
    embedding_flat = embedding_np.flatten()
    
    if method == "variance_pca":
        return _analyze_importance_pca(embedding_flat)
    elif method == "clustering": 
        return _analyze_importance_clustering(embedding_flat)
    elif method == "magnitude":
        return _analyze_importance_magnitude(embedding_flat)
    else:
        # Fallback - равномерные веса
        return np.ones_like(embedding_flat) / len(embedding_flat)


def _analyze_importance_pca(embedding: np.ndarray) -> np.ndarray:
    """
    Улучшенный анализ важности через PCA компоненты.
    
    Использует множественные подходы для более точного определения важности элементов.
    """
    try:
        # Подход 1: Анализ статистических свойств
        embedding_std = np.std(embedding)
        magnitude_importance = np.abs(embedding)
        
        # Подход 2: Локальная вариабельность (как PCA заменитель)
        window_size = max(1, len(embedding) // 50)  # 2% окна для детального анализа
        local_variance = np.zeros_like(embedding)
        
        for i in range(len(embedding)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(embedding), i + window_size + 1)
            local_window = embedding[start_idx:end_idx]
            local_variance[i] = np.var(local_window) if len(local_window) > 1 else np.abs(embedding[i])
        
        # Подход 3: Градиентный анализ (изменения между соседними элементами)
        gradient_importance = np.zeros_like(embedding)
        for i in range(len(embedding)):
            if i > 0 and i < len(embedding) - 1:
                # Центральная разность
                gradient_importance[i] = abs(embedding[i+1] - embedding[i-1]) / 2
            elif i == 0:
                # Прямая разность
                gradient_importance[i] = abs(embedding[i+1] - embedding[i]) if len(embedding) > 1 else abs(embedding[i])
            else:
                # Обратная разность
                gradient_importance[i] = abs(embedding[i] - embedding[i-1])
        
        # Подход 4: Относительное отклонение от среднего
        mean_val = np.mean(embedding)
        deviation_importance = np.abs(embedding - mean_val)
        
        # Подход 5: Квантильный анализ
        percentiles = np.percentile(np.abs(embedding), [25, 50, 75, 90, 95])
        quantile_importance = np.zeros_like(embedding)
        for i, val in enumerate(embedding):
            abs_val = abs(val)
            if abs_val >= percentiles[4]:  # Top 5%
                quantile_importance[i] = 1.0
            elif abs_val >= percentiles[3]:  # Top 10%
                quantile_importance[i] = 0.9
            elif abs_val >= percentiles[2]:  # Top 25%
                quantile_importance[i] = 0.7
            elif abs_val >= percentiles[1]:  # Above median
                quantile_importance[i] = 0.5
            elif abs_val >= percentiles[0]:  # Above Q1
                quantile_importance[i] = 0.3
            else:
                quantile_importance[i] = 0.1
        
        # Комбинируем все подходы с оптимальными весами
        combined_importance = (
            0.25 * magnitude_importance / (np.max(magnitude_importance) + 1e-8) +
            0.20 * local_variance / (np.max(local_variance) + 1e-8) +
            0.20 * gradient_importance / (np.max(gradient_importance) + 1e-8) +
            0.15 * deviation_importance / (np.max(deviation_importance) + 1e-8) +
            0.20 * quantile_importance
        )
        
        # Дополнительное усиление контраста
        # Применяем степенную функцию для увеличения различий
        enhanced_importance = np.power(combined_importance, 1.5)  # Усиливаем различия
        
        # Нормализация к [0,1]
        final_importance = enhanced_importance / (np.max(enhanced_importance) + 1e-8)
        
        return final_importance
        
    except Exception as e:
        # Fallback - улучшенный анализ на основе величины и позиции
        magnitude_importance = np.abs(embedding)
        n = len(embedding)
        
        # Позиционные веса (элементы в начале и конце часто важнее)
        position_weights = np.ones(n)
        # Усиливаем начало (первые 20%)
        position_weights[:n//5] *= 1.5
        # Усиливаем конец (последние 20%)
        position_weights[4*n//5:] *= 1.3
        # Усиливаем середину (центральные 20%)
        center_start, center_end = 2*n//5, 3*n//5
        position_weights[center_start:center_end] *= 1.2
        
        combined = magnitude_importance * position_weights
        return combined / (np.max(combined) + 1e-8)


def _analyze_importance_clustering(embedding: np.ndarray) -> np.ndarray:
    """
    Улучшенный анализ важности через кластеризацию и локальный анализ.
    
    Использует адаптивное окно и множественные метрики для точного определения важности.
    """
    try:
        # Адаптивный размер окна в зависимости от статистических свойств
        embedding_std = np.std(embedding)
        base_window = max(3, len(embedding) // 30)  # Базовый размер окна ~3% 
        
        # Анализируем на нескольких масштабах
        importance_scales = []
        
        # Масштаб 1: Мелкие окна (детальный анализ)
        small_window = max(2, base_window // 2)
        importance_small = np.zeros_like(embedding)
        
        for i in range(len(embedding)):
            start_idx = max(0, i - small_window)
            end_idx = min(len(embedding), i + small_window + 1)
            window = embedding[start_idx:end_idx]
            
            if len(window) > 1:
                # Комбинируем статистики
                local_std = np.std(window)
                local_range = np.max(window) - np.min(window)
                local_energy = np.sum(np.square(window))
                
                importance_small[i] = local_std + 0.5 * local_range + 0.3 * np.sqrt(local_energy)
            else:
                importance_small[i] = np.abs(window[0])
        
        importance_scales.append(importance_small)
        
        # Масштаб 2: Средние окна (структурный анализ)
        medium_window = base_window
        importance_medium = np.zeros_like(embedding)
        
        for i in range(len(embedding)):
            start_idx = max(0, i - medium_window)
            end_idx = min(len(embedding), i + medium_window + 1)
            window = embedding[start_idx:end_idx]
            
            if len(window) > 2:
                # Анализ тренда в окне
                x = np.arange(len(window))
                trend_coefficient = np.abs(np.corrcoef(x, window)[0, 1]) if len(window) > 1 else 0
                local_complexity = np.std(np.diff(window)) if len(window) > 1 else 0
                
                importance_medium[i] = trend_coefficient + local_complexity + np.abs(np.mean(window))
            else:
                importance_medium[i] = np.abs(np.mean(window))
        
        importance_scales.append(importance_medium)
        
        # Масштаб 3: Большие окна (глобальный контекст)
        large_window = base_window * 2
        importance_large = np.zeros_like(embedding)
        
        for i in range(len(embedding)):
            start_idx = max(0, i - large_window)
            end_idx = min(len(embedding), i + large_window + 1)
            window = embedding[start_idx:end_idx]
            
            # Относительная важность в большом контексте
            element_rank = np.sum(np.abs(window) < np.abs(embedding[i])) / len(window)
            local_contrast = np.abs(embedding[i] - np.median(window))
            
            importance_large[i] = element_rank + local_contrast
        
        importance_scales.append(importance_large)
        
        # Комбинируем масштабы с убывающими весами (детальный анализ важнее)
        weights = [0.5, 0.3, 0.2]  # Мелкий, средний, крупный масштаб
        combined_importance = np.zeros_like(embedding)
        
        for scale_importance, weight in zip(importance_scales, weights):
            normalized_scale = scale_importance / (np.max(scale_importance) + 1e-8)
            combined_importance += weight * normalized_scale
        
        # Усиление контраста для лучшего различения
        enhanced_importance = np.power(combined_importance, 1.3)
        
        # Финальная нормализация
        final_importance = enhanced_importance / (np.max(enhanced_importance) + 1e-8)
        
        return final_importance
        
    except Exception as e:
        # Fallback - простой но эффективный анализ
        magnitude = np.abs(embedding)
        
        # Локальная вариация
        local_var = np.zeros_like(embedding)
        for i in range(len(embedding)):
            start = max(0, i - 2)
            end = min(len(embedding), i + 3)
            local_var[i] = np.var(embedding[start:end])
        
        combined = 0.7 * magnitude + 0.3 * local_var
        return combined / (np.max(combined) + 1e-8)


def _analyze_importance_magnitude(embedding: np.ndarray) -> np.ndarray:
    """
    Улучшенный анализ важности на основе величины значений.
    
    Использует множественные метрики для более точного определения важности элементов.
    """
    n = len(embedding)
    
    # Базовая важность по величине
    magnitude_importance = np.abs(embedding)
    
    # Позиционная важность (адаптивная схема)
    position_weights = np.ones(n)
    
    # Усиливаем края (первые и последние элементы часто содержат метаинформацию)
    edge_size = max(1, n // 20)  # 5% с каждого края
    position_weights[:edge_size] *= 1.4  # Начало
    position_weights[-edge_size:] *= 1.4  # Конец
    
    # Усиливаем центральную область (где обычно основная семантика)
    center_start, center_end = n//3, 2*n//3
    position_weights[center_start:center_end] *= 1.2
    
    # Квартильная важность (элементы в верхних квартилях важнее)
    percentiles = np.percentile(magnitude_importance, [50, 75, 90, 95, 99])
    quartile_weights = np.ones_like(magnitude_importance)
    
    for i, mag in enumerate(magnitude_importance):
        if mag >= percentiles[4]:  # Top 1%
            quartile_weights[i] = 2.0
        elif mag >= percentiles[3]:  # Top 5%
            quartile_weights[i] = 1.8
        elif mag >= percentiles[2]:  # Top 10%
            quartile_weights[i] = 1.5
        elif mag >= percentiles[1]:  # Top 25%
            quartile_weights[i] = 1.3
        elif mag >= percentiles[0]:  # Above median
            quartile_weights[i] = 1.0
        else:  # Below median
            quartile_weights[i] = 0.7
    
    # Анализ локальных максимумов (пики важности)
    local_maxima_weights = np.ones_like(magnitude_importance)
    window_size = max(2, n // 50)  # 2% окно для поиска локальных максимумов
    
    for i in range(window_size, n - window_size):
        window = magnitude_importance[i-window_size:i+window_size+1]
        if magnitude_importance[i] == np.max(window):
            local_maxima_weights[i] = 1.5  # Усиливаем локальные максимумы
    
    # Дифференциальная важность (элементы с большими градиентами)
    gradient_importance = np.ones_like(magnitude_importance)
    for i in range(1, n-1):
        left_diff = abs(embedding[i] - embedding[i-1])
        right_diff = abs(embedding[i+1] - embedding[i])
        max_diff = max(left_diff, right_diff)
        
        # Нормализуем к глобальному максимальному градиенту
        max_gradient = np.max([abs(embedding[i+1] - embedding[i]) for i in range(n-1)])
        if max_gradient > 0:
            gradient_importance[i] = 1.0 + (max_diff / max_gradient) * 0.5
    
    # Статистическая аномальность (элементы далекие от среднего)
    mean_val = np.mean(magnitude_importance)
    std_val = np.std(magnitude_importance)
    anomaly_weights = np.ones_like(magnitude_importance)
    
    for i, val in enumerate(magnitude_importance):
        z_score = abs(val - mean_val) / (std_val + 1e-8)
        if z_score > 2.0:  # Статистически значимые аномалии
            anomaly_weights[i] = 1.0 + min(z_score * 0.2, 1.0)  # Максимум +100%
    
    # Комбинируем все компоненты с оптимальными весами
    combined_importance = (
        magnitude_importance * (
            0.40 * position_weights +
            0.25 * quartile_weights +
            0.15 * local_maxima_weights +
            0.10 * gradient_importance +
            0.10 * anomaly_weights
        )
    )
    
    # Дополнительное усиление контраста
    enhanced_importance = np.power(combined_importance / np.max(combined_importance), 1.2)
    
    # Финальная нормализация
    return enhanced_importance / (np.max(enhanced_importance) + 1e-8)


def create_adaptive_transformation_strategy(
    embedding: Union[torch.Tensor, np.ndarray],
    target_shape: Tuple[int, int, int],
    importance_method: str = "variance_pca"
) -> Dict[str, Any]:
    """
    Создание адаптивной стратегии трансформации с учетом важности элементов.
    
    Args:
        embedding: Входной эмбединг
        target_shape: Целевая 3D форма
        importance_method: Метод анализа важности
        
    Returns:
        Словарь с параметрами адаптивной трансформации
    """
    if isinstance(embedding, torch.Tensor):
        embedding_np = embedding.detach().cpu().numpy()
    else:
        embedding_np = embedding
    
    embedding_flat = embedding_np.flatten()
    
    # Анализ важности элементов
    importance_weights = analyze_embedding_importance(embedding_flat, importance_method)
    
    # Создаем карту размещения в 3D пространстве
    placement_map = _create_3d_placement_map(importance_weights, target_shape)
    
    # Параметры для оптимальной трансформации
    strategy = {
        'importance_weights': importance_weights,
        'placement_map': placement_map,
        'target_shape': target_shape,
        'optimization_params': {
            'preserve_high_importance': True,
            'spatial_locality': True,
            'minimize_distortion': True
        },
        'quality_threshold': 0.98  # Целевой порог качества
    }
    
    return strategy


def _create_3d_placement_map(
    importance_weights: np.ndarray, 
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Создание карты размещения элементов в 3D пространстве с учетом их важности.
    
    Важные элементы размещаются ближе к центру куба для лучшего сохранения семантики.
    """
    d, h, w = target_shape
    total_elements = d * h * w
    
    if len(importance_weights) != total_elements:
        raise ValueError(f"Размерности не совпадают: {len(importance_weights)} vs {total_elements}")
    
    # Создаем 3D координаты
    coords = []
    for z in range(d):
        for y in range(h):
            for x in range(w):
                coords.append((z, y, x))
    
    # Вычисляем расстояние от центра для каждой позиции
    center_z, center_y, center_x = d/2, h/2, w/2
    distances = []
    for z, y, x in coords:
        dist = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
        distances.append(dist)
    
    # Сортируем позиции по расстоянию от центра (ближние позиции для важных элементов)
    sorted_indices = np.argsort(distances)
    importance_sorted_indices = np.argsort(importance_weights)[::-1]  # От важных к менее важным
    
    # Создаем карту размещения
    placement_map = np.zeros(total_elements, dtype=int)
    for i, (spatial_idx, importance_idx) in enumerate(zip(sorted_indices, importance_sorted_indices)):
        placement_map[importance_idx] = spatial_idx
    
    return placement_map


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
        embedding_type: Тип эмбедингов ('random', 'gaussian', 'normalized', 'diverse')
        
    Returns:
        Список тестовых эмбедингов
    """
    embeddings = []
    
    if embedding_type == "diverse":
        # Создаем разнообразные эмбединги для комплексного тестирования
        patterns = [
            ("sparse", 0.1),           # Разреженный (10% ненулевых)
            ("dense_positive", 1.0),   # Плотный положительный
            ("mixed_signs", 0.5),      # Смешанные знаки
            ("gaussian_high", 2.0),    # Высокая дисперсия
            ("gaussian_low", 0.1),     # Низкая дисперсия
        ]
        
        # Расширяем паттерны до нужного количества
        extended_patterns = []
        for i in range(count):
            pattern_name, param = patterns[i % len(patterns)]
            extended_patterns.append((pattern_name, param, i))
        
        for pattern_name, param, seed in extended_patterns:
            np.random.seed(seed + 42)  # Детерминированная генерация
            
            if pattern_name == "sparse":
                # Разреженный эмбединг
                emb = np.zeros(dim, dtype=np.float32)
                num_nonzero = int(dim * param)
                indices = np.random.choice(dim, num_nonzero, replace=False)
                emb[indices] = np.random.normal(0, 1, num_nonzero)
                
            elif pattern_name == "dense_positive":
                # Плотный положительный
                emb = np.random.exponential(param, dim).astype(np.float32)
                
            elif pattern_name == "mixed_signs":
                # Смешанные знаки
                emb = np.random.normal(0, 1, dim).astype(np.float32)
                # Половина отрицательных
                neg_indices = np.random.choice(dim, dim//2, replace=False)
                emb[neg_indices] *= -1
                
            elif pattern_name == "gaussian_high":
                # Высокая дисперсия
                emb = np.random.normal(0, param, dim).astype(np.float32)
                
            elif pattern_name == "gaussian_low":
                # Низкая дисперсия
                emb = np.random.normal(0, param, dim).astype(np.float32)
            
            embeddings.append(emb)
    else:
        # Обычная генерация
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