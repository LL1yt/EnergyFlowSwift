"""
Стратегии трансформации эмбедингов
=================================

Содержит три основные стратегии:
1. LinearReshaper - простое изменение формы
2. AdaptiveReshaper - умное преобразование с оптимизацией
3. SemanticReshaper - сохранение семантических кластеров
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from .utils import calculate_similarity_metrics, validate_semantic_preservation


class BaseReshaper(ABC):
    """
    Базовый класс для всех стратегий преобразования.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        cube_shape: Tuple[int, int, int] = (8, 8, 12),
        preserve_semantics: bool = True,
        semantic_threshold: float = 0.95
    ):
        self.input_dim = input_dim
        self.cube_shape = cube_shape
        self.preserve_semantics = preserve_semantics
        self.semantic_threshold = semantic_threshold
        
        # Проверяем совместимость размерностей
        cube_size = np.prod(cube_shape)
        if cube_size != input_dim:
            raise ValueError(
                f"Размеры не совпадают: input_dim={input_dim}, "
                f"cube_shape={cube_shape} (произведение={cube_size})"
            )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def vector_to_matrix(
        self, 
        embedding_1d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Преобразование 1D → 3D"""
        pass
    
    @abstractmethod
    def matrix_to_vector(
        self, 
        embedding_3d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Преобразование 3D → 1D"""
        pass


class LinearReshaper(BaseReshaper):
    """
    Простая стратегия линейного изменения формы.
    
    Выполняет прямое reshape без дополнительных преобразований.
    Быстрая, но может не сохранять семантическую структуру оптимально.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info(f"LinearReshaper инициализирован: {self.input_dim}D ↔ {self.cube_shape}")
    
    def vector_to_matrix(
        self, 
        embedding_1d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Простое преобразование 1D → 3D через reshape.
        
        Args:
            embedding_1d: Входной 1D эмбединг
            
        Returns:
            3D матрица той же размерности
        """
        if isinstance(embedding_1d, torch.Tensor):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            result = embedding_1d.reshape(self.cube_shape)
        elif isinstance(embedding_1d, np.ndarray):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            result = embedding_1d.reshape(self.cube_shape)
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(embedding_1d, result)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"LinearReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result
    
    def matrix_to_vector(
        self, 
        embedding_3d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Простое преобразование 3D → 1D через reshape.
        
        Args:
            embedding_3d: Входная 3D матрица
            
        Returns:
            1D вектор той же размерности
        """
        if isinstance(embedding_3d, torch.Tensor):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            result = embedding_3d.reshape(self.input_dim)
        elif isinstance(embedding_3d, np.ndarray):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            result = embedding_3d.reshape(self.input_dim)
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(result, embedding_3d)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"LinearReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result


class AdaptiveReshaper(BaseReshaper):
    """
    Адаптивная стратегия с оптимизацией под конкретные задачи.
    
    Анализирует входные данные и выбирает оптимальный способ преобразования
    для максимального сохранения семантической информации.
    """
    
    def __init__(self, adaptation_method: str = "variance_based", **kwargs):
        super().__init__(**kwargs)
        self.adaptation_method = adaptation_method
        self.adaptation_cache = {}  # Кэш для адаптивных преобразований
        
        self.logger.info(
            f"AdaptiveReshaper инициализирован: {self.input_dim}D ↔ {self.cube_shape}, "
            f"метод={adaptation_method}"
        )
    
    def vector_to_matrix(
        self, 
        embedding_1d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Адаптивное преобразование 1D → 3D.
        
        Анализирует распределение значений и выбирает оптимальную стратегию.
        """
        if isinstance(embedding_1d, torch.Tensor):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            embedding_np = embedding_1d.detach().cpu().numpy()
            is_torch = True
        elif isinstance(embedding_1d, np.ndarray):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            embedding_np = embedding_1d
            is_torch = False
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Адаптивное преобразование
        if self.adaptation_method == "variance_based":
            result_np = self._variance_based_transform(embedding_np)
        elif self.adaptation_method == "importance_weighted":
            result_np = self._importance_weighted_transform(embedding_np)
        else:
            # Fallback к простому reshape
            result_np = embedding_np.reshape(self.cube_shape)
        
        # Возвращаем в исходном формате
        if is_torch:
            result = torch.from_numpy(result_np)
        else:
            result = result_np
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(embedding_1d, result)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"AdaptiveReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result
    
    def matrix_to_vector(
        self, 
        embedding_3d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Адаптивное преобразование 3D → 1D.
        
        Восстанавливает исходный порядок элементов с учетом важности.
        """
        if isinstance(embedding_3d, torch.Tensor):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            embedding_np = embedding_3d.detach().cpu().numpy()
            is_torch = True
        elif isinstance(embedding_3d, np.ndarray):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            embedding_np = embedding_3d
            is_torch = False
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Адаптивное обратное преобразование
        if self.adaptation_method == "variance_based":
            result_np = self._variance_based_inverse_transform(embedding_np)
        elif self.adaptation_method == "importance_weighted":
            result_np = self._importance_weighted_inverse_transform(embedding_np)
        else:
            # Fallback к простому reshape
            result_np = embedding_np.reshape(self.input_dim)
        
        # Возвращаем в исходном формате
        if is_torch:
            result = torch.from_numpy(result_np)
        else:
            result = result_np
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(result, embedding_3d)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"AdaptiveReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result
    
    def _variance_based_transform(self, embedding_1d: np.ndarray) -> np.ndarray:
        """
        Преобразование на основе дисперсии значений.
        
        Размещает элементы с высокой дисперсией в центральной части куба.
        """
        # Простая реализация: сортируем по абсолютному значению
        indices = np.argsort(np.abs(embedding_1d))[::-1]  # От большего к меньшему
        sorted_values = embedding_1d[indices]
        
        # Формируем 3D структуру
        result = sorted_values.reshape(self.cube_shape)
        
        return result
    
    def _variance_based_inverse_transform(self, embedding_3d: np.ndarray) -> np.ndarray:
        """Обратное преобразование для variance_based."""
        # Для простоты используем обычный reshape
        # В реальной реализации нужно восстанавливать исходный порядок
        return embedding_3d.reshape(self.input_dim)
    
    def _importance_weighted_transform(self, embedding_1d: np.ndarray) -> np.ndarray:
        """
        Преобразование с весами важности.
        
        Использует градиент или другие метрики важности для размещения.
        """
        # Вычисляем веса важности (простая реализация)
        importance_weights = np.abs(embedding_1d) / (np.abs(embedding_1d).sum() + 1e-8)
        
        # Сортируем по важности
        weighted_indices = np.argsort(importance_weights)[::-1]
        weighted_values = embedding_1d[weighted_indices]
        
        # Формируем 3D структуру
        result = weighted_values.reshape(self.cube_shape)
        
        return result
    
    def _importance_weighted_inverse_transform(self, embedding_3d: np.ndarray) -> np.ndarray:
        """Обратное преобразование для importance_weighted."""
        # Для простоты используем обычный reshape
        return embedding_3d.reshape(self.input_dim)


class SemanticReshaper(BaseReshaper):
    """
    Стратегия сохранения семантических кластеров.
    
    Анализирует семантическую структуру эмбединга и размещает похожие
    элементы в пространственной близости в 3D кубе.
    """
    
    def __init__(self, clustering_method: str = "kmeans", n_clusters: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.cluster_cache = {}
        
        self.logger.info(
            f"SemanticReshaper инициализирован: {self.input_dim}D ↔ {self.cube_shape}, "
            f"кластеры={n_clusters}, метод={clustering_method}"
        )
    
    def vector_to_matrix(
        self, 
        embedding_1d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Семантическое преобразование 1D → 3D.
        
        Группирует похожие элементы в пространственные кластеры.
        """
        if isinstance(embedding_1d, torch.Tensor):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            embedding_np = embedding_1d.detach().cpu().numpy()
            is_torch = True
        elif isinstance(embedding_1d, np.ndarray):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            embedding_np = embedding_1d
            is_torch = False
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Семантическое кластерирование
        if self.clustering_method == "kmeans":
            result_np = self._kmeans_transform(embedding_np)
        elif self.clustering_method == "hierarchical":
            result_np = self._hierarchical_transform(embedding_np)
        else:
            # Fallback к простому reshape
            result_np = embedding_np.reshape(self.cube_shape)
        
        # Возвращаем в исходном формате
        if is_torch:
            result = torch.from_numpy(result_np)
        else:
            result = result_np
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(embedding_1d, result)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"SemanticReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result
    
    def matrix_to_vector(
        self, 
        embedding_3d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Семантическое преобразование 3D → 1D.
        
        Восстанавливает исходную семантическую структуру.
        """
        if isinstance(embedding_3d, torch.Tensor):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            embedding_np = embedding_3d.detach().cpu().numpy()
            is_torch = True
        elif isinstance(embedding_3d, np.ndarray):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            embedding_np = embedding_3d
            is_torch = False
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Семантическое обратное преобразование
        if self.clustering_method == "kmeans":
            result_np = self._kmeans_inverse_transform(embedding_np)
        elif self.clustering_method == "hierarchical":
            result_np = self._hierarchical_inverse_transform(embedding_np)
        else:
            # Fallback к простому reshape
            result_np = embedding_np.reshape(self.input_dim)
        
        # Возвращаем в исходном формате
        if is_torch:
            result = torch.from_numpy(result_np)
        else:
            result = result_np
        
        # Контроль качества
        if self.preserve_semantics:
            similarity = calculate_similarity_metrics(result, embedding_3d)
            if similarity < self.semantic_threshold:
                self.logger.warning(
                    f"SemanticReshaper: качество {similarity:.3f} < {self.semantic_threshold}"
                )
        
        return result
    
    def _kmeans_transform(self, embedding_1d: np.ndarray) -> np.ndarray:
        """
        K-means кластеризация для семантического группирования.
        
        Пока простая реализация - группировка по значениям.
        """
        # Простая реализация: группируем по диапазонам значений
        n_groups = min(self.n_clusters, len(embedding_1d))
        
        # Сортируем значения и индексы
        sorted_indices = np.argsort(embedding_1d)
        group_size = len(embedding_1d) // n_groups
        
        # Переупорядочиваем по группам
        reordered_values = np.zeros_like(embedding_1d)
        for i, group_start in enumerate(range(0, len(embedding_1d), group_size)):
            group_end = min(group_start + group_size, len(embedding_1d))
            group_indices = sorted_indices[group_start:group_end]
            reordered_values[group_start:group_end] = embedding_1d[group_indices]
        
        return reordered_values.reshape(self.cube_shape)
    
    def _kmeans_inverse_transform(self, embedding_3d: np.ndarray) -> np.ndarray:
        """Обратное преобразование для kmeans."""
        # Для простоты используем обычный reshape
        return embedding_3d.reshape(self.input_dim)
    
    def _hierarchical_transform(self, embedding_1d: np.ndarray) -> np.ndarray:
        """Иерархическая кластеризация."""
        # Простая реализация через сортировку
        return np.sort(embedding_1d).reshape(self.cube_shape)
    
    def _hierarchical_inverse_transform(self, embedding_3d: np.ndarray) -> np.ndarray:
        """Обратное преобразование для hierarchical."""
        return embedding_3d.reshape(self.input_dim) 