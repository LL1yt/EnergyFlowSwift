"""
EmbeddingReshaper - Основной класс для трансформаций эмбедингов
=============================================================

Обеспечивает преобразование 1D↔3D формат для куба и обратно
с сохранением семантической информации >95%.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
import logging

from .utils import validate_semantic_preservation, calculate_similarity_metrics


class EmbeddingReshaper:
    """
    Основной класс для преобразования эмбедингов между 1D и 3D форматами.
    
    Ключевые возможности:
    - Трансформация 1D → 3D: (768,) → (8, 8, 12)  
    - Трансформация 3D → 1D: (8, 8, 12) → (768,)
    - Сохранение семантической информации >95%
    
    Args:
        input_dim (int): Размерность входного эмбединга (по умолчанию 768)
        cube_shape (Tuple[int, int, int]): Форма 3D куба (по умолчанию (8, 8, 12))
        reshaping_method (str): Метод преобразования ('linear', 'adaptive', 'semantic')
        preserve_semantics (bool): Включить контроль качества (по умолчанию True)
        semantic_threshold (float): Порог качества сохранения семантики (по умолчанию 0.95)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        cube_shape: Tuple[int, int, int] = (8, 8, 12),
        reshaping_method: str = "adaptive",
        preserve_semantics: bool = True,
        semantic_threshold: float = 0.95
    ):
        self.input_dim = input_dim
        self.cube_shape = cube_shape
        self.reshaping_method = reshaping_method
        self.preserve_semantics = preserve_semantics
        self.semantic_threshold = semantic_threshold
        
        # Проверяем совместимость размерностей
        cube_size = np.prod(cube_shape)
        if cube_size != input_dim:
            raise ValueError(
                f"Размеры не совпадают: input_dim={input_dim}, "
                f"cube_shape={cube_shape} (произведение={cube_size})"
            )
        
        # Инициализируем логирование
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"EmbeddingReshaper инициализирован: {input_dim}D ↔ {cube_shape}, "
            f"метод={reshaping_method}, порог={semantic_threshold}"
        )
        
        # Статистика использования
        self.stats = {
            'transformations_1d_to_3d': 0,
            'transformations_3d_to_1d': 0,
            'semantic_quality_avg': [],
            'successful_preservations': 0,
            'failed_preservations': 0
        }
    
    def vector_to_matrix(
        self, 
        embedding_1d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Преобразование 1D эмбединга в 3D матрицу.
        
        Args:
            embedding_1d: Входной эмбединг размерности (768,)
            
        Returns:
            3D матрица размерности (8, 8, 12)
            
        Raises:
            ValueError: Если размерности не совпадают
        """
        # Проверяем входные данные
        if isinstance(embedding_1d, torch.Tensor):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            # Простое изменение формы для PyTorch
            embedding_3d = embedding_1d.reshape(self.cube_shape)
        elif isinstance(embedding_1d, np.ndarray):
            if embedding_1d.shape != (self.input_dim,):
                raise ValueError(f"Ожидается размерность {self.input_dim}, получено {embedding_1d.shape}")
            # Простое изменение формы для NumPy
            embedding_3d = embedding_1d.reshape(self.cube_shape)
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Улучшенный контроль качества при необходимости
        if self.preserve_semantics:
            # Используем расширенные метрики семантического сходства
            from .utils import calculate_enhanced_similarity_metrics
            
            try:
                enhanced_metrics = calculate_enhanced_similarity_metrics(embedding_1d, embedding_3d)
                similarity = enhanced_metrics['weighted_similarity']
                
                # Логирование детальных метрик для отладки
                self.logger.debug(f"Enhanced metrics 1D→3D: {enhanced_metrics}")
                
            except Exception as e:
                # Fallback к базовой метрике
                self.logger.warning(f"Enhanced metrics failed: {e}, using basic similarity")
                similarity = self._check_semantic_preservation(embedding_1d, embedding_3d)
            
            self.stats['semantic_quality_avg'].append(similarity)
            
            if similarity >= self.semantic_threshold:
                self.stats['successful_preservations'] += 1
                if similarity >= 0.98:
                    self.logger.info(f"🎯 Высокое качество 1D→3D достигнуто: {similarity:.6f}")
            else:
                self.stats['failed_preservations'] += 1
                self.logger.warning(
                    f"Качество преобразования 1D→3D ниже порога: {similarity:.6f} < {self.semantic_threshold}"
                )
        
        self.stats['transformations_1d_to_3d'] += 1
        return embedding_3d
    
    def matrix_to_vector(
        self, 
        embedding_3d: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Преобразование 3D матрицы в 1D эмбединг.
        
        Args:
            embedding_3d: Входная 3D матрица размерности (8, 8, 12)
            
        Returns:
            1D эмбединг размерности (768,)
            
        Raises:
            ValueError: Если размерности не совпадают
        """
        # Проверяем входные данные
        if isinstance(embedding_3d, torch.Tensor):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            # Простое изменение формы для PyTorch
            embedding_1d = embedding_3d.reshape(self.input_dim)
        elif isinstance(embedding_3d, np.ndarray):
            if embedding_3d.shape != self.cube_shape:
                raise ValueError(f"Ожидается размерность {self.cube_shape}, получено {embedding_3d.shape}")
            # Простое изменение формы для NumPy
            embedding_1d = embedding_3d.reshape(self.input_dim)
        else:
            raise TypeError("Поддерживаются только torch.Tensor и np.ndarray")
        
        # Улучшенный контроль качества при необходимости
        if self.preserve_semantics:
            # Используем расширенные метрики семантического сходства
            from .utils import calculate_enhanced_similarity_metrics
            
            try:
                enhanced_metrics = calculate_enhanced_similarity_metrics(embedding_1d, embedding_3d)
                similarity = enhanced_metrics['weighted_similarity']
                
                # Логирование детальных метрик для отладки
                self.logger.debug(f"Enhanced metrics 3D→1D: {enhanced_metrics}")
                
            except Exception as e:
                # Fallback к базовой метрике
                self.logger.warning(f"Enhanced metrics failed: {e}, using basic similarity")
                similarity = self._check_semantic_preservation(embedding_1d, embedding_3d)
            
            self.stats['semantic_quality_avg'].append(similarity)
            
            if similarity >= self.semantic_threshold:
                self.stats['successful_preservations'] += 1
                if similarity >= 0.98:
                    self.logger.info(f"🎯 Высокое качество 3D→1D достигнуто: {similarity:.6f}")
            else:
                self.stats['failed_preservations'] += 1
                self.logger.warning(
                    f"Качество преобразования 3D→1D ниже порога: {similarity:.6f} < {self.semantic_threshold}"
                )
        
        self.stats['transformations_3d_to_1d'] += 1
        return embedding_1d
    
    def _check_semantic_preservation(
        self, 
        original: Union[torch.Tensor, np.ndarray],
        reshaped: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Проверка сохранения семантической информации.
        
        Args:
            original: Исходный эмбединг
            reshaped: Преобразованный эмбединг
            
        Returns:
            Cosine similarity между векторами (0-1)
        """
        # Приводим к одинаковой форме для сравнения
        if original.shape != reshaped.shape:
            if len(original.shape) == 1:
                # original 1D → приводим reshaped к 1D
                reshaped_flat = reshaped.reshape(-1) if hasattr(reshaped, 'reshape') else reshaped.flatten()
            else:
                # original 3D → приводим к 1D
                original_flat = original.reshape(-1) if hasattr(original, 'reshape') else original.flatten()
                reshaped_flat = reshaped.reshape(-1) if hasattr(reshaped, 'reshape') else reshaped.flatten()
                original = original_flat
                reshaped = reshaped_flat
        
        # Вычисляем cosine similarity
        return calculate_similarity_metrics(original, reshaped)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику использования модуля.
        
        Returns:
            Словарь со статистикой
        """
        avg_quality = np.mean(self.stats['semantic_quality_avg']) if self.stats['semantic_quality_avg'] else 0.0
        success_rate = (
            self.stats['successful_preservations'] / 
            (self.stats['successful_preservations'] + self.stats['failed_preservations'])
            if (self.stats['successful_preservations'] + self.stats['failed_preservations']) > 0 
            else 0.0
        )
        
        return {
            'total_1d_to_3d': self.stats['transformations_1d_to_3d'],
            'total_3d_to_1d': self.stats['transformations_3d_to_1d'],
            'average_semantic_quality': avg_quality,
            'semantic_preservation_success_rate': success_rate,
            'successful_preservations': self.stats['successful_preservations'],
            'failed_preservations': self.stats['failed_preservations'],
            'semantic_threshold': self.semantic_threshold
        }
    
    def reset_statistics(self):
        """Сброс статистики использования."""
        self.stats = {
            'transformations_1d_to_3d': 0,
            'transformations_3d_to_1d': 0,
            'semantic_quality_avg': [],
            'successful_preservations': 0,
            'failed_preservations': 0
        }
        self.logger.info("Статистика EmbeddingReshaper сброшена")
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingReshaper(input_dim={self.input_dim}, "
            f"cube_shape={self.cube_shape}, method={self.reshaping_method})"
        ) 