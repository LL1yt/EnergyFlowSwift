"""
Preprocessing utilities for embeddings.
"""

import logging
from typing import Union, Optional

import torch
import numpy as np
from torch import Tensor


logger = logging.getLogger(__name__)


class EmbeddingPreprocessor:
    """Препроцессор для эмбедингов."""
    
    def __init__(self):
        """Инициализация препроцессора."""
        self.statistics = {}
    
    def preprocess(self, 
                   embeddings: Union[Tensor, np.ndarray],
                   normalize: bool = True,
                   center: bool = True,
                   clip_outliers: bool = False,
                   outlier_std: float = 3.0) -> Tensor:
        """
        Основная функция предобработки эмбедингов.
        
        Args:
            embeddings: Исходные эмбединги
            normalize: Нормализовать ли векторы (L2 normalization)
            center: Центрировать ли данные (вычесть среднее)
            clip_outliers: Обрезать ли выбросы
            outlier_std: Количество стандартных отклонений для обрезки
            
        Returns:
            torch.Tensor: Предобработанные эмбединги
        """
        # Конвертируем в tensor если нужно
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        elif not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        original_shape = embeddings.shape
        logger.info(f"Preprocessing embeddings with shape: {original_shape}")
        
        # Сохраняем исходную статистику
        self._compute_statistics(embeddings, prefix="original")
        
        # Центрирование
        if center:
            embeddings = self._center_embeddings(embeddings)
        
        # Обрезка выбросов
        if clip_outliers:
            embeddings = self._clip_outliers(embeddings, outlier_std)
        
        # Нормализация
        if normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Сохраняем финальную статистику
        self._compute_statistics(embeddings, prefix="processed")
        
        logger.info(f"Preprocessing completed. Shape: {embeddings.shape}")
        return embeddings
    
    def _center_embeddings(self, embeddings: Tensor) -> Tensor:
        """
        Центрирование эмбедингов (вычитание среднего).
        
        Args:
            embeddings: Исходные эмбединги
            
        Returns:
            torch.Tensor: Центрированные эмбединги
        """
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean
        
        self.statistics['mean'] = mean
        logger.info("Embeddings centered (mean subtracted)")
        
        return centered
    
    def _normalize_embeddings(self, embeddings: Tensor) -> Tensor:
        """
        L2 нормализация эмбедингов.
        
        Args:
            embeddings: Исходные эмбединги
            
        Returns:
            torch.Tensor: Нормализованные эмбединги
        """
        # Вычисляем L2 норму для каждого вектора
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        
        # Избегаем деления на ноль
        norms = torch.clamp(norms, min=1e-8)
        
        normalized = embeddings / norms
        
        logger.info("Embeddings L2 normalized")
        return normalized
    
    def _clip_outliers(self, embeddings: Tensor, std_threshold: float = 3.0) -> Tensor:
        """
        Обрезка выбросов в эмбедингах.
        
        Args:
            embeddings: Исходные эмбединги
            std_threshold: Пороговое значение в стандартных отклонениях
            
        Returns:
            torch.Tensor: Эмбединги с обрезанными выбросами
        """
        mean = embeddings.mean()
        std = embeddings.std()
        
        # Определяем границы
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        # Обрезаем выбросы
        clipped = torch.clamp(embeddings, lower_bound, upper_bound)
        
        outliers_count = (embeddings != clipped).sum().item()
        logger.info(f"Clipped {outliers_count} outlier values "
                   f"(threshold: {std_threshold} std)")
        
        return clipped
    
    def _compute_statistics(self, embeddings: Tensor, prefix: str = "") -> None:
        """
        Вычисление статистик эмбедингов.
        
        Args:
            embeddings: Эмбединги для анализа
            prefix: Префикс для ключей статистик
        """
        prefix = f"{prefix}_" if prefix else ""
        
        stats = {
            f"{prefix}shape": embeddings.shape,
            f"{prefix}mean": float(embeddings.mean()),
            f"{prefix}std": float(embeddings.std()),
            f"{prefix}min": float(embeddings.min()),
            f"{prefix}max": float(embeddings.max()),
            f"{prefix}norm_mean": float(torch.norm(embeddings, dim=1).mean()),
        }
        
        self.statistics.update(stats)
    
    def standardize_embeddings(self, embeddings: Tensor) -> Tensor:
        """
        Стандартизация эмбедингов (z-score normalization).
        
        Args:
            embeddings: Исходные эмбединги
            
        Returns:
            torch.Tensor: Стандартизованные эмбединги
        """
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True)
        
        # Избегаем деления на ноль
        std = torch.clamp(std, min=1e-8)
        
        standardized = (embeddings - mean) / std
        
        logger.info("Embeddings standardized (z-score)")
        return standardized
    
    def whiten_embeddings(self, embeddings: Tensor) -> Tensor:
        """
        Отбеливание эмбедингов (whitening/decorrelation).
        
        Args:
            embeddings: Исходные эмбединги
            
        Returns:
            torch.Tensor: Отбеленные эмбединги
        """
        # Центрируем данные
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Вычисляем ковариационную матрицу
        cov_matrix = torch.mm(centered.T, centered) / (centered.shape[0] - 1)
        
        # Eigen-декомпозиция
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Избегаем численных проблем
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        
        # Матрица отбеливания
        whitening_matrix = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Применяем отбеливание
        whitened = torch.mm(centered, whitening_matrix)
        
        logger.info("Embeddings whitened (decorrelated)")
        return whitened
    
    def reduce_dimensions(self, 
                         embeddings: Tensor, 
                         target_dim: int,
                         method: str = "pca") -> Tensor:
        """
        Уменьшение размерности эмбедингов.
        
        Args:
            embeddings: Исходные эмбединги
            target_dim: Целевая размерность
            method: Метод уменьшения размерности ('pca', 'random')
            
        Returns:
            torch.Tensor: Эмбединги с уменьшенной размерностью
        """
        if target_dim >= embeddings.shape[1]:
            logger.warning(f"Target dimension {target_dim} >= current dimension {embeddings.shape[1]}")
            return embeddings
        
        if method == "pca":
            return self._pca_reduction(embeddings, target_dim)
        elif method == "random":
            return self._random_projection(embeddings, target_dim)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def _pca_reduction(self, embeddings: Tensor, target_dim: int) -> Tensor:
        """PCA уменьшение размерности."""
        # Центрируем данные
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # SVD разложение
        U, S, V = torch.svd(centered)
        
        # Берем первые target_dim компонент
        reduced = torch.mm(centered, V[:, :target_dim])
        
        logger.info(f"PCA reduction: {embeddings.shape[1]} -> {target_dim}")
        return reduced
    
    def _random_projection(self, embeddings: Tensor, target_dim: int) -> Tensor:
        """Случайная проекция для уменьшения размерности."""
        original_dim = embeddings.shape[1]
        
        # Создаем случайную матрицу проекции
        projection_matrix = torch.randn(original_dim, target_dim)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)
        
        # Применяем проекцию
        reduced = torch.mm(embeddings, projection_matrix)
        
        logger.info(f"Random projection: {original_dim} -> {target_dim}")
        return reduced
    
    def get_statistics(self) -> dict:
        """Получение статистик предобработки."""
        return self.statistics.copy()
    
    def reset_statistics(self) -> None:
        """Сброс сохраненных статистик."""
        self.statistics.clear() 