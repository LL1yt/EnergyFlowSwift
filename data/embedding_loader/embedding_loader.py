"""
Main EmbeddingLoader class for loading and preprocessing various embedding formats.
"""

import os
import pickle
import logging
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

import torch
import numpy as np
from torch import Tensor

from .format_handlers import FormatHandler, Word2VecHandler, GloVeHandler, BertHandler
from .preprocessing import EmbeddingPreprocessor


logger = logging.getLogger(__name__)


class EmbeddingLoader:
    """
    Загрузчик эмбедингов различных форматов.
    
    Поддерживаемые форматы:
    - Word2Vec (.bin, .txt)
    - GloVe (.txt)
    - BERT embeddings (.pt, .pkl)
    """
    
    def __init__(self, cache_dir: str = "./data/cache/", max_cache_size: str = "2GB"):
        """
        Инициализация загрузчика эмбедингов.
        
        Args:
            cache_dir: Директория для кэширования
            max_cache_size: Максимальный размер кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # Инициализация обработчиков форматов
        self.format_handlers: Dict[str, FormatHandler] = {
            'word2vec': Word2VecHandler(),
            'glove': GloVeHandler(),
            'bert': BertHandler()
        }
        
        # Инициализация препроцессора
        self.preprocessor = EmbeddingPreprocessor()
        
        # Кэш загруженных эмбедингов
        self._embedding_cache: Dict[str, Tensor] = {}
        
        logger.info(f"EmbeddingLoader initialized with cache_dir: {cache_dir}")
    
    def load_embeddings(self, 
                       path: Union[str, Path], 
                       format_type: str,
                       preprocess: bool = True) -> Tensor:
        """
        Загрузка эмбедингов из файла.
        
        Args:
            path: Путь к файлу с эмбедингами
            format_type: Тип формата ('word2vec', 'glove', 'bert')
            preprocess: Применять ли предобработку
            
        Returns:
            torch.Tensor: Загруженные эмбединги
            
        Raises:
            ValueError: Если формат не поддерживается
            FileNotFoundError: Если файл не найден
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        
        if format_type not in self.format_handlers:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {list(self.format_handlers.keys())}")
        
        # Проверяем кэш
        cache_key = f"{path.stem}_{format_type}"
        if cache_key in self._embedding_cache:
            logger.info(f"Loading embeddings from cache: {cache_key}")
            return self._embedding_cache[cache_key]
        
        # Загружаем эмбединги
        handler = self.format_handlers[format_type]
        logger.info(f"Loading embeddings from {path} using {format_type} handler")
        
        embeddings = handler.load(str(path))
        
        # Предобработка
        if preprocess:
            embeddings = self.preprocessor.preprocess(embeddings)
        
        # Конвертируем в torch.Tensor если это не tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.from_numpy(embeddings).float()
        
        # Кэшируем
        self._embedding_cache[cache_key] = embeddings
        
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    def preprocess_embeddings(self, 
                            embeddings: Tensor,
                            normalize: bool = True,
                            center: bool = True) -> Tensor:
        """
        Предобработка эмбедингов.
        
        Args:
            embeddings: Исходные эмбединги
            normalize: Нормализовать ли вектора
            center: Центрировать ли вектора
            
        Returns:
            torch.Tensor: Предобработанные эмбединги
        """
        return self.preprocessor.preprocess(
            embeddings, 
            normalize=normalize, 
            center=center
        )
    
    def cache_embeddings(self, 
                        embeddings: Tensor, 
                        cache_key: str) -> None:
        """
        Кэширование эмбедингов.
        
        Args:
            embeddings: Эмбединги для кэширования
            cache_key: Ключ для кэша
        """
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        try:
            torch.save(embeddings, cache_path)
            logger.info(f"Cached embeddings to: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
    
    def load_from_cache(self, cache_key: str) -> Optional[Tensor]:
        """
        Загрузка эмбедингов из кэша.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            torch.Tensor или None: Загруженные эмбединги или None если не найдено
        """
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        if not cache_path.exists():
            return None
        
        try:
            embeddings = torch.load(cache_path)
            logger.info(f"Loaded embeddings from cache: {cache_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
    
    def get_embedding_info(self, embeddings: Tensor) -> Dict[str, Union[int, float]]:
        """
        Получение информации об эмбедингах.
        
        Args:
            embeddings: Эмбединги для анализа
            
        Returns:
            Dict: Информация об эмбедингах
        """
        return {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'device': str(embeddings.device),
            'memory_mb': embeddings.element_size() * embeddings.nelement() / (1024 * 1024),
            'min_value': float(embeddings.min()),
            'max_value': float(embeddings.max()),
            'mean_value': float(embeddings.mean()),
            'std_value': float(embeddings.std())
        }
    
    def clear_cache(self) -> None:
        """Очистка кэша эмбедингов."""
        self._embedding_cache.clear()
        logger.info("Memory cache cleared")
    
    def get_supported_formats(self) -> list:
        """Получение списка поддерживаемых форматов."""
        return list(self.format_handlers.keys()) 