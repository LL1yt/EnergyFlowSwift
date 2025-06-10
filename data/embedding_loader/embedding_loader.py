"""
Main EmbeddingLoader class for loading and preprocessing various embedding formats.
"""

import os
import pickle
import logging
import time
from typing import Dict, Optional, Union, Tuple, List, Any
from pathlib import Path

import torch
import numpy as np
from torch import Tensor

from .format_handlers import FormatHandler, Word2VecHandler, GloVeHandler, BertHandler, LLMHandler, create_llm_handler, SUPPORTED_LLM_MODELS
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
    
    def __init__(self, cache_dir: str = "./cache", config_path: str = None):
        """
        Инициализация EmbeddingLoader.
        
        Args:
            cache_dir: Директория для кэширования
            config_path: Путь к конфигурационному файлу
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Инициализируем обработчики форматов
        self.handlers = {
            'word2vec': Word2VecHandler(),
            'glove': GloVeHandler(),
            'bert': BertHandler(),
            'llm': None  # Ленивая инициализация для LLM
        }
        
        # Статистики
        self.stats = {
            'loaded_files': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_generations': 0,
            'total_texts_processed': 0
        }
        
        # Инициализируем препроцессор
        from .preprocessing import EmbeddingPreprocessor
        self.preprocessor = EmbeddingPreprocessor()
        
        # Кэш загруженных эмбедингов в памяти
        self._embedding_cache: Dict[str, Tensor] = {}
        
        # Загружаем конфигурацию
        self.config = self._load_config(config_path)
        
        logger.info(f"EmbeddingLoader initialized with cache: {cache_dir}")
    
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
        
        if format_type not in self.handlers:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {list(self.handlers.keys())}")
        
        # Проверяем кэш
        cache_key = f"{path.stem}_{format_type}"
        if cache_key in self._embedding_cache:
            logger.info(f"Loading embeddings from cache: {cache_key}")
            return self._embedding_cache[cache_key]
        
        # Загружаем эмбединги
        handler = self.handlers[format_type]
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
        return list(self.handlers.keys())
    
    def load_from_llm(self, texts: List[str], model_key: str = "distilbert", 
                     pooling_strategy: str = "mean", 
                     use_cache: bool = True) -> Tensor:
        """
        Генерация эмбедингов из текстов через LLM (Knowledge Distillation).
        
        Args:
            texts: Список текстов для обработки
            model_key: Ключ LLM модели из SUPPORTED_LLM_MODELS
            pooling_strategy: Стратегия агрегации ("mean", "cls", "max")
            use_cache: Использовать кэширование
            
        Returns:
            torch.Tensor: Эмбединги текстов
        """
        if not texts:
            raise ValueError("Empty text list provided")
        
        # Создаем уникальный ключ кэша
        cache_key = self._create_llm_cache_key(texts, model_key, pooling_strategy)
        cache_path = self.cache_dir / f"llm_{cache_key}.pt"
        
        # Проверяем кэш
        if use_cache and cache_path.exists():
            logger.info(f"Loading LLM embeddings from cache: {cache_path}")
            embeddings = torch.load(cache_path)
            self.stats['cache_hits'] += 1
            return embeddings
        
        # Инициализируем LLM handler если нужно
        if model_key not in SUPPORTED_LLM_MODELS:
            raise ValueError(f"Unsupported LLM model: {model_key}")
        
        llm_handler = create_llm_handler(model_key)
        
        # Генерируем эмбединги
        logger.info(f"Generating embeddings for {len(texts)} texts using {model_key}")
        embeddings = llm_handler.generate_embeddings(texts, pooling_strategy)
        
        # Обновляем статистики
        self.stats['llm_generations'] += 1
        self.stats['total_texts_processed'] += len(texts)
        self.stats['cache_misses'] += 1
        
        # Сохраняем в кэш
        if use_cache:
            torch.save(embeddings, cache_path)
            logger.info(f"Cached LLM embeddings to: {cache_path}")
        
        return embeddings
    
    def batch_load_from_llm(self, texts: List[str], model_key: str = "distilbert",
                           batch_size: int = 16, **kwargs) -> Tensor:
        """
        Батчевая генерация эмбедингов через LLM.
        
        Args:
            texts: Список текстов
            model_key: Ключ LLM модели
            batch_size: Размер батча
            **kwargs: Дополнительные параметры для load_from_llm
            
        Returns:
            torch.Tensor: Объединенные эмбединги
        """
        llm_handler = create_llm_handler(model_key)
        embeddings = llm_handler.batch_generate_embeddings(texts, batch_size)
        
        self.stats['llm_generations'] += 1
        self.stats['total_texts_processed'] += len(texts)
        
        return embeddings
    
    def create_knowledge_distillation_dataset(self, texts: List[str], 
                                            teacher_model: str = "llama2-7b",
                                            save_path: str = None) -> Dict[str, Tensor]:
        """
        Создание датасета для Knowledge Distillation.
        
        Генерирует эмбединги teacher модели (LLM) для обучения student модели (3D CNN).
        
        Args:
            texts: Тексты для обработки
            teacher_model: Модель-учитель (LLM)
            save_path: Путь для сохранения датасета
            
        Returns:
            Dict с эмбедингами teacher модели и метаданными
        """
        logger.info(f"Creating Knowledge Distillation dataset with {teacher_model}")
        
        # Генерируем эмбединги teacher модели
        teacher_embeddings = self.load_from_llm(
            texts=texts, 
            model_key=teacher_model,
            pooling_strategy="mean"
        )
        
        # Создаем датасет
        dataset = {
            'teacher_embeddings': teacher_embeddings,
            'texts': texts,
            'teacher_model': teacher_model,
            'num_samples': len(texts),
            'embedding_dim': teacher_embeddings.shape[1],
            'created_at': torch.tensor(time.time())
        }
        
        # Сохраняем если указан путь
        if save_path:
            torch.save(dataset, save_path)
            logger.info(f"Knowledge Distillation dataset saved to: {save_path}")
        
        return dataset
    
    def _create_llm_cache_key(self, texts: List[str], model_key: str, 
                             pooling_strategy: str) -> str:
        """Создание ключа кэша для LLM эмбедингов."""
        import hashlib
        
        # Создаем хэш из текстов и параметров
        content = f"{model_key}_{pooling_strategy}_{'|'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get_llm_info(self, model_key: str) -> Dict[str, Any]:
        """Получение информации о LLM модели."""
        if model_key not in SUPPORTED_LLM_MODELS:
            raise ValueError(f"Unknown model key: {model_key}")
        
        llm_handler = create_llm_handler(model_key)
        return llm_handler.get_model_info()
    
    def list_supported_llm_models(self) -> List[str]:
        """Список поддерживаемых LLM моделей."""
        return list(SUPPORTED_LLM_MODELS.keys())
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Загрузка конфигурации через ConfigManager или из YAML файла."""
        try:
            # Сначала пробуем загрузить через ConfigManager
            if config_path is None:
                try:
                    from utils.config_manager import get_global_config_manager
                    config_manager = get_global_config_manager()
                    
                    if config_manager:
                        # Пробуем получить конфигурацию embedding_loader
                        config = config_manager.get_config('embedding_loader')
                        if config:
                            logger.info("[OK] Loaded embedding_loader config from ConfigManager")
                            return config
                        
                        # Пробуем секции с префиксом embedding_loader_*
                        full_config = config_manager.get_config()
                        embedding_config = {}
                        
                        for section_name, section_data in full_config.items():
                            if section_name.startswith('embedding_loader_'):
                                subsection = section_name.replace('embedding_loader_', '')
                                embedding_config[subsection] = section_data
                        
                        if embedding_config:
                            logger.info("[OK] Loaded embedding_loader config from ConfigManager (prefixed sections)")
                            return embedding_config
                            
                except ImportError:
                    logger.warning("ConfigManager not available, falling back to file loading")
                except Exception as e:
                    logger.warning(f"ConfigManager error: {e}, falling back to file loading")
            
            # Fallback: загружаем из файла
            if config_path is None:
                config_path = Path(__file__).parent / "config" / "embedding_config.yaml"
            
            import yaml
            
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from file: {config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
                
        except ImportError:
            logger.warning("PyYAML not installed, using default configuration")
            return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию если YAML не доступен."""
        return {
            'cache': {
                'enabled': True,
                'directory': './data/cache/',
                'max_size': '2GB'
            },
            'llm': {
                'default_model': 'distilbert',
                'default_pooling': 'mean',
                'batch_size': 16,
                'cache_embeddings': True,
                'device': 'auto'
            },
            'knowledge_distillation': {
                'enabled': True,
                'default_teacher': 'distilbert',
                'save_datasets': True,
                'dataset_save_dir': './data/distillation_datasets/'
            }
        }

    def _create_cache_key(self, path: str, format_type: str) -> str:
        """Создание ключа кэша для файла."""
        import hashlib
        
        # Создаем хэш из пути и типа
        content = f"{format_type}_{path}"
        return hashlib.md5(content.encode()).hexdigest()[:16] 