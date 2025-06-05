"""
Format handlers for different embedding file formats.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Union, Dict, Any

import torch
import numpy as np
from torch import Tensor


logger = logging.getLogger(__name__)


class FormatHandler(ABC):
    """Базовый абстрактный класс для обработчиков форматов эмбедингов."""
    
    @abstractmethod
    def load(self, path: str) -> Union[Tensor, np.ndarray]:
        """
        Загрузка эмбедингов из файла.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Загруженные эмбединги
        """
        pass
    
    @abstractmethod
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """
        Получение словаря токен -> индекс.
        
        Args:
            path: Путь к файлу
            
        Returns:
            Словарь токенов
        """
        pass


class TextFormatHandler(FormatHandler):
    """Базовый обработчик для текстовых форматов."""
    
    def _parse_text_line(self, line: str) -> tuple:
        """
        Парсинг строки из текстового файла.
        
        Args:
            line: Строка для парсинга
            
        Returns:
            (token, vector) кортеж
        """
        parts = line.strip().split()
        if len(parts) < 2:
            return None, None
        
        token = parts[0]
        try:
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            return token, vector
        except ValueError:
            logger.warning(f"Failed to parse line: {line[:50]}...")
            return None, None


class Word2VecHandler(TextFormatHandler):
    """Обработчик Word2Vec формата (.txt и .bin)."""
    
    def load(self, path: str) -> np.ndarray:
        """
        Загрузка Word2Vec эмбедингов.
        
        Args:
            path: Путь к файлу (.txt или .bin)
            
        Returns:
            numpy.ndarray: Матрица эмбедингов
        """
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.bin':
            return self._load_binary(path)
        elif file_ext == '.txt':
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported Word2Vec file extension: {file_ext}")
    
    def _load_text(self, path: str) -> np.ndarray:
        """Загрузка из текстового формата."""
        embeddings = []
        vocab = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            # Первая строка может содержать размеры
            first_line = f.readline().strip()
            if len(first_line.split()) == 2:
                vocab_size, embed_dim = map(int, first_line.split())
                logger.info(f"Word2Vec info: vocab_size={vocab_size}, embed_dim={embed_dim}")
            else:
                # Если первая строка - это эмбединг, вернемся к началу
                f.seek(0)
            
            for idx, line in enumerate(f):
                token, vector = self._parse_text_line(line)
                if token is not None and vector is not None:
                    vocab[token] = idx
                    embeddings.append(vector)
        
        if not embeddings:
            raise ValueError("No valid embeddings found in file")
        
        self._vocabulary = vocab
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Loaded Word2Vec embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def _load_binary(self, path: str) -> np.ndarray:
        """Загрузка из бинарного формата (требует gensim)."""
        try:
            from gensim.models import KeyedVectors
            
            logger.info(f"Loading Word2Vec binary file: {path}")
            model = KeyedVectors.load_word2vec_format(path, binary=True)
            
            # Сохраняем словарь
            self._vocabulary = {word: idx for idx, word in enumerate(model.index_to_key)}
            
            embeddings = model.vectors
            logger.info(f"Loaded Word2Vec binary embeddings: {embeddings.shape}")
            
            return embeddings
            
        except ImportError:
            raise ImportError("gensim is required for loading binary Word2Vec files. "
                            "Install with: pip install gensim")
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """Получение словаря Word2Vec."""
        if not hasattr(self, '_vocabulary'):
            # Загружаем эмбединги, чтобы построить словарь
            self.load(path)
        return self._vocabulary


class GloVeHandler(TextFormatHandler):
    """Обработчик GloVe формата (.txt)."""
    
    def load(self, path: str) -> np.ndarray:
        """
        Загрузка GloVe эмбедингов.
        
        Args:
            path: Путь к .txt файлу
            
        Returns:
            numpy.ndarray: Матрица эмбедингов
        """
        embeddings = []
        vocab = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token, vector = self._parse_text_line(line)
                if token is not None and vector is not None:
                    vocab[token] = idx
                    embeddings.append(vector)
        
        if not embeddings:
            raise ValueError("No valid embeddings found in GloVe file")
        
        self._vocabulary = vocab
        embeddings_array = np.vstack(embeddings)
        logger.info(f"Loaded GloVe embeddings: {embeddings_array.shape}")
        
        return embeddings_array
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """Получение словаря GloVe."""
        if not hasattr(self, '_vocabulary'):
            self.load(path)
        return self._vocabulary


class BertHandler(FormatHandler):
    """Обработчик BERT эмбедингов (.pt, .pkl)."""
    
    def load(self, path: str) -> Union[Tensor, np.ndarray]:
        """
        Загрузка BERT эмбедингов.
        
        Args:
            path: Путь к файлу (.pt или .pkl)
            
        Returns:
            torch.Tensor или numpy.ndarray: Эмбединги
        """
        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext == '.pt':
            return self._load_pytorch(path)
        elif file_ext == '.pkl':
            return self._load_pickle(path)
        else:
            raise ValueError(f"Unsupported BERT file extension: {file_ext}")
    
    def _load_pytorch(self, path: str) -> Tensor:
        """Загрузка из PyTorch формата."""
        try:
            embeddings = torch.load(path, map_location='cpu')
            logger.info(f"Loaded BERT PyTorch embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch embeddings: {e}")
    
    def _load_pickle(self, path: str) -> np.ndarray:
        """Загрузка из pickle формата."""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Данные могут быть в разных форматах
            if isinstance(data, dict):
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                elif 'vectors' in data:
                    embeddings = data['vectors']
                else:
                    # Берем первое tensor/array значение
                    for key, value in data.items():
                        if isinstance(value, (np.ndarray, torch.Tensor)):
                            embeddings = value
                            break
                    else:
                        raise ValueError("No embeddings found in pickle file")
            else:
                embeddings = data
            
            # Конвертируем в numpy если нужно
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            
            logger.info(f"Loaded BERT pickle embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise ValueError(f"Failed to load pickle embeddings: {e}")
    
    def get_vocabulary(self, path: str) -> Dict[str, int]:
        """
        Получение словаря BERT.
        
        Note: BERT обычно использует словарь токенайзера,
        который не всегда доступен в файле эмбедингов.
        """
        logger.warning("BERT vocabulary not available from embedding file. "
                      "Use tokenizer vocabulary instead.")
        return {} 