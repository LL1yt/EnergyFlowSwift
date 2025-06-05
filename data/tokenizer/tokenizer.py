"""
TokenizerManager - 3D Cellular Neural Network

Центральный класс для управления различными типами токенайзеров.
Обеспечивает единый интерфейс для токенизации текста и интеграции
с системой эмбедингов и 3D решеткой.

Автор: 3D CNN Team
Дата: Декабрь 2025
"""

import os
import logging
import yaml
from typing import List, Dict, Optional, Union, Any, Tuple
import torch
import numpy as np
from pathlib import Path

# Импорты для различных токенайзеров
try:
    from transformers import (
        AutoTokenizer, 
        BertTokenizer, 
        GPT2Tokenizer,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedTokenizer = None

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

from .text_processor import TextProcessor
from .tokenizer_adapters import (
    TokenizerAdapter, 
    BertTokenizerAdapter,
    GPTTokenizerAdapter,
    SentencePieceAdapter,
    BasicTokenizerAdapter
)


class TokenizerManager:
    """
    Центральный менеджер для всех типов токенайзеров.
    
    Поддерживает:
    - BERT (bert-base-uncased, bert-base-cased)
    - GPT-2 (gpt2, gpt2-medium)
    - SentencePiece (custom models)
    - Basic (whitespace tokenization)
    """
    
    def __init__(self, 
                 tokenizer_type: str = 'bert-base-uncased',
                 config: Optional[Dict] = None,
                 config_path: Optional[str] = None):
        """
        Инициализация TokenizerManager.
        
        Args:
            tokenizer_type: Тип токенайзера ('bert-base-uncased', 'gpt2', etc.)
            config: Словарь конфигурации (опционально)
            config_path: Путь к файлу конфигурации YAML (опционально)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing TokenizerManager with type: {tokenizer_type}")
        
        # Загрузка конфигурации
        self.config = self._load_config(config, config_path)
        
        # Установка типа токенайзера
        self.tokenizer_type = tokenizer_type or self.config.get('tokenizer', {}).get('type', 'bert-base-uncased')
        
        # Инициализация компонентов
        self.text_processor = TextProcessor(self.config.get('text_processing', {}))
        self.adapter: Optional[TokenizerAdapter] = None
        
        # Кэш токенизации
        self._token_cache: Dict[str, List[int]] = {}
        self._cache_enabled = self.config.get('caching', {}).get('enabled', True)
        self._cache_max_size = self.config.get('caching', {}).get('max_size', 10000)
        
        # Метрики производительности
        self._metrics = {
            'total_tokenizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens': 0,
            'avg_tokens_per_sec': 0.0
        }
        
        # Инициализация токенайзера
        self._initialize_tokenizer()
    
    def _load_config(self, config: Optional[Dict], config_path: Optional[str]) -> Dict:
        """Загрузка конфигурации из файла или словаря."""
        if config:
            return config
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Попытка загрузить конфигурацию по умолчанию
        default_config_path = Path(__file__).parent / 'config' / 'tokenizer_config.yaml'
        if default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Возврат базовой конфигурации
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Получение базовой конфигурации по умолчанию."""
        return {
            'tokenizer': {
                'type': 'bert-base-uncased',
                'max_length': 512,
                'padding': True,
                'truncation': True,
                'add_special_tokens': True
            },
            'text_processing': {
                'lowercase': True,
                'strip_whitespace': True,
                'normalize_unicode': True
            },
            'caching': {
                'enabled': True,
                'max_size': 10000,
                'ttl': 3600
            },
            'batch_processing': {
                'batch_size': 32,
                'num_workers': 4,
                'show_progress': True
            }
        }
    
    def _initialize_tokenizer(self) -> None:
        """Инициализация адаптера токенайзера."""
        try:
            self.logger.info(f"Initializing tokenizer adapter for: {self.tokenizer_type}")
            
            # Фабрика адаптеров
            if self.tokenizer_type.startswith('bert'):
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers library not available for BERT tokenizer")
                self.adapter = BertTokenizerAdapter(self.tokenizer_type, self.config)
                
            elif self.tokenizer_type.startswith('gpt'):
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers library not available for GPT tokenizer")
                self.adapter = GPTTokenizerAdapter(self.tokenizer_type, self.config)
                
            elif self.tokenizer_type == 'sentencepiece':
                if not SENTENCEPIECE_AVAILABLE:
                    raise ImportError("sentencepiece library not available")
                self.adapter = SentencePieceAdapter(self.tokenizer_type, self.config)
                
            elif self.tokenizer_type == 'basic':
                self.adapter = BasicTokenizerAdapter(self.tokenizer_type, self.config)
                
            else:
                raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
            
            self.logger.info(f"Successfully initialized {self.tokenizer_type} tokenizer")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer {self.tokenizer_type}: {str(e)}")
            
            # Fallback к базовому токенайзеру
            fallback_type = self.config.get('error_handling', {}).get('fallback_tokenizer', 'basic')
            if self.tokenizer_type != fallback_type:
                self.logger.info(f"Falling back to {fallback_type} tokenizer")
                self.tokenizer_type = fallback_type
                self.adapter = BasicTokenizerAdapter(fallback_type, self.config)
            else:
                raise RuntimeError(f"Failed to initialize tokenizer and fallback: {str(e)}")
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Кодирование текста в список токен ID.
        
        Args:
            text: Входной текст для токенизации
            **kwargs: Дополнительные параметры (max_length, padding, etc.)
            
        Returns:
            Список ID токенов
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        # Проверка кэша
        cache_key = f"{text}_{hash(str(sorted(kwargs.items())))}"
        if self._cache_enabled and cache_key in self._token_cache:
            self._metrics['cache_hits'] += 1
            return self._token_cache[cache_key]
        
        # Предобработка текста
        processed_text = self.text_processor.preprocess(text)
        
        # Токенизация через адаптер
        try:
            token_ids = self.adapter.encode(processed_text, **kwargs)
            
            # Обновление метрик
            self._metrics['total_tokenizations'] += 1
            self._metrics['total_tokens'] += len(token_ids)
            self._metrics['cache_misses'] += 1
            
            # Кэширование результата
            if self._cache_enabled:
                if len(self._token_cache) >= self._cache_max_size:
                    # Простое LRU - удаляем первый элемент
                    self._token_cache.pop(next(iter(self._token_cache)))
                self._token_cache[cache_key] = token_ids
            
            return token_ids
            
        except Exception as e:
            self.logger.error(f"Tokenization failed for text: {text[:50]}... Error: {str(e)}")
            raise
    
    def decode(self, tokens: List[int]) -> str:
        """
        Декодирование списка токен ID обратно в текст.
        
        Args:
            tokens: Список ID токенов
            
        Returns:
            Декодированный текст
        """
        if not tokens or not isinstance(tokens, list):
            raise ValueError("Tokens must be a non-empty list")
        
        try:
            return self.adapter.decode(tokens)
        except Exception as e:
            self.logger.error(f"Decoding failed for tokens: {tokens[:10]}... Error: {str(e)}")
            raise
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста в список строковых токенов.
        
        Args:
            text: Входной текст
            
        Returns:
            Список строковых токенов
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        processed_text = self.text_processor.preprocess(text)
        
        try:
            return self.adapter.tokenize(processed_text)
        except Exception as e:
            self.logger.error(f"Tokenization failed for text: {text[:50]}... Error: {str(e)}")
            raise
    
    def batch_encode(self, texts: List[str], **kwargs) -> List[List[int]]:
        """
        Batch кодирование списка текстов.
        
        Args:
            texts: Список входных текстов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список списков ID токенов
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")
        
        # Простая реализация - последовательная обработка
        # TODO: Добавить параллельную обработку
        results = []
        for text in texts:
            try:
                encoded = self.encode(text, **kwargs)
                results.append(encoded)
            except Exception as e:
                self.logger.warning(f"Failed to encode text in batch: {str(e)}")
                results.append([])  # Пустой результат для проблемных текстов
        
        return results
    
    def batch_decode(self, token_lists: List[List[int]]) -> List[str]:
        """
        Batch декодирование списка списков токенов.
        
        Args:
            token_lists: Список списков ID токенов
            
        Returns:
            Список декодированных текстов
        """
        if not token_lists or not isinstance(token_lists, list):
            raise ValueError("Token lists must be a non-empty list")
        
        results = []
        for tokens in token_lists:
            try:
                decoded = self.decode(tokens)
                results.append(decoded)
            except Exception as e:
                self.logger.warning(f"Failed to decode tokens in batch: {str(e)}")
                results.append("")  # Пустая строка для проблемных токенов
        
        return results
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря токенайзера."""
        if hasattr(self.adapter, 'get_vocab_size'):
            return self.adapter.get_vocab_size()
        return -1
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов и их ID."""
        if hasattr(self.adapter, 'get_special_tokens'):
            return self.adapter.get_special_tokens()
        return {}
    
    def is_available(self) -> bool:
        """Проверка доступности токенайзера."""
        return self.adapter is not None
    
    def prepare_for_lattice(self, text: str, lattice_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Подготовка токенизированного текста для входной грани 3D решетки.
        
        Args:
            text: Входной текст
            lattice_size: Размер решетки (x, y, z)
            
        Returns:
            PyTorch тензор готовый для подачи на решетку
        """
        # Рассчитываем размер входной грани (например, x*y для грани z=0)
        input_face_size = lattice_size[0] * lattice_size[1]
        
        # Токенизируем с ограничением длины
        tokens = self.encode(text, max_length=input_face_size, padding='max_length', truncation=True)
        
        # Преобразуем в тензор и изменяем форму под грань решетки
        tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Обрезаем или дополняем до нужного размера
        if len(tensor) > input_face_size:
            tensor = tensor[:input_face_size]
        elif len(tensor) < input_face_size:
            pad_size = input_face_size - len(tensor)
            pad_token_id = self.get_special_tokens().get('pad_token', 0)
            padding = torch.full((pad_size,), pad_token_id, dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        
        # Изменяем форму под входную грань решетки
        return tensor.view(lattice_size[0], lattice_size[1])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        metrics = self._metrics.copy()
        
        # Рассчитываем дополнительные метрики
        if metrics['total_tokenizations'] > 0:
            cache_hit_rate = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
            metrics['cache_hit_rate'] = cache_hit_rate
            
            avg_tokens_per_text = metrics['total_tokens'] / metrics['total_tokenizations']
            metrics['avg_tokens_per_text'] = avg_tokens_per_text
        
        metrics['cache_size'] = len(self._token_cache)
        metrics['tokenizer_type'] = self.tokenizer_type
        
        return metrics
    
    def clear_cache(self) -> None:
        """Очистка кэша токенизации."""
        self._token_cache.clear()
        self.logger.info("Token cache cleared")
    
    def __repr__(self) -> str:
        return f"TokenizerManager(type='{self.tokenizer_type}', available={self.is_available()})"
    
    def __str__(self) -> str:
        metrics = self.get_metrics()
        return (f"TokenizerManager:\n"
                f"  Type: {self.tokenizer_type}\n"
                f"  Available: {self.is_available()}\n"
                f"  Vocab Size: {self.get_vocab_size()}\n"
                f"  Total Tokenizations: {metrics['total_tokenizations']}\n"
                f"  Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.2%}") 