"""
TokenizerAdapters - 3D Cellular Neural Network

Адаптеры для различных типов токенайзеров.
Унифицируют интерфейс для работы с BERT, GPT, SentencePiece и базовыми токенайзерами.

Автор: 3D CNN Team
Дата: Декабрь 2025
"""

import os
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import string

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

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


class TokenizerAdapter(ABC):
    """
    Абстрактный базовый класс для всех адаптеров токенайзеров.
    Определяет единый интерфейс для токенизации.
    """
    
    def __init__(self, tokenizer_type: str, config: Dict):
        self.tokenizer_type = tokenizer_type
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
    
    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Кодирование текста в ID токенов."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов в текст."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста в строковые токены."""
        pass
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря."""
        return -1
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов."""
        return {}
    
    def is_available(self) -> bool:
        """Проверка доступности токенайзера."""
        return self.tokenizer is not None


class BertTokenizerAdapter(TokenizerAdapter):
    """
    Адаптер для BERT токенайзера.
    Поддерживает bert-base-uncased, bert-base-cased и другие BERT модели.
    """
    
    def __init__(self, tokenizer_type: str, config: Dict):
        super().__init__(tokenizer_type, config)
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Инициализация BERT токенайзера."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available for BERT tokenizer")
        
        try:
            # Получение конфигурации токенайзера
            tokenizer_config = self.config.get('tokenizer', {})
            cache_dir = tokenizer_config.get('cache_dir', './cache/tokenizers')
            
            # Создание директории кэша
            os.makedirs(cache_dir, exist_ok=True)
            
            # Загрузка токенайзера
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_type,
                cache_dir=cache_dir,
                local_files_only=tokenizer_config.get('local_files_only', False),
                trust_remote_code=tokenizer_config.get('trust_remote_code', False)
            )
            
            # Настройки токенайзера
            self.max_length = tokenizer_config.get('max_length', 512)
            self.padding = tokenizer_config.get('padding', True)
            self.truncation = tokenizer_config.get('truncation', True)
            self.add_special_tokens = tokenizer_config.get('add_special_tokens', True)
            
            self.logger.info(f"Successfully loaded BERT tokenizer: {self.tokenizer_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load BERT tokenizer: {str(e)}")
            raise
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Кодирование текста в ID токенов."""
        # Объединение параметров
        encode_params = {
            'max_length': kwargs.get('max_length', self.max_length),
            'padding': kwargs.get('padding', self.padding),
            'truncation': kwargs.get('truncation', self.truncation),
            'add_special_tokens': kwargs.get('add_special_tokens', self.add_special_tokens),
            'return_tensors': None  # Возвращаем список, а не тензоры
        }
        
        # Кодирование
        result = self.tokenizer(text, **encode_params)
        
        # Возврат списка ID токенов
        return result['input_ids']
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов в текст."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста в строковые токены."""
        return self.tokenizer.tokenize(text)
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов."""
        special_tokens = {}
        
        if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id:
            special_tokens['cls_token'] = self.tokenizer.cls_token_id
        if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id:
            special_tokens['sep_token'] = self.tokenizer.sep_token_id
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id:
            special_tokens['pad_token'] = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id:
            special_tokens['unk_token'] = self.tokenizer.unk_token_id
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id:
            special_tokens['mask_token'] = self.tokenizer.mask_token_id
        
        return special_tokens


class GPTTokenizerAdapter(TokenizerAdapter):
    """
    Адаптер для GPT токенайзера.
    Поддерживает gpt2, gpt2-medium и другие GPT модели.
    """
    
    def __init__(self, tokenizer_type: str, config: Dict):
        super().__init__(tokenizer_type, config)
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Инициализация GPT токенайзера."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available for GPT tokenizer")
        
        try:
            # Получение конфигурации токенайзера
            tokenizer_config = self.config.get('tokenizer', {})
            cache_dir = tokenizer_config.get('cache_dir', './cache/tokenizers')
            
            # Создание директории кэша
            os.makedirs(cache_dir, exist_ok=True)
            
            # Загрузка токенайзера
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_type,
                cache_dir=cache_dir,
                local_files_only=tokenizer_config.get('local_files_only', False),
                trust_remote_code=tokenizer_config.get('trust_remote_code', False)
            )
            
            # Установка pad_token для GPT (у него нет по умолчанию)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Настройки токенайзера
            self.max_length = tokenizer_config.get('max_length', 1024)  # GPT поддерживает больше
            self.padding = tokenizer_config.get('padding', True)
            self.truncation = tokenizer_config.get('truncation', True)
            self.add_special_tokens = tokenizer_config.get('add_special_tokens', True)
            
            self.logger.info(f"Successfully loaded GPT tokenizer: {self.tokenizer_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to load GPT tokenizer: {str(e)}")
            raise
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Кодирование текста в ID токенов."""
        # Объединение параметров
        encode_params = {
            'max_length': kwargs.get('max_length', self.max_length),
            'padding': kwargs.get('padding', self.padding),
            'truncation': kwargs.get('truncation', self.truncation),
            'add_special_tokens': kwargs.get('add_special_tokens', self.add_special_tokens),
            'return_tensors': None
        }
        
        # Кодирование
        result = self.tokenizer(text, **encode_params)
        
        return result['input_ids']
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов в текст."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста в строковые токены."""
        return self.tokenizer.tokenize(text)
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов."""
        special_tokens = {}
        
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id:
            special_tokens['bos_token'] = self.tokenizer.bos_token_id
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
            special_tokens['eos_token'] = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id:
            special_tokens['pad_token'] = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id:
            special_tokens['unk_token'] = self.tokenizer.unk_token_id
        
        return special_tokens


class SentencePieceAdapter(TokenizerAdapter):
    """
    Адаптер для SentencePiece токенайзера.
    Поддерживает кастомные SentencePiece модели.
    """
    
    def __init__(self, tokenizer_type: str, config: Dict):
        super().__init__(tokenizer_type, config)
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Инициализация SentencePiece токенайзера."""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("sentencepiece library not available")
        
        try:
            # Получение пути к модели из конфигурации
            supported_tokenizers = self.config.get('supported_tokenizers', {})
            tokenizer_info = supported_tokenizers.get(self.tokenizer_type, {})
            model_path = tokenizer_info.get('model_path', './models/sentencepiece.model')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
            
            # Инициализация токенайзера
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(model_path)
            
            # Настройки
            self.vocab_size = tokenizer_info.get('vocab_size', self.tokenizer.get_piece_size())
            
            self.logger.info(f"Successfully loaded SentencePiece tokenizer from: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SentencePiece tokenizer: {str(e)}")
            raise
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Кодирование текста в ID токенов."""
        # SentencePiece кодирование
        token_ids = self.tokenizer.encode_as_ids(text)
        
        # Обработка max_length
        max_length = kwargs.get('max_length')
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов в текст."""
        return self.tokenizer.decode_ids(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста в строковые токены."""
        return self.tokenizer.encode_as_pieces(text)
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря."""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов."""
        special_tokens = {}
        
        # SentencePiece обычно использует стандартные ID для специальных токенов
        # 0 - unknown, 1 - bos, 2 - eos
        special_tokens['unk_token'] = 0
        special_tokens['bos_token'] = 1
        special_tokens['eos_token'] = 2
        
        return special_tokens


class BasicTokenizerAdapter(TokenizerAdapter):
    """
    Адаптер для базового токенайзера.
    Простая токенизация по пробелам с обработкой пунктуации.
    Используется как fallback когда другие токенайзеры недоступны.
    """
    
    def __init__(self, tokenizer_type: str, config: Dict):
        super().__init__(tokenizer_type, config)
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Инициализация базового токенайзера."""
        # Получение настроек из конфигурации
        supported_tokenizers = self.config.get('supported_tokenizers', {})
        basic_config = supported_tokenizers.get('basic', {})
        
        self.case_sensitive = basic_config.get('case_sensitive', False)
        
        # Создание простого словаря для демонстрации
        self.vocab = {}
        self.vocab_size = 10000  # Заглушка
        
        # Простые специальные токены
        self.special_tokens = {
            'pad_token': 0,
            'unk_token': 1,
            'bos_token': 2,
            'eos_token': 3
        }
        
        # Компиляция паттернов
        self.word_pattern = re.compile(r'\b\w+\b')
        
        self.tokenizer = self  # Сам себе токенайзер
        
        self.logger.info("Initialized basic tokenizer")
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Кодирование текста в ID токенов."""
        tokens = self.tokenize(text)
        
        # Простое преобразование токенов в ID
        token_ids = []
        for token in tokens:
            # Простой хэш как ID (заглушка)
            token_id = abs(hash(token)) % (self.vocab_size - 10) + 10  # Оставляем место для специальных
            token_ids.append(token_id)
        
        # Обработка max_length
        max_length = kwargs.get('max_length')
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length and kwargs.get('padding'):
                pad_id = self.special_tokens['pad_token']
                token_ids.extend([pad_id] * (max_length - len(token_ids)))
        
        # Добавление специальных токенов
        if kwargs.get('add_special_tokens', True):
            bos_id = self.special_tokens['bos_token']
            eos_id = self.special_tokens['eos_token']
            token_ids = [bos_id] + token_ids + [eos_id]
        
        return token_ids
    
    def decode(self, tokens: List[int]) -> str:
        """Декодирование ID токенов в текст."""
        # Удаление специальных токенов
        special_ids = set(self.special_tokens.values())
        filtered_tokens = [t for t in tokens if t not in special_ids]
        
        # Простое декодирование (заглушка)
        # В реальной реализации нужен обратный словарь
        decoded_tokens = [f"token_{t}" for t in filtered_tokens]
        
        return ' '.join(decoded_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста в строковые токены."""
        if not self.case_sensitive:
            text = text.lower()
        
        # Извлечение слов с помощью регулярного выражения
        tokens = self.word_pattern.findall(text)
        
        return tokens
    
    def get_vocab_size(self) -> int:
        """Получение размера словаря."""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Получение специальных токенов."""
        return self.special_tokens.copy()


# Фабрика для создания адаптеров
def create_tokenizer_adapter(tokenizer_type: str, config: Dict) -> TokenizerAdapter:
    """
    Фабрика для создания подходящего адаптера токенайзера.
    
    Args:
        tokenizer_type: Тип токенайзера
        config: Конфигурация
        
    Returns:
        Экземпляр адаптера токенайзера
    """
    if tokenizer_type.startswith('bert'):
        return BertTokenizerAdapter(tokenizer_type, config)
    elif tokenizer_type.startswith('gpt'):
        return GPTTokenizerAdapter(tokenizer_type, config)
    elif tokenizer_type == 'sentencepiece':
        return SentencePieceAdapter(tokenizer_type, config)
    elif tokenizer_type == 'basic':
        return BasicTokenizerAdapter(tokenizer_type, config)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


# Экспорт всех классов
__all__ = [
    'TokenizerAdapter',
    'BertTokenizerAdapter', 
    'GPTTokenizerAdapter',
    'SentencePieceAdapter',
    'BasicTokenizerAdapter',
    'create_tokenizer_adapter'
] 