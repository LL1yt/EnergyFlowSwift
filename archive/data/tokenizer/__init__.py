"""
Tokenizer Module for 3D Cellular Neural Network

Модуль для конвертации между текстом и токенами, интеграция с популярными токенайзерами.

Основные компоненты:
- TokenizerManager: Основной класс управления токенайзерами
- TokenizerAdapters: Адаптеры для различных типов токенайзеров
- TextProcessor: Предобработка текста

Автор: 3D CNN Team
Дата: Декабрь 2025
"""

from .tokenizer import TokenizerManager
from .text_processor import TextProcessor

__version__ = "1.0.0"
__author__ = "3D CNN Team"

# Основные экспорты модуля
__all__ = [
    'TokenizerManager',
    'TextProcessor',
]

# Метаданные модуля
SUPPORTED_TOKENIZERS = [
    'bert-base-uncased',
    'gpt2',
    'sentencepiece',
    'basic'
]

DEFAULT_CONFIG = {
    'tokenizer_type': 'bert-base-uncased',
    'max_length': 512,
    'padding': True,
    'truncation': True,
    'add_special_tokens': True
} 