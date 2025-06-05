"""
Data Embedding Loader Module

Модуль для загрузки и предобработки векторных представлений (эмбедингов) различных типов.
Поддерживает популярные форматы: Word2Vec, GloVe, BERT embeddings.
"""

from .embedding_loader import EmbeddingLoader
from .format_handlers import (
    FormatHandler,
    Word2VecHandler, 
    GloVeHandler,
    BertHandler
)
from .preprocessing import EmbeddingPreprocessor

__version__ = "1.0.0"
__author__ = "3D CNN Team"

__all__ = [
    "EmbeddingLoader",
    "FormatHandler", 
    "Word2VecHandler",
    "GloVeHandler", 
    "BertHandler",
    "EmbeddingPreprocessor"
] 