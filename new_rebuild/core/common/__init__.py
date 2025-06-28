#!/usr/bin/env python3
"""
Общие компоненты для обучения и инференса
=========================================

Модули, используемые как в процессе обучения, так и в продуктивной работе куба.
Включает преобразование эмбедингов и базовые интерфейсы.
"""

from .embedding_transformer import EmbeddingTransformer, HierarchicalEmbeddingTransformer
from .interfaces import CubeInterface, EmbeddingProcessor

__all__ = [
    "EmbeddingTransformer", 
    "HierarchicalEmbeddingTransformer",
    "CubeInterface",
    "EmbeddingProcessor"
]