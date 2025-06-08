"""
🔧 Embedding Adapter Module
Универсальная система конвертации эмбедингов между любыми размерностями
"""

from .universal_adapter import (
    UniversalEmbeddingAdapter,
    AdapterManager,
    KNOWN_MODELS,
    create_adapter_for_cube
)

__version__ = "1.0.0"
__author__ = "3D Cellular Neural Network Project"

__all__ = [
    "UniversalEmbeddingAdapter",
    "AdapterManager", 
    "KNOWN_MODELS",
    "create_adapter_for_cube"
] 