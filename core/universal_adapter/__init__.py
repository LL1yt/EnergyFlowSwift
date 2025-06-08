"""
Universal Adapter module для работы с различными embedding моделями.

Этот модуль предоставляет интерфейс для конвертации эмбедингов
между различными моделями и размерностями.
"""

# Re-export компонентов из правильного места
from data.embedding_adapter.universal_adapter import (
    UniversalEmbeddingAdapter,
    AdapterManager,
    KNOWN_MODELS,
    create_adapter_for_cube
)

__all__ = [
    'UniversalEmbeddingAdapter',
    'AdapterManager', 
    'KNOWN_MODELS',
    'create_adapter_for_cube'
] 