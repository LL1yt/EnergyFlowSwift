"""
Universal Embedding Adapter wrapper для совместимости.

Этот модуль предоставляет wrapper для UniversalEmbeddingAdapter
из data.embedding_adapter для совместимости с кодом, который
ищет core.universal_adapter.universal_embedding_adapter.
"""

# Re-export всех компонентов из правильного места
from data.embedding_adapter.universal_adapter import *

# Explicit re-export основных классов для IDE support
from data.embedding_adapter.universal_adapter import (
    UniversalEmbeddingAdapter,
    AdapterManager,
    KNOWN_MODELS,
    create_adapter_for_cube
) 