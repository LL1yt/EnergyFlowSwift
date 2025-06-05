"""
Модуль загрузки и предобработки эмбедингов.

Поддерживает:
- Традиционные форматы: Word2Vec, GloVe, BERT
- LLM модели для Knowledge Distillation: LLaMA, Mistral, GPT и др.
- Real-time генерацию эмбедингов из текстов
- Полный pipeline для обучения 3D CNN через knowledge distillation
"""

from .embedding_loader import EmbeddingLoader
from .format_handlers import (
    FormatHandler,
    Word2VecHandler, 
    GloVeHandler, 
    BertHandler,
    LLMHandler,
    create_llm_handler,
    SUPPORTED_LLM_MODELS
)
from .preprocessing import EmbeddingPreprocessor

__all__ = [
    # Основные классы
    'EmbeddingLoader',
    'EmbeddingPreprocessor',
    
    # Format handlers (традиционные)
    'FormatHandler',
    'Word2VecHandler',
    'GloVeHandler', 
    'BertHandler',
    
    # LLM & Knowledge Distillation
    'LLMHandler',
    'create_llm_handler',
    'SUPPORTED_LLM_MODELS',
]

# Версия модуля
__version__ = "2.0.0"  # Обновлена для поддержки LLM

# Метаданные для knowledge distillation
KNOWLEDGE_DISTILLATION_READY = True
SUPPORTED_TEACHER_MODELS = list(SUPPORTED_LLM_MODELS.keys())
PHASE_3_INTEGRATION_READY = True 