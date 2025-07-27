"""
Провайдеры данных для dataset модуля
====================================
"""

from .base_provider import BaseDataProvider
from .teacher_model import TeacherModelProvider, create_teacher_model_provider
from .snli_provider import SNLIProvider, create_snli_provider
from .precomputed_provider import PrecomputedProvider, create_precomputed_provider

__all__ = [
    'BaseDataProvider',
    'TeacherModelProvider', 
    'SNLIProvider',
    'PrecomputedProvider',
    'create_teacher_model_provider',
    'create_snli_provider', 
    'create_precomputed_provider'
]