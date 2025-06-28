#!/usr/bin/env python3
"""
Модули для обучения 3D куба
===========================

Компоненты, необходимые только в процессе обучения:
- Тренеры и оптимизаторы
- Loss функции и метрики
- Валидация и мониторинг
- Загрузчики данных
"""

from .embedding_trainer import EmbeddingTrainer
from .loss_functions import EmbeddingLosses
from .validation import SemanticValidator, ProbingValidator
from .data_loaders import EmbeddingDataLoader

__all__ = [
    "EmbeddingTrainer",
    "EmbeddingLosses", 
    "SemanticValidator",
    "ProbingValidator",
    "EmbeddingDataLoader"
]