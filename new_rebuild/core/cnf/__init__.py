#!/usr/bin/env python3
"""
CNF (Continuous Normalizing Flow) Module для 3D CNN
==================================================

НОВАЯ АРХИТЕКТУРА: Только для Distant Expert в MoE
CNF используется исключительно в Distant Expert для долгосрочной памяти.

СТАТУС КОМПОНЕНТОВ:
- LightweightCNF: АКТИВНЫЙ (используется в Distant Expert)
- NeuralODE: АКТИВНЫЙ (часть LightweightCNF)
- EulerSolver: АКТИВНЫЙ (интеграция для CNF)
- ConnectionClassifier: DEPRECATED (заменен на MoE GatingNetwork)

MoE АРХИТЕКТУРА:
Новые MoE компоненты находятся в отдельном модуле core/moe/
Доступны через импорт: from new_rebuild.core.moe import ...

DEPRECATED:
- HybridConnectionProcessor: заменен на MoEConnectionProcessor
- ConnectionClassifier: заменен на GatingNetwork в MoE
"""

from .lightweight_cnf import LightweightCNF, NeuralODE, ConnectionType
from .euler_solver import EulerSolver

# DEPRECATED компоненты (для обратной совместимости)
try:
    from .connection_classifier import (
        ConnectionClassifier,
        ConnectionCategory,
    )  # DEPRECATED

    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False

# Основные активные компоненты
__all__ = [
    # Активные CNF компоненты для Distant Expert
    "LightweightCNF",
    "NeuralODE",
    "ConnectionType",
    "EulerSolver",
]

# Добавляем deprecated компоненты если доступны
if _CLASSIFIER_AVAILABLE:
    __all__.extend(["ConnectionClassifier", "ConnectionCategory"])  # DEPRECATED
