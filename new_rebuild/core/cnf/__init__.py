"""
Lightweight CNF Module for 3D Cellular Neural Network
===================================================

Continuous Normalizing Flows для functional и distant connections.
Заменяет сложную пластичность для дальних связей на эффективную
continuous dynamics с минимальными вычислениями.

КОМПОНЕНТЫ:
- LightweightCNF: основной CNF класс с Neural ODE
- EulerSolver: 3-step Euler integration (7x быстрее RK4)
- ConnectionClassifier: классификация связей по типам
- HybridConnectionProcessor: MoE архитектура для разных связей

ИСПОЛЬЗОВАНИЕ:
    from new_rebuild.core.cnf import LightweightCNF, HybridConnectionProcessor

    cnf_processor = HybridConnectionProcessor(config)
    new_states = cnf_processor(states, neighbor_classification)
"""

from .lightweight_cnf import LightweightCNF, ConnectionType
from .euler_solver import EulerSolver
from .connection_classifier import ConnectionClassifier, ConnectionCategory
from .hybrid_connection_processor import HybridConnectionProcessor

__all__ = [
    "LightweightCNF",
    "ConnectionType",
    "EulerSolver",
    "ConnectionClassifier",
    "ConnectionCategory",
    "HybridConnectionProcessor",
]
