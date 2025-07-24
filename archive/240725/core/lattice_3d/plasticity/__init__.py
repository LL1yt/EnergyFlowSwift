"""
Модуль пластичности для 3D Решетки
==================================

Содержит механизмы синаптической пластичности:
- AdaptiveThreshold: BCM правило и адаптивные пороги активности
- STDPMechanism: классический и BCM-enhanced STDP
- CompetitiveLearning: Winner-Take-All и латеральное торможение
- PlasticityMixin: объединяющий mixin класс

Архитектура построена модульно для легкости тестирования и расширения.
"""

from .adaptive_threshold import AdaptiveThreshold
from .stdp import STDPMechanism
from .competitive_learning import CompetitiveLearning
from .plasticity_mixin import PlasticityMixin

__all__ = [
    "AdaptiveThreshold",
    "STDPMechanism",
    "CompetitiveLearning",
    "PlasticityMixin",
]
