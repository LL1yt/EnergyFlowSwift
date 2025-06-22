"""
Модуль пластичности для 3D Решетки - Clean Implementation

⚠️ DEPRECATED: Plasticity Module

Этот модуль заменен на CNF (Continuous Normalizing Flows) подход.
Сохранен для справки и тестирования.

Используйте CNF Integration из Phase 3.6+ для адаптивности.


Переносит механизмы синаптической пластичности из Legacy проекта
с адаптацией под новую архитектуру:

- AdaptiveThreshold: BCM правило и адаптивные пороги активности
- STDPMechanism: классический и BCM-enhanced STDP
- CompetitiveLearning: Winner-Take-All и латеральное торможение
- PlasticityManager: объединяющий управляющий класс для HybridCellV2

Биологические принципы:
- Гомеостатическая пластичность для стабильности
- Spike-timing dependent plasticity для обучения
- Конкурентное обучение для специализации
- Метапластичность (BCM) для адаптивности
"""

from .adaptive_threshold import AdaptiveThreshold
from .stdp import STDPMechanism
from .competitive_learning import CompetitiveLearning
from .plasticity_manager import PlasticityManager

__all__ = [
    "AdaptiveThreshold",
    "STDPMechanism",
    "CompetitiveLearning",
    "PlasticityManager",
]
