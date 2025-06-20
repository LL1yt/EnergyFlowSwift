"""
Модуль пластичности для 3D Решетки (LEGACY IMPORT)
==================================================

ВНИМАНИЕ: Этот файл остается для обратной совместимости.
Все функциональность перенесена в модульный пакет plasticity/

Новая структура:
- plasticity/adaptive_threshold.py - BCM правило и адаптивные пороги
- plasticity/stdp.py - STDP механизм
- plasticity/competitive_learning.py - конкурентное обучение
- plasticity/plasticity_mixin.py - основной mixin класс

Для новых проектов используйте прямые импорты из подмодулей.
"""

# Legacy импорты для обратной совместимости
from .plasticity.adaptive_threshold import AdaptiveThreshold
from .plasticity.stdp import STDPMechanism
from .plasticity.competitive_learning import CompetitiveLearning
from .plasticity.plasticity_mixin import PlasticityMixin

# Экспорт всех классов
__all__ = [
    "AdaptiveThreshold",
    "STDPMechanism",
    "CompetitiveLearning",
    "PlasticityMixin",
]

# Показать предупреждение при импорте
import logging

logger = logging.getLogger(__name__)
logger.info(
    "REFACTORING NOTICE: plasticity.py has been split into modular components. "
    "Consider importing directly from core.lattice_3d.plasticity.* for better organization."
)
