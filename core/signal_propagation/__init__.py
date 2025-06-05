"""
Signal Propagation Module

Этот модуль управляет временной динамикой сигналов в 3D решетке клеток.

Основные компоненты:
- TimeManager: Управление временными шагами
- SignalPropagator: Распространение сигналов через решетку
- PropagationPatterns: Анализ паттернов распространения
- ConvergenceDetector: Детекция сходимости состояний
"""

from .time_manager import TimeManager, TimeConfig
from .signal_propagator import SignalPropagator, PropagationConfig
from .propagation_patterns import PropagationPatterns, PatternAnalyzer
from .convergence_detector import ConvergenceDetector, ConvergenceConfig

__all__ = [
    # Основные классы
    'TimeManager',
    'SignalPropagator', 
    'PropagationPatterns',
    'ConvergenceDetector',
    
    # Конфигурации
    'TimeConfig',
    'PropagationConfig',
    'ConvergenceConfig',
    'PatternAnalyzer',
] 