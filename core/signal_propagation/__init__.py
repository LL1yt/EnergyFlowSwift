"""
Signal Propagation Module

Этот модуль управляет временной динамикой сигналов в 3D решетке клеток.

Основные компоненты:
- TimeManager: Управление временными шагами
- SignalPropagator: Распространение сигналов через решетку
- PropagationPatterns: Анализ паттернов распространения
- ConvergenceDetector: Детекция сходимости состояний
"""

from .time_manager import TimeManager, TimeConfig, TimeMode
from .signal_propagator import SignalPropagator, PropagationConfig, PropagationMode
from .propagation_patterns import PropagationPatterns, PatternAnalyzer
from .convergence_detector import ConvergenceDetector, ConvergenceConfig, ConvergenceMode

__all__ = [
    # Основные классы
    'TimeManager',
    'SignalPropagator', 
    'PropagationPatterns',
    'ConvergenceDetector',
    'PatternAnalyzer',
    
    # Конфигурации
    'TimeConfig',
    'PropagationConfig',
    'ConvergenceConfig',
    
    # Enums
    'TimeMode',
    'PropagationMode',
    'ConvergenceMode',
] 