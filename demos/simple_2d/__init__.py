"""
Simple 2D Demo - Демонстрация концепции клеточной нейронной сети

Этот модуль предоставляет наглядную демонстрацию работы клеточной нейронной сети
на простом 2D примере для лучшего понимания концепции.
"""

from .simple_2d_demo import (
    Simple2DLattice,
    PatternGenerator, 
    Demo2DVisualizer,
    run_wave_demo,
    run_pulse_demo,
    run_interference_demo,
    main
)

__all__ = [
    'Simple2DLattice',
    'PatternGenerator',
    'Demo2DVisualizer', 
    'run_wave_demo',
    'run_pulse_demo',
    'run_interference_demo',
    'main'
]

__version__ = "1.0.0" 