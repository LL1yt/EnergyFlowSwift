"""
Модуль Cell Prototype - Основа 3D Клеточной Нейронной Сети

Этот модуль содержит прототип "умной клетки" - базовый строительный блок
для построения 3D решетки нейронов.

Биологическая аналогия:
Каждая клетка как нейрон в коре головного мозга - получает сигналы от соседей,
обрабатывает их и передает дальше.
"""

from .main import CellPrototype, create_cell_from_config

__all__ = ['CellPrototype', 'create_cell_from_config']

__version__ = '0.1.0'
__author__ = 'Research Team'
__description__ = 'Cell prototype for 3D cellular neural network' 