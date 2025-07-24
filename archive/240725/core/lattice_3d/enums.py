"""
Модуль перечислений для 3D решетки
==================================

Содержит все основные перечисления (Enums), используемые в модуле
3D решетки, для централизованного управления константами.
"""

from enum import Enum


class BoundaryCondition(Enum):
    """Типы граничных условий для решетки"""

    WALLS = "walls"  # Границы блокируют сигналы
    PERIODIC = "periodic"  # Решетка замыкается в тор
    ABSORBING = "absorbing"  # Сигналы затухают на границах
    REFLECTING = "reflecting"  # Сигналы отражаются от границ


class Face(Enum):
    """Грани решетки для ввода/вывода"""

    FRONT = "front"  # Z = 0
    BACK = "back"  # Z = max
    LEFT = "left"  # X = 0
    RIGHT = "right"  # X = max
    TOP = "top"  # Y = max
    BOTTOM = "bottom"  # Y = 0


class PlacementStrategy(Enum):
    """Стратегии размещения точек ввода/вывода"""

    PROPORTIONAL = "proportional"  # Пропорциональное автоматическое масштабирование
    RANDOM = "random"  # Случайное размещение
    CORNERS = "corners"  # Размещение в углах
    CORNERS_CENTER = "corners_center"  # Углы + центр
    FULL_FACE = "full_face"  # Полное покрытие грани (текущая реализация)


class NeighborStrategy(Enum):
    """Стратегии поиска соседей"""

    LOCAL = "local"
    RANDOM_SAMPLE = "random_sample"
    HYBRID = "hybrid"
    TIERED = "tiered"
