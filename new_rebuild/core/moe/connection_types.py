#!/usr/bin/env python3
"""
Connection Types - типы и структуры данных для классификации связей
================================================================

Базовые типы данных для системы классификации связей между клетками.
Вынесены в отдельный модуль для лучшей организации кода.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConnectionCategory(Enum):
    """Категории связей"""

    LOCAL = "local"  # 10% - ближайшие соседи
    FUNCTIONAL = "functional"  # 55% - функциональные связи
    DISTANT = "distant"  # 35% - дальние связи


@dataclass
class ConnectionInfo:
    """Информация о связи между клетками"""

    source_idx: int
    target_idx: int
    euclidean_distance: float
    manhattan_distance: float
    category: ConnectionCategory
    strength: float = 1.0  # Сила связи (может модулироваться STDP)
    functional_similarity: Optional[float] = None  # Функциональная близость
