#!/usr/bin/env python3
"""
Connection Types - типы и структуры данных для классификации связей
================================================================

Базовые типы данных для системы классификации связей между клетками.
Вынесены в отдельный модуль для лучшей организации кода.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List, Any
from ...config import get_project_config


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
    strength: float = None  # Значение по умолчанию берётся из централизованного конфига
    functional_similarity: Optional[float] = (
        None  # Значение по умолчанию берётся из централизованного конфига
    )

    def __post_init__(self):
        cfg = get_project_config().connection
        if self.strength is None:
            self.strength = cfg.strength
        if self.functional_similarity is None:
            self.functional_similarity = cfg.functional_similarity
