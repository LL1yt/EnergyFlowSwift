"""
Модуль функциональной кластеризации для 3D решетки клеток.

Основные компоненты:
- BasicFunctionalClustering: базовая кластеризация по сходству состояний
- CoordinationInterface: интерфейс для координации кластеров
- ClusteringMixin: интеграция с Lattice3D

Архитектура готова к расширению:
- Пользовательское управление (будущее)
- Обученная координация (будущее)
- История решений для анализа
"""

from .basic_clustering import BasicFunctionalClustering
from .coordination_interface import CoordinationInterface
from .clustering_mixin import ClusteringMixin

__all__ = [
    "BasicFunctionalClustering",
    "CoordinationInterface",
    "ClusteringMixin",
]
