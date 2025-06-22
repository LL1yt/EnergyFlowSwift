#!/usr/bin/env python3
"""
Connection Classifier - классификация связей по типам
=====================================================

Классифицирует соседские связи на три типа:
- Local (10%): близкие соседи → GNN обработка
- Functional (60%): средние расстояния → CNF эволюция
- Distant (30%): дальние связи → CNF эволюция

ПРИНЦИПЫ КЛАССИФИКАЦИИ:
1. Расстояние в 3D пространстве (Euclidean + Manhattan)
2. Функциональная близость (similarity состояний)
3. Топологические особенности решетки
4. Адаптивная классификация на основе активности
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionCategory(Enum):
    """Категории связей"""

    LOCAL = "local"  # 10% - ближайшие соседи
    FUNCTIONAL = "functional"  # 60% - функциональные связи
    DISTANT = "distant"  # 30% - дальние связи


@dataclass
class ConnectionInfo:
    """Информация о связи между клетками"""

    source_idx: int
    target_idx: int
    euclidean_distance: float
    manhattan_distance: float
    category: ConnectionCategory
    strength: float = 1.0  # Сила связи (может модулироваться STDP)


class DistanceCalculator:
    """Вычисление расстояний в 3D решетке"""

    def __init__(self, lattice_dimensions: Tuple[int, int, int]):
        self.width, self.height, self.depth = lattice_dimensions
        self.total_cells = self.width * self.height * self.depth

    def linear_to_3d(self, linear_idx: int) -> Tuple[int, int, int]:
        """Преобразование линейного индекса в 3D координаты"""
        z = linear_idx // (self.width * self.height)
        remainder = linear_idx % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width
        return x, y, z

    def euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Евклидово расстояние между клетками"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)

        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

    def manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Манхэттенское расстояние между клетками"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)

        return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)


class FunctionalSimilarity(nn.Module):
    """
    Вычисление функциональной близости между клетками

    Основано на similarity состояний, активности и паттернах взаимодействия
    """

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

        # Learnable weights для различных аспектов similarity
        self.cosine_weight = nn.Parameter(torch.tensor(0.5))
        self.euclidean_weight = nn.Parameter(torch.tensor(0.3))
        self.correlation_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        """
        Вычисление функциональной близости между состояниями

        Args:
            state1, state2: [batch, state_size] - состояния клеток

        Returns:
            similarity: [batch] - функциональная близость [0, 1]
        """
        # Cosine similarity
        cosine_sim = F.cosine_similarity(state1, state2, dim=-1)
        cosine_sim = (cosine_sim + 1) / 2  # Нормализация к [0, 1]

        # Euclidean similarity (обратное к расстоянию)
        euclidean_dist = torch.norm(state1 - state2, dim=-1)
        max_dist = torch.norm(torch.ones_like(state1), dim=-1)
        euclidean_sim = 1.0 - (euclidean_dist / max_dist)

        # Correlation similarity (упрощенная версия)
        # В реальной версии можно использовать более сложные метрики
        correlation_sim = torch.sigmoid(torch.sum(state1 * state2, dim=-1))

        # Взвешенная комбинация
        total_similarity = (
            self.cosine_weight * cosine_sim
            + self.euclidean_weight * euclidean_sim
            + self.correlation_weight * correlation_sim
        )

        return torch.clamp(total_similarity, 0, 1)


class ConnectionClassifier(nn.Module):
    """
    Классификатор связей на local/functional/distant категории

    АЛГОРИТМ:
    1. Вычисляем пространственные расстояния
    2. Анализируем функциональную близость
    3. Применяем пороговые значения
    4. Балансируем пропорции (10%/60%/30%)
    """

    def __init__(
        self,
        lattice_dimensions: Tuple[int, int, int],
        state_size: int,
        neighbor_strategy_config: Optional[Dict] = None,
    ):
        super().__init__()

        config = get_project_config()

        self.lattice_dimensions = lattice_dimensions
        self.state_size = state_size

        # Конфигурация пропорций
        self.strategy_config = (
            neighbor_strategy_config or config.neighbor_strategy_config
        )
        self.local_ratio = self.strategy_config["local_tier"]  # 0.1
        self.functional_ratio = self.strategy_config["functional_tier"]  # 0.6
        self.distant_ratio = self.strategy_config["distant_tier"]  # 0.3

        # Калькуляторы
        self.distance_calc = DistanceCalculator(lattice_dimensions)
        self.functional_similarity = FunctionalSimilarity(state_size)

        # Пороговые значения для расстояний (learnable)
        self.local_distance_threshold = nn.Parameter(torch.tensor(2.0))
        self.distant_distance_threshold = nn.Parameter(torch.tensor(8.0))

        # Пороги для функциональной близости
        self.functional_similarity_threshold = nn.Parameter(torch.tensor(0.3))

        logger.info(
            f"ConnectionClassifier initialized: "
            f"lattice={lattice_dimensions}, "
            f"ratios=({self.local_ratio:.1f}, {self.functional_ratio:.1f}, {self.distant_ratio:.1f})"
        )

    def _classify_by_distance(
        self, cell_idx: int, neighbor_indices: List[int]
    ) -> Dict[str, List[int]]:
        """
        Первичная классификация по пространственному расстоянию
        """
        classifications = {"local": [], "functional": [], "distant": []}

        for neighbor_idx in neighbor_indices:
            euclidean_dist = self.distance_calc.euclidean_distance(
                cell_idx, neighbor_idx
            )

            if euclidean_dist <= self.local_distance_threshold:
                classifications["local"].append(neighbor_idx)
            elif euclidean_dist >= self.distant_distance_threshold:
                classifications["distant"].append(neighbor_idx)
            else:
                classifications["functional"].append(neighbor_idx)

        return classifications

    def _refine_by_functionality(
        self,
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_indices: List[int],
        initial_classification: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """
        Уточнение классификации через функциональную близость
        """
        if len(neighbor_indices) == 0:
            return initial_classification

        # Вычисляем функциональную близость со всеми соседями
        similarities = self.functional_similarity(
            cell_state.unsqueeze(0).expand(len(neighbor_indices), -1), neighbor_states
        )

        refined_classification = {
            "local": initial_classification["local"].copy(),
            "functional": [],
            "distant": initial_classification["distant"].copy(),
        }

        # Перераспределяем functional connections на основе similarity
        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx in initial_classification["functional"]:
                similarity = similarities[i].item()

                if similarity >= self.functional_similarity_threshold:
                    refined_classification["functional"].append(neighbor_idx)
                else:
                    # Слабо связанные functional переводим в distant
                    refined_classification["distant"].append(neighbor_idx)

        return refined_classification

    def _balance_proportions(
        self, classification: Dict[str, List[int]], total_neighbors: int
    ) -> Dict[str, List[int]]:
        """
        Балансировка пропорций согласно целевым значениям
        """
        target_local = max(1, int(total_neighbors * self.local_ratio))
        target_functional = max(1, int(total_neighbors * self.functional_ratio))
        target_distant = max(1, int(total_neighbors * self.distant_ratio))

        # Начинаем с имеющихся классификаций
        balanced = {
            "local": classification["local"][:target_local],
            "functional": classification["functional"][:target_functional],
            "distant": classification["distant"][:target_distant],
        }

        # Собираем всех назначенных соседей
        assigned_neighbors = set()
        for neighbors in balanced.values():
            assigned_neighbors.update(neighbors)

        # Собираем всех доступных соседей
        all_available = set()
        for neighbors in classification.values():
            all_available.update(neighbors)

        # Найдем неназначенных соседей
        unassigned_neighbors = list(all_available - assigned_neighbors)

        if unassigned_neighbors:
            # Распределяем неназначенных в functional (основная категория)
            balanced["functional"].extend(unassigned_neighbors)

        return balanced

    def classify_connections(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """
        Основной метод классификации связей

        Args:
            cell_idx: индекс центральной клетки
            neighbor_indices: список индексов соседей
            cell_state: [state_size] - состояние центральной клетки
            neighbor_states: [num_neighbors, state_size] - состояния соседей

        Returns:
            classified_connections: словарь с классифицированными связями
        """
        if len(neighbor_indices) == 0:
            return {category: [] for category in ConnectionCategory}

        # Шаг 1: Классификация по расстоянию
        distance_classification = self._classify_by_distance(cell_idx, neighbor_indices)

        # Шаг 2: Уточнение через функциональную близость
        functional_classification = self._refine_by_functionality(
            cell_state, neighbor_states, neighbor_indices, distance_classification
        )

        # Шаг 3: Балансировка пропорций
        balanced_classification = self._balance_proportions(
            functional_classification, len(neighbor_indices)
        )

        # Шаг 4: Создание ConnectionInfo объектов
        classified_connections = {}

        for category_name, neighbor_list in balanced_classification.items():
            category = ConnectionCategory(category_name)
            connections = []

            for neighbor_idx in neighbor_list:
                euclidean_dist = self.distance_calc.euclidean_distance(
                    cell_idx, neighbor_idx
                )
                manhattan_dist = self.distance_calc.manhattan_distance(
                    cell_idx, neighbor_idx
                )

                connection_info = ConnectionInfo(
                    source_idx=cell_idx,
                    target_idx=neighbor_idx,
                    euclidean_distance=euclidean_dist,
                    manhattan_distance=manhattan_dist,
                    category=category,
                    strength=1.0,  # Может быть модулировано STDP
                )
                connections.append(connection_info)

            classified_connections[category] = connections

        return classified_connections

    def get_classification_stats(
        self, classified_connections: Dict[ConnectionCategory, List[ConnectionInfo]]
    ) -> Dict[str, Any]:
        """Статистика классификации"""
        stats = {}
        total_connections = sum(
            len(connections) for connections in classified_connections.values()
        )

        for category, connections in classified_connections.items():
            count = len(connections)
            ratio = count / max(total_connections, 1)

            if connections:
                avg_euclidean = sum(c.euclidean_distance for c in connections) / count
                avg_manhattan = sum(c.manhattan_distance for c in connections) / count
            else:
                avg_euclidean = avg_manhattan = 0.0

            stats[category.value] = {
                "count": count,
                "ratio": ratio,
                "avg_euclidean_distance": avg_euclidean,
                "avg_manhattan_distance": avg_manhattan,
            }

        stats["total_connections"] = total_connections
        stats["thresholds"] = {
            "local_distance": self.local_distance_threshold.item(),
            "distant_distance": self.distant_distance_threshold.item(),
            "functional_similarity": self.functional_similarity_threshold.item(),
        }

        return stats
