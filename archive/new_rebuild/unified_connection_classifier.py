#!/usr/bin/env python3
"""
Unified Connection Classifier - новый классификатор связей без deprecated зависимостей
====================================================================================

Объединяет всю функциональность из deprecated ConnectionClassifier с улучшениями:
- Batch processing для производительности на больших решетках
- Централизованная конфигурация
- Оптимизированные вычисления расстояний
- Улучшенная функциональная близость

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. Три типа связей: Local (10%) / Functional (55%) / Distant (35%)
2. Batch классификация для эффективности
3. Learnable пороги для адаптивной классификации
4. Интеграция с spatial optimization
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


class DistanceCalculator:
    """Оптимизированный вычислитель расстояний в 3D решетке"""

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

    def batch_linear_to_3d(self, linear_indices: torch.Tensor) -> torch.Tensor:
        """Batch преобразование линейных индексов в 3D координаты

        Args:
            linear_indices: [batch] - линейные индексы

        Returns:
            coords_3d: [batch, 3] - 3D координаты (x, y, z)
        """
        z = linear_indices // (self.width * self.height)
        remainder = linear_indices % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width

        return torch.stack([x, y, z], dim=1).float()

    def euclidean_distance_batch(
        self, idx1: torch.Tensor, idx2: torch.Tensor
    ) -> torch.Tensor:
        """Batch вычисление евклидовых расстояний

        Args:
            idx1: [batch] - индексы первых клеток
            idx2: [batch] - индексы вторых клеток

        Returns:
            distances: [batch] - евклидовы расстояния
        """
        coords1 = self.batch_linear_to_3d(idx1)  # [batch, 3]
        coords2 = self.batch_linear_to_3d(idx2)  # [batch, 3]

        diff = coords1 - coords2  # [batch, 3]
        distances = torch.norm(diff, dim=1)  # [batch]

        return distances

    def manhattan_distance_batch(
        self, idx1: torch.Tensor, idx2: torch.Tensor
    ) -> torch.Tensor:
        """Batch вычисление манхэттенских расстояний"""
        coords1 = self.batch_linear_to_3d(idx1)  # [batch, 3]
        coords2 = self.batch_linear_to_3d(idx2)  # [batch, 3]

        diff = torch.abs(coords1 - coords2)  # [batch, 3]
        distances = torch.sum(diff, dim=1)  # [batch]

        return distances

    def euclidean_distance(self, idx1: int, idx2: int) -> float:
        """Единичное вычисление евклидова расстояния (backward compatibility)"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

    def manhattan_distance(self, idx1: int, idx2: int) -> float:
        """Единичное вычисление манхэттенского расстояния (backward compatibility)"""
        x1, y1, z1 = self.linear_to_3d(idx1)
        x2, y2, z2 = self.linear_to_3d(idx2)
        return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)


class FunctionalSimilarityAnalyzer(nn.Module):
    """
    Улучшенный анализатор функциональной близости с batch processing

    Основано на:
    - Cosine similarity состояний
    - Euclidean distance в feature space
    - Learnable weights для комбинирования метрик
    """

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

        # Learnable weights для различных метрик similarity
        self.cosine_weight = nn.Parameter(torch.tensor(0.5))
        self.euclidean_weight = nn.Parameter(torch.tensor(0.3))
        self.dot_product_weight = nn.Parameter(torch.tensor(0.2))

        # Дополнительные learnable компоненты для более сложной similarity
        self.similarity_transform = nn.Linear(state_size, state_size // 2, bias=False)

        logger.info(
            f"FunctionalSimilarityAnalyzer initialized: state_size={state_size}"
        )

    def forward(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """
        Batch вычисление функциональной близости

        Args:
            states1: [batch, state_size] - первые состояния
            states2: [batch, state_size] - вторые состояния

        Returns:
            similarities: [batch] - функциональная близость [0, 1]
        """
        # 1. Cosine similarity
        cosine_sim = F.cosine_similarity(states1, states2, dim=-1)
        cosine_sim = (cosine_sim + 1) / 2  # Нормализация к [0, 1]

        # 2. Euclidean similarity в оригинальном пространстве
        euclidean_dist = torch.norm(states1 - states2, dim=-1)
        max_possible_dist = torch.norm(torch.ones_like(states1), dim=-1)
        euclidean_sim = 1.0 - (euclidean_dist / (max_possible_dist + 1e-8))
        euclidean_sim = torch.clamp(euclidean_sim, 0, 1)

        # 3. Dot product similarity в трансформированном пространстве
        transformed1 = self.similarity_transform(states1)
        transformed2 = self.similarity_transform(states2)
        dot_product = torch.sum(transformed1 * transformed2, dim=-1)
        dot_product_sim = torch.sigmoid(dot_product)  # Нормализация к [0, 1]

        # 4. Взвешенная комбинация
        total_similarity = (
            self.cosine_weight * cosine_sim
            + self.euclidean_weight * euclidean_sim
            + self.dot_product_weight * dot_product_sim
        )

        return torch.clamp(total_similarity, 0, 1)

    def single_similarity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Единичное вычисление similarity (backward compatibility)"""
        with torch.no_grad():
            states1 = state1.unsqueeze(0)  # [1, state_size]
            states2 = state2.unsqueeze(0)  # [1, state_size]
            similarity = self.forward(states1, states2)
            return similarity.item()


class UnifiedConnectionClassifier(nn.Module):
    """
    Новый классификатор связей без deprecated зависимостей

    УЛУЧШЕНИЯ:
    - Batch processing для больших решеток
    - Оптимизированные вычисления расстояний
    - Централизованная конфигурация
    - Learnable пороги для адаптивности
    - Интеграция с spatial optimization
    """

    def __init__(self, lattice_dimensions: Tuple[int, int, int]):
        super().__init__()

        config = get_project_config()

        self.lattice_dimensions = lattice_dimensions
        self.state_size = config.gnn_state_size  # 32

        # Конфигурация пропорций из централизованного конфига
        self.local_ratio = config.local_tier  # 0.10
        self.functional_ratio = config.functional_tier  # 0.55
        self.distant_ratio = config.distant_tier  # 0.35

        # Основные компоненты
        self.distance_calculator = DistanceCalculator(lattice_dimensions)
        self.functional_analyzer = FunctionalSimilarityAnalyzer(self.state_size)

        # Learnable пороги для классификации
        self.local_distance_threshold = nn.Parameter(torch.tensor(2.0))
        self.distant_distance_threshold = nn.Parameter(torch.tensor(8.0))
        self.functional_similarity_threshold = nn.Parameter(torch.tensor(0.3))

        # Статистика использования
        self.classification_stats = {
            "total_classifications": 0,
            "batch_classifications": 0,
            "avg_neighbors_per_cell": 0.0,
        }

        logger.info(
            f"UnifiedConnectionClassifier initialized: "
            f"lattice={lattice_dimensions}, "
            f"ratios=(local={self.local_ratio:.2f}, "
            f"functional={self.functional_ratio:.2f}, "
            f"distant={self.distant_ratio:.2f})"
        )

    def classify_connections_batch(
        self,
        cell_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch классификация для производительности

        Args:
            cell_indices: [batch] - индексы центральных клеток
            neighbor_indices: [batch, max_neighbors] - индексы соседей (padded)
            states: [total_cells, state_size] - все состояния клеток

        Returns:
            classifications: Dict[str, torch.Tensor] - классификации по типам
        """
        batch_size, max_neighbors = neighbor_indices.shape
        device = cell_indices.device

        # Подготовка данных
        valid_mask = neighbor_indices >= 0  # [batch, max_neighbors]

        # Flatten для batch обработки
        cell_indices_expanded = cell_indices.unsqueeze(1).expand(
            -1, max_neighbors
        )  # [batch, max_neighbors]

        # Вычисляем расстояния batch-wise
        valid_pairs_mask = valid_mask.flatten()
        if valid_pairs_mask.sum() == 0:
            return self._empty_classification_result(batch_size, max_neighbors, device)

        valid_cell_indices = cell_indices_expanded[valid_mask]
        valid_neighbor_indices = neighbor_indices[valid_mask]

        # Batch вычисление расстояний
        euclidean_distances = self.distance_calculator.euclidean_distance_batch(
            valid_cell_indices, valid_neighbor_indices
        )

        # Первичная классификация по расстоянию
        local_mask = euclidean_distances <= self.local_distance_threshold
        distant_mask = euclidean_distances >= self.distant_distance_threshold
        functional_mask = ~(local_mask | distant_mask)

        # Уточнение через функциональную близость для functional connections
        if functional_mask.sum() > 0:
            functional_indices = torch.where(functional_mask)[0]
            func_cell_indices = valid_cell_indices[functional_indices]
            func_neighbor_indices = valid_neighbor_indices[functional_indices]

            # Получаем состояния для similarity анализа
            cell_states = states[func_cell_indices]
            neighbor_states = states[func_neighbor_indices]

            similarities = self.functional_analyzer(cell_states, neighbor_states)

            # Перераспределяем на основе similarity
            low_similarity_mask = similarities < self.functional_similarity_threshold

            # Обновляем маски
            low_sim_global_indices = functional_indices[low_similarity_mask]
            distant_mask[low_sim_global_indices] = True
            functional_mask[low_sim_global_indices] = False

        # Создаем результат
        result = self._create_batch_classification_result(
            batch_size,
            max_neighbors,
            valid_mask,
            local_mask,
            functional_mask,
            distant_mask,
            device,
        )

        # Обновляем статистику
        self.classification_stats["batch_classifications"] += 1
        self.classification_stats["total_classifications"] += batch_size

        return result

    def classify_connections(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """
        Единичная классификация (backward compatibility)

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
        distance_classification = self._classify_by_distance_single(
            cell_idx, neighbor_indices
        )

        # Шаг 2: Уточнение через функциональную близость
        functional_classification = self._refine_by_functionality_single(
            cell_state, neighbor_states, neighbor_indices, distance_classification
        )

        # Шаг 3: Балансировка пропорций
        balanced_classification = self._balance_proportions_single(
            functional_classification, len(neighbor_indices)
        )

        # Шаг 4: Создание ConnectionInfo объектов
        classified_connections = {}

        for category_name, neighbor_list in balanced_classification.items():
            category = ConnectionCategory(category_name)
            connections = []

            for neighbor_idx in neighbor_list:
                euclidean_dist = self.distance_calculator.euclidean_distance(
                    cell_idx, neighbor_idx
                )
                manhattan_dist = self.distance_calculator.manhattan_distance(
                    cell_idx, neighbor_idx
                )

                connection_info = ConnectionInfo(
                    source_idx=cell_idx,
                    target_idx=neighbor_idx,
                    euclidean_distance=euclidean_dist,
                    manhattan_distance=manhattan_dist,
                    category=category,
                    strength=1.0,
                )
                connections.append(connection_info)

            classified_connections[category] = connections

        self.classification_stats["total_classifications"] += 1
        return classified_connections

    def _empty_classification_result(
        self, batch_size: int, max_neighbors: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Создает пустой результат классификации"""
        return {
            "local": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "functional": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "distant": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
        }

    def _create_batch_classification_result(
        self,
        batch_size: int,
        max_neighbors: int,
        valid_mask: torch.Tensor,
        local_mask: torch.Tensor,
        functional_mask: torch.Tensor,
        distant_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Создает результат batch классификации"""

        # Инициализируем результирующие маски
        result_local = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        result_functional = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        result_distant = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )

        # Заполняем результаты только для валидных соседей
        result_local[valid_mask] = local_mask
        result_functional[valid_mask] = functional_mask
        result_distant[valid_mask] = distant_mask

        return {
            "local": result_local,
            "functional": result_functional,
            "distant": result_distant,
        }

    def _classify_by_distance_single(
        self, cell_idx: int, neighbor_indices: List[int]
    ) -> Dict[str, List[int]]:
        """Единичная классификация по расстоянию"""
        classifications = {"local": [], "functional": [], "distant": []}

        for neighbor_idx in neighbor_indices:
            euclidean_dist = self.distance_calculator.euclidean_distance(
                cell_idx, neighbor_idx
            )

            if euclidean_dist <= self.local_distance_threshold:
                classifications["local"].append(neighbor_idx)
            elif euclidean_dist >= self.distant_distance_threshold:
                classifications["distant"].append(neighbor_idx)
            else:
                classifications["functional"].append(neighbor_idx)

        return classifications

    def _refine_by_functionality_single(
        self,
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        neighbor_indices: List[int],
        initial_classification: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """Единичное уточнение классификации через функциональную близость"""
        if len(neighbor_indices) == 0:
            return initial_classification

        # Вычисляем функциональную близость со всеми соседями
        similarities = self.functional_analyzer(
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

    def _balance_proportions_single(
        self, classification: Dict[str, List[int]], total_neighbors: int
    ) -> Dict[str, List[int]]:
        """Единичная балансировка пропорций"""
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

    def get_classification_stats(self) -> Dict[str, Any]:
        """Получить статистику классификации"""
        stats = self.classification_stats.copy()
        stats.update(
            {
                "thresholds": {
                    "local_distance": self.local_distance_threshold.item(),
                    "distant_distance": self.distant_distance_threshold.item(),
                    "functional_similarity": self.functional_similarity_threshold.item(),
                },
                "target_ratios": {
                    "local": self.local_ratio,
                    "functional": self.functional_ratio,
                    "distant": self.distant_ratio,
                },
            }
        )
        return stats

    def reset_stats(self):
        """Сброс статистики"""
        self.classification_stats = {
            "total_classifications": 0,
            "batch_classifications": 0,
            "avg_neighbors_per_cell": 0.0,
        }


# === РЕФАКТОРИНГ И ОБРАТНАЯ СОВМЕСТИМОСТЬ ===
#
# ВНИМАНИЕ: Этот файл был разбит на более мелкие модули для улучшения читаемости:
#
# Новые модули:
# - connection_types.py - типы данных (ConnectionCategory, ConnectionInfo)
# - distance_calculator.py - вычисление расстояний в 3D решетке
# - functional_similarity.py - анализ функциональной близости
# - connection_classifier.py - основной классификатор связей (упрощенный)
#
# Рекомендуемые импорты для нового кода:
# from .connection_classifier import UnifiedConnectionClassifier  # Упрощенная версия
# from .connection_types import ConnectionCategory, ConnectionInfo
# from .distance_calculator import DistanceCalculator
# from .functional_similarity import FunctionalSimilarityAnalyzer
#
# Этот файл сохранен для обратной совместимости с существующим кодом.
