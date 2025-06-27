#!/usr/bin/env python3
"""
Connection Classifier - классификация связей между клетками
=========================================================

Основной классификатор связей, использующий модульную архитектуру
для определения типов связей в 3D нейронной решетке.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from .connection_types import ConnectionCategory, ConnectionInfo
from .distance_calculator import DistanceCalculator
from .functional_similarity import FunctionalSimilarityAnalyzer
from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedConnectionClassifier(nn.Module):
    """
    Унифицированный классификатор связей с модульной архитектурой

    Использует:
    - DistanceCalculator для пространственного анализа
    - FunctionalSimilarityAnalyzer для функциональной близости
    - Learnable пороги для адаптивной классификации
    """

    def __init__(self, lattice_dimensions: Tuple[int, int, int]):
        super().__init__()

        config = get_project_config()

        self.lattice_dimensions = lattice_dimensions
        self.state_size = config.gnn.state_size

        # Модульные компоненты
        self.distance_calculator = DistanceCalculator(lattice_dimensions)
        self.similarity_analyzer = FunctionalSimilarityAnalyzer(self.state_size)

        # Learnable пороги для классификации
        self.local_distance_threshold = nn.Parameter(
            torch.tensor(config.expert.connections.local_distance_threshold)
        )
        self.functional_distance_threshold = nn.Parameter(
            torch.tensor(config.expert.connections.functional_distance_threshold)
        )
        self.distant_distance_threshold = nn.Parameter(
            torch.tensor(config.expert.connections.distant_distance_threshold)
        )
        self.functional_similarity_threshold = nn.Parameter(
            torch.tensor(config.expert.connections.functional_similarity_threshold)
        )

        # Целевые пропорции из конфига
        self.target_ratios = {
            "local": config.neighbors.local_tier,
            "functional": config.neighbors.functional_tier,
            "distant": config.neighbors.distant_tier,
        }

        # Статистика использования
        self.reset_stats()

        logger.info(f"UnifiedConnectionClassifier initialized for {lattice_dimensions}")

    def classify_connections_batch(
        self,
        cell_indices: torch.Tensor,
        neighbor_indices: torch.Tensor,
        states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch классификация связей для эффективности

        Args:
            cell_indices: [batch] - индексы клеток
            neighbor_indices: [batch, max_neighbors] - индексы соседей
            states: [total_cells, state_size] - состояния всех клеток

        Returns:
            Dict с масками для каждого типа связей
        """
        try:
            logger.debug(f"🔍 classify_connections_batch: входные данные - cell_indices.shape={cell_indices.shape}, neighbor_indices.shape={neighbor_indices.shape}, states.shape={states.shape}")
            logger.debug(f"🔍 cell_indices.dtype={cell_indices.dtype}, neighbor_indices.dtype={neighbor_indices.dtype}, states.dtype={states.dtype}")
            
            batch_size, max_neighbors = neighbor_indices.shape
            device = cell_indices.device

            # Создаем валидную маску (исключаем -1 padding)
            valid_mask = neighbor_indices >= 0
            logger.debug(f"🔍 valid_mask.shape={valid_mask.shape}, valid_mask.dtype={valid_mask.dtype}")
        except Exception as e:
            import traceback
            logger.error(f"❌ ОШИБКА в classify_connections_batch (начало): {e}")
            logger.error(f"📍 Traceback:\n{traceback.format_exc()}")
            raise

        if valid_mask.sum().item() == 0:  # Используем .sum().item() вместо .any()
            return self._empty_classification_result(batch_size, max_neighbors, device)

        # Извлекаем валидные пары
        valid_cells = cell_indices.unsqueeze(1).expand(-1, max_neighbors)[valid_mask]
        valid_neighbors = neighbor_indices[valid_mask]

        # 1. Расчет расстояний
        euclidean_distances = self.distance_calculator.euclidean_distance_batch(
            valid_cells, valid_neighbors
        )

        # 2. Классификация по расстоянию
        local_mask_flat = euclidean_distances <= self.local_distance_threshold
        distant_mask_flat = euclidean_distances >= self.distant_distance_threshold
        # Функциональные связи: между local и functional_distance_threshold
        functional_candidate_mask = (
            euclidean_distances > self.local_distance_threshold
        ) * (euclidean_distances <= self.functional_distance_threshold)
        # Средние связи: между functional_distance и distant_threshold (будут проверены на similarity)
        middle_mask = (euclidean_distances > self.functional_distance_threshold) * (
            euclidean_distances < self.distant_distance_threshold
        )

        # 3. Уточнение функциональных связей
        # Прямые функциональные (близкие по расстоянию)
        functional_mask_flat = functional_candidate_mask.clone()

        # Проверяем средние связи на функциональную близость
        if middle_mask.sum().item() > 0:  # Используем .sum().item() вместо .any()
            middle_cells = valid_cells[middle_mask]
            middle_neighbors = valid_neighbors[middle_mask]

            cell_states = states[middle_cells]
            neighbor_states = states[middle_neighbors]

            similarities = self.similarity_analyzer(cell_states, neighbor_states)
            high_similarity = similarities > self.functional_similarity_threshold

            # Добавляем средние связи с высокой функциональной близостью к functional
            middle_indices = torch.where(middle_mask)[0]
            functional_middle_indices = middle_indices[high_similarity]
            functional_mask_flat[functional_middle_indices] = True

        # 4. Восстанавливаем форму масок
        local_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        functional_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )
        distant_mask = torch.zeros(
            batch_size, max_neighbors, dtype=torch.bool, device=device
        )

        local_mask[valid_mask] = local_mask_flat
        functional_mask[valid_mask] = functional_mask_flat
        distant_mask[valid_mask] = distant_mask_flat

        # Обновляем статистику
        self._update_stats_batch(local_mask, functional_mask, distant_mask)

        return self._create_batch_classification_result(
            batch_size,
            max_neighbors,
            valid_mask,
            local_mask,
            functional_mask,
            distant_mask,
            device,
        )

    def classify_connections(
        self,
        cell_idx: int,
        neighbor_indices: List[int],
        cell_state: torch.Tensor,
        neighbor_states: torch.Tensor,
    ) -> Dict[ConnectionCategory, List[ConnectionInfo]]:
        """Единичная классификация связей (backward compatibility)"""
        # Проверяем пустые neighbor_indices (может быть тензором или списком)
        if torch.is_tensor(neighbor_indices):
            if neighbor_indices.numel() == 0:
                return {cat: [] for cat in ConnectionCategory}
        else:
            if not neighbor_indices:
                return {cat: [] for cat in ConnectionCategory}

        # Конвертируем в batch формат
        cell_tensor = torch.tensor([cell_idx], device=cell_state.device)
        if torch.is_tensor(neighbor_indices):
            neighbor_tensor = neighbor_indices.unsqueeze(0)  # [1, num_neighbors]
        else:
            neighbor_tensor = torch.tensor([neighbor_indices], device=cell_state.device)

        # Создаем полный тензор состояний
        try:
            logger.debug(f"🔍 concat debug: cell_state.shape={cell_state.shape}, neighbor_states.shape={neighbor_states.shape}")
            
            # Нормализуем размеры для корректной конкатенации
            # cell_state должен быть [state_size] или [1, state_size]
            if cell_state.dim() == 0:
                raise ValueError(f"cell_state не может быть скаляром, получено: {cell_state.shape}")
            elif cell_state.dim() == 1:
                # [state_size] -> [1, state_size]
                cell_state_normalized = cell_state.unsqueeze(0)
            elif cell_state.dim() == 2 and cell_state.shape[0] == 1:
                # [1, state_size] - уже правильный формат
                cell_state_normalized = cell_state
            else:
                raise ValueError(f"Неожиданная размерность cell_state: {cell_state.shape}, ожидалось [state_size] или [1, state_size]")
            
            # neighbor_states должен быть [num_neighbors, state_size]
            if neighbor_states.dim() == 0:
                raise ValueError(f"neighbor_states не может быть скаляром")
            elif neighbor_states.dim() == 1:
                # [state_size] -> [1, state_size] (один сосед)
                neighbor_states_normalized = neighbor_states.unsqueeze(0)
            elif neighbor_states.dim() == 2:
                # [num_neighbors, state_size] - уже правильный формат
                neighbor_states_normalized = neighbor_states
            elif neighbor_states.dim() == 3 and neighbor_states.shape[0] == 1:
                # [1, num_neighbors, state_size] -> [num_neighbors, state_size]
                neighbor_states_normalized = neighbor_states.squeeze(0)
            else:
                raise ValueError(f"Неожиданная размерность neighbor_states: {neighbor_states.shape}, ожидалось [num_neighbors, state_size]")
            
            logger.debug(f"🔍 after normalization: cell_state_normalized.shape={cell_state_normalized.shape}, neighbor_states_normalized.shape={neighbor_states_normalized.shape}")
            
            # Проверяем совместимость размеров state_size
            if cell_state_normalized.shape[-1] != neighbor_states_normalized.shape[-1]:
                raise ValueError(f"Несовместимые размеры state_size: cell={cell_state_normalized.shape[-1]}, neighbors={neighbor_states_normalized.shape[-1]}")
            
            # Объединяем: [1, state_size] + [num_neighbors, state_size] -> [1+num_neighbors, state_size]
            all_states = torch.cat([cell_state_normalized, neighbor_states_normalized], dim=0)
            logger.debug(f"🔍 concat result: all_states.shape={all_states.shape}")
        except Exception as e:
            logger.error(f"❌ concat error: {e}")
            logger.error(f"🔍 cell_state.shape={cell_state.shape}, neighbor_states.shape={neighbor_states.shape}")
            logger.error(f"🔍 cell_state_normalized.shape={locals().get('cell_state_normalized', 'не определено')}")
            logger.error(f"🔍 neighbor_states_normalized.shape={locals().get('neighbor_states_normalized', 'не определено')}")
            raise

        # Вызываем batch версию
        batch_result = self.classify_connections_batch(
            cell_tensor, neighbor_tensor, all_states
        )

        # Конвертируем результат обратно
        result = {cat: [] for cat in ConnectionCategory}

        for i, neighbor_idx in enumerate(neighbor_indices):
            if batch_result["local_mask"][0, i].item():
                category = ConnectionCategory.LOCAL
            elif batch_result["functional_mask"][0, i].item():
                category = ConnectionCategory.FUNCTIONAL
            elif batch_result["distant_mask"][0, i].item():
                category = ConnectionCategory.DISTANT
            else:
                continue  # Исключенный сосед

            # Вычисляем расстояния
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
            )

            result[category].append(connection_info)

        return result

    def _empty_classification_result(
        self, batch_size: int, max_neighbors: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Пустой результат классификации"""
        return {
            "local_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "functional_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "distant_mask": torch.zeros(
                batch_size, max_neighbors, dtype=torch.bool, device=device
            ),
            "valid_mask": torch.zeros(
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
        """Создание результата batch классификации"""
        return {
            "local_mask": local_mask,
            "functional_mask": functional_mask,
            "distant_mask": distant_mask,
            "valid_mask": valid_mask,
        }

    def _update_stats_batch(
        self,
        local_mask: torch.Tensor,
        functional_mask: torch.Tensor,
        distant_mask: torch.Tensor,
    ):
        """Обновление статистики использования"""
        self.usage_stats["local_count"] += local_mask.sum().item()
        self.usage_stats["functional_count"] += functional_mask.sum().item()
        self.usage_stats["distant_count"] += distant_mask.sum().item()
        self.usage_stats["total_classifications"] += 1

    def get_classification_stats(self) -> Dict[str, Any]:
        """Получить статистику классификации"""
        total = max(
            1,
            self.usage_stats["local_count"]
            + self.usage_stats["functional_count"]
            + self.usage_stats["distant_count"],
        )

        return {
            "local_ratio": self.usage_stats["local_count"] / total,
            "functional_ratio": self.usage_stats["functional_count"] / total,
            "distant_ratio": self.usage_stats["distant_count"] / total,
            "total_connections": total,
            "total_classifications": self.usage_stats["total_classifications"],
            "thresholds": {
                "local_distance": self.local_distance_threshold.item(),
                "functional_distance": self.functional_distance_threshold.item(),
                "distant_distance": self.distant_distance_threshold.item(),
                "functional_similarity": self.functional_similarity_threshold.item(),
            },
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.usage_stats = {
            "local_count": 0,
            "functional_count": 0,
            "distant_count": 0,
            "total_classifications": 0,
        }
