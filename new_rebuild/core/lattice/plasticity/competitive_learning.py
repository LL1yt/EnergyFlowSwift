"""
Конкурентное обучение для синаптической пластичности - Clean Implementation
========================================================================

Реализует механизмы стабилизации STDP для HybridCellV2:
- Winner-Take-All механизм
- Латеральное торможение
- Нормализация весов
- Предотвращение "убегания" весов

Биологическая основа:
- Гомеостатическая пластичность поддерживает стабильные уровни активности
- Конкуренция между синапсами за ограниченные ресурсы
- Активные связи усиливаются, неактивные ослабляются

Адаптировано для модулированных GNN связей.
"""

from typing import Dict, Any, List
import torch
from ....utils.logging import get_logger

logger = get_logger(__name__)


class CompetitiveLearning:
    """
    Механизм конкурентного обучения для стабилизации STDP в HybridCellV2.

    Включает:
    - Нормализацию весов для предотвращения "убегания"
    - Winner-Take-All механизм
    - Латеральное торможение неактивных связей
    """

    def __init__(
        self,
        winner_boost_factor: float = 1.05,
        lateral_inhibition_factor: float = 0.98,
        enable_weight_normalization: bool = True,
        normalization_method: str = "proportional",
        max_winner_connections: int = 8,
        update_frequency: int = 1,
        weight_bounds: tuple = (0.1, 2.0),
    ):
        """
        Инициализация конкурентного обучения.

        Args:
            winner_boost_factor: Коэффициент усиления связей-победителей
            lateral_inhibition_factor: Коэффициент ослабления неактивных связей
            enable_weight_normalization: Включить нормализацию весов
            normalization_method: Метод нормализации ("proportional", "sum")
            max_winner_connections: Максимум усиленных связей на клетку
            update_frequency: Частота применения (каждые N шагов)
            weight_bounds: (min, max) границы весов
        """
        self.winner_boost_factor = winner_boost_factor
        self.lateral_inhibition_factor = lateral_inhibition_factor
        self.enable_weight_normalization = enable_weight_normalization
        self.normalization_method = normalization_method
        self.max_winner_connections = max_winner_connections
        self.update_frequency = update_frequency
        self.weight_bounds = weight_bounds

        # Счетчик для частоты обновлений
        self.step_counter = 0

        logger.info("Competitive learning initialized:")
        logger.info(f"  Winner boost: {winner_boost_factor}")
        logger.info(f"  Lateral inhibition: {lateral_inhibition_factor}")
        logger.info(f"  Max winners: {max_winner_connections}")

    def apply_competitive_update_to_gnn(
        self,
        gnn_cell,
        current_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Применяет конкурентное обучение к GNN весам в HybridCellV2.

        Args:
            gnn_cell: ModulatedGNNCell объект из HybridCellV2
            current_states: Текущие состояния клеток [batch, num_cells, state_size]
            neighbor_indices: Индексы соседей [num_cells, neighbor_count]

        Returns:
            Dict со статистикой конкурентного обучения
        """
        # Проверяем частоту обновлений
        self.step_counter += 1
        if self.step_counter % self.update_frequency != 0:
            return {"message": "Skipped - update frequency"}

        # Вычисляем активности клеток
        batch_size, num_cells, state_size = current_states.shape
        activity_levels = torch.norm(current_states, dim=-1)  # [batch, num_cells]
        mean_activity = activity_levels.mean(
            dim=0
        )  # [num_cells] - средняя активность по batch

        # Статистика
        normalized_weights = 0
        winner_updates = 0
        lateral_inhibition_updates = 0

        with torch.no_grad():
            # Применяем конкурентное обучение к GNN message network
            if hasattr(gnn_cell, "message_network") and hasattr(
                gnn_cell.message_network, "weight"
            ):

                # 1. Нормализация весов
                if self.enable_weight_normalization:
                    normalized_weights = self._normalize_gnn_weights(gnn_cell)

                # 2. Winner-Take-All для активных соединений
                for cell_idx in range(min(num_cells, neighbor_indices.shape[0])):
                    cell_activity = mean_activity[cell_idx]

                    if cell_activity < 1e-6:  # Пропускаем неактивные клетки
                        continue

                    # Получаем соседей
                    cell_neighbors = neighbor_indices[cell_idx]

                    # Применяем конкуренцию
                    cell_winner_updates, cell_inhibition_updates = (
                        self._apply_competition_to_cell(
                            gnn_cell, cell_idx, cell_neighbors, mean_activity
                        )
                    )

                    winner_updates += cell_winner_updates
                    lateral_inhibition_updates += cell_inhibition_updates

        return {
            "normalized_weights": normalized_weights,
            "winner_updates": winner_updates,
            "lateral_inhibition_updates": lateral_inhibition_updates,
            "step_counter": self.step_counter,
            "active_cells": int((mean_activity > 1e-6).sum().item()),
        }

    def _normalize_gnn_weights(self, gnn_cell) -> int:
        """
        Нормализует веса GNN для предотвращения убегания.

        Returns:
            1 если нормализация была применена, 0 иначе
        """
        if not hasattr(gnn_cell.message_network, "weight"):
            return 0

        weights = gnn_cell.message_network.weight

        if self.normalization_method == "proportional":
            # Сохраняем среднее значение весов
            current_mean = weights.mean()
            if current_mean > 0:
                # Нормализуем, сохраняя пропорции
                normalized_weights = weights / weights.mean() * current_mean
                gnn_cell.message_network.weight.data = normalized_weights
                return 1
        elif self.normalization_method == "sum":
            # Нормализация к единице
            weight_sum = weights.sum()
            if weight_sum > 0:
                gnn_cell.message_network.weight.data = weights / weight_sum
                return 1

        return 0

    def _apply_competition_to_cell(
        self,
        gnn_cell,
        cell_idx: int,
        cell_neighbors: torch.Tensor,
        activity_levels: torch.Tensor,
    ) -> tuple:
        """
        Применяет Winner-Take-All и латеральное торможение для одной клетки.

        Returns:
            (winner_updates, lateral_inhibition_updates)
        """
        winner_updates = 0
        lateral_inhibition_updates = 0

        # Определяем активности соседей
        neighbor_activities = []
        valid_neighbors = []

        for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
            if (
                neighbor_cell_idx < len(activity_levels)
                and neighbor_cell_idx != cell_idx
            ):
                neighbor_activities.append(activity_levels[neighbor_cell_idx].item())
                valid_neighbors.append((neighbor_idx, neighbor_cell_idx))
            else:
                neighbor_activities.append(0.0)
                valid_neighbors.append((neighbor_idx, neighbor_cell_idx))

        if not neighbor_activities:
            return winner_updates, lateral_inhibition_updates

        # Находим наиболее активных соседей (winners)
        neighbor_activities_tensor = torch.tensor(neighbor_activities)
        _, top_indices = torch.topk(
            neighbor_activities_tensor,
            min(self.max_winner_connections, len(neighbor_activities)),
        )

        # Применяем boost к winners и inhibition к остальным
        for i, (neighbor_idx, neighbor_cell_idx) in enumerate(valid_neighbors):
            if not hasattr(gnn_cell.message_network, "weight"):
                continue

            # Проверяем размерности
            if neighbor_idx >= gnn_cell.message_network.weight.shape[0]:
                continue

            if i in top_indices:
                # Winner: усиливаем связь
                gnn_cell.message_network.weight[
                    neighbor_idx
                ] *= self.winner_boost_factor
                winner_updates += 1
            else:
                # Lateral inhibition: ослабляем связь
                gnn_cell.message_network.weight[
                    neighbor_idx
                ] *= self.lateral_inhibition_factor
                lateral_inhibition_updates += 1

            # Применяем bounds
            gnn_cell.message_network.weight[neighbor_idx] = torch.clamp(
                gnn_cell.message_network.weight[neighbor_idx],
                self.weight_bounds[0],
                self.weight_bounds[1],
            )

        return winner_updates, lateral_inhibition_updates

    def get_configuration(self) -> Dict[str, Any]:
        """Получить текущую конфигурацию конкурентного обучения."""
        return {
            "winner_boost_factor": self.winner_boost_factor,
            "lateral_inhibition_factor": self.lateral_inhibition_factor,
            "enable_weight_normalization": self.enable_weight_normalization,
            "normalization_method": self.normalization_method,
            "max_winner_connections": self.max_winner_connections,
            "update_frequency": self.update_frequency,
            "weight_bounds": self.weight_bounds,
        }

    def reset_counters(self):
        """Сброс счетчиков."""
        self.step_counter = 0
        logger.info("Competitive learning counters reset")
