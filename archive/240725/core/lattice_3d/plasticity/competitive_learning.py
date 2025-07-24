"""
Конкурентное обучение для синаптической пластичности
=================================================

Реализует механизмы стабилизации STDP:
- Winner-Take-All механизм
- Латеральное торможение
- Нормализация весов
- Предотвращение "убегания" весов

Биологическая основа:
- Гомеостатическая пластичность поддерживает стабильные уровни активности
- Конкуренция между синапсами за ограниченные ресурсы
- Активные связи усиливаются, неактивные ослабляются
"""

from typing import Dict, Any, List
import torch
import logging


class CompetitiveLearning:
    """
    Механизм конкурентного обучения для стабилизации STDP.

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
        lateral_inhibition_radius: float = 3.0,
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
            lateral_inhibition_radius: Радиус латерального торможения
            update_frequency: Частота применения (каждые N шагов)
            weight_bounds: (min, max) границы весов
        """
        self.winner_boost_factor = winner_boost_factor
        self.lateral_inhibition_factor = lateral_inhibition_factor
        self.enable_weight_normalization = enable_weight_normalization
        self.normalization_method = normalization_method
        self.max_winner_connections = max_winner_connections
        self.lateral_inhibition_radius = lateral_inhibition_radius
        self.update_frequency = update_frequency
        self.weight_bounds = weight_bounds

        # Счетчик для частоты обновлений
        self.step_counter = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info("Competitive learning initialized:")
        self.logger.info(f"  Winner boost: {winner_boost_factor}")
        self.logger.info(f"  Lateral inhibition: {lateral_inhibition_factor}")
        self.logger.info(f"  Max winners: {max_winner_connections}")

    def apply_competitive_update(
        self,
        connection_weights: torch.Tensor,
        neighbor_indices: torch.Tensor,
        current_active: torch.Tensor,
        current_state_changes: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Применяет конкурентное обучение для стабилизации STDP.

        Args:
            connection_weights: Тензор весов связей [total_cells, max_neighbors]
            neighbor_indices: Индексы соседей для каждой клетки
            current_active: Маска активных клеток
            current_state_changes: Уровни активности клеток

        Returns:
            Dict со статистикой конкурентного обучения
        """
        # Проверяем частоту обновлений
        self.step_counter += 1
        if self.step_counter % self.update_frequency != 0:
            return {"message": "Skipped - update frequency"}

        # Статистика
        normalized_cells = 0
        winner_updates = 0
        lateral_inhibition_updates = 0
        total_cells = connection_weights.shape[0]

        with torch.no_grad():
            for cell_idx in range(total_cells):
                # 1. Нормализация весов для предотвращения "убегания"
                if self.enable_weight_normalization:
                    normalized_cells += self._normalize_cell_weights(
                        connection_weights, cell_idx
                    )

                # 2. Winner-Take-All для активных клеток
                if current_active[cell_idx]:
                    cell_neighbors = neighbor_indices[cell_idx]

                    # Обновления для этой клетки
                    cell_winner_updates, cell_inhibition_updates = (
                        self._apply_competition(
                            connection_weights,
                            cell_idx,
                            cell_neighbors,
                            current_active,
                            current_state_changes,
                        )
                    )

                    winner_updates += cell_winner_updates
                    lateral_inhibition_updates += cell_inhibition_updates

        return {
            "normalized_cells": normalized_cells,
            "winner_updates": winner_updates,
            "lateral_inhibition_updates": lateral_inhibition_updates,
            "active_cells": int(current_active.sum()),
            "step_counter": self.step_counter,
        }

    def _normalize_cell_weights(
        self, connection_weights: torch.Tensor, cell_idx: int
    ) -> int:
        """
        Нормализует веса связей для одной клетки.

        Returns:
            1 если нормализация была применена, 0 иначе
        """
        cell_weights = connection_weights[cell_idx, :]

        if self.normalization_method == "proportional":
            # Сохраняем среднее значение весов (proportional method)
            target_sum = float(cell_weights.sum().item())
            if target_sum > 0:
                # Нормализуем, сохраняя пропорции
                normalized_weights = cell_weights / cell_weights.sum() * target_sum
                connection_weights[cell_idx, :] = normalized_weights
                return 1
        elif self.normalization_method == "sum":
            # Нормализация до единичной суммы
            weight_sum = cell_weights.sum()
            if weight_sum > 0:
                connection_weights[cell_idx, :] = cell_weights / weight_sum
                return 1

        return 0

    def _apply_competition(
        self,
        connection_weights: torch.Tensor,
        cell_idx: int,
        cell_neighbors: torch.Tensor,
        current_active: torch.Tensor,
        current_state_changes: torch.Tensor,
    ) -> tuple:
        """
        Применяет Winner-Take-All и латеральное торможение для одной клетки.

        Returns:
            (winner_updates, lateral_inhibition_updates)
        """
        winner_updates = 0
        lateral_inhibition_updates = 0

        # Найдем "победителей" среди соседей по активности
        neighbor_activities = []
        valid_neighbors = []

        for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
            if neighbor_cell_idx != cell_idx and neighbor_cell_idx < len(
                current_active
            ):
                neighbor_activities.append(current_state_changes[neighbor_cell_idx])
                valid_neighbors.append((neighbor_idx, neighbor_cell_idx))
            else:
                neighbor_activities.append(0.0)
                valid_neighbors.append((neighbor_idx, neighbor_cell_idx))

        if not neighbor_activities:
            return winner_updates, lateral_inhibition_updates

        # Найдем наиболее активных соседей
        neighbor_activities_tensor = torch.tensor(neighbor_activities)

        # Получаем индексы топ-K активных соседей
        top_k = min(self.max_winner_connections, len(valid_neighbors))
        if top_k > 0:
            _, top_indices = torch.topk(neighbor_activities_tensor, top_k)
            winner_indices = set(top_indices.tolist())

            # Применяем конкуренцию
            for neighbor_idx, neighbor_cell_idx in valid_neighbors:
                if neighbor_cell_idx == cell_idx:
                    continue

                if neighbor_idx in winner_indices:
                    # Усиливаем связь с победителем
                    old_weight = connection_weights[cell_idx, neighbor_idx].item()
                    new_weight = min(
                        self.weight_bounds[1],
                        old_weight * self.winner_boost_factor,
                    )
                    if new_weight > old_weight:
                        connection_weights[cell_idx, neighbor_idx] = new_weight
                        winner_updates += 1

                elif (
                    neighbor_cell_idx < len(current_active)
                    and not current_active[neighbor_cell_idx]
                ):
                    # Латеральное торможение неактивных соседей
                    old_weight = connection_weights[cell_idx, neighbor_idx].item()
                    new_weight = max(
                        self.weight_bounds[0],
                        old_weight * self.lateral_inhibition_factor,
                    )
                    if new_weight < old_weight:
                        connection_weights[cell_idx, neighbor_idx] = new_weight
                        lateral_inhibition_updates += 1

        return winner_updates, lateral_inhibition_updates

    def get_configuration(self) -> Dict[str, Any]:
        """Получить текущую конфигурацию конкурентного обучения."""
        return {
            "winner_boost_factor": self.winner_boost_factor,
            "lateral_inhibition_factor": self.lateral_inhibition_factor,
            "enable_weight_normalization": self.enable_weight_normalization,
            "normalization_method": self.normalization_method,
            "max_winner_connections": self.max_winner_connections,
            "lateral_inhibition_radius": self.lateral_inhibition_radius,
            "update_frequency": self.update_frequency,
            "weight_bounds": self.weight_bounds,
        }

    def reset_counters(self):
        """Сброс внутренних счетчиков."""
        self.step_counter = 0
