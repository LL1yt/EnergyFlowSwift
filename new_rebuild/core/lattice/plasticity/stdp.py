"""
STDP (Spike-Timing Dependent Plasticity) механизм - Clean Implementation
======================================================================

Реализует биологически правдоподобную синаптическую пластичность:
- Классический STDP: LTP/LTD на основе временной последовательности
- BCM-enhanced STDP: интеграция с адаптивными порогами
- Оптимизированный для HybridCellV2 и GNN connection weights

Биологическая основа:
- LTP: усиление связи если пре-нейрон активен ДО пост-нейрона
- LTD: ослабление связи если пре-нейрон активен ПОСЛЕ пост-нейрона

Адаптировано для работы с модулированными GNN соединениями.
"""

from typing import Dict, Any, Optional
import torch
from ....utils.logging import get_logger

logger = get_logger(__name__)


class STDPMechanism:
    """
    Механизм STDP для обновления весов синаптических связей в HybridCellV2.

    Поддерживает:
    - Классический STDP (временные окна)
    - BCM-enhanced STDP (адаптивные пороги)
    - Интеграция с GNN connection weights
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        weight_bounds: tuple = (0.1, 2.0),
        enable_bcm: bool = False,
        bcm_learning_rate_factor: float = 0.5,
    ):
        """
        Инициализация STDP механизма.

        Args:
            learning_rate: Базовая скорость обучения
            A_plus: Амплитуда LTP (потенциация)
            A_minus: Амплитуда LTD (депрессия)
            tau_plus: Временная константа LTP (мс)
            tau_minus: Временная константа LTD (мс)
            weight_bounds: (min, max) границы весов
            enable_bcm: Включить BCM enhancement
            bcm_learning_rate_factor: Коэффициент для BCM относительно основного LR
        """
        self.learning_rate = learning_rate
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.weight_bounds = weight_bounds
        self.enable_bcm = enable_bcm
        self.bcm_learning_rate_factor = bcm_learning_rate_factor

        logger.info("STDP mechanism initialized:")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  LTP/LTD: {A_plus}/{A_minus}")
        logger.info(f"  BCM enabled: {enable_bcm}")

    def compute_activity_levels(
        self, states: torch.Tensor, previous_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет уровни активности на основе изменения состояний.

        Args:
            states: Текущие состояния клеток [batch, num_cells, state_size]
            previous_states: Предыдущие состояния клеток [batch, num_cells, state_size]

        Returns:
            activity_levels: [batch, num_cells] - уровни активности
        """
        # Вычисляем норму изменения состояния как меру активности
        state_changes = states - previous_states
        activity_levels = torch.norm(state_changes, dim=-1)  # [batch, num_cells]

        return activity_levels

    def apply_stdp_to_gnn_weights(
        self,
        gnn_cell,
        current_states: torch.Tensor,
        previous_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        adaptive_threshold: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Применяет STDP к весам GNN клетки в HybridCellV2.

        Args:
            gnn_cell: ModulatedGNNCell объект из HybridCellV2
            current_states: Текущие состояния клеток [batch, num_cells, state_size]
            previous_states: Предыдущие состояния клеток [batch, num_cells, state_size]
            neighbor_indices: Индексы соседей [num_cells, neighbor_count]
            adaptive_threshold: AdaptiveThreshold объект (optional)

        Returns:
            Dict со статистикой STDP обновления
        """
        # Вычисляем активности
        current_activity = self.compute_activity_levels(current_states, previous_states)

        # Статистика
        ltp_updates = 0
        ltd_updates = 0
        bcm_updates = 0
        total_weight_change = 0.0

        batch_size, num_cells = current_activity.shape

        with torch.no_grad():
            # Применяем STDP к connection weights в GNN
            if hasattr(gnn_cell, "message_network") and hasattr(
                gnn_cell.message_network, "weight"
            ):
                # Обновляем веса message network
                for cell_idx in range(min(num_cells, neighbor_indices.shape[0])):
                    cell_activity = current_activity[
                        :, cell_idx
                    ].mean()  # Средняя активность по batch

                    if cell_activity < 1e-6:  # Пропускаем неактивные клетки
                        continue

                    # Получаем соседей
                    cell_neighbors = neighbor_indices[cell_idx]

                    for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
                        if (
                            neighbor_cell_idx >= num_cells
                            or neighbor_cell_idx == cell_idx
                        ):
                            continue

                        # Активность соседа
                        neighbor_activity = current_activity[
                            :, neighbor_cell_idx
                        ].mean()

                        # Вычисляем STDP update
                        delta_w = self._compute_classical_stdp(
                            cell_activity.item(), neighbor_activity.item()
                        )

                        # BCM enhancement если включено
                        if self.enable_bcm and adaptive_threshold is not None:
                            bcm_factor = adaptive_threshold.get_plasticity_factor(
                                neighbor_activity.unsqueeze(0),
                                cell_activity.unsqueeze(0),
                            )
                            delta_w += self.bcm_learning_rate_factor * bcm_factor.item()
                            bcm_updates += 1

                        # Применяем обновление если есть изменение
                        if abs(delta_w) > 1e-6:
                            # Находим соответствующие веса в message network
                            # Упрощенная версия - обновляем веса пропорционально
                            weight_update_factor = delta_w * self.learning_rate

                            # Обновляем первые веса message network (упрощение)
                            if neighbor_idx < gnn_cell.message_network.weight.shape[0]:
                                old_weight = gnn_cell.message_network.weight[
                                    neighbor_idx
                                ].clone()
                                gnn_cell.message_network.weight[
                                    neighbor_idx
                                ] += weight_update_factor

                                # Применяем bounds
                                gnn_cell.message_network.weight[neighbor_idx] = (
                                    torch.clamp(
                                        gnn_cell.message_network.weight[neighbor_idx],
                                        self.weight_bounds[0],
                                        self.weight_bounds[1],
                                    )
                                )

                                total_weight_change += (
                                    torch.abs(
                                        gnn_cell.message_network.weight[neighbor_idx]
                                        - old_weight
                                    )
                                    .sum()
                                    .item()
                                )

                            if delta_w > 0:
                                ltp_updates += 1
                            else:
                                ltd_updates += 1

        return {
            "stdp_updates": {
                "ltp_updates": ltp_updates,
                "ltd_updates": ltd_updates,
                "bcm_updates": bcm_updates,
            },
            "total_weight_change": total_weight_change,
            "avg_weight_change": total_weight_change
            / max(1, ltp_updates + ltd_updates),
            "active_connections": ltp_updates + ltd_updates,
        }

    def _compute_classical_stdp(
        self, post_activity: float, pre_activity: float
    ) -> float:
        """
        Вычисляет классическое STDP обновление.

        Упрощенная версия: используем активности вместо точного времени спайков.

        Args:
            post_activity: Активность постсинаптической клетки
            pre_activity: Активность пресинаптической клетки

        Returns:
            delta_w: Изменение веса связи
        """
        # Упрощенное STDP правило на основе относительной активности
        if post_activity > pre_activity:
            # LTP: постсинаптическая клетка более активна
            delta_t_effective = post_activity - pre_activity
            delta_w = self.A_plus * torch.exp(
                torch.tensor(-delta_t_effective / self.tau_plus)
            )
        else:
            # LTD: пресинаптическая клетка более активна
            delta_t_effective = pre_activity - post_activity
            delta_w = -self.A_minus * torch.exp(
                torch.tensor(-delta_t_effective / self.tau_minus)
            )

        return float(delta_w.item())

    def get_configuration(self) -> Dict[str, Any]:
        """Получить текущую конфигурацию STDP."""
        return {
            "learning_rate": self.learning_rate,
            "A_plus": self.A_plus,
            "A_minus": self.A_minus,
            "tau_plus": self.tau_plus,
            "tau_minus": self.tau_minus,
            "weight_bounds": self.weight_bounds,
            "enable_bcm": self.enable_bcm,
            "bcm_learning_rate_factor": self.bcm_learning_rate_factor,
        }
