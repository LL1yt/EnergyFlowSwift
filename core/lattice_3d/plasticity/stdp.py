"""
STDP (Spike-Timing Dependent Plasticity) механизм
================================================

Реализует биологически правдоподобную синаптическую пластичность:
- Классический STDP: LTP/LTD на основе временной последовательности
- BCM-enhanced STDP: интеграция с адаптивными порогами
- Векторизованные операции для GPU эффективности

Биологическая основа:
- LTP: усиление связи если пре-нейрон активен ДО пост-нейрона
- LTD: ослабление связи если пре-нейрон активен ПОСЛЕ пост-нейрона
"""

from typing import Dict, Any, Optional
import torch
import logging


class STDPMechanism:
    """
    Механизм STDP для обновления весов синаптических связей.

    Поддерживает:
    - Классический STDP (временные окна)
    - BCM-enhanced STDP (адаптивные пороги)
    - Векторизованные операции
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

        self.logger = logging.getLogger(__name__)
        self.logger.info("STDP mechanism initialized:")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  LTP/LTD: {A_plus}/{A_minus}")
        self.logger.info(f"  BCM enabled: {enable_bcm}")

    def apply_stdp_update(
        self,
        connection_weights: torch.Tensor,
        neighbor_indices: torch.Tensor,
        current_active: torch.Tensor,
        previous_active: torch.Tensor,
        current_state_changes: torch.Tensor,
        previous_state_changes: torch.Tensor,
        adaptive_threshold=None,
    ) -> Dict[str, Any]:
        """
        Применяет STDP правило для обновления весов связей.

        Args:
            connection_weights: Тензор весов связей [total_cells, max_neighbors]
            neighbor_indices: Индексы соседей для каждой клетки
            current_active: Маска активных клеток (текущий шаг)
            previous_active: Маска активных клеток (предыдущий шаг)
            current_state_changes: Уровни активности (текущий шаг)
            previous_state_changes: Уровни активности (предыдущий шаг)
            adaptive_threshold: BCM AdaptiveThreshold объект (optional)

        Returns:
            Dict со статистикой STDP обновления
        """
        # Статистика для отчета
        ltp_updates = 0
        ltd_updates = 0
        bcm_ltp_updates = 0
        bcm_ltd_updates = 0
        total_weight_change = 0.0

        total_cells = connection_weights.shape[0]

        # Batch processing для эффективности
        with torch.no_grad():
            # Vectorized STDP update для всех клеток
            for cell_idx in range(total_cells):
                if not current_active[cell_idx]:
                    continue  # Клетка не активна - не обновляем веса

                # Получаем соседей этой клетки
                cell_neighbors = neighbor_indices[cell_idx]
                cell_post_activity = current_state_changes[cell_idx]

                for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
                    if neighbor_cell_idx == cell_idx:
                        continue  # Пропускаем self-connections

                    neighbor_pre_activity = previous_state_changes[neighbor_cell_idx]

                    # === ОСНОВНОЕ STDP ПРАВИЛО ===
                    delta_w_stdp = self._compute_classical_stdp(
                        current_active[cell_idx],
                        previous_active[neighbor_cell_idx],
                        current_active,
                        previous_active,
                        cell_idx,
                        neighbor_cell_idx,
                    )

                    if delta_w_stdp > 0:
                        ltp_updates += 1
                    elif delta_w_stdp < 0:
                        ltd_updates += 1

                    # === BCM ENHANCEMENT ===
                    delta_w_bcm = 0.0
                    if self.enable_bcm and adaptive_threshold is not None:
                        delta_w_bcm = self._compute_bcm_enhancement(
                            neighbor_pre_activity,
                            cell_post_activity,
                            adaptive_threshold.get_threshold_for_cell(cell_idx),
                        )

                        if delta_w_bcm > 0:
                            bcm_ltp_updates += 1
                        elif delta_w_bcm < 0:
                            bcm_ltd_updates += 1

                    # Комбинируем STDP и BCM updates
                    delta_w_total = delta_w_stdp + delta_w_bcm

                    if abs(delta_w_total) > 1e-6:
                        # Обновляем вес связи
                        old_weight = connection_weights[cell_idx, neighbor_idx].item()
                        new_weight = old_weight + delta_w_total

                        # Применяем bounds checking
                        new_weight = max(
                            self.weight_bounds[0],
                            min(self.weight_bounds[1], new_weight),
                        )

                        connection_weights[cell_idx, neighbor_idx] = new_weight
                        total_weight_change += abs(new_weight - old_weight)

        # Статистика для мониторинга
        active_cells_count = int(current_active.sum())

        stats = {
            "active_cells": active_cells_count,
            "stdp_updates": {
                "ltp_updates": ltp_updates,
                "ltd_updates": ltd_updates,
            },
            "total_weight_change": total_weight_change,
            "avg_weight_change": total_weight_change
            / max(1, ltp_updates + ltd_updates + bcm_ltp_updates + bcm_ltd_updates),
        }

        # Добавляем BCM статистику если включена
        if self.enable_bcm:
            stats["bcm_updates"] = {
                "bcm_ltp_updates": bcm_ltp_updates,
                "bcm_ltd_updates": bcm_ltd_updates,
            }

        return stats

    def _compute_classical_stdp(
        self,
        current_cell_active: bool,
        previous_neighbor_active: bool,
        current_active: torch.Tensor,
        previous_active: torch.Tensor,
        cell_idx: int,
        neighbor_cell_idx: int,
    ) -> float:
        """
        Вычисляет классическое STDP обновление.

        Returns:
            delta_w: Изменение веса (положительное = LTP, отрицательное = LTD)
        """
        if previous_neighbor_active and current_cell_active:
            # LTP: сосед был активен на предыдущем шаге, текущая клетка активна сейчас
            return self.A_plus * self.learning_rate
        elif current_cell_active and not previous_neighbor_active:
            # LTD: текущая клетка активна, но сосед НЕ был активен ранее
            return -self.A_minus * self.learning_rate
        else:
            return 0.0

    def _compute_bcm_enhancement(
        self, pre_activity: float, post_activity: float, adaptive_threshold: float
    ) -> float:
        """
        Вычисляет BCM enhancement для STDP.

        BCM правило: Δw = η * pre * post * (post - θ)

        Returns:
            delta_w_bcm: BCM вклад в изменение веса
        """
        # BCM пластичность: Δw = η * pre * post * (post - θ)
        bcm_factor = post_activity - adaptive_threshold

        if abs(bcm_factor) > 1e-6:  # Избегаем численной нестабильности
            return (
                self.learning_rate
                * self.bcm_learning_rate_factor
                * pre_activity
                * bcm_factor
            )
        else:
            return 0.0

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
