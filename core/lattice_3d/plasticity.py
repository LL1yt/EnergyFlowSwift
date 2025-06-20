"""
Модуль пластичности для 3D Решетки
==================================

Содержит механизмы синаптической пластичности:
- STDP (Spike-Timing Dependent Plasticity)
- Конкурентное обучение (Winner-Take-All, латеральное торможение)
- Нормализация весов связей

Выделен из lattice.py для улучшения модульности и читаемости кода.
"""

from typing import Dict, Any
import torch
import logging
import collections


class PlasticityMixin:
    """
    Mixin класс для добавления пластичности в Lattice3D.

    Содержит методы для:
    - STDP обновлений
    - Конкурентного обучения
    - Объединенной пластичности

    Предполагает наличие атрибутов от основного класса Lattice3D.
    """

    def _init_plasticity(self):
        """Инициализация механизмов пластичности"""
        # === STDP и пластичность ===
        self.enable_stdp = self.config.enable_plasticity
        if self.enable_stdp:
            # Параметры STDP
            stdp_config = self.config.stdp_config
            self.activity_threshold = stdp_config.get("activity_threshold", 0.1)
            self.learning_rate = stdp_config.get("learning_rate", 0.01)
            self.A_plus = stdp_config.get("A_plus", 0.01)
            self.A_minus = stdp_config.get("A_minus", 0.01)
            self.tau_plus = stdp_config.get("tau_plus", 20)
            self.tau_minus = stdp_config.get("tau_minus", 20)
            self.weight_bounds = stdp_config.get("weight_bounds", [0.1, 2.0])

            # Circular buffer для истории активности
            self.activity_history = collections.deque(
                maxlen=self.config.activity_history_size
            )

            # Буфер для предыдущих состояний
            self.previous_states = None

            self.logger.info("STDP mechanism initialized:")
            self.logger.info(f"  Activity threshold: {self.activity_threshold}")
            self.logger.info(f"  Learning rate: {self.learning_rate}")
            self.logger.info(f"  Weight bounds: {self.weight_bounds}")
        else:
            self.activity_history = None
            self.previous_states = None

        # === Конкурентное обучение ===
        self.enable_competitive = self.config.enable_competitive_learning
        if self.enable_competitive:
            # Параметры конкурентного обучения
            comp_config = self.config.competitive_config
            self.winner_boost_factor = comp_config.get("winner_boost_factor", 1.05)
            self.lateral_inhibition_factor = comp_config.get(
                "lateral_inhibition_factor", 0.98
            )
            self.enable_weight_normalization = comp_config.get(
                "enable_weight_normalization", True
            )
            self.normalization_method = comp_config.get(
                "normalization_method", "proportional"
            )
            self.competitive_update_frequency = comp_config.get("update_frequency", 1)
            self.max_winner_connections = comp_config.get("max_winner_connections", 8)
            self.lateral_inhibition_radius = comp_config.get(
                "lateral_inhibition_radius", 3.0
            )

            # Счетчик для частоты обновлений
            self.competitive_step_counter = 0

            self.logger.info("Competitive learning initialized:")
            self.logger.info(f"  Winner boost factor: {self.winner_boost_factor}")
            self.logger.info(
                f"  Lateral inhibition factor: {self.lateral_inhibition_factor}"
            )
            self.logger.info(f"  Max winner connections: {self.max_winner_connections}")
        else:
            self.competitive_step_counter = 0

    def _track_activity_for_stdp(self, new_states: torch.Tensor):
        """Отслеживание активности для STDP"""
        if not self.enable_stdp:
            return

        # Сохраняем предыдущие состояния
        if self.previous_states is None:
            self.previous_states = new_states.clone()
            return

        # Вычисляем активность как норму изменения состояния
        state_change = torch.norm(new_states - self.previous_states, dim=1)
        active_cells = (state_change > self.activity_threshold).detach().cpu().numpy()

        # Добавляем в circular buffer
        self.activity_history.append(
            {
                "step": self.perf_stats["total_steps"],
                "active_cells": active_cells,
                "state_change": state_change.detach().cpu().numpy(),
            }
        )

        # Обновляем предыдущие состояния
        self.previous_states = new_states.clone()

    def apply_stdp_update(self) -> Dict[str, Any]:
        """
        Применяет STDP правило для обновления весов связей.

        Основано на биологическом правиле:
        - LTP (Long Term Potentiation): если сосед активен ДО текущей клетки → вес++
        - LTD (Long Term Depression): если сосед активен ПОСЛЕ текущей клетки → вес--

        Returns:
            Dict с статистикой STDP обновления
        """
        if not self.enable_stdp or len(self.activity_history) < 2:
            return {"message": "STDP not enabled or insufficient history"}

        # Получаем два последних временных шага
        current_activity = self.activity_history[-1]
        previous_activity = self.activity_history[-2]

        current_active = current_activity["active_cells"]
        previous_active = previous_activity["active_cells"]

        # Статистика для отчета
        ltp_updates = 0
        ltd_updates = 0
        total_weight_change = 0.0

        # Batch processing для эффективности
        with torch.no_grad():
            # Получаем индексы всех соседей
            neighbor_indices = self.topology.get_all_neighbor_indices_batched()

            # Vectorized STDP update для всех клеток
            for cell_idx in range(self.config.total_cells):
                if not current_active[cell_idx]:
                    continue  # Клетка не активна - не обновляем веса

                # Получаем соседей этой клетки
                cell_neighbors = neighbor_indices[cell_idx]

                for neighbor_idx, neighbor_cell_idx in enumerate(cell_neighbors):
                    if neighbor_cell_idx == cell_idx:
                        continue  # Пропускаем self-connections

                    # STDP правило
                    delta_w = 0.0

                    if previous_active[neighbor_cell_idx] and current_active[cell_idx]:
                        # LTP: сосед был активен на предыдущем шаге, текущая клетка активна сейчас
                        delta_w = self.A_plus * self.learning_rate
                        ltp_updates += 1
                    elif (
                        current_active[cell_idx]
                        and not previous_active[neighbor_cell_idx]
                    ):
                        # LTD: текущая клетка активна, но сосед НЕ был активен ранее
                        delta_w = -self.A_minus * self.learning_rate
                        ltd_updates += 1

                    if delta_w != 0.0:
                        # Обновляем вес связи
                        old_weight = self.connection_weights[
                            cell_idx, neighbor_idx
                        ].item()
                        new_weight = old_weight + delta_w

                        # Применяем bounds checking
                        new_weight = max(
                            self.weight_bounds[0],
                            min(self.weight_bounds[1], new_weight),
                        )

                        self.connection_weights[cell_idx, neighbor_idx] = new_weight
                        total_weight_change += abs(new_weight - old_weight)

        # Статистика для мониторинга
        active_cells_count = int(current_active.sum())

        return {
            "active_cells": active_cells_count,
            "ltp_updates": ltp_updates,
            "ltd_updates": ltd_updates,
            "total_weight_change": total_weight_change,
            "avg_weight_change": total_weight_change
            / max(1, ltp_updates + ltd_updates),
            "connection_weights_stats": {
                "min": float(self.connection_weights.min().item()),
                "max": float(self.connection_weights.max().item()),
                "mean": float(self.connection_weights.mean().item()),
                "std": float(self.connection_weights.std().item()),
            },
        }

    def apply_competitive_learning(self) -> Dict[str, Any]:
        """
        Применяет конкурентное обучение для стабилизации STDP.

        Включает:
        1. Нормализацию весов (∑weights = const)
        2. Winner-Take-All механизм
        3. Латеральное торможение неактивных связей

        Returns:
            Dict с статистикой конкурентного обучения
        """
        if (
            not self.enable_competitive
            or not self.enable_stdp
            or len(self.activity_history) < 1
        ):
            return {"message": "Competitive learning not available"}

        # Проверяем частоту обновлений
        self.competitive_step_counter += 1
        if self.competitive_step_counter % self.competitive_update_frequency != 0:
            return {"message": "Skipped - update frequency"}

        current_activity = self.activity_history[-1]
        current_active = current_activity["active_cells"]

        # Статистика
        normalized_cells = 0
        winner_updates = 0
        lateral_inhibition_updates = 0

        with torch.no_grad():
            neighbor_indices = self.topology.get_all_neighbor_indices_batched()

            for cell_idx in range(self.config.total_cells):
                # 1. Нормализация весов для предотвращения "убегания"
                if self.enable_weight_normalization:
                    cell_weights = self.connection_weights[cell_idx, :]

                    # Сохраняем среднее значение весов (proportional method)
                    if self.normalization_method == "proportional":
                        target_sum = float(cell_weights.sum().item())
                        if target_sum > 0:
                            # Нормализуем, сохраняя пропорции
                            normalized_weights = (
                                cell_weights / cell_weights.sum() * target_sum
                            )
                            self.connection_weights[cell_idx, :] = normalized_weights
                            normalized_cells += 1

                # 2. Winner-Take-All для активных клеток
                if current_active[cell_idx]:
                    cell_neighbors = neighbor_indices[cell_idx]

                    # Найдем "победителя" среди соседей по активности
                    neighbor_activities = []
                    for neighbor_cell_idx in cell_neighbors:
                        if neighbor_cell_idx < len(current_active):
                            neighbor_activities.append(
                                current_activity["state_change"][neighbor_cell_idx]
                            )
                        else:
                            neighbor_activities.append(0.0)

                    if neighbor_activities:
                        # Найдем наиболее активного соседа
                        neighbor_activities_tensor = torch.tensor(neighbor_activities)
                        winner_idx = torch.argmax(neighbor_activities_tensor)
                        winner_neighbor = cell_neighbors[winner_idx]

                        # Ограничиваем количество победителей
                        winners_so_far = 0

                        # Усиливаем связь с победителем
                        if (
                            winner_neighbor != cell_idx
                            and winners_so_far < self.max_winner_connections
                        ):
                            old_weight = self.connection_weights[
                                cell_idx, winner_idx
                            ].item()
                            new_weight = min(
                                self.weight_bounds[1],
                                old_weight * self.winner_boost_factor,
                            )
                            self.connection_weights[cell_idx, winner_idx] = new_weight
                            if new_weight > old_weight:
                                winner_updates += 1
                                winners_so_far += 1

                        # 3. Латеральное торможение неактивных соседей
                        for neighbor_idx, neighbor_cell_idx in enumerate(
                            cell_neighbors
                        ):
                            if (
                                neighbor_cell_idx != cell_idx
                                and neighbor_cell_idx < len(current_active)
                                and not current_active[neighbor_cell_idx]
                                and neighbor_idx != winner_idx
                            ):

                                old_weight = self.connection_weights[
                                    cell_idx, neighbor_idx
                                ].item()
                                new_weight = max(
                                    self.weight_bounds[0],
                                    old_weight * self.lateral_inhibition_factor,
                                )
                                self.connection_weights[cell_idx, neighbor_idx] = (
                                    new_weight
                                )
                                if new_weight < old_weight:
                                    lateral_inhibition_updates += 1

        return {
            "normalized_cells": normalized_cells,
            "winner_updates": winner_updates,
            "lateral_inhibition_updates": lateral_inhibition_updates,
            "active_cells": int(current_active.sum()),
            "step_counter": self.competitive_step_counter,
            "connection_weights_stats": {
                "min": float(self.connection_weights.min().item()),
                "max": float(self.connection_weights.max().item()),
                "mean": float(self.connection_weights.mean().item()),
                "std": float(self.connection_weights.std().item()),
            },
        }

    def apply_combined_plasticity(self) -> Dict[str, Any]:
        """
        Применяет STDP + конкурентное обучение последовательно.

        Это основной метод для биологически правдоподобной пластичности.

        Returns:
            Dict с объединенной статистикой
        """
        # Сначала STDP
        stdp_stats = self.apply_stdp_update()

        # Затем конкурентное обучение для стабилизации
        competitive_stats = self.apply_competitive_learning()

        # Объединенная статистика
        return {
            "stdp": stdp_stats,
            "competitive": competitive_stats,
            "combined_stats": {
                "total_active_cells": competitive_stats.get("active_cells", 0),
                "plasticity_operations": (
                    stdp_stats.get("ltp_updates", 0)
                    + stdp_stats.get("ltd_updates", 0)
                    + competitive_stats.get("winner_updates", 0)
                    + competitive_stats.get("lateral_inhibition_updates", 0)
                ),
                "weight_stability": {
                    "min": float(self.connection_weights.min().item()),
                    "max": float(self.connection_weights.max().item()),
                    "mean": float(self.connection_weights.mean().item()),
                    "std": float(self.connection_weights.std().item()),
                },
            },
        }
