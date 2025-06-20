"""
Основной Mixin для пластичности
==============================

Объединяет все механизмы пластичности в единый интерфейс:
- Инициализация всех механизмов пластичности
- Координация между STDP, BCM и конкурентным обучением
- Отслеживание активности и управление обновлениями

Рефакторированная версия с модульной архитектурой.
"""

from typing import Dict, Any, Optional
import torch
import logging
import collections

from .adaptive_threshold import AdaptiveThreshold
from .stdp import STDPMechanism
from .competitive_learning import CompetitiveLearning


class PlasticityMixin:
    """
    Mixin класс для добавления пластичности в Lattice3D.

    Содержит методы для:
    - STDP обновлений
    - Конкурентного обучения
    - Метапластичности (BCM правило)
    - Объединенной пластичности

    Предполагает наличие атрибутов от основного класса Lattice3D.
    """

    def _init_plasticity(self):
        """Инициализация механизмов пластичности"""
        self.logger = logging.getLogger(__name__)

        # === STDP и пластичность ===
        self.enable_stdp = self.config.enable_plasticity
        if self.enable_stdp:
            # Параметры STDP
            stdp_config = self.config.stdp_config
            self.activity_threshold = stdp_config.get("activity_threshold", 0.1)
            learning_rate = stdp_config.get("learning_rate", 0.01)
            A_plus = stdp_config.get("A_plus", 0.01)
            A_minus = stdp_config.get("A_minus", 0.01)
            tau_plus = stdp_config.get("tau_plus", 20)
            tau_minus = stdp_config.get("tau_minus", 20)
            self.weight_bounds = stdp_config.get("weight_bounds", [0.1, 2.0])

            # Circular buffer для истории активности
            self.activity_history = collections.deque(
                maxlen=self.config.activity_history_size
            )

            # Буфер для предыдущих состояний
            self.previous_states = None

            # === Метапластичность (BCM правило) ===
            self.enable_metaplasticity = getattr(
                self.config, "enable_metaplasticity", False
            )

            if self.enable_metaplasticity:
                # Параметры BCM правила
                bcm_config = getattr(self.config, "bcm_config", {})
                tau_theta = bcm_config.get("tau_theta", 1000.0)
                initial_threshold = bcm_config.get(
                    "initial_threshold", self.activity_threshold
                )
                min_threshold = bcm_config.get("min_threshold", 0.001)
                max_threshold = bcm_config.get("max_threshold", 0.5)

                # Создаем адаптивные пороги
                self.adaptive_threshold = AdaptiveThreshold(
                    total_cells=self.config.total_cells,
                    tau_theta=tau_theta,
                    initial_threshold=initial_threshold,
                    min_threshold=min_threshold,
                    max_threshold=max_threshold,
                    device=self.device,
                )

                self.logger.info("BCM metaplasticity initialized:")
                self.logger.info(f"  tau_theta: {tau_theta}")
                self.logger.info(
                    f"  threshold range: [{min_threshold}, {max_threshold}]"
                )
            else:
                self.adaptive_threshold = None

            # Создаем STDP механизм
            self.stdp_mechanism = STDPMechanism(
                learning_rate=learning_rate,
                A_plus=A_plus,
                A_minus=A_minus,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                weight_bounds=tuple(self.weight_bounds),
                enable_bcm=self.enable_metaplasticity,
                bcm_learning_rate_factor=0.5,
            )

            self.logger.info("STDP mechanism initialized:")
            self.logger.info(f"  Activity threshold: {self.activity_threshold}")
            self.logger.info(f"  Learning rate: {learning_rate}")
            self.logger.info(f"  Weight bounds: {self.weight_bounds}")
        else:
            self.activity_history = None
            self.previous_states = None
            self.adaptive_threshold = None
            self.stdp_mechanism = None

        # === Конкурентное обучение ===
        self.enable_competitive = self.config.enable_competitive_learning
        if self.enable_competitive:
            # Параметры конкурентного обучения
            comp_config = self.config.competitive_config
            winner_boost_factor = comp_config.get("winner_boost_factor", 1.05)
            lateral_inhibition_factor = comp_config.get(
                "lateral_inhibition_factor", 0.98
            )
            enable_weight_normalization = comp_config.get(
                "enable_weight_normalization", True
            )
            normalization_method = comp_config.get(
                "normalization_method", "proportional"
            )
            max_winner_connections = comp_config.get("max_winner_connections", 8)
            lateral_inhibition_radius = comp_config.get(
                "lateral_inhibition_radius", 3.0
            )
            competitive_update_frequency = comp_config.get("update_frequency", 1)

            # Создаем конкурентное обучение
            self.competitive_learning = CompetitiveLearning(
                winner_boost_factor=winner_boost_factor,
                lateral_inhibition_factor=lateral_inhibition_factor,
                enable_weight_normalization=enable_weight_normalization,
                normalization_method=normalization_method,
                max_winner_connections=max_winner_connections,
                lateral_inhibition_radius=lateral_inhibition_radius,
                update_frequency=competitive_update_frequency,
                weight_bounds=(
                    tuple(self.weight_bounds) if self.enable_stdp else (0.1, 2.0)
                ),
            )

            self.logger.info("Competitive learning initialized:")
            self.logger.info(f"  Winner boost factor: {winner_boost_factor}")
            self.logger.info(
                f"  Lateral inhibition factor: {lateral_inhibition_factor}"
            )
            self.logger.info(f"  Max winner connections: {max_winner_connections}")
        else:
            self.competitive_learning = None

        # Проверка интеграции
        if self.enable_competitive and self.enable_stdp:
            self.logger.info(
                "BCM metaplasticity will work together with competitive learning. "
                "This is a powerful combination for stable learning."
            )

    def _track_activity_for_stdp(self, new_states: torch.Tensor):
        """Отслеживание активности для STDP и метапластичности"""
        if not self.enable_stdp:
            return

        # Сохраняем предыдущие состояния
        if self.previous_states is None:
            self.previous_states = new_states.clone()
            return

        # Вычисляем активность как норму изменения состояния
        state_change = torch.norm(new_states - self.previous_states, dim=1)

        # Определяем активные клетки с адаптивными порогами
        if self.enable_metaplasticity and self.adaptive_threshold is not None:
            # BCM: используем адаптивные пороги для каждой клетки
            adaptive_thresholds = self.adaptive_threshold.thresholds
            active_cells = (state_change > adaptive_thresholds).detach().cpu().numpy()

            # Обновляем адаптивные пороги
            threshold_stats = self.adaptive_threshold.update_thresholds(state_change)
        else:
            # Классический STDP: фиксированный порог
            active_cells = (
                (state_change > self.activity_threshold).detach().cpu().numpy()
            )
            threshold_stats = {"message": "Fixed threshold used"}

        # Добавляем в circular buffer
        activity_record = {
            "step": self.perf_stats["total_steps"],
            "active_cells": active_cells,
            "state_change": state_change.detach().cpu().numpy(),
        }

        # Добавляем статистику адаптивных порогов если доступна
        if self.enable_metaplasticity:
            activity_record["bcm_stats"] = threshold_stats

        self.activity_history.append(activity_record)

        # Обновляем предыдущие состояния
        self.previous_states = new_states.clone()

    def apply_stdp_update(self) -> Dict[str, Any]:
        """
        Применяет STDP правило для обновления весов связей.

        Использует модульный STDPMechanism для обработки.
        """
        if (
            not self.enable_stdp
            or not self.stdp_mechanism
            or len(self.activity_history) < 2
        ):
            return {"message": "STDP not enabled or insufficient history"}

        # Получаем два последних временных шага
        current_activity = self.activity_history[-1]
        previous_activity = self.activity_history[-2]

        current_active = torch.tensor(current_activity["active_cells"])
        previous_active = torch.tensor(previous_activity["active_cells"])
        current_state_changes = torch.tensor(current_activity["state_change"])
        previous_state_changes = torch.tensor(previous_activity["state_change"])

        # Получаем индексы всех соседей
        neighbor_indices = self.topology.get_all_neighbor_indices_batched()

        # Применяем STDP через модульный механизм
        stats = self.stdp_mechanism.apply_stdp_update(
            connection_weights=self.connection_weights,
            neighbor_indices=neighbor_indices,
            current_active=current_active,
            previous_active=previous_active,
            current_state_changes=current_state_changes,
            previous_state_changes=previous_state_changes,
            adaptive_threshold=self.adaptive_threshold,
        )

        # Добавляем статистику весов
        stats["connection_weights_stats"] = {
            "min": float(self.connection_weights.min().item()),
            "max": float(self.connection_weights.max().item()),
            "mean": float(self.connection_weights.mean().item()),
            "std": float(self.connection_weights.std().item()),
        }

        # Добавляем BCM статистику если включена метапластичность
        if self.enable_metaplasticity and "bcm_stats" in current_activity:
            stats["adaptive_thresholds"] = current_activity["bcm_stats"]

        return stats

    def apply_competitive_learning(self) -> Dict[str, Any]:
        """
        Применяет конкурентное обучение для стабилизации STDP.

        Использует модульный CompetitiveLearning для обработки.
        """
        if (
            not self.enable_competitive
            or not self.competitive_learning
            or not self.enable_stdp
            or len(self.activity_history) < 1
        ):
            return {"message": "Competitive learning not available"}

        current_activity = self.activity_history[-1]
        current_active = torch.tensor(current_activity["active_cells"])
        current_state_changes = torch.tensor(current_activity["state_change"])

        # Получаем индексы всех соседей
        neighbor_indices = self.topology.get_all_neighbor_indices_batched()

        # Применяем конкурентное обучение через модульный механизм
        stats = self.competitive_learning.apply_competitive_update(
            connection_weights=self.connection_weights,
            neighbor_indices=neighbor_indices,
            current_active=current_active,
            current_state_changes=current_state_changes,
        )

        # Добавляем статистику весов
        stats["connection_weights_stats"] = {
            "min": float(self.connection_weights.min().item()),
            "max": float(self.connection_weights.max().item()),
            "mean": float(self.connection_weights.mean().item()),
            "std": float(self.connection_weights.std().item()),
        }

        return stats

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
                    stdp_stats.get("stdp_updates", {}).get("ltp_updates", 0)
                    + stdp_stats.get("stdp_updates", {}).get("ltd_updates", 0)
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
