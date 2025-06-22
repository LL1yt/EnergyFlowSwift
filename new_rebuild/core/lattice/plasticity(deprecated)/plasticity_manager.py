"""
Plasticity Manager - Управление пластичностью для HybridCellV2
=============================================================

Объединяет все механизмы пластичности в единый интерфейс:
- Координация между STDP, BCM и конкурентным обучением
- Интеграция с HybridCellV2 архитектурой
- Отслеживание активности и управление обновлениями
- Биологически правдоподобная пластичность

Заменяет Legacy PlasticityMixin с адаптацией под Clean архитектуру.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import collections
from ....utils.logging import get_logger
from ....config import get_project_config

from .adaptive_threshold import AdaptiveThreshold
from .stdp import STDPMechanism
from .competitive_learning import CompetitiveLearning

logger = get_logger(__name__)


class PlasticityManager:
    """
    Управляющий класс для всех механизмов пластичности в HybridCellV2.

    Содержит методы для:
    - STDP обновлений GNN весов
    - Конкурентного обучения
    - Метапластичности (BCM правило)
    - Интеграции с NCA модуляцией

    Предназначен для работы с HybridCellV2 и Lattice3D.
    """

    def __init__(
        self,
        total_cells: int,
        enable_stdp: bool = True,
        enable_competitive: bool = True,
        enable_metaplasticity: bool = True,
        activity_history_size: int = 100,
        device: Optional[torch.device] = None,
    ):
        """
        Инициализация менеджера пластичности.

        Args:
            total_cells: Общее количество клеток в решетке
            enable_stdp: Включить STDP механизм
            enable_competitive: Включить конкурентное обучение
            enable_metaplasticity: Включить BCM метапластичность
            activity_history_size: Размер буфера истории активности
            device: Устройство для тензоров
        """
        self.total_cells = total_cells
        self.enable_stdp = enable_stdp
        self.enable_competitive = enable_competitive
        self.enable_metaplasticity = enable_metaplasticity
        self.device = device or torch.device("cpu")

        # Получаем конфигурацию
        config = get_project_config()

        # === Инициализация компонентов ===

        # 1. Адаптивные пороги (BCM метапластичность)
        if self.enable_metaplasticity:
            self.adaptive_threshold = AdaptiveThreshold(
                total_cells=total_cells,
                tau_theta=1000.0,
                initial_threshold=0.05,
                min_threshold=0.001,
                max_threshold=0.5,
                device=self.device,
            )
            logger.info("BCM metaplasticity enabled")
        else:
            self.adaptive_threshold = None

        # 2. STDP механизм
        if self.enable_stdp:
            self.stdp_mechanism = STDPMechanism(
                learning_rate=0.01,
                A_plus=0.01,
                A_minus=0.01,
                tau_plus=20.0,
                tau_minus=20.0,
                weight_bounds=(0.1, 2.0),
                enable_bcm=self.enable_metaplasticity,
                bcm_learning_rate_factor=0.5,
            )
            logger.info("STDP mechanism enabled")
        else:
            self.stdp_mechanism = None

        # 3. Конкурентное обучение
        if self.enable_competitive:
            self.competitive_learning = CompetitiveLearning(
                winner_boost_factor=1.05,
                lateral_inhibition_factor=0.98,
                enable_weight_normalization=True,
                normalization_method="proportional",
                max_winner_connections=8,
                update_frequency=1,
                weight_bounds=(0.1, 2.0),
            )
            logger.info("Competitive learning enabled")
        else:
            self.competitive_learning = None

        # === Буферы состояний ===
        self.activity_history = collections.deque(maxlen=activity_history_size)
        self.previous_states = None
        self.step_counter = 0

        logger.info(f"PlasticityManager initialized for {total_cells} cells")
        logger.info(
            f"  STDP: {enable_stdp}, Competitive: {enable_competitive}, BCM: {enable_metaplasticity}"
        )

    def update_plasticity(
        self,
        hybrid_cell,
        current_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Основной метод обновления пластичности для HybridCellV2.

        Args:
            hybrid_cell: HybridCellV2 объект
            current_states: Текущие состояния клеток [batch, num_cells, state_size]
            neighbor_indices: Индексы соседей [num_cells, neighbor_count]

        Returns:
            Dict со статистикой всех обновлений пластичности
        """
        self.step_counter += 1

        # Инициализация previous_states если первый вызов
        if self.previous_states is None:
            self.previous_states = current_states.clone()
            return {"message": "Initializing previous states"}

        stats = {"step": self.step_counter}

        # Вычисляем активности для BCM
        activity_levels = self._compute_activity_levels(current_states)

        # === 1. Обновление адаптивных порогов (BCM) ===
        if self.enable_metaplasticity and self.adaptive_threshold is not None:
            threshold_stats = self.adaptive_threshold.update_thresholds(
                activity_levels.mean(dim=0)
            )
            stats["bcm_thresholds"] = threshold_stats

        # === 2. STDP обновления ===
        if self.enable_stdp and self.stdp_mechanism is not None:
            stdp_stats = self.stdp_mechanism.apply_stdp_to_gnn_weights(
                gnn_cell=hybrid_cell.gnn_cell,
                current_states=current_states,
                previous_states=self.previous_states,
                neighbor_indices=neighbor_indices,
                adaptive_threshold=self.adaptive_threshold,
            )
            stats["stdp"] = stdp_stats

        # === 3. Конкурентное обучение ===
        if self.enable_competitive and self.competitive_learning is not None:
            competitive_stats = (
                self.competitive_learning.apply_competitive_update_to_gnn(
                    gnn_cell=hybrid_cell.gnn_cell,
                    current_states=current_states,
                    neighbor_indices=neighbor_indices,
                )
            )
            stats["competitive"] = competitive_stats

        # === 4. Интеграция с NCA модуляцией ===
        if self.enable_metaplasticity and self.adaptive_threshold is not None:
            nca_modulation_stats = self._integrate_with_nca_modulation(
                hybrid_cell=hybrid_cell, activity_levels=activity_levels
            )
            stats["nca_modulation"] = nca_modulation_stats

        # Обновляем буферы
        self._update_activity_history(activity_levels)
        self.previous_states = current_states.clone()

        return stats

    def _compute_activity_levels(self, current_states: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет уровни активности клеток.

        Args:
            current_states: [batch, num_cells, state_size]

        Returns:
            activity_levels: [batch, num_cells]
        """
        if self.previous_states is None:
            return torch.zeros(current_states.shape[:2], device=self.device)

        # Активность как норма изменения состояния
        state_changes = current_states - self.previous_states
        activity_levels = torch.norm(state_changes, dim=-1)

        return activity_levels

    def _integrate_with_nca_modulation(
        self, hybrid_cell, activity_levels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Интеграция BCM порогов с NCA модуляцией.

        Args:
            hybrid_cell: HybridCellV2 объект
            activity_levels: [batch, num_cells] активности клеток

        Returns:
            Dict со статистикой интеграции
        """
        if not self.enable_metaplasticity or self.adaptive_threshold is None:
            return {"message": "BCM not enabled"}

        # Получаем средние активности
        mean_activity = activity_levels.mean(dim=0)  # [num_cells]

        # Получаем адаптивные пороги
        thresholds = self.adaptive_threshold.thresholds  # [num_cells]

        # Вычисляем модуляционный фактор для NCA
        # Высокая активность относительно порога → сильная модуляция
        modulation_factor = torch.sigmoid(mean_activity - thresholds)

        # Статистика
        stats = {
            "mean_modulation_factor": float(modulation_factor.mean().item()),
            "modulation_range": [
                float(modulation_factor.min().item()),
                float(modulation_factor.max().item()),
            ],
            "high_modulation_cells": int((modulation_factor > 0.7).sum().item()),
            "low_modulation_cells": int((modulation_factor < 0.3).sum().item()),
        }

        return stats

    def _update_activity_history(self, activity_levels: torch.Tensor):
        """
        Обновляет историю активности для анализа паттернов.

        Args:
            activity_levels: [batch, num_cells] активности клеток
        """
        # Сохраняем средние активности по batch
        mean_activity = activity_levels.mean(dim=0).detach().cpu().numpy()

        activity_record = {
            "step": self.step_counter,
            "activity_levels": mean_activity,
            "max_activity": float(mean_activity.max()),
            "mean_activity": float(mean_activity.mean()),
        }

        self.activity_history.append(activity_record)

    def get_plasticity_statistics(self) -> Dict[str, Any]:
        """Получить полную статистику пластичности."""
        stats = {
            "step_counter": self.step_counter,
            "total_cells": self.total_cells,
            "enabled_mechanisms": {
                "stdp": self.enable_stdp,
                "competitive": self.enable_competitive,
                "metaplasticity": self.enable_metaplasticity,
            },
        }

        # Статистика компонентов
        if self.adaptive_threshold is not None:
            stats["adaptive_thresholds"] = self.adaptive_threshold.get_statistics()

        if self.stdp_mechanism is not None:
            stats["stdp_config"] = self.stdp_mechanism.get_configuration()

        if self.competitive_learning is not None:
            stats["competitive_config"] = self.competitive_learning.get_configuration()

        # История активности
        if self.activity_history:
            recent_activities = [
                record["mean_activity"] for record in list(self.activity_history)[-10:]
            ]
            stats["recent_activity"] = {
                "mean": float(sum(recent_activities) / len(recent_activities)),
                "trend": (
                    "increasing"
                    if recent_activities[-1] > recent_activities[0]
                    else "decreasing"
                ),
            }

        return stats

    def reset_plasticity(self):
        """Сброс всех состояний пластичности."""
        if self.adaptive_threshold is not None:
            self.adaptive_threshold.reset_thresholds()

        if self.competitive_learning is not None:
            self.competitive_learning.reset_counters()

        self.activity_history.clear()
        self.previous_states = None
        self.step_counter = 0

        logger.info("All plasticity mechanisms reset")
