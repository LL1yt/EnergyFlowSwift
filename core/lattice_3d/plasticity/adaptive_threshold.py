"""
Адаптивные пороги активности для BCM правила
===========================================

Реализует Bienenstock-Cooper-Munro (BCM) правило обучения:
- Адаптивные пороги активности на основе истории клеток
- Автоматическая стабилизация обучения
- Предотвращение насыщения и катастрофического забывания

Биологическая основа:
- Нейроны с высокой активностью повышают порог → труднее активировать LTP
- Нейроны с низкой активностью понижают порог → легче активировать LTP
- Результат: автобалансировка активности нейронной сети
"""

from typing import Dict, Any, Optional
import torch
import logging


class AdaptiveThreshold:
    """
    Адаптивные пороги активности для BCM правила (метапластичность).

    Реализует Bienenstock-Cooper-Munro (BCM) правило обучения:
    - Порог модификации θ зависит от средней недавней активности клетки
    - Автоматическая стабилизация обучения через экспоненциальное скользящее среднее
    - Предотвращение насыщения и катастрофического забывания

    Биологическая основа:
    - Нейроны с высокой активностью повышают порог → труднее активировать LTP
    - Нейроны с низкой активностью понижают порог → легче активировать LTP
    - Результат: автобалансировка активности нейронной сети
    """

    def __init__(
        self,
        total_cells: int,
        tau_theta: float = 1000.0,
        initial_threshold: float = 0.05,
        min_threshold: float = 0.001,
        max_threshold: float = 0.5,
        device: torch.device = None,
    ):
        """
        Инициализация адаптивных порогов для всех клеток.

        Args:
            total_cells: Общее количество клеток в решетке
            tau_theta: Временная константа для скользящего среднего (шаги)
            initial_threshold: Начальное значение порога для всех клеток
            min_threshold: Минимальный допустимый порог
            max_threshold: Максимальный допустимый порог
            device: Устройство для тензоров (CPU/GPU)
        """
        self.total_cells = total_cells
        self.tau_theta = tau_theta
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.device = device or torch.device("cpu")

        # Адаптивные пороги для каждой клетки
        self.thresholds = torch.full(
            (total_cells,), initial_threshold, dtype=torch.float32, device=self.device
        )

        # Скользящее среднее активности для каждой клетки (для BCM правила)
        self.activity_averages = torch.zeros(
            total_cells, dtype=torch.float32, device=self.device
        )

        # Счетчик обновлений для каждой клетки
        self.update_counts = torch.zeros(
            total_cells, dtype=torch.long, device=self.device
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AdaptiveThreshold initialized for {total_cells} cells")
        self.logger.info(
            f"  tau_theta: {tau_theta}, range: [{min_threshold}, {max_threshold}]"
        )

    def update_thresholds(self, activity_levels: torch.Tensor) -> Dict[str, Any]:
        """
        Обновляет адаптивные пороги на основе текущей активности клеток.

        BCM правило: θ_i(t+1) = θ_i(t) + (activity_i^2 - θ_i(t)) / τ_θ

        Args:
            activity_levels: Tensor [total_cells] - уровни активности всех клеток

        Returns:
            Dict со статистикой обновления порогов
        """
        activity_levels = activity_levels.to(self.device)

        # BCM правило: экспоненциальное скользящее среднее квадратов активности
        activity_squared = activity_levels**2

        with torch.no_grad():
            # Обновляем скользящее среднее
            alpha = 1.0 / self.tau_theta  # Коэффициент обучения для EMA
            self.activity_averages = (
                1 - alpha
            ) * self.activity_averages + alpha * activity_squared

            # Обновляем пороги на основе скользящего среднего
            self.thresholds = torch.clamp(
                self.activity_averages, min=self.min_threshold, max=self.max_threshold
            )

            # Обновляем счетчики
            self.update_counts += 1

        # Статистика для мониторинга
        return {
            "thresholds_stats": {
                "min": float(self.thresholds.min().item()),
                "max": float(self.thresholds.max().item()),
                "mean": float(self.thresholds.mean().item()),
                "std": float(self.thresholds.std().item()),
            },
            "activity_stats": {
                "min": float(activity_levels.min().item()),
                "max": float(activity_levels.max().item()),
                "mean": float(activity_levels.mean().item()),
                "std": float(activity_levels.std().item()),
            },
            "adaptation_rate": float(alpha),
            "total_updates": int(self.update_counts.min().item()),
        }

    def get_plasticity_factor(
        self, pre_activity: torch.Tensor, post_activity: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет фактор пластичности для BCM правила.

        BCM правило пластичности:
        Δw = η * pre * post * (post - θ)

        Где:
        - η: learning rate
        - pre: активность пресинаптического нейрона
        - post: активность постсинаптического нейрона
        - θ: адаптивный порог

        Args:
            pre_activity: Активность пресинаптических нейронов
            post_activity: Активность постсинаптических нейронов

        Returns:
            Tensor с факторами пластичности для каждой связи
        """
        post_activity = post_activity.to(self.device)

        # BCM правило: (post - threshold)
        # Если post > threshold → LTP (positive plasticity)
        # Если post < threshold → LTD (negative plasticity)
        plasticity_factor = post_activity - self.thresholds

        return plasticity_factor

    def get_threshold_for_cell(self, cell_idx: int) -> float:
        """Получить текущий порог для конкретной клетки."""
        return float(self.thresholds[cell_idx].item())

    def reset_thresholds(self, initial_threshold: Optional[float] = None):
        """Сброс всех порогов к начальным значениям."""
        if initial_threshold is None:
            initial_threshold = 0.05

        self.thresholds.fill_(initial_threshold)
        self.activity_averages.zero_()
        self.update_counts.zero_()

        self.logger.info(f"Thresholds reset to {initial_threshold}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получить текущую статистику адаптивных порогов."""
        return {
            "current_thresholds": {
                "min": float(self.thresholds.min().item()),
                "max": float(self.thresholds.max().item()),
                "mean": float(self.thresholds.mean().item()),
                "std": float(self.thresholds.std().item()),
            },
            "activity_averages": {
                "min": float(self.activity_averages.min().item()),
                "max": float(self.activity_averages.max().item()),
                "mean": float(self.activity_averages.mean().item()),
                "std": float(self.activity_averages.std().item()),
            },
            "total_updates": int(self.update_counts.min().item()),
            "tau_theta": self.tau_theta,
        }
