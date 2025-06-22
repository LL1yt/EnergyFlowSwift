#!/usr/bin/env python3
"""
NCA Modulator - биологическая модуляция межклеточной коммуникации
===============================================================

Модулятор влияния NCA на GNN без преобразования размерностей.

NCA (4D) генерирует модулирующие сигналы для GNN операций:
- Attention weights modulation - селективность восприятия
- Message passing efficiency - интенсивность сообщений
- State update gates - скорость обновления состояния

Биологический аналог: внутриклеточные процессы влияют на
эффективность синаптической передачи через нейромодуляторы.
"""

import torch
import torch.nn as nn
from typing import Dict

from ...utils.logging import get_logger

logger = get_logger(__name__)


class NCAModulator(nn.Module):
    """
    Модулятор влияния NCA на GNN без преобразования размерностей

    NCA (4D) генерирует модулирующие сигналы для GNN операций:
    - Attention weights modulation
    - Message passing efficiency
    - State update gates

    Биологический аналог: внутриклеточные процессы влияют на
    эффективность синаптической передачи
    """

    def __init__(self, nca_state_size: int, gnn_components: int = 3):
        super().__init__()

        self.nca_state_size = nca_state_size
        self.gnn_components = gnn_components  # attention, messages, update

        # NCA состояние → модулирующие коэффициенты для GNN
        self.modulation_network = nn.Sequential(
            nn.Linear(nca_state_size, nca_state_size * 2, bias=True),
            nn.Tanh(),  # ограничиваем модуляцию
            nn.Linear(nca_state_size * 2, gnn_components, bias=True),
            nn.Sigmoid(),  # модуляция в диапазоне [0, 1]
        )

        # Инициализация для нейтральной модуляции (0.5)
        with torch.no_grad():
            self.modulation_network[-2].bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, nca_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Генерация модулирующих коэффициентов из NCA состояния

        Args:
            nca_state: [batch, nca_state_size] - состояние NCA

        Returns:
            modulation: dict с коэффициентами для разных GNN компонентов
        """
        # Получаем модулирующие коэффициенты
        modulation_coeffs = self.modulation_network(
            nca_state
        )  # [batch, gnn_components]

        return {
            "attention_modulation": modulation_coeffs[:, 0:1],  # [batch, 1]
            "message_modulation": modulation_coeffs[:, 1:2],  # [batch, 1]
            "update_modulation": modulation_coeffs[:, 2:3],  # [batch, 1]
        }
