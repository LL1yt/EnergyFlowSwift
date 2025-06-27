#!/usr/bin/env python3
"""
Базовый интерфейс для всех клеток 3D решетки
==========================================

Определяет общий интерфейс для NCA и gMLP клеток.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Импорты для логирования
from ...utils.logging import get_logger

logger = get_logger(__name__)


class BaseCell(nn.Module, ABC):
    """
    Базовый интерфейс для всех клеток

    Принципы:
    1. Единый интерфейс forward()
    2. Информация о параметрах
    3. Сброс состояния (для модулей с памятью)
    """

    def __init__(self):
        super().__init__()
        self.state_size: int = 0
        self.target_params: Optional[int] = None

    @abstractmethod
    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass клетки

        Args:
            neighbor_states: [batch, neighbor_count, state_size] - состояния соседей
            own_state: [batch, state_size] - собственное состояние
            external_input: [batch, external_input_size] - внешний вход (опционально)

        Returns:
            new_state: [batch, state_size] - новое состояние
        """
        pass

    def reset_memory(self):
        """
        Сброс внутреннего состояния (для модулей с памятью)

        NCA клетки: не используют память
        gMLP клетки: могут иметь память (но в нашей архитектуре отключена)
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Информация о клетке для логирования и отладки
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "type": self.__class__.__name__,
            "state_size": self.state_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "target_params": self.target_params,
            "params_efficiency": (
                total_params / self.target_params if self.target_params else None
            ),
        }

    def log_info(self, logger):
        """Удобное логирование информации о клетке"""
        info = self.get_info()
        logger.info(
            f"[{info['type']}] Parameters: {info['total_params']:,} "
            f"(target: {info['target_params']:,}, "
            f"efficiency: {info['params_efficiency']:.2f}x)"
        )


# DEPRECATED: CellFactory удален. Используйте create_cell() из __init__.py
