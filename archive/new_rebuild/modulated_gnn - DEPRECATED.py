#!/usr/bin/env python3
"""
Modulated GNN Cell - GNN клетка с NCA модуляцией
==============================================

GNN клетка с модуляцией от NCA для биологически правдоподобного
взаимодействия внутриклеточной и межклеточной динамики.

Расширяет базовую GNN клетку возможностью модуляции от NCA:
- Модулирует attention weights - селективность восприятия
- Модулирует эффективность message passing - интенсивность сообщений
- Модулирует интенсивность state update - скорость адаптации

Биологический аналог: синапсы с нейрохимической модуляцией.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .gnn_cell import GNNCell
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ModulatedGNNCell(GNNCell):
    """
    GNN клетка с модуляцией от NCA

    Расширяет базовую GNN клетку возможностью модуляции от NCA:
    - Модулирует attention weights
    - Модулирует эффективность message passing
    - Модулирует интенсивность state update
    """

    def __init__(self, *args, **kwargs):
        """Инициализация с вызовом родительского конструктора"""
        super().__init__(*args, **kwargs)

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        nca_modulation: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Модулированный GNN forward pass

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size]
            nca_modulation: dict с модулирующими коэффициентами от NCA

        Returns:
            new_state: [batch, state_size] - модулированное новое состояние
        """
        batch_size = own_state.shape[0]
        num_neighbors = neighbor_states.shape[1]

        # Обработка external input
        if external_input is not None:
            ext_input = external_input
        else:
            ext_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # === STEP 1: MESSAGE CREATION ===
        # Создаем сообщения от соседей к текущей клетке
        messages = []
        for i in range(self.neighbor_count):
            neighbor_state = neighbor_states[:, i, :]  # [batch, state_size]
            message = self.message_network(neighbor_state, own_state)
            messages.append(message)

        messages = torch.stack(messages, dim=1)  # [batch, neighbor_count, message_dim]

        # МОДУЛЯЦИЯ: влияние NCA на эффективность сообщений
        if nca_modulation and "message_modulation" in nca_modulation:
            message_mod = nca_modulation["message_modulation"]  # [batch, 1]
            message_mod = message_mod.unsqueeze(1)  # [batch, 1, 1]
            messages = messages * message_mod  # модулируем интенсивность сообщений

        # === STEP 2: ATTENTION AGGREGATION ===
        if self.use_attention:
            aggregated_message = self.aggregator(messages, own_state)

            # МОДУЛЯЦИЯ: влияние NCA на attention mechanism
            if nca_modulation and "attention_modulation" in nca_modulation:
                attention_mod = nca_modulation["attention_modulation"]  # [batch, 1]
                # Модулируем силу attention (ближе к 0.5 = меньше селективность)
                attention_strength = (attention_mod - 0.5) * 2  # [-1, 1]
                aggregated_message = aggregated_message * (
                    1.0 + attention_strength * 0.3
                )
        else:
            # Простая агрегация через среднее
            aggregated_message = torch.mean(messages, dim=1)

        # === STEP 3: STATE UPDATE ===
        new_state = self.state_updater(own_state, aggregated_message, ext_input)

        # МОДУЛЯЦИЯ: влияние NCA на интенсивность обновления состояния
        if nca_modulation and "update_modulation" in nca_modulation:
            update_mod = nca_modulation["update_modulation"]  # [batch, 1]
            # Интерполяция между старым и новым состоянием
            new_state = own_state * (1 - update_mod) + new_state * update_mod

        return new_state
