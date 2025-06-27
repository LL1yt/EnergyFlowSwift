#!/usr/bin/env python3
"""
DEPRECATED: GNN Cell - замена gMLP для оптимизированной коммуникации
=================================================================

🚨 УСТАРЕЛ: Используйте `new_rebuild.core.cells.vectorized_gnn_cell.VectorizedGNNCell` для лучшей производительности!

ЗАМЕНЕН НА: `vectorized_gnn_cell.py`
ПРИЧИНА: `VectorizedGNNCell` использует векторизованные операции, что значительно быстрее.
DEPRECATED с 27 июня 2025. Будет удален в версии 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from .base_cell import BaseCell
from ...config import get_project_config
from ...utils.logging import (
    get_logger,
    log_cell_init,
    log_cell_forward,
    log_cell_component_params,
)

logger = get_logger(__name__)


class MessageNetwork(nn.Module):
    """
    Создание контекстно-зависимых сообщений между клетками

    Каждое сообщение зависит от состояний ОБЕИХ сторон:
    sender_state + receiver_state → meaningful_message
    """

    def __init__(self, state_size: int, message_dim: int, hidden_dim: int):
        super().__init__()

        # Объединяем состояния отправителя и получателя
        combined_size = state_size * 2  # sender + receiver

        self.message_creator = nn.Sequential(
            nn.Linear(combined_size, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, message_dim, bias=True),
        )

    def forward(
        self, sender_state: torch.Tensor, receiver_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Создание сообщения от отправителя к получателю

        Args:
            sender_state: [batch, state_size] - состояние отправителя
            receiver_state: [batch, state_size] - состояние получателя

        Returns:
            message: [batch, message_dim] - контекстное сообщение
        """
        # Комбинируем состояния для создания контекстного сообщения
        combined = torch.cat([sender_state, receiver_state], dim=-1)
        message = self.message_creator(combined)
        return message


class AttentionAggregator(nn.Module):
    """
    Attention mechanism для селективной агрегации сообщений

    Клетка "выбирает" на какие сообщения обратить внимание
    → эмерджентный эффект селективного восприятия
    """

    def __init__(self, message_dim: int, state_size: int):
        super().__init__()

        # Attention weights зависят от текущего состояния клетки
        self.attention_network = nn.Sequential(
            nn.Linear(message_dim + state_size, message_dim, bias=True),
            nn.Tanh(),
            nn.Linear(message_dim, 1, bias=True),
        )

    def forward(
        self, messages: torch.Tensor, receiver_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Селективная агрегация сообщений через attention

        Args:
            messages: [batch, num_neighbors, message_dim] - все сообщения
            receiver_state: [batch, state_size] - состояние получателя

        Returns:
            aggregated: [batch, message_dim] - агрегированное сообщение
        """
        batch_size, num_neighbors, message_dim = messages.shape

        # Расширяем состояние получателя для каждого сообщения
        receiver_expanded = receiver_state.unsqueeze(1).expand(-1, num_neighbors, -1)

        # Комбинируем сообщения с состоянием для attention
        attention_input = torch.cat([messages, receiver_expanded], dim=-1)

        # Вычисляем attention weights
        attention_logits = self.attention_network(
            attention_input
        )  # [batch, num_neighbors, 1]
        attention_weights = F.softmax(
            attention_logits, dim=1
        )  # [batch, num_neighbors, 1]

        # Взвешенная агрегация сообщений
        aggregated = torch.sum(
            messages * attention_weights, dim=1
        )  # [batch, message_dim]

        return aggregated


class StateUpdater(nn.Module):
    """
    GRU-style обновление состояния для стабильности

    Использует gates для контролируемого обновления состояния
    → предотвращает взрывы и нестабильность
    """

    def __init__(self, state_size: int, message_dim: int, external_input_size: int):
        super().__init__()

        input_size = message_dim + external_input_size

        # Update gate: "насколько сильно обновлять состояние"
        self.update_gate = nn.Linear(state_size + input_size, state_size, bias=True)

        # Reset gate: "что забыть из старого состояния"
        self.reset_gate = nn.Linear(state_size + input_size, state_size, bias=True)

        # Candidate state: "новая информация для интеграции"
        self.candidate_network = nn.Linear(
            state_size + input_size, state_size, bias=True
        )

    def forward(
        self,
        current_state: torch.Tensor,
        aggregated_message: torch.Tensor,
        external_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Обновление состояния через GRU-style gates

        Args:
            current_state: [batch, state_size] - текущее состояние
            aggregated_message: [batch, message_dim] - агрегированное сообщение
            external_input: [batch, external_input_size] - внешний вход

        Returns:
            new_state: [batch, state_size] - новое состояние
        """
        # Комбинируем все входы
        combined_input = torch.cat([aggregated_message, external_input], dim=-1)
        full_input = torch.cat([current_state, combined_input], dim=-1)

        # Вычисляем gates
        update = torch.sigmoid(self.update_gate(full_input))
        reset = torch.sigmoid(self.reset_gate(full_input))

        # Candidate state с reset gate
        reset_state = reset * current_state
        candidate_input = torch.cat([reset_state, combined_input], dim=-1)
        candidate = torch.tanh(self.candidate_network(candidate_input))

        # Финальное обновление состояния
        new_state = (1 - update) * current_state + update * candidate

        return new_state


class GNNCell(BaseCell):
    """
    Graph Neural Network Cell для эмерджентной коммуникации

    АРХИТЕКТУРА:
    1. MessageNetwork: создание контекстно-зависимых сообщений
    2. AttentionAggregator: селективная агрегация сообщений
    3. StateUpdater: стабильное обновление состояния

    ПРЕИМУЩЕСТВА:
    - Намного меньше параметров чем gMLP (8k vs 113k)
    - Богатая коммуникация через контекстные сообщения
    - Естественная интеграция с STDP весами
    - Биологически правдоподобная архитектура
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        message_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        external_input_size: Optional[int] = None,
        activation: Optional[str] = None,
        target_params: Optional[int] = None,
        use_attention: Optional[bool] = None,
        **kwargs,
    ):
        """
        GNN клетка с оптимизированной коммуникацией

        Args:
            Все параметры опциональны - берутся из ProjectConfig если не указаны
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_project_config()
        gnn_config = config.get_gnn_config()

        # Используем параметры из конфигурации если не переданы
        self.state_size = state_size or gnn_config["state_size"]
        self.neighbor_count = neighbor_count or gnn_config["neighbor_count"]
        self.message_dim = message_dim or gnn_config["message_dim"]
        self.hidden_dim = hidden_dim or gnn_config["hidden_dim"]
        self.external_input_size = (
            external_input_size or gnn_config["external_input_size"]
        )
        self.target_params = target_params or gnn_config["target_params"]
        self.use_attention = (
            use_attention if use_attention is not None else gnn_config["use_attention"]
        )

        # === GNN АРХИТЕКТУРА ===

        # 1. Message Network - создание контекстных сообщений
        self.message_network = MessageNetwork(
            state_size=self.state_size,
            message_dim=self.message_dim,
            hidden_dim=self.hidden_dim,
        )

        # 2. Aggregation - селективная или простая агрегация
        if self.use_attention:
            self.aggregator = AttentionAggregator(
                message_dim=self.message_dim,
                state_size=self.state_size,
            )
        else:
            # Простая агрегация через среднее
            self.aggregator = None

        # 3. State Update - стабильное обновление состояния
        self.state_updater = StateUpdater(
            state_size=self.state_size,
            message_dim=self.message_dim,
            external_input_size=self.external_input_size,
        )

        # Логирование параметров
        if config.logging.debug_mode:
            self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование параметров через централизованную систему"""
        total_params = sum(p.numel() for p in self.parameters())

        # Используем специализированную функцию логирования клеток
        log_cell_init(
            cell_type="GNN",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            hidden_dim=self.hidden_dim,
            neighbor_count=self.neighbor_count,
            external_input_size=self.external_input_size,
            message_dim=self.message_dim,
            use_attention=self.use_attention,
        )

        # Детализация по компонентам
        component_params = {}
        for name, param in self.named_parameters():
            component = name.split(".")[0] if "." in name else name
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()

        log_cell_component_params(component_params, total_params)

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        connection_weights: Optional[torch.Tensor] = None,  # STDP веса
        flexible_neighbor_count: bool = False,  # NEW: поддержка переменного количества соседей
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass с message passing

        Args:
            neighbor_states: [batch, neighbor_count, state_size] - состояния соседей
            own_state: [batch, state_size] - собственное состояние
            external_input: [batch, external_input_size] - внешний вход
            connection_weights: [batch, neighbor_count] - STDP веса для модуляции

        Returns:
            new_state: [batch, state_size] - новое состояние
        """
        batch_size = own_state.shape[0]
        device = own_state.device

        # Обработка размерностей neighbor_states
        if neighbor_states.dim() == 2:
            # [num_neighbors, state_size] -> [1, num_neighbors, state_size]
            neighbor_states = neighbor_states.unsqueeze(0)

        # Логирование forward pass (только в debug режиме)
        config = get_project_config()
        if config.logging.debug_mode:
            input_shapes = {
                "neighbor_states": neighbor_states.shape,
                "own_state": own_state.shape,
                "external_input": (
                    external_input.shape if external_input is not None else None
                ),
                "connection_weights": (
                    connection_weights.shape if connection_weights is not None else None
                ),
            }
            log_cell_forward("GNN", input_shapes)

        # Подготовка external_input
        if external_input is None:
            external_input = torch.zeros(
                batch_size, self.external_input_size, device=device
            )
        elif external_input.shape[-1] != self.external_input_size:
            # Адаптация размера external_input если нужно
            if external_input.shape[-1] < self.external_input_size:
                padding = torch.zeros(
                    batch_size,
                    self.external_input_size - external_input.shape[-1],
                    device=device,
                )
                external_input = torch.cat([external_input, padding], dim=-1)
            else:
                external_input = external_input[:, : self.external_input_size]

        # === MESSAGE PASSING ===

        # 1. Создание сообщений от каждого соседа к текущей клетке
        messages = []
        # Используем фактическое количество соседей если flexible_neighbor_count=True
        actual_neighbor_count = (
            neighbor_states.shape[1] if flexible_neighbor_count else self.neighbor_count
        )

        for i in range(actual_neighbor_count):
            neighbor_state = neighbor_states[:, i, :]  # [batch, state_size]

            # Создаем контекстно-зависимое сообщение
            message = self.message_network(neighbor_state, own_state)
            messages.append(message)

        messages = torch.stack(
            messages, dim=1
        )  # [batch, actual_neighbor_count, message_dim]

        # 2. Модуляция сообщений через STDP веса (если есть)
        if connection_weights is not None:
            # connection_weights: [batch, actual_neighbor_count] → [batch, actual_neighbor_count, 1]
            stdp_weights = connection_weights.unsqueeze(-1)
            messages = messages * stdp_weights

        # 3. Агрегация сообщений
        if self.use_attention:
            # Селективная агрегация через attention
            aggregated_message = self.aggregator(messages, own_state)
        else:
            # Простая агрегация через среднее
            aggregated_message = torch.mean(messages, dim=1)  # [batch, message_dim]

        # 4. Обновление состояния
        new_state = self.state_updater(own_state, aggregated_message, external_input)

        return new_state

    def get_message_statistics(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Получить статистику сообщений для анализа эмерджентности

        Returns:
            Dict с ключами:
            - message_diversity: разнообразие сообщений
            - attention_entropy: энтропия attention (если используется)
            - message_magnitudes: величины сообщений
        """
        with torch.no_grad():
            # Создание сообщений
            messages = []
            for i in range(self.neighbor_count):
                neighbor_state = neighbor_states[:, i, :]
                message = self.message_network(neighbor_state, own_state)
                messages.append(message)

            messages = torch.stack(
                messages, dim=1
            )  # [batch, neighbor_count, message_dim]

            # Статистики
            stats = {
                "message_diversity": torch.std(messages, dim=1).mean(dim=-1),  # [batch]
                "message_magnitudes": torch.norm(
                    messages, dim=-1
                ),  # [batch, neighbor_count]
            }

            # Attention entropy если используется
            if self.use_attention:
                # Повторно вычисляем attention для статистики
                batch_size, num_neighbors, message_dim = messages.shape
                receiver_expanded = own_state.unsqueeze(1).expand(-1, num_neighbors, -1)
                attention_input = torch.cat([messages, receiver_expanded], dim=-1)
                attention_logits = self.aggregator.attention_network(
                    attention_input
                ).squeeze(-1)
                attention_weights = F.softmax(attention_logits, dim=1)

                # Entropy: -sum(p * log(p))
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=1
                )
                stats["attention_entropy"] = attention_entropy

            return stats
