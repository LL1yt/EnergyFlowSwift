#!/usr/bin/env python3
"""
Gating Network - сеть управления экспертами в MoE
===============================================

Learnable Gating Network для адаптивного взвешивания результатов
различных экспертов в Mixture of Experts архитектуре.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class GatingNetwork(nn.Module):
    """
    Learnable Gating Network для комбинирования результатов экспертов

    Цель: 808 параметров но нам не столько важно попасть в target_params, сколько иметь возможность настроить это через конфиг центральный
    Принцип: адаптивное взвешивание результатов на основе контекста
    """

    def __init__(self, state_size: Optional[int] = None, num_experts: int = 3):
        super().__init__()

        config = get_project_config()

        self.state_size = state_size or config.gnn.state_size
        self.num_experts = num_experts
        self.target_params = config.expert.gating.params

        # Рассчитываем архитектуру для достижения 808 параметров
        # Input: state_size + neighbor_activity = 32 + 32 = 64
        input_size = self.state_size * 2

        # Получаем hidden_dim из централизованной конфигурации
        hidden_dim = config.expert.gating.hidden_dim  # Централизованное значение

        self.gating_network = nn.Sequential(
            nn.Linear(input_size, hidden_dim, bias=True),  # 64*11 + 11 = 715
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts, bias=True),  # 11*3 + 3 = 36
            nn.Softmax(dim=-1),
        )
        # Итого: 715 + 36 = 751

        # Добавляем normalization для достижения 808
        # Нужно добавить: 808 - 751 = 57 параметров
        self.context_norm = nn.LayerNorm(self.state_size, bias=True)  # 32*2 = 64
        # Итого: 751 + 64 = 815 (близко к 808)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"GatingNetwork: {total_params} параметров (цель: {self.target_params})"
        )

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_activity: torch.Tensor,
        expert_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление весов экспертов и комбинирование результатов

        Args:
            current_state: [batch, state_size]
            neighbor_activity: [batch, state_size]
            expert_outputs: List[Tensor] - результаты от каждого эксперта

        Returns:
            combined_output: [batch, state_size] - взвешенная комбинация
            expert_weights: [batch, num_experts] - веса экспертов
        """
        # 1. Подготовка контекста для принятия решения
        normalized_state = self.context_norm(current_state)
        context = torch.cat([normalized_state, neighbor_activity], dim=-1)

        # 2. Вычисление весов экспертов
        expert_weights = self.gating_network(context)  # [batch, num_experts]

        # 3. Взвешенное комбинирование результатов экспертов
        stacked_outputs = torch.stack(
            expert_outputs, dim=1
        )  # [batch, num_experts, state_size]
        weights_expanded = expert_weights.unsqueeze(-1)  # [batch, num_experts, 1]

        combined_output = torch.sum(
            stacked_outputs * weights_expanded, dim=1
        )  # [batch, state_size]

        return combined_output, expert_weights

    def get_parameter_count(self) -> int:
        """Получить количество параметров"""
        return sum(p.numel() for p in self.parameters())

    def get_expert_weights_stats(self, expert_weights: torch.Tensor) -> dict:
        """Получить статистику использования экспертов"""
        with torch.no_grad():
            mean_weights = expert_weights.mean(dim=0)
            return {
                "local_expert_usage": mean_weights[0].item(),
                "functional_expert_usage": mean_weights[1].item(),
                "distant_expert_usage": mean_weights[2].item(),
                "entropy": -torch.sum(
                    mean_weights * torch.log(mean_weights + 1e-8)
                ).item(),
            }
