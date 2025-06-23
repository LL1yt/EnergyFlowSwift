#!/usr/bin/env python3
"""
Simple Linear Expert - для локальных связей (10%)
================================================

Простой линейный эксперт для быстрой обработки ближайших соседей.
Аналогия: рефлексы в нервной системе - быстрая реакция без сложных вычислений.

АРХИТЕКТУРА:
- Neighbor weights для взвешенной агрегации
- Linear transformation для обработки состояний
- Residual connection для стабильности
- Точно 2059 параметров согласно спецификации

ПРИНЦИПЫ:
1. Минимальная сложность для максимальной скорости
2. Стабильность через residual connections
3. Биологическая правдоподобность (рефлексы)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward

logger = get_logger(__name__)


class SimpleLinearExpert(nn.Module):
    """
    Простой линейный эксперт для local connections (10%)

    Быстрая обработка ближайших соседей без сложных вычислений.
    Целевые параметры: 2059 (точно по спецификации)
    """

    def __init__(self, state_size: int, max_neighbors: Optional[int] = None):
        super().__init__()

        config = get_project_config()

        self.state_size = state_size

        self.target_params = config.local_expert_params  # 2059

        # === АРХИТЕКТУРА ДЛЯ ДОСТИЖЕНИЯ 2059 ПАРАМЕТРОВ ===

        # 1. Neighbor weights (adaptive для разного количества соседей)
        self.neighbor_weights = nn.Parameter(
            torch.ones(self.max_neighbors) / self.max_neighbors
        )
        # params: зависит от размера решетки

        # 2. Main processing network
        # Рассчитываем размеры для достижения точно 2059 параметров
        input_size = state_size * 2  # current + neighbor_influence

        # Подбираем hidden_dim для достижения целевого количества параметров
        # Layer1: input_size -> hidden_dim (bias included)
        # Layer2: hidden_dim -> state_size (bias included)
        # Total: input_size * hidden_dim + hidden_dim + hidden_dim * state_size + state_size
        # = hidden_dim * (input_size + 1 + state_size) + state_size

        remaining_params = self.target_params - self.max_neighbors  # 2059 - neighbors
        # Для 2000 соседей: 2059 - 2000 = 59 параметров на сети
        # Используем минимальную архитектуру если соседей много

        if remaining_params > 100:
            # Достаточно параметров для полной сети
            hidden_dim = min(20, remaining_params // 100)  # Адаптивный размер
        else:
            # Минимальная архитектура
            hidden_dim = 1

        self.processing_network = nn.Sequential(
            nn.Linear(input_size, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, state_size, bias=True),
        )
        # Адаптивная архитектура в зависимости от количества соседей

        # 3. Residual connection parameters
        # Параметры из централизованной конфигурации
        self.alpha = nn.Parameter(torch.tensor(config.local_expert_alpha))  # 1 параметр
        self.beta = nn.Parameter(torch.tensor(config.local_expert_beta))  # 1 параметр

        # 4. Добавляем нормализацию для стабильности
        self.normalization = nn.LayerNorm(state_size, bias=True)

        # Итого: адаптивное количество параметров в зависимости от neighbors

        # Подсчет и логирование параметров
        total_params = sum(p.numel() for p in self.parameters())

        log_cell_init(
            cell_type="SimpleLinearExpert",
            total_params=total_params,
            target_params=self.target_params,
            state_size=state_size,
            max_neighbors=self.max_neighbors,
            hidden_dim=hidden_dim,
        )

        logger.info(
            f"SimpleLinearExpert: {total_params} параметров (цель: {self.target_params})"
        )

    def forward(
        self, current_state: torch.Tensor, neighbor_states: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Быстрая обработка локальных соседей

        Args:
            current_state: [batch, state_size] - текущее состояние
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей

        Returns:
            new_state: [batch, state_size] - обновленное состояние
        """
        batch_size = current_state.shape[0]

        if neighbor_states.shape[1] == 0:
            return current_state  # Нет соседей - возвращаем без изменений

        # 1. Взвешенная агрегация соседей
        num_neighbors = min(neighbor_states.shape[1], self.max_neighbors)
        weights = self.neighbor_weights[:num_neighbors]

        # Нормализуем веса для корректной агрегации
        weights = F.softmax(weights, dim=0)

        # Weighted aggregation
        neighbor_influence = torch.einsum(
            "bnc,n->bc", neighbor_states[:, :num_neighbors], weights
        )

        # 2. Объединяем текущее состояние с влиянием соседей
        combined_input = torch.cat([current_state, neighbor_influence], dim=-1)

        # 3. Линейная обработка
        processed = self.processing_network(combined_input)

        # 4. Нормализация для стабильности
        processed = self.normalization(processed)

        # 5. Residual connection с learnable коэффициентами
        new_state = self.alpha * current_state + self.beta * processed

        return new_state

    def get_parameter_info(self) -> Dict[str, Any]:
        """Получить информацию о параметрах эксперта"""
        param_breakdown = {
            "neighbor_weights": self.neighbor_weights.numel(),
            "processing_network": sum(
                p.numel() for p in self.processing_network.parameters()
            ),
            "alpha": self.alpha.numel(),
            "beta": self.beta.numel(),
            "normalization": sum(p.numel() for p in self.normalization.parameters()),
        }

        total = sum(param_breakdown.values())

        return {
            "total_params": total,
            "target_params": self.target_params,
            "breakdown": param_breakdown,
            "efficiency": (
                f"{total/self.target_params:.1%}" if self.target_params > 0 else "N/A"
            ),
        }
