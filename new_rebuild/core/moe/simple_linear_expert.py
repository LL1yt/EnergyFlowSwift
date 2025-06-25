#!/usr/bin/env python3
"""
Optimized Simple Linear Expert - для локальных связей (10%)
=========================================================

Оптимизированный линейный эксперт с фиксированной архитектурой.
Аналогия: рефлексы в нервной системе - быстрая реакция без сложных вычислений.

АРХИТЕКТУРА:
- ФИКСИРОВАННАЯ архитектура независимо от max_neighbors
- Attention-based агрегация для переменного количества соседей
- Динамические веса адаптируются к любому количеству соседей
- Все параметры настраиваются через централизованный конфиг

ПРИНЦИПЫ:
1. Фиксированное количество параметров
2. Адаптивность к переменному количеству соседей
3. Биологическая правдоподобность (рефлексы)
4. Настройка через централизованный конфиг
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward

logger = get_logger(__name__)


class OptimizedSimpleLinearExpert(nn.Module):
    """
    Оптимизированный эксперт с фиксированными параметрами для local connections (10%)

    Фиксированная архитектура независимо от количества соседей.
    Адаптивная агрегация для переменного количества соседей.
    """

    def __init__(self, state_size: int):
        super().__init__()

        config = get_project_config()
        local_config = config.expert.local

        self.state_size = state_size
        self.target_params = local_config.params  # Из конфига

        # === ФИКСИРОВАННАЯ АРХИТЕКТУРА ===

        # 1. Neighbor aggregator - фиксированные размеры из конфига
        self.neighbor_aggregator = nn.Sequential(
            nn.Linear(
                state_size, local_config.neighbor_agg_hidden1, bias=True
            ),  # state_size * hidden1 + hidden1
            nn.GELU(),
            nn.Linear(
                local_config.neighbor_agg_hidden1,
                local_config.neighbor_agg_hidden2,
                bias=True,
            ),  # hidden1 * hidden2 + hidden2
        )

        # 2. State processor - комбинирует состояние + агрегацию
        processor_input_size = state_size + local_config.neighbor_agg_hidden2
        self.state_processor = nn.Sequential(
            nn.Linear(
                processor_input_size, local_config.processor_hidden, bias=True
            ),  # (state_size + hidden2) * processor_hidden + processor_hidden
            nn.GELU(),
            nn.Linear(
                local_config.processor_hidden, state_size, bias=True
            ),  # processor_hidden * state_size + state_size
        )

        # 3. Residual connection parameters из конфига
        self.alpha = nn.Parameter(torch.tensor(local_config.alpha))  # 1 параметр
        self.beta = nn.Parameter(torch.tensor(local_config.beta))  # 1 параметр

        # 4. Нормализация для стабильности
        self.normalization = nn.LayerNorm(state_size, bias=True)

        # 5. Настройки для adaptive агрегации
        self.max_neighbors_buffer = local_config.max_neighbors_buffer
        self.use_attention = local_config.use_attention

        # Подсчет и логирование параметров
        total_params = sum(p.numel() for p in self.parameters())

        log_cell_init(
            cell_type="OptimizedSimpleLinearExpert",
            total_params=total_params,
            target_params=self.target_params,
            state_size=state_size,
            config=local_config,
        )

        logger.info(
            f"OptimizedSimpleLinearExpert: {total_params} параметров "
            f"(архитектура: {local_config.neighbor_agg_hidden1}->{local_config.neighbor_agg_hidden2} | "
            f"{processor_input_size}->{local_config.processor_hidden}->{state_size})"
        )

    def forward(
        self, current_state: torch.Tensor, neighbor_states: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Быстрая обработка локальных соседей с фиксированной архитектурой

        Args:
            current_state: [batch, state_size] - текущее состояние
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей

        Returns:
            new_state: [batch, state_size] - обновленное состояние
        """
        batch_size, num_neighbors, _ = neighbor_states.shape

        if num_neighbors == 0:
            # Нет соседей - возвращаем нормализованное состояние
            return self.normalization(current_state)

        # 1. Адаптивная агрегация соседей
        if self.use_attention and num_neighbors > 1:
            # Attention-based агрегация (независимо от количества соседей)
            attention_weights = F.softmax(
                torch.sum(neighbor_states * current_state.unsqueeze(1), dim=-1), dim=1
            )
            aggregated_neighbors = torch.sum(
                neighbor_states * attention_weights.unsqueeze(-1), dim=1
            )
        else:
            # Простое усреднение для одного соседа или fallback
            aggregated_neighbors = torch.mean(neighbor_states, dim=1)

        # 2. Обработка агрегированных соседей через фиксированную сеть
        neighbor_features = self.neighbor_aggregator(aggregated_neighbors)

        # 3. Объединяем текущее состояние с обработанными соседями
        combined_input = torch.cat([current_state, neighbor_features], dim=-1)

        # 4. Основная обработка через фиксированную архитектуру
        processed = self.state_processor(combined_input)

        # 5. Нормализация для стабильности
        processed = self.normalization(processed)

        # 6. Residual connection с learnable коэффициентами
        new_state = self.alpha * current_state + self.beta * processed

        return new_state

    def get_parameter_info(self) -> Dict[str, Any]:
        """Получить информацию о параметрах эксперта"""
        param_breakdown = {
            "neighbor_aggregator": sum(
                p.numel() for p in self.neighbor_aggregator.parameters()
            ),
            "state_processor": sum(
                p.numel() for p in self.state_processor.parameters()
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
            "architecture": "fixed",
            "adaptive_neighbors": True,
            "use_attention": self.use_attention,
        }


# Backward compatibility alias
SimpleLinearExpert = OptimizedSimpleLinearExpert
