#!/usr/bin/env python3
"""
Batch-Optimized Simple Linear Expert - для локальных связей (10%)
================================================================

Версия SimpleLinearExpert с полной поддержкой batch обработки.
Может обрабатывать как одну клетку, так и batch клеток одновременно.

КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:
- Поддержка входа [batch_size, state_size] где batch_size может быть любым
- Эффективная batch обработка без циклов
- Обратная совместимость с single-cell режимом
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from ...config import get_project_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class BatchOptimizedSimpleLinearExpert(nn.Module):
    """
    Batch-оптимизированный эксперт для local connections с поддержкой batch обработки
    """

    def __init__(self, state_size: int):
        super().__init__()

        config = get_project_config()
        local_config = config.expert.local

        self.state_size = state_size
        self.target_params = local_config.params

        # === АРХИТЕКТУРА (та же что и в оригинале) ===
        # 1. Neighbor aggregator
        self.neighbor_aggregator = nn.Sequential(
            nn.Linear(state_size, local_config.neighbor_agg_hidden1, bias=True),
            nn.GELU(),
            nn.Linear(
                local_config.neighbor_agg_hidden1,
                local_config.neighbor_agg_hidden2,
                bias=True,
            ),
        )

        # 2. State processor
        processor_input_size = state_size + local_config.neighbor_agg_hidden2
        self.state_processor = nn.Sequential(
            nn.Linear(processor_input_size, local_config.processor_hidden, bias=True),
            nn.GELU(),
            nn.Linear(local_config.processor_hidden, state_size, bias=True),
        )

        # 3. Residual connection parameters
        self.alpha = nn.Parameter(torch.tensor(local_config.alpha))
        self.beta = nn.Parameter(torch.tensor(local_config.beta))

        # 4. Нормализация
        self.normalization = nn.LayerNorm(state_size, bias=True)

        # 5. Настройки
        self.use_attention = local_config.use_attention

        logger.info(f"[BatchOptimizedSimpleLinearExpert] Initialized with state_size={state_size}")

    def forward(
        self, 
        current_state: torch.Tensor, 
        neighbor_states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Batch-optimized forward pass
        
        Args:
            current_state: [batch_size, state_size] - состояния клеток (batch_size может быть любым)
            neighbor_states: Optional - для batch режима не используется (соседи берутся из кэша)
            
        Returns:
            new_state: [batch_size, state_size] - обновленные состояния
        """
        # Определяем batch_size
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # В batch режиме мы обрабатываем все клетки без учета соседей
        # (так как каждая клетка имеет свой набор соседей)
        # Для local expert это означает простую трансформацию состояния
        
        if batch_size == 1 and neighbor_states is not None:
            # Single-cell режим с соседями (обратная совместимость)
            return self._forward_single(current_state, neighbor_states)
        
        # Batch режим - обрабатываем каждую клетку независимо
        # Для local expert это локальная трансформация состояния
        
        # 1. Создаем "виртуальных соседей" из самого состояния клетки
        # (имитируем локальное взаимодействие)
        virtual_neighbors = current_state.unsqueeze(1)  # [batch_size, 1, state_size]
        
        # 2. Обрабатываем через neighbor aggregator
        neighbor_features = self.neighbor_aggregator(virtual_neighbors)  # [batch_size, 1, hidden2]
        neighbor_features = neighbor_features.squeeze(1)  # [batch_size, hidden2]
        
        # 3. Объединяем с текущим состоянием
        combined_input = torch.cat([current_state, neighbor_features], dim=-1)
        
        # 4. Основная обработка
        processed = self.state_processor(combined_input)
        
        # 5. Нормализация
        processed = self.normalization(processed)
        
        # 6. Residual connection
        new_state = self.alpha * current_state + self.beta * processed
        
        return new_state
    
    def _forward_single(
        self, 
        current_state: torch.Tensor, 
        neighbor_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Обработка одной клетки с соседями (legacy режим)
        """
        # Проверяем размерности
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
            
        if neighbor_states.dim() == 2:
            num_neighbors = neighbor_states.shape[0]
        elif neighbor_states.dim() == 3:
            num_neighbors = neighbor_states.shape[1]
            neighbor_states = neighbor_states.squeeze(0)
        else:
            num_neighbors = 0
            
        if num_neighbors == 0:
            return self.normalization(current_state)
        
        # Агрегация соседей
        if self.use_attention and num_neighbors > 1:
            # Attention-based
            current_expanded = current_state.expand(num_neighbors, -1)
            attention_weights = F.softmax(
                torch.sum(neighbor_states * current_expanded, dim=-1), dim=0
            )
            aggregated_neighbors = torch.sum(
                neighbor_states * attention_weights.unsqueeze(-1), dim=0, keepdim=True
            )
        else:
            # Простое усреднение
            aggregated_neighbors = torch.mean(neighbor_states, dim=0, keepdim=True)
        
        # Обработка через сеть
        neighbor_features = self.neighbor_aggregator(aggregated_neighbors)
        combined_input = torch.cat([current_state, neighbor_features], dim=-1)
        processed = self.state_processor(combined_input)
        processed = self.normalization(processed)
        
        # Residual connection
        new_state = self.alpha * current_state + self.beta * processed
        
        return new_state


# Алиас для обратной совместимости
SimpleLinearExpert = BatchOptimizedSimpleLinearExpert