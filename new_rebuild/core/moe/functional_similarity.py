#!/usr/bin/env python3
"""
Functional Similarity Analyzer - анализ функциональной близости
==============================================================

Анализирует функциональную близость между состояниями клеток
для определения функциональных связей в нейронной сети.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)


class FunctionalSimilarityAnalyzer(nn.Module):
    """
    Улучшенный анализатор функциональной близости с batch processing

    Основано на:
    - Cosine similarity состояний
    - Euclidean distance в feature space
    - Learnable weights для комбинирования метрик
    """

    def __init__(self, state_size: int):
        super().__init__()
        self.state_size = state_size

        # Learnable weights для различных метрик similarity
        self.cosine_weight = nn.Parameter(torch.tensor(0.5))
        self.euclidean_weight = nn.Parameter(torch.tensor(0.3))
        self.dot_product_weight = nn.Parameter(torch.tensor(0.2))

        # Дополнительные learnable компоненты для более сложной similarity
        self.similarity_transform = nn.Linear(state_size, state_size // 2, bias=False)

        logger.info(
            f"FunctionalSimilarityAnalyzer initialized: state_size={state_size}"
        )

    def forward(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """
        Batch вычисление функциональной близости

        Args:
            states1: [batch, state_size] - первые состояния
            states2: [batch, state_size] - вторые состояния

        Returns:
            similarities: [batch] - функциональная близость [0, 1]
        """
        # 1. Cosine similarity
        cosine_sim = F.cosine_similarity(states1, states2, dim=-1)
        cosine_sim = (cosine_sim + 1) / 2  # Нормализация к [0, 1]

        # 2. Euclidean similarity в оригинальном пространстве
        euclidean_dist = torch.norm(states1 - states2, dim=-1)

        # Это значение должно быть скаляром
        reference_vector = torch.ones(self.state_size, device=states1.device)
        max_possible_dist = torch.norm(reference_vector)

        euclidean_sim = 1.0 - (euclidean_dist / (max_possible_dist + 1e-8))
        euclidean_sim = torch.clamp(euclidean_sim, 0, 1)

        # 3. Dot product similarity в трансформированном пространстве
        transformed1 = self.similarity_transform(states1)
        transformed2 = self.similarity_transform(states2)
        dot_product = torch.sum(transformed1 * transformed2, dim=-1)
        dot_product_sim = torch.sigmoid(dot_product)  # Нормализация к [0, 1]

        # 4. Взвешенная комбинация
        total_similarity = (
            self.cosine_weight * cosine_sim
            + self.euclidean_weight * euclidean_sim
            + self.dot_product_weight * dot_product_sim
        )

        return torch.clamp(total_similarity, 0, 1)

    def single_similarity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Единичное вычисление similarity (backward compatibility)"""
        with torch.no_grad():
            states1 = state1.unsqueeze(0)  # [1, state_size]
            states2 = state2.unsqueeze(0)  # [1, state_size]
            similarity = self.forward(states1, states2)
            return similarity.item()

    def get_similarity_weights(self) -> dict:
        """Получить текущие веса similarity метрик"""
        return {
            "cosine": self.cosine_weight.item(),
            "euclidean": self.euclidean_weight.item(),
            "dot_product": self.dot_product_weight.item(),
        }
