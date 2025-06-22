#!/usr/bin/env python3
"""
Hybrid Connection Processor - MoE архитектура для разных типов связей
====================================================================

Mixture of Experts (MoE) для обработки трех типов связей:
- Local connections (10%): SimpleLinear - быстрые рефлексы
- Functional connections (55%): GNN/CNF - основная обработка информации
- Distant connections (35%): LightweightCNF - интуиция и долгосрочная память

ПРИНЦИПЫ:
1. Классификация связей по типам через ConnectionClassifier
2. Применение специализированного эксперта для каждого типа
3. Комбинирование результатов через learnable gating network
4. Настраиваемое количество параметров для CNF (500-4000+)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .lightweight_cnf import LightweightCNF, ConnectionType
from .connection_classifier import ConnectionClassifier, ConnectionCategory
from ..cells.gnn_cell import GNNCell
from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward

logger = get_logger(__name__)


class SimpleLinearExpert(nn.Module):
    """
    Простой линейный эксперт для local connections (10%)

    Быстрая обработка ближайших соседей без сложных вычислений.
    Аналогия: рефлексы в нервной системе.
    """

    def __init__(self, state_size: int):
        super().__init__()

        self.state_size = state_size

        # Минимальный LinearExpert - только необходимое
        self.neighbor_weights = nn.Parameter(torch.ones(10) / 10)  # До 10 local соседей
        self.linear_transform = nn.Linear(state_size * 2, state_size, bias=False)
        self.activation = nn.Tanh()  # Стабильная активация

        # Residual connection
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Малое влияние для стабильности

        logger.info(
            f"SimpleLinearExpert created: {sum(p.numel() for p in self.parameters())} parameters"
        )

    def forward(
        self, current_state: torch.Tensor, neighbor_states: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            current_state: [batch, state_size]
            neighbor_states: [batch, num_neighbors, state_size]
        Returns:
            processed_state: [batch, state_size]
        """
        batch_size = current_state.shape[0]

        if neighbor_states.shape[1] == 0:
            return current_state  # Нет соседей - возвращаем без изменений

        # Простая агрегация соседей
        num_neighbors = min(neighbor_states.shape[1], len(self.neighbor_weights))
        weights = self.neighbor_weights[:num_neighbors]

        # Weighted sum neighbors
        neighbor_influence = torch.einsum(
            "bnc,n->bc", neighbor_states[:, :num_neighbors], weights
        )

        # Linear transformation
        combined_input = torch.cat([current_state, neighbor_influence], dim=-1)
        delta = self.linear_transform(combined_input)
        delta = self.activation(delta)

        # Residual update
        new_state = current_state + self.alpha * delta

        return new_state


class HybridConnectionProcessor(nn.Module):
    """
    MoE архитектура для разных типов связей

    ЭКСПЕРТЫ:
    - LocalExpert (10%): SimpleLinear для рефлексов (~50 params)
    - FunctionalExpert (55%): GNN/CNF для мышления (~4000 params)
    - DistantExpert (35%): LightweightCNF для интуиции (~500-4000 params)

    АРХИТЕКТУРА:
    1. ConnectionClassifier определяет типы связей
    2. Каждый эксперт обрабатывает свои связи
    3. GatingNetwork комбинирует результаты
    4. Fallback к GNN если CNF отключен
    """

    def __init__(
        self,
        state_size: int,
        lattice_dimensions: Tuple[int, int, int],
        neighbor_count: int = 26,
        enable_cnf: Optional[bool] = None,
        neighbor_strategy_config: Optional[Dict] = None,
        cnf_target_params: Optional[int] = None,  # NEW: настраиваемые параметры CNF
        functional_expert_type: str = "gnn",  # "gnn", "cnf", "hybrid"
    ):
        super().__init__()

        config = get_project_config()

        self.state_size = state_size
        self.lattice_dimensions = lattice_dimensions
        self.neighbor_count = neighbor_count
        self.enable_cnf = enable_cnf if enable_cnf is not None else config.enable_cnf
        self.functional_expert_type = functional_expert_type

        # Настраиваемые параметры CNF
        self.cnf_target_params = (
            cnf_target_params or config.cnf_target_params_per_connection
        )

        # Обновленные пропорции: 10%/55%/35%
        if neighbor_strategy_config is None:
            neighbor_strategy_config = {
                "local_tier": 0.1,  # 10% - рефлексы
                "functional_tier": 0.55,  # 55% - мышление (уменьшено)
                "distant_tier": 0.35,  # 35% - интуиция (увеличено)
                "local_grid_cell_size": 8,
            }

        self.strategy_config = neighbor_strategy_config

        # Классификатор связей
        self.connection_classifier = ConnectionClassifier(
            lattice_dimensions=lattice_dimensions,
            state_size=state_size,
            neighbor_strategy_config=neighbor_strategy_config,
        )

        # === ЭКСПЕРТЫ ===

        # 1. Local Expert (10%) - простые рефлексы
        self.local_expert = SimpleLinearExpert(state_size)

        # 2. Functional Expert (55%) - основная обработка
        if functional_expert_type == "gnn":
            self.functional_expert = GNNCell(
                state_size=state_size,
                neighbor_count=neighbor_count,
                message_dim=min(16, state_size // 2),
                hidden_dim=min(24, state_size // 2 + 8),
                external_input_size=8,
            )
            self.functional_cnf = None
        elif functional_expert_type == "cnf":
            # Только CNF для functional
            self.functional_expert = LightweightCNF(
                state_size=state_size,
                connection_type=ConnectionType.FUNCTIONAL,
                integration_steps=3,
                adaptive_step_size=True,
                target_params=self.cnf_target_params,
            )
            self.functional_cnf = self.functional_expert
        elif functional_expert_type == "hybrid":
            # Hybrid: GNN + CNF для functional
            self.functional_expert = GNNCell(
                state_size=state_size,
                neighbor_count=neighbor_count,
                message_dim=min(16, state_size // 2),
                hidden_dim=min(24, state_size // 2 + 8),
                external_input_size=8,
            )
            self.functional_cnf = LightweightCNF(
                state_size=state_size,
                connection_type=ConnectionType.FUNCTIONAL,
                integration_steps=3,
                adaptive_step_size=True,
                target_params=self.cnf_target_params,
            )
        else:
            raise ValueError(
                f"Unknown functional_expert_type: {functional_expert_type}"
            )

        # 3. Distant Expert (35%) - интуиция и память
        if self.enable_cnf:
            self.distant_expert = LightweightCNF(
                state_size=state_size,
                connection_type=ConnectionType.DISTANT,
                integration_steps=3,
                adaptive_step_size=True,
                target_params=self.cnf_target_params,
            )
        else:
            # Fallback: все через GNN
            self.distant_expert = GNNCell(
                state_size=state_size,
                neighbor_count=neighbor_count,
                message_dim=min(16, state_size // 2),
                hidden_dim=min(24, state_size // 2 + 8),
                external_input_size=8,
            )

        # === GATING NETWORK ===
        # Learnable веса для комбинации экспертов
        self.gating_network = nn.Sequential(
            nn.Linear(state_size + 3, 3),  # state + 3 expert_info -> 3 weights
            nn.Softmax(dim=-1),
        )

        # Статистика использования экспертов
        self.expert_usage_stats = {
            "local": 0,
            "functional": 0,
            "distant": 0,
            "total_calls": 0,
        }

        # Подсчет параметров экспертов
        local_params = sum(p.numel() for p in self.local_expert.parameters())
        functional_params = sum(p.numel() for p in self.functional_expert.parameters())
        distant_params = sum(p.numel() for p in self.distant_expert.parameters())

        total_params = sum(p.numel() for p in self.parameters())
        target_params = 15000  # Примерная цель для MoE: 3 эксперта + gating

        log_cell_init(
            cell_type="HybridConnectionProcessor",
            total_params=total_params,
            target_params=target_params,
            state_size=state_size,
            lattice_dimensions=lattice_dimensions,
            enable_cnf=self.enable_cnf,
            functional_expert_type=functional_expert_type,
            cnf_target_params=self.cnf_target_params,
            local_expert_params=local_params,
            functional_expert_params=functional_params,
            distant_expert_params=distant_params,
            gating_params=sum(p.numel() for p in self.gating_network.parameters()),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        cell_idx: int,
        neighbor_indices: List[int],
        connection_weights: Optional[torch.Tensor] = None,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        MoE forward pass

        Args:
            current_state: [batch, state_size] - состояние текущей клетки
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей
            cell_idx: индекс клетки в решетке
            neighbor_indices: список индексов соседей
            connection_weights: [batch, num_neighbors] - STDP веса (опционально)
            external_input: [batch, external_input_size] - внешний вход

        Returns:
            dict с processed_state и дополнительной информацией
        """
        batch_size = current_state.shape[0]
        num_neighbors = len(neighbor_indices)

        if num_neighbors == 0:
            return {
                "processed_state": current_state,
                "expert_outputs": {},
                "gating_weights": torch.zeros(batch_size, 3),
                "classification_stats": {},
                "expert_usage": self.expert_usage_stats.copy(),
            }

        # === STEP 1: КЛАССИФИКАЦИЯ СВЯЗЕЙ ===
        classified_connections = self.connection_classifier.classify_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            cell_state=current_state[0],  # Берем первый элемент batch
            neighbor_states=neighbor_states[0],  # Берем первый элемент batch
        )

        # === STEP 2: ПОДГОТОВКА ВХОДОВ ДЛЯ ЭКСПЕРТОВ ===
        expert_outputs = {}
        expert_info = torch.zeros(
            batch_size, 3
        )  # Информация о количестве связей каждого типа

        # Local connections (10%)
        local_connections = classified_connections.get(ConnectionCategory.LOCAL, [])
        if local_connections:
            local_indices = [conn.target_idx for conn in local_connections]
            local_mask = torch.tensor(
                [idx in local_indices for idx in neighbor_indices]
            )
            local_neighbor_states = neighbor_states[:, local_mask]

            expert_outputs["local"] = self.local_expert(
                current_state=current_state,
                neighbor_states=local_neighbor_states,
            )
            expert_info[:, 0] = len(local_connections)
            self.expert_usage_stats["local"] += 1

        # Functional connections (55%)
        functional_connections = classified_connections.get(
            ConnectionCategory.FUNCTIONAL, []
        )
        if functional_connections:
            functional_indices = [conn.target_idx for conn in functional_connections]
            functional_mask = torch.tensor(
                [idx in functional_indices for idx in neighbor_indices]
            )
            functional_neighbor_states = neighbor_states[:, functional_mask]
            functional_connection_weights = (
                connection_weights[:, functional_mask]
                if connection_weights is not None
                else None
            )

            if self.functional_expert_type == "gnn":
                # GNN эксперт - нужно передать правильное количество соседей
                # Для GNN создаем версию которая может работать с любым количеством соседей
                expert_outputs["functional"] = self.functional_expert(
                    neighbor_states=functional_neighbor_states,
                    own_state=current_state,
                    external_input=external_input,
                    connection_weights=functional_connection_weights,
                    flexible_neighbor_count=True,  # Флаг для гибкого количества соседей
                )
            elif self.functional_expert_type == "cnf":
                expert_outputs["functional"] = self.functional_expert(
                    current_state=current_state,
                    neighbor_states=functional_neighbor_states,
                    connection_weights=functional_connection_weights,
                )
            elif self.functional_expert_type == "hybrid":
                # Комбинируем GNN и CNF
                gnn_output = self.functional_expert(
                    neighbor_states=functional_neighbor_states,
                    own_state=current_state,
                    external_input=external_input,
                    connection_weights=functional_connection_weights,
                    flexible_neighbor_count=True,  # Флаг для гибкого количества соседей
                )
                cnf_output = self.functional_cnf(
                    current_state=current_state,
                    neighbor_states=functional_neighbor_states,
                    connection_weights=functional_connection_weights,
                )
                # Learnable combination (50/50 for now)
                expert_outputs["functional"] = 0.5 * gnn_output + 0.5 * cnf_output

            expert_info[:, 1] = len(functional_connections)
            self.expert_usage_stats["functional"] += 1

        # Distant connections (35%)
        distant_connections = classified_connections.get(ConnectionCategory.DISTANT, [])
        if distant_connections:
            distant_indices = [conn.target_idx for conn in distant_connections]
            distant_mask = torch.tensor(
                [idx in distant_indices for idx in neighbor_indices]
            )
            distant_neighbor_states = neighbor_states[:, distant_mask]
            distant_connection_weights = (
                connection_weights[:, distant_mask]
                if connection_weights is not None
                else None
            )

            if self.enable_cnf:
                expert_outputs["distant"] = self.distant_expert(
                    current_state=current_state,
                    neighbor_states=distant_neighbor_states,
                    connection_weights=distant_connection_weights,
                )
            else:
                # Fallback к GNN
                expert_outputs["distant"] = self.distant_expert(
                    neighbor_states=distant_neighbor_states,
                    own_state=current_state,
                    external_input=external_input,
                    connection_weights=distant_connection_weights,
                    flexible_neighbor_count=True,  # Флаг для гибкого количества соседей
                )
            expert_info[:, 2] = len(distant_connections)
            self.expert_usage_stats["distant"] += 1

        # === STEP 3: GATING NETWORK ===
        # Вычисляем веса для комбинации экспертов
        gating_input = torch.cat([current_state, expert_info], dim=-1)
        gating_weights = self.gating_network(gating_input)  # [batch, 3]

        # === STEP 4: КОМБИНИРОВАНИЕ РЕЗУЛЬТАТОВ ===
        combined_output = current_state.clone()  # Начинаем с исходного состояния

        if "local" in expert_outputs:
            combined_output = combined_output + gating_weights[:, 0:1] * (
                expert_outputs["local"] - current_state
            )

        if "functional" in expert_outputs:
            combined_output = combined_output + gating_weights[:, 1:2] * (
                expert_outputs["functional"] - current_state
            )

        if "distant" in expert_outputs:
            combined_output = combined_output + gating_weights[:, 2:3] * (
                expert_outputs["distant"] - current_state
            )

        # Обновляем статистику
        self.expert_usage_stats["total_calls"] += 1

        # Собираем статистику классификации
        classification_stats = self.connection_classifier.get_classification_stats(
            classified_connections
        )

        return {
            "processed_state": combined_output,
            "expert_outputs": expert_outputs,
            "gating_weights": gating_weights,
            "classification_stats": classification_stats,
            "expert_usage": self.expert_usage_stats.copy(),
        }

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Статистика использования экспертов"""
        total = max(self.expert_usage_stats["total_calls"], 1)

        return {
            "raw_counts": self.expert_usage_stats.copy(),
            "percentages": {
                "local": self.expert_usage_stats["local"] / total * 100,
                "functional": self.expert_usage_stats["functional"] / total * 100,
                "distant": self.expert_usage_stats["distant"] / total * 100,
            },
            "total_calls": total,
            "cnf_enabled": self.enable_cnf,
            "functional_expert_type": self.functional_expert_type,
            "cnf_target_params": self.cnf_target_params,
            "strategy_config": self.strategy_config,
        }

    def reset_stats(self):
        """Сброс статистики использования экспертов"""
        self.expert_usage_stats = {
            "local": 0,
            "functional": 0,
            "distant": 0,
            "total_calls": 0,
        }
