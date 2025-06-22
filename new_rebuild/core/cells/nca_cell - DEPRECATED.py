#!/usr/bin/env python3
"""
NCA Cell - перенос из Legacy minimal_nca_cell.py
===============================================

Минимальная Neural Cellular Automata клетка для clean архитектуры.
Основано на core/cell_prototype/architectures/minimal_nca_cell.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .base_cell import BaseCell
from ...config import get_project_config
from ...utils.logging import (
    get_logger,
    log_cell_init,
    log_cell_forward,
    log_cell_component_params,
)

logger = get_logger(__name__)


class NCACell(BaseCell):
    """
    Минимальная Neural Cellular Automata клетка

    Перенос из Legacy MinimalNCACell с адаптацией под clean архитектуру:
    - Использует ProjectConfig для параметров
    - Упрощенное логирование
    - Совместимый интерфейс через BaseCell
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        external_input_size: Optional[int] = None,
        activation: Optional[str] = None,
        target_params: Optional[int] = None,
        **kwargs,  # Дополнительные параметры для совместимости
    ):
        """
        NCA клетка с параметрами из ProjectConfig

        Args:
            Все параметры опциональны - берутся из ProjectConfig если не указаны
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_project_config()
        nca_config = config.get_nca_config()

        # Используем параметры из конфигурации если не переданы
        self.state_size = state_size or nca_config["state_size"]
        self.hidden_dim = hidden_dim or nca_config["hidden_dim"]
        self.external_input_size = (
            external_input_size or nca_config["external_input_size"]
        )
        self.neighbor_count = neighbor_count or nca_config["neighbor_count"]
        self.target_params = target_params or nca_config["target_params"]

        # Функция активации
        activation_name = activation or nca_config["activation"]
        if activation_name == "tanh":
            self.activation = nn.Tanh()
        elif activation_name == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # === АРХИТЕКТУРА (копия из Legacy) ===

        # 1. Neighbor weights (learnable aggregation)
        self.neighbor_weights = nn.Parameter(
            torch.ones(self.neighbor_count) / self.neighbor_count
        )

        # 2. Perception layer
        perception_input_size = self.state_size + self.external_input_size
        self.perception = nn.Linear(perception_input_size, self.hidden_dim, bias=False)

        # 3. Update rule
        self.update_rule = nn.Linear(self.hidden_dim, self.state_size, bias=False)

        # 4. NCA update parameters (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Update strength
        self.beta = nn.Parameter(torch.tensor(0.05))  # Neighbor influence

        # Логирование
        if config.debug_mode:
            self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование параметров через централизованную систему"""
        total_params = sum(p.numel() for p in self.parameters())

        # Используем специализированную функцию логирования клеток
        log_cell_init(
            cell_type="NCA",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            hidden_dim=self.hidden_dim,
            neighbor_count=self.neighbor_count,
            external_input_size=self.external_input_size,
        )

        # Детализация по компонентам (если в debug режиме)
        config = get_project_config()
        if config.debug_mode:
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass (копия логики из Legacy)

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size]

        Returns:
            new_state: [batch, state_size]
        """
        batch_size = own_state.shape[0]

        # Логирование forward pass (только в debug режиме)
        config = get_project_config()
        if config.debug_mode:
            input_shapes = {
                "neighbor_states": neighbor_states.shape,
                "own_state": own_state.shape,
            }
            if external_input is not None:
                input_shapes["external_input"] = external_input.shape

            log_cell_forward("NCA", input_shapes)

        # === STEP 1: NEIGHBOR AGGREGATION ===
        if neighbor_states.numel() > 0:
            # Weighted aggregation of neighbors
            weighted_neighbors = torch.einsum(
                "bnc,n->bc", neighbor_states, self.neighbor_weights
            )
        else:
            # No neighbors case
            weighted_neighbors = torch.zeros_like(own_state)

        # === STEP 2: EXTERNAL INPUT HANDLING ===
        if external_input is not None:
            # Используем переданный external_input
            ext_input = external_input
        else:
            # Создаем нулевой external_input
            ext_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # === STEP 3: PERCEPTION ===
        # Объединяем own_state и external_input
        perception_input = torch.cat([own_state, ext_input], dim=-1)

        # Применяем perception layer
        hidden = self.perception(perception_input)
        hidden = self.activation(hidden)

        # === STEP 4: UPDATE RULE ===
        delta = self.update_rule(hidden)

        # === STEP 5: NCA UPDATE ===
        # Комбинируем neighbor influence и собственное обновление
        neighbor_influence = self.beta * weighted_neighbors
        state_update = self.alpha * delta

        # Новое состояние
        new_state = own_state + state_update + neighbor_influence

        return new_state
