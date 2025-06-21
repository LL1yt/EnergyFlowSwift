#!/usr/bin/env python3
"""
gMLP Cell - перенос из Legacy gmlp_opt_connections.py
===================================================

Optimized Gated MLP Cell без bottleneck архитектуры.
Основано на core/cell_prototype/architectures/gmlp_opt_connections.py
ОПТИМИЗАЦИЯ: убран bottleneck для полноценной производительности
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


class SGUOptimized(nn.Module):
    """
    Оптимизированная Spatial Gating Unit (перенос из Legacy)
    БЕЗ bottleneck ограничений
    """

    def __init__(self, dim: int, seq_len: int = 27):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Spatial projection (как в Legacy)
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial gating (копия из Legacy)"""
        u, v = x.chunk(2, dim=-1)

        # Spatial transformation
        v = v.transpose(-1, -2)
        v = self.spatial_proj(v)
        v = v.transpose(-1, -2)

        # Simple gating
        out = u * torch.sigmoid(v)
        out = self.norm(out)

        return out


class GMLPCell(BaseCell):
    """
    Оптимизированная Gated MLP Cell для clean архитектуры

    ОПТИМИЗАЦИИ относительно Legacy:
    1. ❌ УБРАН bottleneck (bottleneck_dim, input_bottleneck, compressed_residual)
    2. ✅ УВЕЛИЧЕН hidden_dim до 64 (полноценная архитектура)
    3. ✅ ПРЯМОЕ подключение вместо compressed residual
    4. ✅ Использует ProjectConfig для параметров
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        external_input_size: Optional[int] = None,
        activation: Optional[str] = None,
        target_params: Optional[int] = None,
        # Убираем bottleneck параметры
        # bottleneck_dim: не используется
        **kwargs,
    ):
        """
        gMLP клетка с оптимизированной архитектурой

        Args:
            Все параметры опциональны - берутся из ProjectConfig если не указаны
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_project_config()
        gmlp_config = config.get_gmlp_config()

        # Используем параметры из конфигурации если не переданы
        self.state_size = state_size or gmlp_config["state_size"]
        self.neighbor_count = neighbor_count or gmlp_config["neighbor_count"]
        self.hidden_dim = hidden_dim or gmlp_config["hidden_dim"]  # 64 вместо 32
        self.external_input_size = (
            external_input_size or gmlp_config["external_input_size"]
        )
        self.target_params = target_params or gmlp_config["target_params"]

        # Функция активации
        activation_name = activation or gmlp_config["activation"]
        if activation_name == "gelu":
            self.activation = nn.GELU()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # === ОПТИМИЗИРОВАННАЯ АРХИТЕКТУРА ===

        # Размеры входов
        neighbor_input_size = self.neighbor_count * self.state_size
        total_input_size = (
            neighbor_input_size + self.state_size + self.external_input_size
        )

        # 1. INPUT PROCESSING (БЕЗ bottleneck)
        self.input_norm = nn.LayerNorm(total_input_size)
        self.input_projection = nn.Linear(total_input_size, self.hidden_dim, bias=True)

        # 2. SPATIAL GATING UNIT (как в Legacy)
        self.pre_gating = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False)
        self.spatial_gating = SGUOptimized(
            dim=self.hidden_dim, seq_len=self.neighbor_count + 1
        )

        # 3. FFN (увеличенный для полноценной архитектуры)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False),
            self.activation,
            nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False),
        )

        # 4. OUTPUT PROJECTION
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.state_size, bias=True)

        # 5. RESIDUAL CONNECTION (прямой, без compressed)
        self.residual_projection = nn.Linear(
            total_input_size, self.state_size, bias=False
        )

        # Логирование
        if config.debug_mode:
            self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование параметров через централизованную систему"""
        total_params = sum(p.numel() for p in self.parameters())

        # Используем специализированную функцию логирования клеток
        log_cell_init(
            cell_type="gMLP",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            hidden_dim=self.hidden_dim,
            neighbor_count=self.neighbor_count,
            external_input_size=self.external_input_size,
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
        connection_weights: Optional[torch.Tensor] = None,  # Legacy совместимость
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass (оптимизированная версия из Legacy)

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size]
            connection_weights: не используется в clean архитектуре

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

            log_cell_forward("gMLP", input_shapes)

        # === STEP 1: INPUT PREPARATION ===

        # Flatten neighbor states
        neighbor_flat = neighbor_states.view(batch_size, -1)

        # External input handling
        if external_input is not None:
            ext_input = external_input
        else:
            ext_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Concatenate all inputs
        full_input = torch.cat([neighbor_flat, own_state, ext_input], dim=-1)

        # Store for residual connection
        residual_input = full_input

        # === STEP 2: INPUT PROCESSING (БЕЗ bottleneck) ===
        x = self.input_norm(full_input)
        x = self.input_projection(x)

        # === STEP 3: SPATIAL GATING ===
        gating_input = self.pre_gating(x)  # [batch, hidden_dim * 2]

        # Reshape для spatial processing (как в Legacy)
        spatial_seq = gating_input.unsqueeze(1).expand(-1, self.neighbor_count + 1, -1)
        gated = self.spatial_gating(spatial_seq)

        # Aggregate (как в Legacy)
        gated = gated.mean(dim=1)

        # === STEP 4: FFN ===
        processed = self.ffn(gated)
        processed = self.output_norm(processed)

        # === STEP 5: OUTPUT PROJECTION ===
        output = self.output_projection(processed)

        # === STEP 6: RESIDUAL CONNECTION (прямой) ===
        residual = self.residual_projection(residual_input)

        # Final output
        new_state = output + residual

        return new_state
