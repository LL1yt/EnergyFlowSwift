#!/usr/bin/env python3
"""
Optimized Gated MLP Cell - целевые 10K параметров
==================================================

Оптимизированная версия GatedMLPCell для эффективной обработки связей:
- Target: ~10,000 параметров (vs 54,892 в оригинале)
- Сохранена память как критический компонент для связей
- Правильное распределение параметров между компонентами
- Биологически правдоподобная обработка соседства

Основано на анализе: текущие 1,888 → масштабирование до 10,000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class OptimizedSpatialGatingUnit(nn.Module):
    """
    Оптимизированная Spatial Gating Unit для эффективной обработки связей
    Урезанная версия оригинальной SGU с focus на параметрическую эффективность
    """

    def __init__(self, dim: int, seq_len: int = 7, init_eps: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Упрощенная spatial projection (основной компонент SGU)
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=True)
        self.norm = nn.LayerNorm(dim)

        # Инициализация для стабильности
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=init_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: [batch, seq_len, dim]
        Returns: [batch, seq_len, dim]
        """
        # Spatial gating
        u, v = x.chunk(2, dim=-1)  # Split на gate и value

        # Spatial transformation
        v = v.transpose(-1, -2)  # [batch, dim, seq_len]
        v = self.spatial_proj(v)  # Spatial mixing
        v = v.transpose(-1, -2)  # [batch, seq_len, dim]

        # Apply gating и normalization
        out = u * v
        out = self.norm(out)

        return out


class OptimizedGatedMLPCell(nn.Module):
    """
    Оптимизированная Gated MLP Cell для обработки связей

    Target: ~10,000 параметров
    Роль: Определение принципов взаимодействия между нейронами

    Архитектура:
    1. Input processing - обработка входных сигналов от соседей
    2. Spatial Gating - key innovation для spatial connectivity
    3. Memory component - критичен для адаптации связей
    4. Output projection - финальная обработка
    """

    def __init__(
        self,
        state_size: int = 36,  # Синхронизировано с NCA
        neighbor_count: int = 26,  # 3D Moore neighborhood
        hidden_dim: int = 48,  # Оптимизировано для 10K params
        external_input_size: int = 8,  # Уменьшено vs 12
        activation: str = "gelu",
        dropout: float = 0.05,  # Уменьшена регуляризация
        use_memory: bool = True,  # КРИТИЧНО для связей
        memory_dim: int = 24,  # Оптимизировано для 10K
        target_params: int = 10000,
    ):
        super().__init__()

        # === КОНФИГУРАЦИЯ ===
        self.state_size = state_size
        self.neighbor_count = neighbor_count
        self.hidden_dim = hidden_dim
        self.external_input_size = external_input_size
        self.use_memory = use_memory
        self.memory_dim = memory_dim
        self.target_params = target_params

        # === ВЫЧИСЛЕННЫЕ РАЗМЕРЫ ===
        neighbor_input_size = neighbor_count * state_size  # 26 * 36 = 936
        total_input_size = (
            neighbor_input_size + state_size + external_input_size
        )  # 936 + 36 + 8 = 980

        # === INPUT PROCESSING LAYER ===
        # Упрощенная обработка входа (основной потребитель параметров)
        self.input_norm = nn.LayerNorm(total_input_size)  # 980*2 = 1,960 params
        self.input_projection = nn.Linear(
            total_input_size, hidden_dim
        )  # 980*48 + 48 = 47,088 params

        # === SPATIAL GATING UNIT (Ключевая инновация) ===
        # Уменьшенная версия для экономии параметров
        self.pre_gating = nn.Linear(
            hidden_dim, hidden_dim * 2
        )  # 48*96 + 96 = 4,704 params
        self.spatial_gating = OptimizedSpatialGatingUnit(
            dim=hidden_dim, seq_len=neighbor_count + 1  # 27 spatial positions
        )

        # === FEED FORWARD NETWORK ===
        # Упрощенная FFN архитектура
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Меньший expansion ratio (1.5x vs 2x)
        ffn_dim = int(hidden_dim * 1.5)  # 48 * 1.5 = 72
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),  # 48*72 + 72 = 3,528 params
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),  # 72*48 + 48 = 3,504 params
        )

        # === MEMORY COMPONENT (КРИТИЧНЫЙ для связей) ===
        if use_memory:
            # GRU для запоминания паттернов взаимодействия
            self.memory_gate = nn.GRU(
                input_size=hidden_dim, hidden_size=memory_dim, batch_first=True
            )  # Approx 48*24*3 + 24*24*3 + 24*6 = ~5,184 params
            self.memory_to_output = nn.Linear(
                memory_dim, hidden_dim
            )  # 24*48 + 48 = 1,200 params

        # === OUTPUT PROJECTION ===
        self.output_norm = nn.LayerNorm(hidden_dim)  # 48*2 = 96 params
        self.output_projection = nn.Linear(
            hidden_dim, state_size
        )  # 48*36 + 36 = 1,764 params

        # === RESIDUAL CONNECTIONS ===
        # Упрощенная residual connection
        if total_input_size != state_size:
            self.input_residual = nn.Linear(
                total_input_size, state_size, bias=False
            )  # 980*36 = 35,280 params
        else:
            self.input_residual = nn.Identity()

        # === INTERNAL STATE ===
        self.memory_state = None

        # Подсчет и логирование параметров
        self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование детального анализа параметров"""
        total_params = sum(p.numel() for p in self.parameters())

        logger.info(f"[OPTIMIZED] OptimizedGatedMLPCell параметры:")
        logger.info(f"   Total: {total_params:,} parameters")
        logger.info(f"   Target: {self.target_params:,} parameters")
        logger.info(f"   Efficiency: {total_params/self.target_params:.3f}x target")

        # Детальный breakdown
        component_params = {}
        for name, param in self.named_parameters():
            component = name.split(".")[0] if "." in name else name
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()

        logger.info(f"[BREAKDOWN] Parameter distribution:")
        for component, count in sorted(
            component_params.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_params) * 100
            logger.info(f"   {component}: {count:,} params ({percentage:.1f}%)")

        if total_params > self.target_params * 1.2:
            logger.warning(
                f"[WARNING] Parameter count exceeds target by {total_params - self.target_params:,}"
            )
        elif total_params < self.target_params * 0.8:
            logger.info(
                f"[UNDER] Parameter count under target by {self.target_params - total_params:,}"
            )
        else:
            logger.info(f"[OK] Parameter count within target range!")

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        connection_weights: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Оптимизированный forward pass для обработки связей

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            connection_weights: [batch, neighbor_count] - веса связей
            external_input: [batch, external_input_size] - опционально

        Returns:
            new_state: [batch, state_size]
        """
        batch_size = own_state.shape[0]

        # === ЭТАП 1: INPUT PREPARATION ===
        # Взвешивание состояний соседей
        weighted_neighbor_states = neighbor_states * connection_weights.unsqueeze(-1)

        # Flatten neighbors
        if weighted_neighbor_states.numel() > 0:
            neighbor_flat = weighted_neighbor_states.view(batch_size, -1)
        else:
            neighbor_flat = torch.zeros(
                batch_size,
                self.neighbor_count * self.state_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # External input handling
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Combine all inputs
        combined_input = torch.cat([neighbor_flat, own_state, external_input], dim=1)

        # === ЭТАП 2: INPUT PROCESSING ===
        x = self.input_norm(combined_input)
        x = self.input_projection(x)  # [batch, hidden_dim]

        # === ЭТАП 3: SPATIAL GATING UNIT ===
        x_gating = self.pre_gating(x)  # [batch, hidden_dim * 2]

        # Reshape для spatial processing
        spatial_seq = x_gating.unsqueeze(1).expand(-1, self.neighbor_count + 1, -1)
        x_gated = self.spatial_gating(spatial_seq)

        # Aggregate spatial sequence
        x = x_gated.mean(dim=1)  # [batch, hidden_dim]

        # === ЭТАП 4: FEED FORWARD ===
        x_residual = x
        x = self.ffn(x)
        x = x + x_residual  # Residual connection

        # === ЭТАП 5: MEMORY PROCESSING (КРИТИЧНЫЙ) ===
        if self.use_memory:
            x_memory = x.unsqueeze(1)  # [batch, 1, hidden_dim]

            if self.memory_state is None or self.memory_state.size(1) != batch_size:
                self.memory_state = torch.zeros(
                    1, batch_size, self.memory_dim, device=x.device, dtype=x.dtype
                )

            # Memory update через GRU
            memory_output, new_memory_state = self.memory_gate(
                x_memory, self.memory_state
            )
            memory_output = memory_output.squeeze(1)

            # Detach memory state
            self.memory_state = new_memory_state.detach()

            # Integrate memory
            memory_contribution = self.memory_to_output(memory_output)
            x = x + memory_contribution

        # === ЭТАП 6: OUTPUT ===
        x = self.output_norm(x)
        new_state = self.output_projection(x)

        # === RESIDUAL CONNECTION ===
        input_residual = self.input_residual(combined_input)
        new_state = new_state + input_residual * 0.1  # Scaled residual

        return new_state

    def reset_memory(self):
        """Сброс memory state"""
        self.memory_state = None

    def get_info(self) -> Dict[str, Any]:
        """Информация о клетке"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "architecture": "OptimizedGatedMLP",
            "state_size": self.state_size,
            "neighbor_count": self.neighbor_count,
            "hidden_dim": self.hidden_dim,
            "memory_enabled": self.use_memory,
            "memory_dim": self.memory_dim,
            "total_parameters": total_params,
            "target_parameters": self.target_params,
            "parameter_efficiency": total_params / max(1, self.target_params),
            "memory_state_active": self.memory_state is not None,
            "optimization_level": "10K_target",
        }


def create_optimized_gmlp_cell_from_config(
    config: Dict[str, Any],
) -> OptimizedGatedMLPCell:
    """
    Создание оптимизированной gMLP клетки из конфигурации
    """
    cell_config = config.get("cell_prototype", {})
    arch_config = cell_config.get("architecture", {})

    params = {
        "state_size": cell_config.get("state_size", 36),
        "neighbor_count": cell_config.get("num_neighbors", 26),
        "hidden_dim": arch_config.get("hidden_dim", 48),
        "external_input_size": cell_config.get("input_size", 8),
        "activation": arch_config.get("activation", "gelu"),
        "dropout": arch_config.get("dropout", 0.05),
        "use_memory": arch_config.get("use_memory", True),
        "memory_dim": arch_config.get("memory_dim", 24),
        "target_params": arch_config.get("target_params", 10000),
    }

    logger.info(f"🔬 Creating OptimizedGatedMLPCell: {params}")
    return OptimizedGatedMLPCell(**params)


def test_optimized_gmlp_cell() -> bool:
    """Базовый тест оптимизированной клетки"""
    try:
        cell = OptimizedGatedMLPCell()

        # Test data
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 36)
        own_state = torch.randn(batch_size, 36)
        connection_weights = torch.randn(batch_size, 26)
        external_input = torch.randn(batch_size, 8)

        # Forward pass
        output = cell(neighbor_states, own_state, connection_weights, external_input)

        logger.info(f"[TEST] Forward pass successful: {output.shape}")
        logger.info(f"[TEST] Parameter info: {cell.get_info()}")

        return True

    except Exception as e:
        logger.error(f"[TEST FAILED] {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Optimized GatedMLPCell...")
    success = test_optimized_gmlp_cell()
    print(f"Test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
