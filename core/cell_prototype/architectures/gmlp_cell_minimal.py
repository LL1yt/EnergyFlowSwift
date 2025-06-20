#!/usr/bin/env python3
"""
Minimal Gated MLP Cell - целевые 10K параметров БЕЗ memory
===========================================================

Кардинально оптимизированная версия для обработки связей:
- Target: ~10,000 параметров
- БЕЗ локальной памяти (полагаемся на пространственную distributed memory)
- Решена проблема Input Residual через bottleneck architecture
- Shared weights architecture для всех клеток решетки

Философия: Memory = topology + connection_weights, НЕ локальный GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MinimalSpatialGatingUnit(nn.Module):
    """
    Минимальная Spatial Gating Unit
    Ключевая инновация без memory overhead
    """

    def __init__(self, dim: int, seq_len: int = 27):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        # Только критически важные компоненты
        self.spatial_proj = nn.Linear(
            seq_len, seq_len, bias=False
        )  # Без bias для экономии
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Minimal spatial gating"""
        u, v = x.chunk(2, dim=-1)

        # Spatial transformation
        v = v.transpose(-1, -2)
        v = self.spatial_proj(v)
        v = v.transpose(-1, -2)

        # Simple gating
        out = u * torch.sigmoid(v)  # Простой gating vs сложный
        out = self.norm(out)

        return out


class MinimalGatedMLPCell(nn.Module):
    """
    Минимальная Gated MLP Cell для shared weights architecture

    Target: ~10,000 параметров
    Philosophy: Spatial distributed memory > Local memory

    Решение Input Residual проблемы:
    - Bottleneck architecture: high-dim input → low-dim processing → output
    - Смысловое сжатие вместо прямого residual connection
    """

    def __init__(
        self,
        state_size: int = 36,
        neighbor_count: int = 26,
        hidden_dim: int = 32,  # Уменьшено для bottleneck
        bottleneck_dim: int = 16,  # НОВОЕ: промежуточное сжатие
        external_input_size: int = 4,  # Минимальный external input
        activation: str = "gelu",
        dropout: float = 0.0,  # Убираем dropout для экономии
        target_params: int = 10000,
    ):
        super().__init__()

        # === КОНФИГУРАЦИЯ ===
        self.state_size = state_size
        self.neighbor_count = neighbor_count
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.external_input_size = external_input_size
        self.target_params = target_params

        # === ВЫЧИСЛЕННЫЕ РАЗМЕРЫ ===
        neighbor_input_size = neighbor_count * state_size  # 26 * 36 = 936
        total_input_size = neighbor_input_size + state_size + external_input_size  # 980

        # === BOTTLENECK INPUT PROCESSING ===
        # Решение Input Residual проблемы: сжимаем в bottleneck
        self.input_norm = nn.LayerNorm(total_input_size)
        self.input_bottleneck = nn.Linear(
            total_input_size, bottleneck_dim, bias=False
        )  # 980*16 = 15,680
        self.bottleneck_to_hidden = nn.Linear(
            bottleneck_dim, hidden_dim
        )  # 16*32 + 32 = 544

        # === SPATIAL GATING UNIT ===
        self.pre_gating = nn.Linear(
            hidden_dim, hidden_dim * 2, bias=False
        )  # 32*64 = 2,048
        self.spatial_gating = MinimalSpatialGatingUnit(
            dim=hidden_dim, seq_len=neighbor_count + 1  # 27
        )

        # === MINIMAL FFN ===
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Очень компактный FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),  # 32*32 = 1,024
            self.activation,
        )

        # === OUTPUT PROJECTION ===
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, state_size)  # 32*36 + 36 = 1,188

        # === СМЫСЛОВОЙ RESIDUAL (вместо прямого) ===
        # Вместо input_residual используем compressed residual
        self.compressed_residual = nn.Linear(
            bottleneck_dim, state_size, bias=False
        )  # 16*36 = 576

        # Подсчет параметров
        self._log_parameter_count()

    def _log_parameter_count(self):
        """Детальный анализ параметров"""
        total_params = sum(p.numel() for p in self.parameters())

        logger.info(f"[MINIMAL] MinimalGatedMLPCell параметры:")
        logger.info(f"   Total: {total_params:,} parameters")
        logger.info(f"   Target: {self.target_params:,} parameters")
        logger.info(f"   Efficiency: {total_params/self.target_params:.3f}x target")

        # Breakdown по компонентам
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

        # Статус относительно target
        if total_params <= self.target_params * 1.1:  # 10% tolerance
            logger.info(f"[SUCCESS] ✅ Parameter count within target!")
        else:
            excess = total_params - self.target_params
            logger.warning(
                f"[OVER] ⚠️ Exceeds target by {excess:,} ({total_params/self.target_params:.2f}x)"
            )

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        connection_weights: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Minimal forward pass с bottleneck architecture
        """
        batch_size = own_state.shape[0]

        # === ЭТАП 1: INPUT PREPARATION ===
        weighted_neighbor_states = neighbor_states * connection_weights.unsqueeze(-1)

        if weighted_neighbor_states.numel() > 0:
            neighbor_flat = weighted_neighbor_states.view(batch_size, -1)
        else:
            neighbor_flat = torch.zeros(
                batch_size,
                self.neighbor_count * self.state_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Combine inputs
        combined_input = torch.cat([neighbor_flat, own_state, external_input], dim=1)

        # === ЭТАП 2: BOTTLENECK PROCESSING ===
        x = self.input_norm(combined_input)

        # Bottleneck compression (решение Input Residual проблемы)
        x_compressed = self.input_bottleneck(x)  # [batch, bottleneck_dim]
        x = self.bottleneck_to_hidden(x_compressed)  # [batch, hidden_dim]

        # === ЭТАП 3: SPATIAL GATING ===
        x_gating = self.pre_gating(x)  # [batch, hidden_dim * 2]

        # Reshape для spatial processing
        spatial_seq = x_gating.unsqueeze(1).expand(-1, self.neighbor_count + 1, -1)
        x_gated = self.spatial_gating(spatial_seq)

        # Aggregate
        x = x_gated.mean(dim=1)

        # === ЭТАП 4: MINIMAL FFN ===
        x = x + self.ffn(x)  # Simple residual

        # === ЭТАП 5: OUTPUT ===
        x = self.output_norm(x)
        new_state = self.output_projection(x)

        # === СМЫСЛОВОЙ RESIDUAL ===
        # Используем compressed representation для residual
        compressed_residual = self.compressed_residual(x_compressed)
        new_state = new_state + compressed_residual * 0.1

        return new_state

    def reset_memory(self):
        """Compatibility method (no-op, так как нет memory)"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Информация о клетке"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "architecture": "MinimalGatedMLP",
            "state_size": self.state_size,
            "neighbor_count": self.neighbor_count,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "memory_enabled": False,  # Принципиально НЕТ локальной памяти
            "distributed_memory": True,  # Полагаемся на spatial topology
            "total_parameters": total_params,
            "target_parameters": self.target_params,
            "parameter_efficiency": total_params / max(1, self.target_params),
            "optimization_level": "10K_minimal_bottleneck",
        }


def create_minimal_gmlp_cell_from_config(config: Dict[str, Any]) -> MinimalGatedMLPCell:
    """Создание минимальной gMLP клетки из конфигурации"""
    cell_config = config.get("cell_prototype", {})
    arch_config = cell_config.get("architecture", {})

    params = {
        "state_size": cell_config.get("state_size", 36),
        "neighbor_count": cell_config.get("num_neighbors", 26),
        "hidden_dim": arch_config.get("hidden_dim", 32),
        "bottleneck_dim": arch_config.get("bottleneck_dim", 16),
        "external_input_size": cell_config.get("input_size", 4),
        "activation": arch_config.get("activation", "gelu"),
        "dropout": arch_config.get("dropout", 0.0),
        "target_params": arch_config.get("target_params", 10000),
    }

    logger.info(f"🧬 Creating MinimalGatedMLPCell: {params}")
    return MinimalGatedMLPCell(**params)


def test_minimal_gmlp_cell() -> bool:
    """Тест минимальной клетки"""
    try:
        cell = MinimalGatedMLPCell()

        # Test data
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 36)
        own_state = torch.randn(batch_size, 36)
        connection_weights = torch.randn(batch_size, 26)
        external_input = torch.randn(batch_size, 4)

        # Forward pass
        output = cell(neighbor_states, own_state, connection_weights, external_input)

        logger.info(f"[TEST] Forward pass successful: {output.shape}")
        logger.info(f"[TEST] Info: {cell.get_info()}")

        return True

    except Exception as e:
        logger.error(f"[TEST FAILED] {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("🧪 Testing Minimal GatedMLPCell...")
    success = test_minimal_gmlp_cell()
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
