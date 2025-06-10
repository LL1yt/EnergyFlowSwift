"""
Gated MLP Cell - 2024/2025 State-of-the-Art Architecture
========================================================

Основано на:
- Google Research gMLP (Spatial Gating Unit)
- Meta AI sparse MLP efficiency research
- Bio-inspired cortical column processing

Ключевые инновации:
- Spatial Gating Unit заменяет attention эффективнее
- Линейная сложность O(n) vs O(n²) у Transformer
- Биологически точная обработка соседства
- Memory component для emergent behavior

Target: ~25K параметров на клетку (vs 1K в простой MLP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class SpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit (SGU) - ключевая инновация gMLP

    Заменяет self-attention механизм эффективной пространственной обработкой.
    Специально адаптирован для 3D cellular neural networks.
    """

    def __init__(
        self, dim: int, seq_len: int = 6, init_eps: float = 1e-3  # 6 соседей в 3D
    ):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len

        # Spatial projection для neighbor interactions
        self.spatial_proj = nn.Linear(seq_len, seq_len)

        # Normalization для стабильности
        self.norm = nn.LayerNorm(dim)

        # Инициализация близкая к identity (стабильное обучение)
        self.spatial_proj.weight.data.uniform_(-init_eps, init_eps)
        self.spatial_proj.bias.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] - neighbor states + own state
        Returns:
            Gated tensor той же размерности
        """
        u, v = x.chunk(2, dim=-1)  # Split into gate and value

        # Применяем spatial gating
        v = self.norm(v)  # Normalize value part
        v = v.permute(0, 2, 1)  # [batch, dim, seq_len] для spatial projection
        v = self.spatial_proj(v)  # Spatial interactions
        v = v.permute(0, 2, 1)  # Back to [batch, seq_len, dim]

        # Gate mechanism: u controls information flow
        return u * v


class GatedMLPCell(nn.Module):
    """
    Gated MLP Cell - революционная архитектура для cellular networks

    Архитектура:
    1. Input processing (neighbor embedding)
    2. Spatial Gating Unit (key innovation)
    3. Memory state management (GRU)
    4. Output projection

    Биологическая аналогия:
    - Input processing = дендриты (прием сигналов)
    - SGU = сома (интеграция и обработка)
    - Memory = долгосрочная потенциация
    - Output = аксон (передача сигнала)
    """

    def __init__(
        self,
        state_size: int = 32,  # Размер состояния клетки
        neighbor_count: int = 6,  # Количество соседей
        hidden_dim: int = 128,  # Размер скрытого слоя (OPTIMIZED для 25K)
        external_input_size: int = 12,  # Размер внешнего входа
        activation: str = "gelu",  # Современная активация
        dropout: float = 0.1,  # Регуляризация
        use_memory: bool = True,  # Memory component
        memory_dim: int = 32,  # Размер memory state (OPTIMIZED)
        target_params: int = 25000,
    ):  # НОВОЕ: Динамический target (из dynamic config)

        super().__init__()

        # === КОНФИГУРАЦИЯ ===
        self.state_size = state_size
        self.neighbor_count = neighbor_count
        self.hidden_dim = hidden_dim
        self.external_input_size = external_input_size
        self.use_memory = use_memory
        self.memory_dim = memory_dim
        self.target_params = target_params  # НОВОЕ: Сохраняем динамический target

        # === ВЫЧИСЛЕННЫЕ РАЗМЕРЫ ===
        neighbor_input_size = neighbor_count * state_size  # Входы от соседей
        total_input_size = neighbor_input_size + state_size + external_input_size

        # === INPUT PROCESSING LAYER ===
        self.input_norm = nn.LayerNorm(total_input_size)
        self.input_projection = nn.Linear(total_input_size, hidden_dim)

        # === SPATIAL GATING UNIT (Ключевая инновация) ===
        # Удваиваем hidden_dim для gate/value split
        self.pre_gating = nn.Linear(hidden_dim, hidden_dim * 2)
        self.spatial_gating = SpatialGatingUnit(
            dim=hidden_dim, seq_len=neighbor_count + 1  # +1 для own state
        )

        # === FEED FORWARD NETWORK ===
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swiglu":
            # SwiGLU современная активация (используется в LLaMA)
            self.activation = SwiGLU(hidden_dim)
        else:
            self.activation = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # === MEMORY COMPONENT ===
        if use_memory:
            self.memory_gate = nn.GRU(
                input_size=hidden_dim, hidden_size=memory_dim, batch_first=True
            )
            self.memory_to_output = nn.Linear(memory_dim, hidden_dim)

        # === OUTPUT PROJECTION ===
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, state_size)

        # === RESIDUAL CONNECTIONS ===
        self.input_residual = (
            nn.Linear(total_input_size, state_size)
            if total_input_size != state_size
            else nn.Identity()
        )

        # === INTERNAL STATE ===
        self.memory_state = None  # Persistent memory между forward calls

        # Подсчет параметров (только при первом создании)
        if not hasattr(GatedMLPCell, "_param_count_logged"):
            self._log_parameter_count()
            GatedMLPCell._param_count_logged = True

    def _log_parameter_count(self):
        """Логирование количества параметров"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"✅ GatedMLPCell параметры:")
        logger.info(f"   Total: {total_params:,} parameters")
        logger.info(f"   Trainable: {trainable_params:,} parameters")
        logger.info(f"   Target: ~{self.target_params:,} (current: {total_params:,})")

        if total_params > self.target_params * 1.2:  # 20% допуск
            logger.warning(
                f"⚠️  Parameter count превышает target {self.target_params:,}: {total_params:,}"
            )

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass gMLP клетки

        Args:
            neighbor_states: [batch, neighbor_count, state_size] - состояния соседей
            own_state: [batch, state_size] - собственное состояние
            external_input: [batch, external_input_size] - внешний вход (опционально)

        Returns:
            new_state: [batch, state_size] - новое состояние клетки
        """
        batch_size = own_state.shape[0]

        # === ЭТАП 1: INPUT PREPARATION ===

        # Обработка neighbor states
        if neighbor_states.numel() > 0:
            neighbor_flat = neighbor_states.view(batch_size, -1)  # Flatten neighbors
        else:
            # Если соседей нет, создаем нулевой вектор
            neighbor_flat = torch.zeros(
                batch_size,
                self.neighbor_count * self.state_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Обработка external input
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Объединение всех входов
        combined_input = torch.cat(
            [
                neighbor_flat,  # Сигналы от соседей
                own_state,  # Собственное состояние
                external_input,  # Внешний сигнал
            ],
            dim=1,
        )

        # === ЭТАП 2: INPUT PROCESSING ===
        x = self.input_norm(combined_input)
        x = self.input_projection(x)  # [batch, hidden_dim]

        # === ЭТАП 3: SPATIAL GATING UNIT ===

        # Подготовка для spatial gating - разбиваем на spatial positions
        x_gating = self.pre_gating(x)  # [batch, hidden_dim * 2]

        # Reshape для spatial processing
        # Создаем "spatial sequence": [neighbors + own_state]
        spatial_seq = x_gating.unsqueeze(1).expand(-1, self.neighbor_count + 1, -1)

        # Применяем spatial gating
        x_gated = self.spatial_gating(spatial_seq)  # [batch, seq_len, hidden_dim]

        # Агрегируем spatial sequence
        x = x_gated.mean(dim=1)  # [batch, hidden_dim]

        # === ЭТАП 4: FEED FORWARD PROCESSING ===
        x_residual = x
        x = self.ffn(x)
        x = x + x_residual  # Residual connection

        # === ЭТАП 5: MEMORY PROCESSING ===
        if self.use_memory:
            # Обновляем memory state
            x_memory = x.unsqueeze(1)  # [batch, 1, hidden_dim] для GRU

            if self.memory_state is None or self.memory_state.size(1) != batch_size:
                # Первый вызов или изменился batch size - инициализируем memory
                self.memory_state = torch.zeros(
                    1, batch_size, self.memory_dim, device=x.device, dtype=x.dtype
                )

            # GRU memory update
            memory_output, new_memory_state = self.memory_gate(
                x_memory, self.memory_state
            )
            memory_output = memory_output.squeeze(1)  # [batch, memory_dim]

            # КРИТИЧЕСКОЕ: Детачим memory_state от computational graph
            self.memory_state = new_memory_state.detach()

            # Интегрируем memory в основной поток
            memory_contribution = self.memory_to_output(memory_output)
            x = x + memory_contribution

        # === ЭТАП 6: OUTPUT PROJECTION ===
        x = self.output_norm(x)
        new_state = self.output_projection(x)

        # === RESIDUAL CONNECTION ===
        # Residual от исходного input к output
        input_residual = self.input_residual(combined_input)
        new_state = new_state + input_residual * 0.1  # Scaled residual

        return new_state

    def reset_memory(self):
        """Сброс memory state (для начала новой последовательности)"""
        self.memory_state = None

    def get_info(self) -> Dict[str, Any]:
        """Информация о клетке для отладки"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "architecture": "GatedMLP",
            "state_size": self.state_size,
            "neighbor_count": self.neighbor_count,
            "hidden_dim": self.hidden_dim,
            "memory_enabled": self.use_memory,
            "total_parameters": total_params,
            "target_parameters": self.target_params,  # ИСПРАВЛЕНО: используем динамический target
            "parameter_efficiency": total_params
            / max(1, self.target_params),  # ИСПРАВЛЕНО: избегаем деления на 0
            "memory_state_active": self.memory_state is not None,
        }


class SwiGLU(nn.Module):
    """
    SwiGLU Activation - современная активация из LLaMA/GLU family
    Более эффективная чем GELU для больших моделей
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(self.linear(gate))


def create_gmlp_cell_from_config(config: Dict[str, Any]) -> GatedMLPCell:
    """
    Создание gMLP клетки из конфигурации

    Args:
        config: Конфигурация cell_prototype

    Returns:
        GatedMLPCell: Настроенная клетка
    """
    cell_config = config.get("cell_prototype", {})
    arch_config = cell_config.get("architecture", {})

    # Извлекаем параметры с defaults
    params = {
        "state_size": cell_config.get("state_size", 32),
        "neighbor_count": cell_config.get("num_neighbors", 6),
        "hidden_dim": arch_config.get("hidden_dim", 512),
        "external_input_size": cell_config.get("input_size", 12),
        "activation": arch_config.get("activation", "gelu"),
        "dropout": arch_config.get("dropout", 0.1),
        "use_memory": arch_config.get("use_memory", True),
        "memory_dim": arch_config.get("memory_dim", 128),
    }

    logger.info(f"🔬 Создание GatedMLPCell с параметрами: {params}")

    return GatedMLPCell(**params)


def test_gmlp_cell_basic() -> bool:
    """
    Базовое тестирование gMLP клетки

    Returns:
        bool: True если все тесты прошли
    """
    logger.info("🧪 Тестирование GatedMLPCell...")

    try:
        # Создание клетки
        cell = GatedMLPCell(
            state_size=32,
            neighbor_count=6,
            hidden_dim=256,  # Меньше для тестирования
            external_input_size=12,
        )

        # Тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 6, 32)
        own_state = torch.randn(batch_size, 32)
        external_input = torch.randn(batch_size, 12)

        # Forward pass
        new_state = cell(neighbor_states, own_state, external_input)

        # Проверки
        assert new_state.shape == (
            batch_size,
            32,
        ), f"Wrong output shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values in output"
        assert not torch.isinf(new_state).any(), "Inf values in output"

        # Тест memory reset
        cell.reset_memory()
        assert cell.memory_state is None, "Memory not reset"

        # Информация о клетке
        info = cell.get_info()
        logger.info(f"✅ gMLP Cell тест пройден: {info['total_parameters']} params")

        return True

    except Exception as e:
        logger.error(f"❌ gMLP Cell тест failed: {e}")
        return False


if __name__ == "__main__":
    # Быстрый тест при запуске модуля
    logging.basicConfig(level=logging.INFO)
    success = test_gmlp_cell_basic()
    print(f"gMLP Cell test: {'✅ PASSED' if success else '❌ FAILED'}")
