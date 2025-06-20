#!/usr/bin/env python3
"""
Минимальная NCA клетка - drop-in replacement для GatedMLPCell
Интеграция μNCA принципов в существующую архитектуру проекта
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MinimalNCACell(nn.Module):
    """
    Минимальная Neural Cellular Automata клетка

    Drop-in replacement для GatedMLPCell с:
    - Совместимым интерфейсом
    - 68-300 параметров вместо 1,888
    - μNCA принципами
    - Масштабируемостью через конфигурацию
    """

    def __init__(
        self,
        state_size: int = 4,  # Фиксированный размер для NCA
        neighbor_count: int = 26,  # 3D Moore neighborhood
        hidden_dim: int = 3,  # Минимальный hidden layer
        external_input_size: int = 1,  # Минимальный внешний вход
        activation: str = "tanh",
        dropout: float = 0.0,
        use_memory: bool = False,  # Не используется в NCA
        memory_dim: int = 4,  # Не используется в NCA
        target_params: int = None,  # Теперь только для логирования
        enable_lattice_scaling: bool = False,  # Отключено по умолчанию
    ):
        """
        Минимальная NCA клетка с фиксированной архитектурой

        Args:
            state_size: Размер состояния клетки (фиксированный 4)
            neighbor_count: Количество соседей (26 для 3D)
            hidden_dim: Размер скрытого слоя (фиксированный 3)
            external_input_size: Размер внешнего входа (фиксированный 1)
            activation: Функция активации
            dropout: Не используется в NCA
            use_memory: Не используется в NCA
            memory_dim: Не используется в NCA
            target_params: Только для логирования (не влияет на архитектуру)
            enable_lattice_scaling: Отключено - используем фиксированные размеры
        """
        super().__init__()

        # Фиксированные размеры для NCA (без масштабирования)
        self.state_size = 4  # Всегда фиксированный
        self.hidden_dim = 3  # Всегда фиксированный
        self.external_input_size = 1  # Всегда фиксированный
        self.neighbor_count = neighbor_count  # Единственный параметр масштабирования

        # Сохраняем target_params только для логирования
        self.target_params = target_params if target_params is not None else 100

        # Отключаем сложное масштабирование
        self.enable_lattice_scaling = False

        logger.info(
            f"[NCA-FIXED] Создание MinimalNCA: state=4, hidden=3, input=1, neighbors={neighbor_count}"
        )

        # 1. Neighbor weights (learnable aggregation)
        self.neighbor_weights = nn.Parameter(
            torch.ones(neighbor_count) / neighbor_count
        )

        # 2. Простая архитектура без сложного масштабирования
        perception_input_size = self.state_size + self.external_input_size  # 4 + 1 = 5

        # 3. Perception layer (минимальный)
        self.perception = nn.Linear(perception_input_size, self.hidden_dim, bias=False)

        # 4. Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # 5. Update rule (минимальная NCA архитектура)
        self.update_rule = nn.Linear(self.hidden_dim, self.state_size, bias=False)

        # 6. NCA update parameters (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Update strength
        self.beta = nn.Parameter(torch.tensor(0.05))  # Neighbor influence

        # Логирование параметров
        self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование количества параметров (совместимо с gMLP)"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"[OK] MinimalNCACell параметры:")
        logger.info(f"   Total: {total_params:,} parameters")
        logger.info(f"   Trainable: {trainable_params:,} parameters")
        logger.info(f"   Target: ~{self.target_params:,} (current: {total_params:,})")

        if total_params <= self.target_params:
            logger.info(f"[SUCCESS] Parameter count в рамках target!")
        elif total_params <= self.target_params * 1.2:
            logger.warning(f"[WARNING] Parameter count близко к target")
        else:
            logger.warning(
                f"[WARNING] Parameter count превышает target {self.target_params:,}: {total_params:,}"
            )

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass совместимый с GatedMLPCell

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size]

        Returns:
            new_state: [batch, state_size]
        """
        batch_size = own_state.shape[0]

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
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # === STEP 3: PERCEPTION ===
        perception_input = torch.cat([own_state, external_input], dim=1)
        perceived = self.perception(perception_input)

        # === STEP 4: UPDATE RULE ===
        # Применяем активацию ПЕРЕД update rule для стабильности NCA
        activated = self.activation(perceived)
        delta = self.update_rule(activated)

        # === STEP 5: NCA STATE UPDATE ===
        # Core NCA principle: gradual state evolution
        new_state = own_state + self.alpha * delta + self.beta * weighted_neighbors

        return new_state

    def reset_memory(self):
        """Совместимость с GatedMLPCell интерфейсом"""
        # NCA не имеет explicit memory state, поэтому это no-op
        pass

    def get_info(self) -> Dict[str, Any]:
        """Информация о клетке (совместимо с GatedMLPCell)"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "architecture": "MinimalNCA",
            "state_size": self.state_size,
            "neighbor_count": self.neighbor_count,
            "hidden_dim": self.hidden_dim,
            "external_input_size": self.external_input_size,
            "memory_enabled": False,  # NCA имеет implicit memory
            "total_parameters": total_params,
            "target_parameters": self.target_params,
            "parameter_efficiency": total_params / max(1, self.target_params),
            "memory_state_active": False,
            "nca_alpha": float(self.alpha.item()),
            "nca_beta": float(self.beta.item()),
            "architecture_optimized": False,  # Всегда фиксированная архитектура
            "lattice_scaling_enabled": False,  # Всегда отключено
            "scaling_mode": "fixed",  # Всегда фиксированный режим
        }


def create_nca_cell_from_config(config: Dict[str, Any]) -> MinimalNCACell:
    """
    Создание NCA клетки из конфигурации
    Теперь без сложного масштабирования и хардкодов

    Args:
        config: Конфигурация

    Returns:
        MinimalNCACell: Настроенная клетка с фиксированными размерами
    """
    # Извлекаем конфигурации
    gmlp_config = config.get("gmlp_config", {})
    nca_config = config.get("nca", {})
    minimal_nca_config = config.get("minimal_nca_cell", {})

    # Простые параметры без сложной логики
    params = {
        "neighbor_count": minimal_nca_config.get(
            "neighbor_count",
            gmlp_config.get("neighbor_count", nca_config.get("neighbor_count", 26)),
        ),
        "activation": minimal_nca_config.get(
            "activation",
            gmlp_config.get("activation", nca_config.get("activation", "tanh")),
        ),
        "target_params": minimal_nca_config.get(
            "target_params",
            gmlp_config.get("target_params", nca_config.get("target_params")),
        ),
        # Фиксированные размеры (игнорируем конфигурацию)
        "state_size": 4,
        "hidden_dim": 3,
        "external_input_size": 1,
        "enable_lattice_scaling": False,
    }

    logger.info(f"🔬 Создание MinimalNCACell: neighbors={params['neighbor_count']}")

    return MinimalNCACell(**params)


def test_nca_cell_basic() -> bool:
    """
    Базовое тестирование NCA клетки с новой архитектурой

    Returns:
        bool: True если все тесты прошли
    """
    logger.info("🧪 Тестирование MinimalNCACell...")

    try:
        # Создание клетки с новой архитектурой
        cell = MinimalNCACell(neighbor_count=26)

        # Тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 26, 4)  # state_size=4
        own_state = torch.randn(batch_size, 4)
        external_input = torch.randn(batch_size, 1)  # external_input_size=1

        # Forward pass
        new_state = cell(neighbor_states, own_state, external_input)

        # Проверки
        assert new_state.shape == (
            batch_size,
            4,
        ), f"Wrong output shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values in output"
        assert not torch.isinf(new_state).any(), "Inf values in output"

        # Тест memory reset (для совместимости)
        cell.reset_memory()

        # Информация о клетке
        info = cell.get_info()
        logger.info(f"[OK] NCA Cell тест пройден: {info['total_parameters']} params")

        return True

    except Exception as e:
        logger.error(f"[ERROR] NCA Cell тест failed: {e}")
        return False


# Convenience function для создания NCA
def create_compatible_nca_cell(**kwargs) -> MinimalNCACell:
    """
    Создает NCA клетку с фиксированной архитектурой
    Игнорирует большинство параметров, использует только neighbor_count
    """
    # Используем только neighbor_count, остальное фиксированно
    nca_params = {
        "neighbor_count": kwargs.get("neighbor_count", 26),
        "activation": kwargs.get("activation", "tanh"),
        "target_params": kwargs.get("target_params"),  # Только для логирования
    }

    return MinimalNCACell(**nca_params)
