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
        state_size: int = 8,
        neighbor_count: int = 6,
        hidden_dim: int = 4,  # Значительно меньше чем в gMLP
        external_input_size: int = 1,
        activation: str = "tanh",
        dropout: float = 0.0,  # NCA обычно не нуждается в dropout
        use_memory: bool = False,  # NCA имеет implicit memory
        memory_dim: int = 4,  # Не используется, для совместимости
        target_params: int = 150,
    ):

        super().__init__()

        # Сохраняем параметры для совместимости
        self.neighbor_count = neighbor_count
        self.original_state_size = state_size  # Сохраняем оригинальные значения
        self.original_hidden_dim = hidden_dim
        self.original_external_input_size = external_input_size
        self.use_memory = use_memory
        self.memory_dim = memory_dim
        self.target_params = target_params

        # === ДИНАМИЧЕСКАЯ NCA АРХИТЕКТУРА ===
        # Адаптируем архитектуру под target_params

        # 1. Neighbor weighting (learnable)
        self.neighbor_weights = nn.Parameter(
            torch.ones(neighbor_count) / neighbor_count
        )

        # 2. Динамическое масштабирование ВСЕЙ архитектуры под target_params

        # Масштабируем все размерности на основе target_params
        if target_params >= 1000:
            # Большие модели: увеличиваем все размерности пропорционально
            scale_factor = (
                target_params / 150
            ) ** 0.5  # Квадратный корень для сбалансированного роста
            self.state_size = max(state_size, int(state_size * scale_factor * 0.3))
            self.hidden_dim = max(hidden_dim, int(hidden_dim * scale_factor * 0.6))
            self.external_input_size = max(
                external_input_size, int(external_input_size * scale_factor * 0.2)
            )
        elif target_params <= 100:
            # Маленькие модели: уменьшаем размерности
            self.state_size = max(4, state_size)
            self.hidden_dim = max(2, hidden_dim)
            self.external_input_size = max(1, external_input_size)
        else:
            # Средние модели: используем config значения
            self.state_size = state_size
            self.hidden_dim = hidden_dim
            self.external_input_size = external_input_size

        perception_input_size = self.state_size + self.external_input_size

        logger.info(
            f"[ARCHITECTURE] Scaled for {target_params} params: "
            f"state={self.state_size}, hidden={self.hidden_dim}, input={self.external_input_size}"
        )

        # 3. Perception layer (оптимизированный)
        self.perception = nn.Linear(perception_input_size, self.hidden_dim, bias=False)

        # 4. Update rule (core NCA component)
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.update_rule = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            self.activation,
            nn.Linear(self.hidden_dim, state_size, bias=False),
        )

        # 5. NCA update parameters (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Update strength
        self.beta = nn.Parameter(torch.tensor(0.05))  # Neighbor influence

        # Логирование параметров только при первом создании
        if not hasattr(MinimalNCACell, "_param_count_logged"):
            self._log_parameter_count()
            MinimalNCACell._param_count_logged = True

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
        delta = self.update_rule(perceived)

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

        # Проверяем, была ли использована оптимизация архитектуры
        architecture_optimized = (
            self.state_size != self.original_state_size
            or self.hidden_dim != self.original_hidden_dim
            or self.external_input_size != self.original_external_input_size
        )

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
            "architecture_optimized": architecture_optimized,
            "original_dimensions": {
                "state_size": self.original_state_size,
                "hidden_dim": self.original_hidden_dim,
                "external_input_size": self.original_external_input_size,
            },
        }


def create_nca_cell_from_config(config: Dict[str, Any]) -> MinimalNCACell:
    """
    Создание NCA клетки из конфигурации (аналог create_gmlp_cell_from_config)

    Args:
        config: Конфигурация с поддержкой как gMLP так и NCA параметров

    Returns:
        MinimalNCACell: Настроенная клетка
    """
    # Извлекаем из gmlp_config (для обратной совместимости)
    gmlp_config = config.get("gmlp_config", {})

    # NCA specific config
    nca_config = config.get("nca_config", {})

    # Параметры с fallback на gMLP значения
    params = {
        "state_size": gmlp_config.get("state_size", nca_config.get("state_size", 8)),
        "neighbor_count": gmlp_config.get(
            "neighbor_count", nca_config.get("neighbor_count", 6)
        ),
        "hidden_dim": nca_config.get("hidden_dim", 4),  # Значительно меньше чем gMLP
        "external_input_size": gmlp_config.get(
            "external_input_size", nca_config.get("external_input_size", 1)
        ),
        "activation": gmlp_config.get(
            "activation", nca_config.get("activation", "tanh")
        ),
        "dropout": nca_config.get("dropout", 0.0),  # NCA обычно без dropout
        "target_params": gmlp_config.get(
            "target_params", nca_config.get("target_params", 150)
        ),
    }

    logger.info(f"🔬 Создание MinimalNCACell с параметрами: {params}")

    return MinimalNCACell(**params)


def test_nca_cell_basic() -> bool:
    """
    Базовое тестирование NCA клетки (аналог test_gmlp_cell_basic)

    Returns:
        bool: True если все тесты прошли
    """
    logger.info("🧪 Тестирование MinimalNCACell...")

    try:
        # Создание клетки
        cell = MinimalNCACell(
            state_size=8,
            neighbor_count=6,
            hidden_dim=4,
            external_input_size=1,
            target_params=150,
        )

        # Тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 6, 8)
        own_state = torch.randn(batch_size, 8)
        external_input = torch.randn(batch_size, 1)

        # Forward pass
        new_state = cell(neighbor_states, own_state, external_input)

        # Проверки
        assert new_state.shape == (
            batch_size,
            8,
        ), f"Wrong output shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values in output"
        assert not torch.isinf(new_state).any(), "Inf values in output"

        # Тест memory reset (для совместимости)
        cell.reset_memory()

        # Информация о клетке
        info = cell.get_info()
        logger.info(f"[OK] NCA Cell тест пройден: {info['total_parameters']} params")

        # Проверка параметров
        if info["total_parameters"] <= info["target_parameters"]:
            logger.info(f"[SUCCESS] Parameters в рамках target!")

        return True

    except Exception as e:
        logger.error(f"[ERROR] NCA Cell тест failed: {e}")
        return False


# Convenience function для прямой замены gMLP
def create_compatible_nca_cell(**kwargs) -> MinimalNCACell:
    """
    Создает NCA клетку с параметрами совместимыми с GatedMLPCell
    Упрощенный интерфейс для быстрой замены
    """
    # Адаптируем gMLP параметры под NCA
    nca_params = {
        "state_size": kwargs.get("state_size", 8),
        "neighbor_count": kwargs.get("neighbor_count", 6),
        "hidden_dim": min(kwargs.get("hidden_dim", 8), 6),  # Ограничиваем для NCA
        "external_input_size": min(
            kwargs.get("external_input_size", 4), 2
        ),  # Уменьшаем
        "activation": kwargs.get("activation", "tanh"),
        "target_params": kwargs.get("target_params", 150),
    }

    return MinimalNCACell(**nca_params)
