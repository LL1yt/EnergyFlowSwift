#!/usr/bin/env python3
"""
NCA Adapter для интеграции в Emergent Training System
Простая замена gMLP → NCA без изменения основной логики
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import sys
import os

# Добавляем путь к корневой директории
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.cell_prototype.architectures.minimal_nca_cell import (
    MinimalNCACell,
    create_nca_cell_from_config,
)

logger = logging.getLogger(__name__)


class EmergentNCACell(MinimalNCACell):
    """
    Enhanced NCA Cell для emergent training

    Расширяет MinimalNCACell для совместимости с EmergentGMLPCell интерфейсом
    """

    def __init__(
        self,
        state_size: int = 8,
        neighbor_count: int = 6,
        hidden_dim: int = 4,
        external_input_size: int = 1,
        memory_dim: int = 4,
        use_memory: bool = False,
        activation: str = "tanh",
        dropout: float = 0.0,
        spatial_connections: bool = True,
        target_params: int = None,
    ):

        # Инициализируем базовую NCA клетку
        super().__init__(
            state_size=state_size,
            neighbor_count=neighbor_count,
            hidden_dim=hidden_dim,
            external_input_size=external_input_size,
            activation=activation,
            dropout=dropout,
            use_memory=use_memory,
            memory_dim=memory_dim,
            target_params=target_params,
        )

        self.spatial_connections = spatial_connections

        # Логирование только один раз для всех cells
        if spatial_connections and not hasattr(EmergentNCACell, "_param_count_logged"):
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(
                f"[BRAIN] EmergentNCACell: {total_params:,} params (target: ~{target_params:,})"
            )
            EmergentNCACell._param_count_logged = True

        # Дополнительные NCA features для emergent behavior
        if spatial_connections:
            # Emergent specialization tracking
            self.register_buffer("specialization_tracker", torch.zeros(1, state_size))

        # Debug tracking (совместимость с EmergentGMLPCell)
        self.forward_count = 0
        self.last_output_id = None

        logger.debug(
            f"[CONFIG] EmergentNCACell created with {self.count_parameters()} parameters"
        )

    def count_parameters(self) -> int:
        """Count total parameters в cell (совместимость)"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enhanced forward pass с emergent features
        """
        # Базовый NCA forward pass
        new_state = super().forward(neighbor_states, own_state, external_input)

        # Emergent specialization tracking
        if self.spatial_connections and hasattr(self, "specialization_tracker"):
            # Простое отслеживание специализации через среднее состояние
            avg_state = new_state.mean(dim=0, keepdim=True)
            self.specialization_tracker = (
                0.9 * self.specialization_tracker + 0.1 * avg_state.detach()
            )

        # Debug tracking
        self.forward_count += 1

        return new_state

    def get_specialization_info(self) -> Dict[str, Any]:
        """Информация о специализации клетки"""
        if hasattr(self, "specialization_tracker"):
            spec_norm = self.specialization_tracker.norm().item()
            return {
                "specialization_strength": spec_norm,
                "forward_count": self.forward_count,
                "spatial_connections": self.spatial_connections,
            }
        return {"specialization_strength": 0.0}


def create_emergent_nca_cell_from_config(config: Dict[str, Any]) -> EmergentNCACell:
    """
    Создание EmergentNCACell из конфигурации
    Drop-in replacement для create EmergentGMLPCell
    """
    # Извлекаем конфигурацию
    nca_config = config.get("nca", {})
    gmlp_config = config.get("gmlp_config", {})  # Fallback для совместимости

    # Параметры с приоритетом NCA конфигурации
    params = {
        "state_size": nca_config.get("state_size", gmlp_config.get("state_size", 8)),
        "neighbor_count": nca_config.get(
            "neighbor_count", gmlp_config.get("neighbor_count", 6)
        ),
        "hidden_dim": nca_config.get("hidden_dim", 4),  # Значительно меньше чем gMLP
        "external_input_size": nca_config.get(
            "external_input_size", gmlp_config.get("external_input_size", 1)
        ),
        "memory_dim": nca_config.get("memory_dim", 4),  # Не используется в NCA
        "use_memory": nca_config.get(
            "use_memory", False
        ),  # NCA обычно без explicit memory
        "activation": nca_config.get(
            "activation", gmlp_config.get("activation", "tanh")
        ),
        "dropout": nca_config.get("dropout", 0.0),
        "spatial_connections": True,  # Всегда включено для emergent training
        "target_params": nca_config.get(
            "target_params", gmlp_config.get("target_params", None)
        ),
    }

    # Убираем спам логирования для каждой клетки
    logger.debug(f"🔬 Создание EmergentNCACell с параметрами: {params}")

    return EmergentNCACell(**params)


def test_nca_adapter():
    """Тестирование NCA адаптера"""

    print("🧪 TESTING NCA ADAPTER")
    print("=" * 50)

    # Тестовая конфигурация
    config = {
        "nca_config": {
            "state_size": 8,
            "neighbor_count": 6,
            "hidden_dim": 4,
            "external_input_size": 1,
            "target_params": None,
            "activation": "tanh",
        }
    }

    # Создание клетки
    cell = create_emergent_nca_cell_from_config(config)

    # Тестовые данные
    batch_size = 4
    neighbor_states = torch.randn(batch_size, 6, 8)
    own_state = torch.randn(batch_size, 8)
    external_input = torch.randn(batch_size, 1)

    # Forward pass
    output = cell(neighbor_states, own_state, external_input)

    print(f"✅ Forward pass successful: {own_state.shape} → {output.shape}")

    # Информация о клетке
    info = cell.get_info()
    spec_info = cell.get_specialization_info()

    print(f"📊 Cell parameters: {info['total_parameters']}")
    print(f"🎯 Target: {info['target_parameters']}")
    print(f"📈 Efficiency: {info['parameter_efficiency']:.2f}x")
    print(f"🧠 Specialization: {spec_info['specialization_strength']:.3f}")

    # Сравнение с gMLP
    gmlp_params = 1888  # Из предыдущего анализа
    reduction = ((gmlp_params - info["total_parameters"]) / gmlp_params) * 100
    print(f"🔥 Parameter reduction vs gMLP: {reduction:.1f}%")

    return cell


if __name__ == "__main__":
    test_nca_adapter()
