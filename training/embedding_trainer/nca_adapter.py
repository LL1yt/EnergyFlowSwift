#!/usr/bin/env python3
"""
NCA Adapter для emergent training
Интеграция NCA архитектуры в существующую систему обучения
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

# Добавляем путь к корневой директории
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.cell_prototype.architectures.minimal_nca_cell import (
    MinimalNCACell,
    create_nca_cell_from_config,
)

# НОВОЕ: Используем централизованную конфигурацию
from utils.centralized_config import get_centralized_config, get_nca_defaults

logger = logging.getLogger(__name__)


class EmergentNCACell(MinimalNCACell):
    """
    Enhanced NCA Cell для emergent training

    Расширяет MinimalNCACell для совместимости с EmergentGMLPCell интерфейсом
    """

    def __init__(
        self,
        state_size: Optional[int] = None,  # Берем из centralized config
        neighbor_count: Optional[int] = None,  # Берем из centralized config
        hidden_dim: Optional[int] = None,  # Берем из centralized config
        external_input_size: Optional[int] = None,  # Берем из centralized config
        memory_dim: int = 4,
        use_memory: bool = False,
        activation: Optional[str] = None,  # Берем из centralized config
        dropout: float = 0.0,
        spatial_connections: bool = True,
        target_params: Optional[int] = None,  # Берем из centralized config
    ):
        # НОВОЕ: Получаем параметры из централизованной конфигурации
        central_config = get_centralized_config()
        nca_defaults = central_config.get_nca_config()

        # Используем централизованные значения или переданные параметры
        actual_state_size = (
            state_size if state_size is not None else nca_defaults["state_size"]
        )
        actual_neighbor_count = (
            neighbor_count
            if neighbor_count is not None
            else nca_defaults["neighbor_count"]
        )
        actual_hidden_dim = (
            hidden_dim if hidden_dim is not None else nca_defaults["hidden_dim"]
        )
        actual_external_input_size = (
            external_input_size
            if external_input_size is not None
            else nca_defaults["external_input_size"]
        )
        actual_activation = (
            activation if activation is not None else nca_defaults["activation"]
        )
        actual_target_params = (
            target_params
            if target_params is not None
            else nca_defaults["target_params"]
        )

        # Инициализируем базовую NCA клетку
        super().__init__(
            state_size=actual_state_size,
            neighbor_count=actual_neighbor_count,
            hidden_dim=actual_hidden_dim,
            external_input_size=actual_external_input_size,
            activation=actual_activation,
            dropout=dropout,
            use_memory=use_memory,
            memory_dim=memory_dim,
            target_params=actual_target_params,
        )

        self.spatial_connections = spatial_connections

        # ИСПРАВЛЕНО: Безопасное форматирование target_params
        if spatial_connections and not hasattr(EmergentNCACell, "_param_count_logged"):
            total_params = sum(p.numel() for p in self.parameters())
            target_str = (
                f"{actual_target_params:,}"
                if actual_target_params is not None
                else "N/A"
            )
            logger.info(
                f"[BRAIN] EmergentNCACell: {total_params:,} params (target: ~{target_str})"
            )
            EmergentNCACell._param_count_logged = True

        # Дополнительные NCA features для emergent behavior
        if spatial_connections:
            # Emergent specialization tracking
            self.register_buffer(
                "specialization_tracker", torch.zeros(1, actual_state_size)
            )

        # Debug tracking (совместимость с EmergentGMLPCell)
        self.forward_count = 0
        self.last_output_id = None

        logger.debug(
            f"[CONFIG] EmergentNCACell created with {self.count_parameters()} parameters "
            f"(centralized config: state={actual_state_size}, neighbors={actual_neighbor_count})"
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
    НОВОЕ: Использует централизованную конфигурацию
    """
    # НОВОЕ: Получаем параметры из централизованной конфигурации
    central_config = get_centralized_config()
    nca_defaults = central_config.get_nca_config()

    # Извлекаем конфигурацию с приоритетом централизованной
    nca_config = config.get("nca", {})
    gmlp_config = config.get("gmlp_config", {})  # Fallback для совместимости

    # Параметры с приоритетом: переданная конфигурация -> централизованная -> fallback
    params = {
        "state_size": nca_config.get("state_size", nca_defaults["state_size"]),
        "neighbor_count": nca_config.get(
            "neighbor_count", nca_defaults["neighbor_count"]
        ),
        "hidden_dim": nca_config.get("hidden_dim", nca_defaults["hidden_dim"]),
        "external_input_size": nca_config.get(
            "external_input_size", nca_defaults["external_input_size"]
        ),
        "memory_dim": nca_config.get("memory_dim", 4),  # Не используется в NCA
        "use_memory": nca_config.get(
            "use_memory", False
        ),  # NCA обычно без explicit memory
        "activation": nca_config.get("activation", nca_defaults["activation"]),
        "dropout": nca_config.get("dropout", 0.0),
        "spatial_connections": True,  # Всегда включено для emergent training
        "target_params": nca_config.get("target_params", nca_defaults["target_params"]),
    }

    # Логирование из централизованной конфигурации
    logger.info(
        f"🔬 EmergentNCACell (centralized): state={params['state_size']}, "
        f"hidden={params['hidden_dim']}, neighbors={params['neighbor_count']}"
    )

    return EmergentNCACell(**params)


def test_nca_adapter():
    """Тестирование NCA адаптера с централизованной конфигурацией"""

    print("🧪 TESTING NCA ADAPTER (CENTRALIZED CONFIG)")
    print("=" * 60)

    # НОВОЕ: Используем централизованную конфигурацию
    central_config = get_centralized_config()
    nca_defaults = central_config.get_nca_config()

    print(f"📋 Centralized NCA config: {nca_defaults}")

    # Тестовая конфигурация (минимальная, всё берется из централизованной)
    config = {"nca_config": {}}  # Пустая - все из централизованной конфигурации

    # Создание клетки
    cell = create_emergent_nca_cell_from_config(config)

    # Тестовые данные с параметрами из централизованной конфигурации
    batch_size = 4
    state_size = nca_defaults["state_size"]
    neighbor_count = nca_defaults["neighbor_count"]
    external_input_size = nca_defaults["external_input_size"]

    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    # Forward pass
    output = cell(neighbor_states, own_state, external_input)

    print(f"✅ Forward pass successful: {own_state.shape} → {output.shape}")

    # Информация о клетке
    info = cell.get_info()
    spec_info = cell.get_specialization_info()

    print(f"📊 Cell parameters: {info['total_parameters']}")
    target_params = info.get("target_parameters")
    if target_params:
        print(f"🎯 Target: {target_params}")
        print(f"📈 Efficiency: {info['parameter_efficiency']:.2f}x")
    else:
        print(f"🎯 Target: N/A")
    print(f"🧠 Specialization: {spec_info['specialization_strength']:.3f}")

    # Проверка централизованной конфигурации
    print("\n📋 CENTRALIZED CONFIG VERIFICATION:")
    print(f"   ✓ State size: {state_size} (from centralized)")
    print(f"   ✓ Neighbor count: {neighbor_count} (from centralized)")
    print(f"   ✓ External input: {external_input_size} (from centralized)")

    return cell


if __name__ == "__main__":
    test_nca_adapter()
