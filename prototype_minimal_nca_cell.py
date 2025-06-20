#!/usr/bin/env python3
"""
Прототип минимальной NCA клетки на основе исследования
Neural Cellular Automata для замены gMLP архитектуры

Цель: 68-300 параметров вместо 1,888 в gMLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MinimalNCACell(nn.Module):
    """
    Минимальная Neural Cellular Automata клетка

    Основа: μNCA (68 параметров) + адаптация для нашего проекта

    Архитектура:
    1. Perception: агрегация соседей + собственное состояние
    2. Update rule: простое нелинейное преобразование
    3. State update: прямое обновление состояния

    NO complex gating, NO memory, NO multiple layers
    """

    def __init__(
        self,
        state_size: int = 8,
        neighbor_count: int = 6,
        hidden_channels: int = 4,
        external_input_size: int = 1,
        target_params: int = 150,
    ):

        super().__init__()

        self.state_size = state_size
        self.neighbor_count = neighbor_count
        self.hidden_channels = hidden_channels
        self.external_input_size = external_input_size
        self.target_params = target_params

        # === MINIMAL ARCHITECTURE ===

        # 1. Perception: простая агрегация входов
        perception_input_size = state_size + external_input_size  # own_state + external
        self.perception = nn.Linear(perception_input_size, hidden_channels, bias=False)

        # 2. Update rule: минимальная нелинейность
        self.update_rule = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.Tanh(),  # Bounded activation важна для stability
            nn.Linear(hidden_channels, state_size, bias=False),
        )

        # 3. Neighbor weighting (опционально)
        self.neighbor_weights = nn.Parameter(
            torch.ones(neighbor_count) / neighbor_count
        )

        # Подсчет и логирование параметров
        self._log_parameters()

    def _log_parameters(self):
        """Подсчет и анализ параметров"""
        total_params = sum(p.numel() for p in self.parameters())

        print(f"=== MINIMAL NCA CELL PARAMETERS ===")
        print(f"Target: {self.target_params} parameters")
        print(f"Actual: {total_params} parameters")
        print(f"Ratio: {total_params/self.target_params:.2f}x")

        print(f"\nDETAILED BREAKDOWN:")
        for name, param in self.named_parameters():
            print(f"  {name}: {param.numel()} params, shape: {list(param.shape)}")

        if total_params <= self.target_params:
            print(f"✅ SUCCESS: Within target!")
        elif total_params <= self.target_params * 1.2:
            print(f"⚠️  CLOSE: Within 20% of target")
        else:
            print(
                f"❌ OVER: {total_params - self.target_params} parameters over target"
            )

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Minimal NCA forward pass

        Args:
            neighbor_states: [batch, neighbor_count, state_size]
            own_state: [batch, state_size]
            external_input: [batch, external_input_size] (optional)

        Returns:
            new_state: [batch, state_size]
        """
        batch_size = own_state.shape[0]

        # === STEP 1: NEIGHBOR AGGREGATION ===
        if neighbor_states.numel() > 0:
            # Weighted sum of neighbors (learnable weights)
            weighted_neighbors = torch.einsum(
                "bnc,n->bc", neighbor_states, self.neighbor_weights
            )
        else:
            weighted_neighbors = torch.zeros_like(own_state)

        # === STEP 2: PERCEPTION ===
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                self.external_input_size,
                device=own_state.device,
                dtype=own_state.dtype,
            )

        # Combine inputs (NO heavy projections like in gMLP)
        perception_input = torch.cat([own_state, external_input], dim=1)
        perceived = self.perception(perception_input)

        # === STEP 3: UPDATE RULE ===
        delta = self.update_rule(perceived)

        # === STEP 4: STATE UPDATE ===
        # NCA principle: current_state + small_update + neighbor_influence
        alpha = 0.1  # Learning rate for state updates
        neighbor_influence = 0.05  # Strength of neighbor influence

        new_state = own_state + alpha * delta + neighbor_influence * weighted_neighbors

        return new_state

    def get_info(self) -> Dict[str, Any]:
        """Информация о клетке"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "architecture": "MinimalNCA",
            "state_size": self.state_size,
            "neighbor_count": self.neighbor_count,
            "hidden_channels": self.hidden_channels,
            "total_parameters": total_params,
            "target_parameters": self.target_params,
            "parameter_efficiency": total_params / self.target_params,
            "parameter_reduction_vs_gmlp": f"{((1888 - total_params) / 1888 * 100):.1f}%",
        }


class UltraMinimalNCACell(nn.Module):
    """
    Ультра-минимальная версия для μNCA принципов (~68 параметров)
    """

    def __init__(
        self, state_size: int = 8, neighbor_count: int = 6, target_params: int = 68
    ):

        super().__init__()

        self.state_size = state_size
        self.neighbor_count = neighbor_count
        self.target_params = target_params

        # ULTRA-MINIMAL: только самое необходимое
        # Одна linear layer для update rule
        self.update_rule = nn.Linear(state_size, state_size, bias=False)

        # Learnable neighbor weights
        self.neighbor_weights = nn.Parameter(
            torch.ones(neighbor_count) / neighbor_count
        )

        self._log_parameters()

    def _log_parameters(self):
        """Подсчет параметров ультра-минимальной версии"""
        total_params = sum(p.numel() for p in self.parameters())

        print(f"\n=== ULTRA-MINIMAL NCA CELL ===")
        print(f"Target: {self.target_params} parameters")
        print(f"Actual: {total_params} parameters")
        print(f"Ratio: {total_params/self.target_params:.2f}x")

        for name, param in self.named_parameters():
            print(f"  {name}: {param.numel()} params, shape: {list(param.shape)}")

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ультра-минимальный forward pass"""

        # Simple neighbor aggregation
        if neighbor_states.numel() > 0:
            weighted_neighbors = torch.einsum(
                "bnc,n->bc", neighbor_states, self.neighbor_weights
            )
        else:
            weighted_neighbors = torch.zeros_like(own_state)

        # Minimal update rule
        delta = self.update_rule(own_state)

        # Simple state update
        new_state = own_state + 0.1 * delta + 0.05 * weighted_neighbors

        return new_state


def test_nca_cells():
    """Тестирование NCA клеток и сравнение с target параметрами"""

    print("🧪 TESTING NCA CELLS vs gMLP")
    print("=" * 50)

    # Test data
    batch_size = 4
    state_size = 8
    neighbor_count = 6

    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, 1)

    # Test 1: Minimal NCA (target ~150 params)
    print("\n1. MINIMAL NCA CELL:")
    minimal_cell = MinimalNCACell(
        state_size=state_size,
        neighbor_count=neighbor_count,
        hidden_channels=4,
        external_input_size=1,
        target_params=None,  # Убираем хардкод 150
    )

    output1 = minimal_cell(neighbor_states, own_state, external_input)
    print(f"   Forward pass: {own_state.shape} → {output1.shape}")
    print(f"   Output range: [{output1.min():.3f}, {output1.max():.3f}]")

    # Test 2: Ultra-minimal NCA (target ~68 params)
    print("\n2. ULTRA-MINIMAL NCA CELL:")
    ultra_cell = UltraMinimalNCACell(
        state_size=state_size, neighbor_count=neighbor_count, target_params=68
    )

    output2 = ultra_cell(neighbor_states, own_state)
    print(f"   Forward pass: {own_state.shape} → {output2.shape}")
    print(f"   Output range: [{output2.min():.3f}, {output2.max():.3f}]")

    # Comparison
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"   gMLP (current):     1,888 parameters")
    print(
        f"   Minimal NCA:       {sum(p.numel() for p in minimal_cell.parameters())} parameters"
    )
    print(
        f"   Ultra-minimal NCA: {sum(p.numel() for p in ultra_cell.parameters())} parameters"
    )
    print(f"   Target:            300 parameters")

    # Parameter reduction
    minimal_params = sum(p.numel() for p in minimal_cell.parameters())
    ultra_params = sum(p.numel() for p in ultra_cell.parameters())

    print(f"\n🎯 PARAMETER REDUCTION vs gMLP:")
    print(
        f"   Minimal NCA:       {((1888 - minimal_params) / 1888 * 100):.1f}% reduction"
    )
    print(
        f"   Ultra-minimal NCA: {((1888 - ultra_params) / 1888 * 100):.1f}% reduction"
    )

    return minimal_cell, ultra_cell


def calculate_theoretical_performance():
    """Теоретическая оценка performance impact"""

    print(f"\n🚀 THEORETICAL PERFORMANCE ANALYSIS:")
    print("=" * 50)

    print("Memory usage (approximate):")
    print(f"  gMLP:              1,888 params × 4 bytes = {1888 * 4 / 1024:.1f} KB")
    print(f"  Minimal NCA:       ~100 params × 4 bytes = {100 * 4 / 1024:.2f} KB")
    print(f"  Ultra-minimal NCA: ~68 params × 4 bytes = {68 * 4 / 1024:.2f} KB")

    print(f"\nTraining speed impact:")
    print(f"  Fewer parameters → faster forward/backward pass")
    print(f"  Expected speedup: 2-5x for minimal architectures")

    print(f"\nBiological plausibility:")
    print(f"  ✅ NCA: Based on cellular automata (biological)")
    print(f"  ⚠️  gMLP: Complex internal gating (less biological)")


def test_minimal_nca_cell(
    state_size: int = 4,
    neighbor_count: int = 26,
    hidden_dim: int = 3,
    external_input_size: int = 1,
    target_params: int = None,  # Убираем хардкод 150
    batch_size: int = 4,
    device: str = "cpu",
) -> bool:

    print("🧪 TESTING NCA CELLS vs gMLP")
    print("=" * 50)

    # Test data
    neighbor_states = torch.randn(batch_size, neighbor_count, state_size)
    own_state = torch.randn(batch_size, state_size)
    external_input = torch.randn(batch_size, external_input_size)

    # Test 1: Minimal NCA (target ~150 params)
    print("\n1. MINIMAL NCA CELL:")
    minimal_cell = MinimalNCACell(
        state_size=state_size,
        neighbor_count=neighbor_count,
        hidden_channels=hidden_dim,
        external_input_size=external_input_size,
        target_params=target_params,
    )

    output1 = minimal_cell(neighbor_states, own_state, external_input)
    print(f"   Forward pass: {own_state.shape} → {output1.shape}")
    print(f"   Output range: [{output1.min():.3f}, {output1.max():.3f}]")

    # Test 2: Ultra-minimal NCA (target ~68 params)
    print("\n2. ULTRA-MINIMAL NCA CELL:")
    ultra_cell = UltraMinimalNCACell(
        state_size=state_size, neighbor_count=neighbor_count, target_params=68
    )

    output2 = ultra_cell(neighbor_states, own_state)
    print(f"   Forward pass: {own_state.shape} → {output2.shape}")
    print(f"   Output range: [{output2.min():.3f}, {output2.max():.3f}]")

    # Comparison
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"   gMLP (current):     1,888 parameters")
    print(
        f"   Minimal NCA:       {sum(p.numel() for p in minimal_cell.parameters())} parameters"
    )
    print(
        f"   Ultra-minimal NCA: {sum(p.numel() for p in ultra_cell.parameters())} parameters"
    )
    print(f"   Target:            300 parameters")

    # Parameter reduction
    minimal_params = sum(p.numel() for p in minimal_cell.parameters())
    ultra_params = sum(p.numel() for p in ultra_cell.parameters())

    print(f"\n🎯 PARAMETER REDUCTION vs gMLP:")
    print(
        f"   Minimal NCA:       {((1888 - minimal_params) / 1888 * 100):.1f}% reduction"
    )
    print(
        f"   Ultra-minimal NCA: {((1888 - ultra_params) / 1888 * 100):.1f}% reduction"
    )

    return minimal_cell, ultra_cell


if __name__ == "__main__":
    test_nca_cells()
    calculate_theoretical_performance()

    print(f"\n🎯 CONCLUSION:")
    print("NCA архитектура может достичь target 300 параметров")
    print("с значительным запасом, оставляя место для future enhancements")
    print("\nNext step: Integrate в существующую систему для real testing")
