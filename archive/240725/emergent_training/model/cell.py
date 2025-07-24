"""
The Emergent gMLP Cell, the core processing unit of the lattice.
"""

import torch
import torch.nn as nn
from typing import Optional


class EmergentGMLPCell(nn.Module):
    """
    An enhanced GatedMLP cell with spatial connectivity capabilities.
    It processes its own state and the states of its neighbors.
    """

    def __init__(
        self,
        state_size: int = 32,
        hidden_dim: int = 32,
        external_input_size: int = 12,
        memory_dim: int = 16,
        use_memory: bool = True,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_size = state_size
        self.use_memory = use_memory

        # A simple linear layer to process the combined state of neighbors
        self.neighbor_processor = nn.Linear(state_size * 6, hidden_dim)

        # The core gMLP block for processing the cell's own state
        self.gmlp_block = nn.Sequential(
            nn.Linear(state_size + hidden_dim + external_input_size, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_size),
        )

        # Optional memory component
        if self.use_memory:
            self.memory_unit = nn.GRUCell(hidden_dim, memory_dim)
            self.memory_state = None

    def forward(
        self,
        own_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            own_state: The cell's own state. [batch, state_size]
            neighbor_states: The concatenated states of 6 neighbors. [batch, 6 * state_size]
            external_input: External input for this cell. [batch, external_input_size]
        """
        processed_neighbors = self.neighbor_processor(neighbor_states)

        # Concatenate all inputs for the main processing block
        if external_input is None:
            external_input = torch.zeros(
                own_state.shape[0],
                self.gmlp_block[0].in_features
                - self.state_size
                - processed_neighbors.shape[1],
                device=own_state.device,
            )

        gmlp_input = torch.cat([own_state, processed_neighbors, external_input], dim=-1)

        # The new state is calculated through the gMLP block
        new_state = self.gmlp_block(gmlp_input)

        # Update memory state if enabled
        if self.use_memory:
            # The GRUCell needs a 'hidden' state, we can use the processed neighbors
            # or some other derivative of the input.
            if (
                self.memory_state is None
                or self.memory_state.shape[0] != gmlp_input.shape[0]
            ):
                self.memory_state = torch.randn(
                    gmlp_input.shape[0],
                    self.memory_unit.hidden_size,
                    device=gmlp_input.device,
                )

            self.memory_state = self.memory_unit(gmlp_input, self.memory_state)
            # Combine new state with memory
            new_state = new_state + self.memory_state  # Simple additive combination

        return new_state

    def reset_memory(self):
        self.memory_state = None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
