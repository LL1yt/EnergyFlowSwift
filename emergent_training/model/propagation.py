"""
Spatial Propagation system for Emergent Training.
"""

import torch
import torch.nn as nn
from typing import Tuple


class EmergentSpatialPropagation(nn.Module):
    """
    Manages spatial propagation of information through the 3D lattice.
    This simplified version uses a series of convolutions to simulate
    information spreading.
    """

    def __init__(
        self, cube_dimensions: Tuple[int, int, int], cell_state_size: int = 32
    ):
        super().__init__()
        # Using a 3D convolution to simulate local communication.
        # Kernel size 3 means each cell looks at its immediate neighbors.
        # Padding 1 keeps the dimensions the same.
        self.propagation_layer = nn.Conv3d(
            in_channels=cell_state_size,
            out_channels=cell_state_size,
            kernel_size=3,
            padding=1,
            groups=cell_state_size,  # Depthwise convolution for efficiency
            bias=False,
        )
        # Simple non-linearity
        self.activation = nn.GELU()

    def forward(self, cube_states: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Performs spatial propagation.

        Args:
            cube_states: The current state of the entire cube, with shape
                         [batch, state_size, depth, height, width].
            depth: The number of propagation steps to perform.

        Returns:
            The updated cube states after propagation.
        """
        propagated_states = cube_states
        for _ in range(depth):
            # The core of propagation is applying the convolutional layer,
            # which mixes information between neighboring cells.
            residual = self.propagation_layer(propagated_states)
            propagated_states = self.activation(
                propagated_states + residual
            )  # Additive skip connection
        return propagated_states
