"""
The main EmergentCubeTrainer class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional, Dict, Any

from emergent_training.config.config import EmergentTrainingConfig
from emergent_training.model.cell import EmergentGMLPCell
from emergent_training.model.loss import EmergentMultiObjectiveLoss
from emergent_training.model.propagation import EmergentSpatialPropagation
from emergent_training.utils.state_management import (
    smart_state_reset,
    lightweight_cleanup,
)
from emergent_training.core.training_step import perform_training_step

from core.lattice_3d import Lattice3D, LatticeConfig
from training.embedding_trainer.neural_cellular_automata import (
    NeuralCellularAutomata,
    create_nca_config,
)

logger = logging.getLogger(__name__)


class EmergentCubeTrainer(nn.Module):
    """
    Orchestrates the emergent training process for the 3D Cellular Neural Network.
    """

    def __init__(self, config: EmergentTrainingConfig, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._initialize_components()
        self.to(self._device)
        logger.info(f"EmergentCubeTrainer initialized on device: {self.device}")

    @property
    def device(self) -> torch.device:
        return self._device

    def _initialize_components(self):
        """Initializes all sub-modules of the trainer."""
        # Main cell
        self.cell = EmergentGMLPCell(**self.config.gmlp_config).to(self.device)

        # 3D Lattice
        lattice_config = LatticeConfig(dimensions=self.config.cube_dimensions)
        self.lattice = Lattice3D(lattice_config, cell=self.cell, device=self.device)

        # Loss function
        self.loss_fn = EmergentMultiObjectiveLoss(self.config).to(self.device)

        # Spatial propagation module
        self.propagation = EmergentSpatialPropagation(
            self.config.cube_dimensions, self.config.gmlp_config.get("state_size", 32)
        ).to(self.device)

        # NCA module
        if self.config.enable_nca:
            self.nca = NeuralCellularAutomata(self.config.nca_config).to(self.device)
        else:
            self.nca = None

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        # GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)

    def forward(self, surface_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the cube.
        1. Inject surface embedding.
        2. Propagate through the cube.
        3. Extract output surface.
        """
        # Reset state before forward pass
        smart_state_reset(self)

        # 1. Inject input
        cube_states = self._inject_surface_to_cube(surface_embeddings)

        # 2. Propagate
        cube_states = self.propagation(
            cube_states, self.config.spatial_propagation_depth
        )

        # Optional NCA step
        if self.nca:
            cube_states = self.nca.forward(cube_states)

        # 3. Extract output
        output_surface = self._extract_output_surface(cube_states)

        return {"output_surface": output_surface, "final_cube_state": cube_states}

    def _inject_surface_to_cube(self, surface_embeddings: torch.Tensor) -> torch.Tensor:
        """Projects and injects the flat surface embedding onto the cube's front face."""
        # This is a placeholder for the complex logic of mapping a flat embedding
        # to the 3D surface of the cube.
        batch_size = surface_embeddings.shape[0]
        D, H, W = self.config.cube_dimensions
        state_size = self.config.gmlp_config.get("state_size", 32)

        # Simplified: Project to the right total size and reshape
        surface_size = H * W
        projection_layer = nn.Linear(
            surface_embeddings.shape[1], surface_size * state_size
        ).to(self.device)
        projected_surface = projection_layer(surface_embeddings)

        # Reshape to fit the front face
        surface_tensor = projected_surface.view(batch_size, state_size, H, W)

        # Create full cube and place the surface on the front face (depth=0)
        full_cube = torch.zeros(batch_size, state_size, D, H, W, device=self.device)
        full_cube[:, :, 0, :, :] = surface_tensor
        return full_cube

    def _extract_output_surface(self, cube_states: torch.Tensor) -> torch.Tensor:
        """Extracts the back face of the cube and flattens it."""
        # Extracts from the back face (depth=-1)
        back_surface = cube_states[:, :, -1, :, :]
        return back_surface.flatten(start_dim=1)

    def train_step(
        self, question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Delegates the training step to the specialized function."""
        return perform_training_step(self, question_embeddings, answer_embeddings)

    def get_debug_tensors(self) -> Dict[str, torch.Tensor]:
        """Provides key tensors for debugging."""
        return {"lattice_state": self.lattice.states}
