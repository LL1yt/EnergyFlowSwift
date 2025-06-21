"""
The main EmergentCubeTrainer class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import json

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

# Import NCA cell types
from training.embedding_trainer.nca_adapter import (
    EmergentNCACell,
    create_emergent_nca_cell_from_config,
)
from core.cell_prototype.architectures.minimal_nca_cell import (
    MinimalNCACell,
    create_nca_cell_from_config,
)

logger = logging.getLogger(__name__)


def _get_caller_info():  # Simple version for this file
    """Возвращает информацию о вызывающем коде (файл, строка, функция)."""
    try:
        import inspect

        stack = inspect.stack()
        for frame_info in stack[2:]:
            if frame_info.filename != __file__:
                return f"{frame_info.filename}:{frame_info.lineno} (in {frame_info.function})"
        return "N/A"
    except Exception:
        return "N/A"


class EmergentCubeTrainer(nn.Module):
    """
    Orchestrates the emergent training process for the 3D Cellular Neural Network.
    """

    def __init__(self, config: EmergentTrainingConfig, device: Optional[str] = None):
        super().__init__()

        # --- Enhanced Initialization Logging ---
        caller_info = _get_caller_info()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            # This config object is complex, we need a custom serializer
            config_dict = config.to_dict()
        except Exception:
            config_dict = {"error": "Failed to serialize EmergentTrainingConfig"}

        logger.info(
            f"🚀 INIT EmergentCubeTrainer @ {timestamp}\n"
            f"     FROM: {caller_info}\n"
            f"     WITH_CONFIG: {json.dumps(config_dict, indent=2, default=str)}"
        )
        # --- End of Logging ---

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
        # Choose cell architecture based on configuration
        if self.config.enable_nca:
            logger.info("🔬 Using NCA Cell Architecture")
            if hasattr(self.config, "nca_config") and self.config.nca_config:
                # Use EmergentNCACell for advanced features
                self.cell = create_emergent_nca_cell_from_config(
                    {
                        "nca": (
                            self.config.nca_config.__dict__
                            if hasattr(self.config.nca_config, "__dict__")
                            else self.config.nca_config
                        ),
                        "gmlp_config": self.config.gmlp_config,
                    }
                ).to(self.device)
            else:
                # Fallback to MinimalNCACell with gMLP config compatibility
                self.cell = create_nca_cell_from_config(
                    {"gmlp_config": self.config.gmlp_config}
                ).to(self.device)
        else:
            logger.info("🧠 Using gMLP Cell Architecture")
            self.cell = EmergentGMLPCell(**self.config.gmlp_config).to(self.device)

        # Get state size from the actual cell BEFORE using it
        if hasattr(self.cell, "state_size"):
            state_size = self.cell.state_size
        elif hasattr(self.cell, "original_state_size"):
            state_size = self.cell.original_state_size
        else:
            state_size = self.config.gmlp_config.get("state_size", 32)

        # 3D Lattice with proper cell configuration
        lattice_config = LatticeConfig(dimensions=self.config.cube_dimensions)

        # Pass cell configuration to lattice so it creates the right cell type
        if self.config.enable_nca:
            # ИСПРАВЛЕНО: Создаем правильную структуру конфигурации для minimal_nca_cell
            nca_cell_config = {
                "state_size": state_size,
                "neighbor_count": 26,
                "hidden_dim": (
                    self.config.nca_config.get("hidden_dim", 3)
                    if self.config.nca_config
                    else 3
                ),
                "external_input_size": (
                    self.config.nca_config.get("external_input_size", 1)
                    if self.config.nca_config
                    else 1
                ),
                "activation": (
                    self.config.nca_config.get("activation", "tanh")
                    if self.config.nca_config
                    else "tanh"
                ),
                "target_params": (
                    self.config.nca_config.get("target_params")
                    if hasattr(self.config, "nca_config") and self.config.nca_config
                    else None
                ),
                "dropout": 0.0,
                "use_memory": False,
                "enable_lattice_scaling": False,
            }

            # Create cell_config для NCA в правильном формате
            cell_config = {
                "prototype_name": "minimal_nca_cell",  # Указываем архитектуру
                "minimal_nca_cell": nca_cell_config,  # Конфигурация для архитектуры
            }
        else:
            # Create cell_config for gMLP
            logger.info(
                f"ERROR: мы больше не использыем gMLP для создания клетки нейрона @ \n"
            )
            raise NotImplementedError(
                "gMLP path in EmergentCubeTrainer is disabled. "
                "The only supported architecture is currently 'minimal_nca_cell'. "
                "Enable NCA or implement the new gmlp_opt_connections path."
            )

        lattice_config.cell_config = cell_config
        self.lattice = Lattice3D(lattice_config)

        # Move lattice to device manually since it's not passed in constructor
        self.lattice = self.lattice.to(self.device)

        # Loss function
        self.loss_fn = EmergentMultiObjectiveLoss(self.config).to(self.device)

        # Spatial propagation module
        self.propagation = EmergentSpatialPropagation(
            self.config.cube_dimensions, state_size
        ).to(self.device)

        # NCA module (separate from cell - this is for additional NCA processing)
        if (
            self.config.enable_nca
            and hasattr(self.config, "nca_config")
            and self.config.nca_config
        ):
            # Create proper NCAConfig from dict if needed
            if isinstance(self.config.nca_config, dict):
                # Extract only the parameters that create_nca_config accepts
                nca_params = {}
                if "update_probability" in self.config.nca_config:
                    nca_params["update_probability"] = self.config.nca_config[
                        "update_probability"
                    ]
                if "residual_learning_rate" in self.config.nca_config:
                    nca_params["residual_learning_rate"] = self.config.nca_config[
                        "residual_learning_rate"
                    ]
                if "pattern_detection_enabled" in self.config.nca_config:
                    nca_params["enable_pattern_detection"] = self.config.nca_config[
                        "pattern_detection_enabled"
                    ]

                nca_config_obj = create_nca_config(**nca_params)
            else:
                nca_config_obj = self.config.nca_config

            self.nca = NeuralCellularAutomata(
                nca_config_obj,
                cube_dimensions=self.config.cube_dimensions,
                state_size=state_size,
            ).to(self.device)
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
