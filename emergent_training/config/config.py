"""
Configuration for Emergent Training
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Assuming NCAConfig is in this path
from training.embedding_trainer.neural_cellular_automata import (
    NCAConfig,
    create_nca_config,
)


@dataclass
class EmergentTrainingConfig:
    """Configuration for emergent training with full cube influence."""

    teacher_model: str = "distilbert-base-uncased"
    teacher_embedding_dim: int = 768
    cube_dimensions: Tuple[int, int, int] = (16, 16, 16)

    enable_full_cube_gradient: bool = True
    spatial_propagation_depth: int = 11
    emergent_specialization: bool = True

    gmlp_config: Optional[Dict[str, Any]] = None
    loss_weights: Optional[Dict[str, float]] = None

    learning_rate: float = 0.001
    batch_size: int = 8
    epochs: int = 15
    warmup_epochs: int = 3

    gradient_balancing: bool = True
    adaptive_loss_weighting: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4

    enable_nca: bool = True
    nca_config: Optional[Any] = None

    def __post_init__(self):
        if self.gmlp_config is None:
            self.gmlp_config = {}  # Start with empty and fill safe defaults

        safe_defaults = {
            "neighbor_count": 6,
            "use_memory": True,
            "activation": "gelu",
            "dropout": 0.1,
            "spatial_connections": True,
        }
        for key, value in safe_defaults.items():
            self.gmlp_config.setdefault(key, value)

        if self.loss_weights is None:
            self.loss_weights = {
                "surface_reconstruction": 0.3,
                "internal_consistency": 0.3,
                "dialogue_similarity": 0.4,
            }

        # Create NCA config if NCA is enabled but config not provided
        if self.enable_nca and self.nca_config is None:
            # Create simple dict-based config compatible with create_emergent_nca_cell_from_config
            self.nca_config = {
                "state_size": self.gmlp_config.get("state_size", 8),
                "neighbor_count": self.gmlp_config.get("neighbor_count", 6),
                "hidden_dim": 4,  # Significantly smaller than gMLP
                "external_input_size": min(
                    self.gmlp_config.get("external_input_size", 12), 2
                ),
                "activation": "tanh",
                "dropout": 0.0,
                "use_memory": False,  # NCA has implicit memory
                "enable_lattice_scaling": False,
            }

    @classmethod
    def from_main_config(cls, main_config: Dict[str, Any]) -> "EmergentTrainingConfig":
        """Create EmergentTrainingConfig from main_config.yaml structure"""

        # Extract relevant sections
        nca_config = main_config.get("nca", {})
        gmlp_config = main_config.get("gmlp", {})
        experimental = main_config.get("experimental", {})
        training_config = main_config.get("training", {})
        lattice_config = main_config.get("lattice", {})

        # Determine if NCA should be enabled
        enable_nca = nca_config.get("enabled", False) or experimental.get(
            "enable_nca", False
        )

        # Create cube dimensions from lattice config
        cube_dimensions = (
            lattice_config.get("xs", 15),
            lattice_config.get("ys", 15),
            lattice_config.get("zs", 11),
        )

        config = cls(
            enable_nca=enable_nca,
            cube_dimensions=cube_dimensions,
            learning_rate=training_config.get("learning_rate", 0.001),
            batch_size=training_config.get("batch_size", 8),
            gmlp_config=gmlp_config.copy() if gmlp_config else None,
        )

        # Override nca_config if provided in main config
        if enable_nca and nca_config:
            config.nca_config = nca_config.copy()

        return config
