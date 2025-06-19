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
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11)

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
    nca_config: Optional[NCAConfig] = None

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

        if self.nca_config is None and self.enable_nca:
            self.nca_config = create_nca_config()
