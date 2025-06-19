"""
Configuration for training stages in production training.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingStage:
    """Represents a training stage with its configuration and targets."""

    name: str
    description: str
    dataset_limit: int
    epochs: int
    target_loss: float
    target_similarity: float
    batch_size: int
    learning_rate: float
    save_checkpoints: bool = True
    early_stopping_patience: int = 10


def get_default_training_stages() -> List[TrainingStage]:
    """Returns the predefined list of training stages."""
    return [
        TrainingStage(
            name="validation",
            description="Quick validation of system integrity and basic learning",
            dataset_limit=50,
            epochs=3,
            target_loss=1.0,
            target_similarity=0.15,
            batch_size=2,
            learning_rate=0.001,
        ),
        TrainingStage(
            name="convergence",
            description="Achieve stable convergence on a larger dataset",
            dataset_limit=400,
            epochs=10,
            target_loss=0.5,
            target_similarity=0.25,
            batch_size=4,
            learning_rate=0.0005,
        ),
        TrainingStage(
            name="production",
            description="Production-like training on a large dataset",
            dataset_limit=1000,
            epochs=5,
            target_loss=0.3,
            target_similarity=0.40,
            batch_size=8,
            learning_rate=0.0003,
            early_stopping_patience=15,
        ),
        TrainingStage(
            name="optimization",
            description="Fine-tuning on the full dataset with low learning rate",
            dataset_limit=-1,  # Use full dataset
            epochs=3,
            target_loss=0.25,
            target_similarity=0.45,
            batch_size=6,
            learning_rate=0.0001,
            early_stopping_patience=10,
        ),
    ]
