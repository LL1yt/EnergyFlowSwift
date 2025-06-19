"""
System Validator for Production Training.
"""

import torch
import logging
from emergent_training import EmergentCubeTrainer, EmergentTrainingConfig
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

logger = logging.getLogger(__name__)


def validate_system(model_name: str, device: str) -> bool:
    """Performs a series of checks to validate the system readiness."""
    try:
        logger.info("[VALIDATOR] Starting system validation...")
        # Simplified tests for brevity
        assert (
            torch.cuda.is_available() if "cuda" in device else True
        ), "CUDA check failed."

        adapter = UniversalEmbeddingAdapter(
            input_dim=768, output_dim=225, strategy="hierarchical"
        )
        logger.info("[OK] Universal Adapter initialized.")

        config = EmergentTrainingConfig(teacher_model=model_name)
        trainer = EmergentCubeTrainer(config, device=device)
        logger.info(f"[OK] EmergentCubeTrainer initialized on {trainer.device}")

        logger.info("[SUCCESS] System validation passed.")
        return True
    except Exception as e:
        logger.error(f"[ERROR] System validation failed: {e}", exc_info=True)
        return False
