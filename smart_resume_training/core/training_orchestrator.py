"""
Training Orchestrator for Smart Resume Training
"""

import logging
import time
import torch
from typing import Dict, Any, Optional

from .checkpoint_manager import CheckpointManager

# This is a placeholder for the actual training function/script.
# In a real scenario, this would be a more robust module.
from real_llama_training_production import main as run_actual_training

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates the training process, deciding whether to resume or start fresh."""

    def __init__(self, config: Dict, metadata: Dict):
        self.config = config
        self.metadata = metadata
        self.checkpoint_manager = CheckpointManager(config)

    def run(
        self, dataset_limit: Optional[int], additional_epochs: int, **training_kwargs
    ):
        """
        Starts the training process by finding compatible checkpoints and
        deciding the training strategy.
        """
        logger.info("Starting training orchestration...")
        compatible_checkpoints = self.checkpoint_manager.find_compatible_checkpoints()

        if compatible_checkpoints:
            self._resume_from_checkpoint(
                compatible_checkpoints[0],
                dataset_limit,
                additional_epochs,
                **training_kwargs,
            )
        else:
            self._start_fresh_training(
                dataset_limit, additional_epochs, **training_kwargs
            )

    def _start_fresh_training(
        self, dataset_limit: Optional[int], epochs: int, **kwargs
    ):
        """Starts a new training session from scratch."""
        logger.warning("No compatible checkpoints found. Starting fresh training.")

        training_args = self._prepare_training_args(dataset_limit, epochs, **kwargs)

        try:
            run_actual_training(self.config, self.metadata, training_args)
            logger.info("Fresh training session completed.")
        except Exception as e:
            logger.error(f"Fresh training failed: {e}", exc_info=True)

    def _resume_from_checkpoint(
        self,
        checkpoint_info: Dict[str, Any],
        dataset_limit: Optional[int],
        additional_epochs: int,
        **kwargs,
    ):
        """Resumes training from a given checkpoint."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_info['path']}")

        training_args = self._prepare_training_args(
            dataset_limit, additional_epochs, **kwargs
        )
        training_args["resume_from_checkpoint"] = checkpoint_info["path"]

        try:
            # The actual training script/function needs to handle the `resume_from_checkpoint` arg.
            run_actual_training(self.config, self.metadata, training_args)
            logger.info(f"Resumed training session completed.")
        except Exception as e:
            logger.error(f"Resumed training failed: {e}", exc_info=True)

    def _prepare_training_args(
        self, dataset_limit: Optional[int], epochs: int, **kwargs
    ) -> Dict:
        """Prepares the arguments for the training script."""
        args = {
            "dataset_limit": dataset_limit,
            "epochs": epochs,
        }
        args.update(kwargs)
        return args
