"""
State management utilities for the EmergentCubeTrainer.
Includes functions for resetting and cleaning up states.
"""

import torch
import logging
import gc

logger = logging.getLogger(__name__)


def full_state_reset(trainer_instance):
    """
    Performs a full, hard reset of the trainer's state.
    This is expensive and should be used sparingly.
    """
    logger.warning("Performing a full state reset. This is an expensive operation.")

    # This requires access to the trainer's internal components.
    # The function now takes the trainer instance as an argument.

    # 1. Reset cell memory
    if hasattr(trainer_instance.cell, "reset_memory"):
        trainer_instance.cell.reset_memory()

    # 2. Reset NCA state if it exists
    if trainer_instance.nca:
        trainer_instance.nca.reset()

    # 3. Reset optimizer state
    trainer_instance.optimizer.state.clear()

    # 4. Reset GradScaler state
    if trainer_instance.scaler:
        trainer_instance.scaler = torch.cuda.amp.GradScaler(
            enabled=trainer_instance.config.mixed_precision
        )

    # 5. Clear CUDA cache and run garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("Full state reset complete.")


def smart_state_reset(trainer_instance):
    """
    Performs a 'smart' reset, only clearing states that are likely
    to cause issues, like gradients and NCA history.
    """
    logger.debug("Performing smart state reset.")

    # This function also needs access to the trainer's state.

    trainer_instance.optimizer.zero_grad(set_to_none=True)

    if trainer_instance.nca:
        if hasattr(trainer_instance.nca, "reset_tracking"):
            trainer_instance.nca.reset_tracking()
        elif hasattr(trainer_instance.nca, "reset"):
            trainer_instance.nca.reset()
        else:
            logger.warning("NCA object has no reset method available")

    if hasattr(trainer_instance.cell, "reset_memory"):
        trainer_instance.cell.reset_memory()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def lightweight_cleanup():
    """A very lightweight cleanup to be run after each step."""
    gc.collect()
