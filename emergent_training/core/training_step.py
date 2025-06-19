"""
Handles the detailed logic for a single training step of the EmergentCubeTrainer.
"""

import torch
from torch.cuda.amp import autocast
import logging
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from emergent_training.core.trainer import EmergentCubeTrainer

logger = logging.getLogger(__name__)


def perform_training_step(
    trainer: "EmergentCubeTrainer",
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Performs a single training step, including forward pass, loss calculation,
    backward pass, and optimizer step.

    This function is designed to be called from the EmergentCubeTrainer.
    """
    trainer.train()

    # Move data to the correct device
    question_embeddings = question_embeddings.to(trainer.device)
    answer_embeddings = answer_embeddings.to(trainer.device)

    # The training process is wrapped in a loop for gradient accumulation
    total_loss = 0.0

    # Reset gradients at the start of the accumulation cycle
    trainer.optimizer.zero_grad()

    # Use mixed precision for the forward pass
    with autocast(enabled=trainer.config.mixed_precision):
        # The forward pass now happens inside the trainer
        outputs = trainer(question_embeddings)

        # Prepare targets for the loss function
        targets = {
            "target_surface_embedding": answer_embeddings,
            "answer_embedding": answer_embeddings,
        }

        # The loss calculation also happens inside the trainer's loss_fn
        # We need the internal states from the forward pass for one of the loss components
        internal_states = outputs["final_cube_state"]
        loss_dict = trainer.loss_fn(outputs, targets, internal_states)

        loss = loss_dict["total_loss"]

    # Scale the loss and perform backward pass
    trainer.scaler.scale(loss).backward()

    # Accumulate scaled loss
    total_loss += loss.item()

    # Perform optimizer step after accumulating gradients
    # Unscale gradients before clipping
    trainer.scaler.unscale_(trainer.optimizer)
    torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)

    # Optimizer step
    trainer.scaler.step(trainer.optimizer)

    # Update the scale for next iteration
    trainer.scaler.update()

    # Log metrics
    metrics = {f"loss_{k}": v.item() for k, v in loss_dict.items() if k != "total_loss"}
    metrics["total_loss"] = total_loss

    return metrics
