"""
Runs a single stage of the production training pipeline.
"""

import torch
import numpy as np
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from production_training.config.training_stages import TrainingStage
from emergent_training import EmergentCubeTrainer

# We'll need to pass the dataset object to this runner
# from training.embedding_trainer.dialogue_dataset import DialogueDataset

logger = logging.getLogger(__name__)


def run_training_stage(
    trainer: EmergentCubeTrainer,
    dataset,  # This would be a DialogueDataset object
    stage: TrainingStage,
    checkpoint_dir: Path,
) -> dict:
    """
    Runs a full training stage, including all epochs and steps.

    Args:
        trainer: The EmergentCubeTrainer instance.
        dataset: The dataset for the stage.
        stage: The configuration for the current stage.
        checkpoint_dir: Path to save checkpoints.

    Returns:
        A dictionary containing the results of the stage.
    """
    logger.info(f"[STAGE RUNNER] Starting stage: {stage.name} - {stage.description}")
    stage_start_time = time.time()

    stage_metrics = {
        "losses": [],
        "similarities": [],
        "epoch_times": [],
        "best_loss": float("inf"),
        "best_similarity": 0.0,
        "converged": False,
        "early_stopped": False,
    }
    patience_counter = 0

    for epoch in range(stage.epochs):
        logger.info(f"  Starting epoch {epoch + 1}/{stage.epochs}...")
        epoch_start = time.time()
        epoch_losses, epoch_similarities = [], []

        for i in range(0, len(dataset), stage.batch_size):
            batch_indices = range(i, min(i + stage.batch_size, len(dataset)))

            questions = torch.stack([dataset[j][0] for j in batch_indices]).to(
                trainer.device
            )
            answers = torch.stack([dataset[j][1] for j in batch_indices]).to(
                trainer.device
            )

            metrics = trainer.train_step(questions, answers)
            epoch_losses.append(metrics.get("total_loss", 0.0))
            # Assuming train_step returns similarity or we derive it
            # For now, placeholder:
            epoch_similarities.append(
                metrics.get("dialogue_loss", 1.0)
            )  # Using dialogue_loss as inverse similarity

        # Process epoch results
        epoch_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
        epoch_similarity = 1.0 - (
            np.mean(epoch_similarities) if epoch_similarities else 1.0
        )

        stage_metrics["losses"].append(epoch_loss)
        stage_metrics["similarities"].append(epoch_similarity)
        stage_metrics["epoch_times"].append(time.time() - epoch_start)

        logger.info(
            f"    [METRICS] Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Similarity={epoch_similarity:.4f}"
        )

        # Update best metrics and check for early stopping
        if epoch_loss < stage_metrics["best_loss"]:
            stage_metrics["best_loss"] = epoch_loss
            patience_counter = 0
            if stage.save_checkpoints:
                _save_checkpoint(trainer, stage, epoch + 1, epoch_loss, checkpoint_dir)
        else:
            patience_counter += 1

        if epoch_similarity > stage_metrics["best_similarity"]:
            stage_metrics["best_similarity"] = epoch_similarity

        if patience_counter >= stage.early_stopping_patience:
            logger.info(f"    [INFO] Early stopping triggered at epoch {epoch + 1}.")
            stage_metrics["early_stopped"] = True
            break

        if (
            epoch_loss <= stage.target_loss
            and epoch_similarity >= stage.target_similarity
        ):
            logger.info(f"    [SUCCESS] Stage targets achieved at epoch {epoch + 1}!")
            stage_metrics["converged"] = True
            break

    # Finalize stage results
    success = (
        stage_metrics["best_loss"] <= stage.target_loss
        or stage_metrics["best_similarity"] >= stage.target_similarity
    )

    return {
        "stage": stage.name,
        "success": success,
        "metrics": stage_metrics,
        "config": asdict(stage),
    }


def _save_checkpoint(
    trainer: EmergentCubeTrainer,
    stage: TrainingStage,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
):
    """Saves a model checkpoint."""
    checkpoint_path = checkpoint_dir / f"{stage.name}_epoch_{epoch}_loss_{loss:.4f}.pt"
    checkpoint = {
        "stage": stage.name,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": trainer.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": asdict(stage),
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"  [SAVE] Checkpoint saved: {checkpoint_path}")
