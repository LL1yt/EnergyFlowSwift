"""
Handles saving of training artifacts like model checkpoints and results.
"""

import logging
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResultsSaver:
    """Saves training results and model checkpoints."""

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {self.output_dir}")

    def save(
        self,
        trainer: Any,  # The EmergentCubeTrainer instance
        dataset_size: int,
        epochs: int,
        best_similarity: float,
        total_time: float,
        training_log: List[Dict],
    ):
        """Saves the model checkpoint and a JSON summary of the results."""
        self._save_checkpoint(trainer, epochs, best_similarity)
        self._save_results_summary(
            dataset_size, epochs, best_similarity, total_time, training_log
        )

    def _save_checkpoint(self, trainer: Any, epochs: int, best_similarity: float):
        """Saves the model state as a checkpoint."""
        checkpoint_path = self.output_dir / "final_model_checkpoint.pt"
        try:
            # Prepare metadata for the checkpoint
            metadata = {
                "epochs": epochs,
                "best_similarity": best_similarity,
                "timestamp": datetime.now().isoformat(),
                "config_mode": self.config.get("_metadata", {}).get("mode", "unknown"),
                "config_scale": self.config.get("_metadata", {}).get(
                    "scale_factor", "unknown"
                ),
            }

            # The trainer should expose a way to get the state dict
            state_dict = trainer.get_model_state_dict()

            torch.save(
                {
                    "model_state_dict": state_dict,
                    "config": self.config,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            logger.info(f"Model checkpoint saved to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}", exc_info=True)

    def _save_results_summary(
        self,
        dataset_size: int,
        epochs: int,
        best_similarity: float,
        total_time: float,
        training_log: List[Dict],
    ):
        """Saves a JSON file with a summary of the training results."""
        results_path = self.output_dir / "training_summary.json"

        summary = {
            "training_parameters": {
                "dataset_size": dataset_size,
                "epochs": epochs,
                "batch_size": self.config.get("training", {}).get("batch_size"),
            },
            "performance": {
                "total_training_time_seconds": total_time,
                "best_similarity": best_similarity,
            },
            "configuration": {
                "mode": self.config.get("_metadata", {}).get("mode", "unknown"),
                "scale": self.config.get("_metadata", {}).get(
                    "scale_factor", "unknown"
                ),
                "lattice_shape": self.config.get("lattice", {}),
                "gmlp_params": self.config.get("gmlp", {}),
            },
            "full_log": training_log,
        }

        try:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            logger.info(f"Training summary saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}", exc_info=True)
