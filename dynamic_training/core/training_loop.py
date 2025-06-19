"""
The main training loop for dynamic training.
"""

import logging
import time
import torch
import gc
from typing import Dict, Any, Optional

from .trainer_factory import TrainerFactory
from .dataset_handler import DatasetHandler
from .results_saver import ResultsSaver

# Assuming warmup_scheduler is available
from warmup_scheduler import create_warmup_scheduler

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Encapsulates the training loop and its logic."""

    def __init__(self, config: Dict):
        self.config = config
        self.trainer_factory = TrainerFactory(config)
        self.dataset_handler = DatasetHandler(config)
        self.results_saver = ResultsSaver(config)

    def run(
        self,
        dataset_limit: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        start_epoch: int = 0,
        fixed_sampling: bool = False,
    ):
        """
        Executes the main training loop.
        """
        training_start_time = time.time()

        # --- Trainer Setup ---
        trainer = self.trainer_factory.create_trainer()
        if resume_from_checkpoint:
            try:
                state_dict = torch.load(resume_from_checkpoint, map_location="cpu")[
                    "model_state_dict"
                ]
                trainer.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Successfully loaded model state from {resume_from_checkpoint}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoint. Starting fresh. Error: {e}",
                    exc_info=True,
                )
                start_epoch = 0  # Reset epoch count

        # --- Dataset Setup ---
        dataset = self.dataset_handler.prepare_dataset(
            limit=dataset_limit, fixed_sampling=fixed_sampling
        )

        # --- Training Parameters ---
        train_cfg = self.config.get("training", {})
        final_epochs = epochs or train_cfg.get("epochs", 10)
        final_batch_size = batch_size or train_cfg.get("batch_size", 32)

        logger.info(
            f"Starting training for {final_epochs} epochs with batch size {final_batch_size}."
        )

        # --- Optimizer and Scheduler ---
        optimizer = torch.optim.AdamW(
            trainer.get_trainable_parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        scheduler = create_warmup_scheduler(
            optimizer,
            warmup_epochs=train_cfg.get("warmup_epochs", 1),
            total_epochs=final_epochs,
        )

        # --- Training ---
        try:
            results = trainer.train(
                dataset=dataset,
                num_epochs=final_epochs,
                batch_size=final_batch_size,
                optimizer=optimizer,
                scheduler=scheduler,
                start_epoch=start_epoch,
            )

            total_time = time.time() - training_start_time
            logger.info(f"Training completed in {total_time:.2f} seconds.")

            # --- Save Results ---
            self.results_saver.save(
                trainer=trainer,
                dataset_size=len(dataset),
                epochs=results["total_epochs"],
                best_similarity=results["best_similarity"],
                total_time=total_time,
                training_log=results["training_log"],
            )
            # This is to communicate with the outer `automated_training` script
            print(f"final_similarity={results['best_similarity']:.4f}")

        except Exception as e:
            logger.error(
                f"An error occurred during the training loop: {e}", exc_info=True
            )
            raise
        finally:
            del trainer, dataset, optimizer, scheduler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
