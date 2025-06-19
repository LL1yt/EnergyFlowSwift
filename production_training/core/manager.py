"""
The main manager for the production training pipeline.
"""

import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from emergent_training import EmergentCubeTrainer, EmergentTrainingConfig
from production_training.config.training_stages import get_default_training_stages
from production_training.core.validator import validate_system
from production_training.core.stage_runner import run_training_stage
from production_training.analysis.results_analyzer import (
    analyze_stage_failure,
    analyze_final_results,
)
from production_training.utils.data_factory import get_dataset_for_stage
from production_training.utils.savers import (
    save_complete_results,
    create_training_visualizations,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "lightweight": "distilbert",
    "production": "llama3-8b-local",
}


class ProductionTrainingManager:
    """Orchestrates the full training pipeline."""

    def __init__(
        self,
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        training_args: Dict[str, Any] = None,
        use_lightweight_model: bool = True,
    ):
        # Store the passed parameters
        self.config = config or {}
        self.metadata = metadata or {}
        self.training_args = training_args or {}

        self.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(f"checkpoints/prod_{self.training_id}")
        self.results_dir = Path(f"results/prod_{self.training_id}")
        self.model_name = (
            MODEL_CONFIG["lightweight"]
            if use_lightweight_model
            else MODEL_CONFIG["production"]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stages = get_default_training_stages()

        # Apply training_args overrides to stages if provided
        if self.training_args:
            self._apply_training_args_to_stages()

        self.history = {
            "stages": [],
            "best_metrics": {"loss": float("inf"), "similarity": 0.0},
            "total_time": 0,
            "decisions": [],
            "model_used": self.model_name,
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Manager initialized for training ID {self.training_id} on device {self.device}"
        )

    def _apply_training_args_to_stages(self):
        """Apply training_args overrides to stages configuration"""
        if (
            "dataset_limit" in self.training_args
            and self.training_args["dataset_limit"]
        ):
            for stage in self.stages:
                stage.dataset_limit = min(
                    stage.dataset_limit, self.training_args["dataset_limit"]
                )

        if "epochs" in self.training_args and self.training_args["epochs"]:
            for stage in self.stages:
                stage.epochs = self.training_args["epochs"]

        if "batch_size" in self.training_args and self.training_args["batch_size"]:
            for stage in self.stages:
                stage.batch_size = self.training_args["batch_size"]

    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Runs the entire pipeline from validation to final analysis."""
        logger.info("Starting full training pipeline...")

        if not validate_system(self.model_name, self.device):
            logger.error("System validation failed. Aborting.")
            return {"status": "failed", "stage": "validation"}

        # Initialize the trainer once
        config = EmergentTrainingConfig(teacher_model=self.model_name)
        trainer = EmergentCubeTrainer(config, device=self.device)

        for stage in self.stages:
            dataset = get_dataset_for_stage(stage, self.model_name)
            stage_result = run_training_stage(
                trainer, dataset, stage, self.checkpoint_dir
            )

            self.history["stages"].append(stage_result)
            self._update_best_metrics(stage_result)

            if not stage_result["success"]:
                decision = analyze_stage_failure(stage, stage_result)
                self.history["decisions"].append(decision)
                if decision == "abort":
                    logger.error(f"Pipeline aborted at stage {stage.name}.")
                    break

        final_analysis = analyze_final_results(self.history)
        save_complete_results(
            self.results_dir, self.training_id, self.history, final_analysis
        )
        create_training_visualizations(self.results_dir, self.history)

        logger.info("Full training pipeline completed.")
        return {"status": "completed", "results": final_analysis}

    def _update_best_metrics(self, stage_result: Dict[str, Any]):
        """Updates the best metrics found so far."""
        stage_best_loss = stage_result["metrics"].get("best_loss", float("inf"))
        stage_best_sim = stage_result["metrics"].get("best_similarity", 0.0)
        if stage_best_loss < self.history["best_metrics"]["loss"]:
            self.history["best_metrics"]["loss"] = stage_best_loss
        if stage_best_sim > self.history["best_metrics"]["similarity"]:
            self.history["best_metrics"]["similarity"] = stage_best_sim
