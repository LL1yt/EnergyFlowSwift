"""
Analyzes the results of training stages and the final outcome.
Also contains utility functions for saving results.
"""

import logging
import numpy as np
from typing import Dict, Any
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from production_training.config.training_stages import TrainingStage

logger = logging.getLogger(__name__)


def analyze_stage_failure(stage: TrainingStage, result: Dict[str, Any]) -> str:
    """
    Analyzes a failed stage and decides the next action.

    Args:
        stage: The configuration of the failed stage.
        result: The results dictionary from the failed stage.

    Returns:
        A string representing the decision: 'abort', 'continue', or 'retry'.
    """
    logger.info(f"[ANALYZER] Analyzing failure of stage: {stage.name}")
    metrics = result["metrics"]
    best_loss = metrics.get("best_loss", float("inf"))

    # Critical failure on validation stage
    if stage.name == "validation" and best_loss > 1.5:
        logger.error("[ANALYZER] Critical validation failure. System is not learning.")
        return "abort"

    # Check if any learning happened at all
    if len(metrics.get("losses", [])) > 5:
        initial_loss = np.mean(metrics["losses"][:3])
        final_loss = np.mean(metrics["losses"][-3:])
        improvement = (
            (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
        )

        if improvement < 0.05:  # Less than 5% improvement
            logger.warning("[ANALYZER] Stagnation detected. No significant learning.")
            # Potentially abort if it's an early stage
            if stage.name == "convergence":
                return "abort"
        else:
            logger.info(
                f"[ANALYZER] Learning detected ({improvement:.1%}). Continuing despite not meeting targets."
            )
            return "continue"

    logger.warning("[ANALYZER] Stage failed, but conditions met to continue.")
    return "continue"


def analyze_final_results(training_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the complete training history and provides a summary and recommendations.

    Args:
        training_history: The accumulated history of all training stages.

    Returns:
        A dictionary with final metrics and recommendations.
    """
    logger.info("[ANALYZER] Analyzing final training results...")

    final_metrics = training_history.get("best_metrics", {})
    performance = {
        "final_loss": final_metrics.get("loss"),
        "final_similarity": final_metrics.get("similarity"),
        "total_training_time": training_history.get("total_time"),
        "stages_completed": len(training_history.get("stages", [])),
    }

    recommendations = []
    if performance["final_similarity"] and performance["final_similarity"] > 0.4:
        recommendations.append(
            "Excellent! Similarity target met. Model is ready for review."
        )
    else:
        recommendations.append(
            "Model did not reach similarity target. Consider more data or architectural tweaks."
        )

    if performance["final_loss"] and performance["final_loss"] < 0.3:
        recommendations.append("Good! Final loss is low, indicating convergence.")
    else:
        recommendations.append(
            "Warning: Final loss is high. The model may not have fully converged."
        )

    return {"metrics": performance, "recommendations": recommendations}


def save_complete_results(
    results_dir: Path, training_id: str, history: dict, analysis: dict
):
    """Saves the complete results of the training pipeline to a JSON file."""
    results_file = results_dir / "complete_results.json"
    complete_results = {
        "training_id": training_id,
        "timestamp": datetime.now().isoformat(),
        "final_analysis": analysis,
        "training_history": history,
        "system_info": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__,
        },
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    logger.info(f"[SAVER] Complete results saved: {results_file}")


def create_training_visualizations(results_dir: Path, history: dict):
    """Creates and saves plots summarizing the training process."""
    if not history or not history.get("stages"):
        return

    plt.style.use("seaborn-v0_8_darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), tight_layout=True)

    all_losses = [
        loss for stage in history["stages"] for loss in stage["metrics"]["losses"]
    ]
    all_similarities = [
        s for stage in history["stages"] for s in stage["metrics"]["similarities"]
    ]

    if not all_losses:
        plt.close()
        return

    axes[0].plot(all_losses, "b-", label="Loss")
    axes[0].set_title("Training Loss Over All Epochs")
    axes[0].grid(True)
    axes[1].plot(all_similarities, "g-", label="Similarity")
    axes[1].set_title("Q→A Similarity Over All Epochs")
    axes[1].grid(True)

    plot_path = results_dir / "training_analysis.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"[PLOT] Training visualization saved: {plot_path}")
    plt.close()
