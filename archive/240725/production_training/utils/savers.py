"""
Utility functions for saving results and visualizations.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

logger = logging.getLogger(__name__)


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
        logger.warning("[PLOT] No stage data available to create visualizations.")
        return

    plt.style.use("seaborn-v0_8_darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), tight_layout=True)

    all_losses = [
        loss for stage in history["stages"] for loss in stage["metrics"]["losses"]
    ]
    all_similarities = [
        s for stage in history["stages"] for s in stage["metrics"]["similarities"]
    ]

    if not all_losses or not all_similarities:
        logger.warning(
            "[PLOT] No losses or similarities recorded to create visualizations."
        )
        plt.close()
        return

    # Plot Loss
    axes[0].plot(all_losses, "b-", label="Loss")
    axes[0].set_title("Training Loss Over All Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Plot Similarity
    axes[1].plot(all_similarities, "g-", label="Similarity")
    axes[1].set_title("Q→A Similarity Over All Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Similarity")
    axes[1].grid(True)

    # Add stage boundaries
    try:
        stage_boundaries = np.cumsum(
            [len(s["metrics"]["losses"]) for s in history["stages"]]
        )
        for boundary in stage_boundaries[:-1]:
            axes[0].axvline(x=boundary - 1, color="r", linestyle="--", alpha=0.7)
            axes[1].axvline(x=boundary - 1, color="r", linestyle="--", alpha=0.7)
    except Exception as e:
        logger.warning(f"Could not draw stage boundaries on plot: {e}")

    plot_path = results_dir / "training_analysis.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"[PLOT] Training visualization saved: {plot_path}")
    plt.close()
