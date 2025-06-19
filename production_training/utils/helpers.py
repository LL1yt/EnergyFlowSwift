"""
Helper functions for the production training pipeline.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
from production_training.config.training_stages import TrainingStage
import matplotlib.pyplot as plt
import seaborn as sns
import torch

logger = logging.getLogger(__name__)

# --- Data Factory Logic ---


def get_dataset_for_stage(stage: TrainingStage, model_name: str):
    """Creates and returns a dataset suitable for the given training stage."""
    logger.info(f"[HELPERS] Creating dataset for stage: {stage.name}")
    dialogue_pairs = _get_dialogue_pairs_for_stage(stage.name)
    dataset = create_dialogue_dataset(
        dialogue_pairs=dialogue_pairs,
        teacher_model=model_name,
        cache_embeddings=(stage.name != "validation"),
        validation_split=0.0,
        normalize_embeddings=True,
        dataset_limit=stage.dataset_limit,
    )
    logger.info(f"[HELPERS] Dataset created with {len(dataset)} pairs.")
    return dataset


def _get_dialogue_pairs_for_stage(stage_name: str) -> List[Dict[str, Any]]:
    """Returns a list of dialogue pairs based on the stage name."""
    if stage_name == "validation":
        return [{"q": "Q1", "a": "A1"}]
    elif stage_name == "convergence":
        return _get_medium_dialogue_dataset()
    else:
        return _get_full_dialogue_dataset()


def _get_medium_dialogue_dataset() -> List[Dict[str, str]]:
    """Provides a medium-sized dataset."""
    return [{"question": f"MedQ{i}", "answer": f"MedA{i}"} for i in range(10)]


def _get_full_dialogue_dataset() -> List[Dict[str, str]]:
    """Provides a large dataset."""
    return _get_medium_dialogue_dataset() + [
        {"question": f"FullQ{i}", "answer": f"FullA{i}"} for i in range(20)
    ]


# --- Results Saving and Visualization Logic ---


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
    logger.info(f"[SAVE] Complete results saved: {results_file}")


def create_training_visualizations(results_dir: Path, history: dict):
    """Creates and saves plots summarizing the training process."""
    if not history or not history.get("stages"):
        logger.warning("[PLOT] No stage data available to create visualizations.")
        return

    plt.style.use("seaborn-v0_8_darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    all_losses = [
        epoch_loss
        for stage in history["stages"]
        for epoch_loss in stage["metrics"]["losses"]
    ]
    all_similarities = [
        s for stage in history["stages"] for s in stage["metrics"]["similarities"]
    ]

    # Plot Loss
    axes[0].plot(all_losses, "b-", label="Loss")
    axes[0].set_title("Training Loss Over All Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # Plot Similarity
    axes[1].plot(all_similarities, "g-", label="Similarity")
    axes[1].set_title("Q→A Similarity Over All Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Similarity")

    plt.tight_layout()
    plot_path = results_dir / "training_analysis.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"[PLOT] Training visualization saved: {plot_path}")
    plt.close()
