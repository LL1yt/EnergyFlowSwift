#!/usr/bin/env python3
"""
[LAUNCHER] Production Real Training

This script is a lightweight launcher for the modular production training pipeline.
All complex logic is handled by the ProductionTrainingManager and its submodules.
"""
import sys
import logging
import io

# --- Basic Logging Setup ---
# A simplified logger setup for the launcher itself.
# The manager and its components will have more detailed logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
logger = logging.getLogger(__name__)

# --- Import and Run the Manager ---
try:
    from production_training.core.manager import ProductionTrainingManager
except ImportError as e:
    logger.error(f"Failed to import ProductionTrainingManager: {e}")
    logger.error(
        "Please ensure the 'production_training' package is correctly structured."
    )
    sys.exit(1)


def main(config: dict, metadata: dict, training_args: dict):
    """
    Main entry point for production training.

    Args:
        config (dict): The main configuration dictionary.
        metadata (dict): The metadata dictionary for the training run.
        training_args (dict): Dictionary with runtime training arguments like
                              'dataset_limit', 'epochs', 'resume_from_checkpoint'.

    Returns:
        dict: A dictionary containing the final training metrics.
    """
    logger.info("--- Production Training Launcher ---")

    # The model type can be determined from the config in a more robust way,
    # but for now we can assume the manager handles it.
    model_type = (
        config.get("emergent_training", {}).get("cell_architecture", "gmlp").upper()
    )
    logger.info(f"Launcher mode: Using {model_type}")
    final_metrics = {}

    try:
        # Pass the arguments to the manager
        manager = ProductionTrainingManager(
            config=config, metadata=metadata, training_args=training_args
        )
        results = manager.run_full_training_pipeline()

        logger.info("--- Training Run Summary ---")
        if results.get("status") == "completed":
            final_metrics = results.get("results", {}).get("metrics", {})
            recommendations = results.get("results", {}).get("recommendations", [])

            logger.info(f"Final Loss: {final_metrics.get('final_loss', 'N/A'):.4f}")
            logger.info(
                f"Final Similarity: {final_metrics.get('final_similarity', 'N/A'):.4f}"
            )
            if recommendations:
                logger.info("Recommendations:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")
            logger.info("[SUCCESS] Pipeline completed successfully.")
        else:
            logger.error(
                f"[FAILURE] Pipeline finished with status: {results.get('status')}"
            )

    except Exception as e:
        logger.critical(
            f"A critical error occurred in the training pipeline: {e}", exc_info=True
        )
        # In case of error, we should still return something
        return {"status": "error", "message": str(e)}

    return final_metrics


if __name__ == "__main__":
    # This part is for direct execution and will need a way to load mock/default configs.
    # For now, it's primarily designed to be called by the orchestrator.
    logger.warning(
        "This script is not meant to be run directly without a proper config."
    )
    logger.warning("It's designed to be launched by 'smart_resume_training.py'.")
    # You could add a simple test run here if needed, e.g.:
    # main({}, {}, {'dataset_limit': 100, 'epochs': 1})
