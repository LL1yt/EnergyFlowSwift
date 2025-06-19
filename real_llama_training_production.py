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


def main():
    """Main entry point for production training."""
    logger.info("--- Production Training Launcher ---")

    use_lightweight = "--distilbert" in sys.argv or "--llama" not in sys.argv
    model_type = "DistilBERT" if use_lightweight else "LLaMA"
    logger.info(f"Launcher mode: Using {model_type}")

    try:
        manager = ProductionTrainingManager(use_lightweight_model=use_lightweight)
        results = manager.run_full_training_pipeline()

        logger.info("--- Training Run Summary ---")
        if results.get("status") == "completed":
            final_metrics = results.get("results", {}).get("metrics", {})
            recommendations = results.get("results", {}).get("recommendations", [])

            logger.info(f"Final Loss: {final_metrics.get('final_loss', 'N/A'):.4f}")
            logger.info(
                f"Final Similarity: {final_metrics.get('final_similarity', 'N/A'):.4f}"
            )
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
        sys.exit(1)


if __name__ == "__main__":
    main()
