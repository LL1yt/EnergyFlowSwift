#!/usr/bin/env python3
"""
[LAUNCHER] Production Real Training

This script is a lightweight launcher for the modular production training pipeline.
All complex logic is handled by the ProductionTrainingManager and its submodules.
"""
import sys
import logging
import io
import argparse
import json

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


def create_cli_parser():
    """Create CLI argument parser for subprocess calls"""
    parser = argparse.ArgumentParser(description="Production Real Training")
    parser.add_argument("--mode", default="development", help="Training mode")
    parser.add_argument("--dataset-limit", type=int, help="Dataset limit override")
    parser.add_argument(
        "--additional-epochs", type=int, help="Additional epochs to run"
    )
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--output-json-path", help="Path to save JSON results")
    parser.add_argument("--scale", type=float, help="Scale factor")
    return parser


def main(config: dict = None, metadata: dict = None, training_args: dict = None):
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

    # If called as subprocess, create default config
    if config is None:
        config = {"emergent_training": {"cell_architecture": "gmlp"}}
    if metadata is None:
        metadata = {"training_id": "subprocess_run"}
    if training_args is None:
        training_args = {}

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

            logger.info(f"Final Loss: {final_metrics.get('final_loss', 'N/A')}")
            logger.info(
                f"Final Similarity: {final_metrics.get('final_similarity', 'N/A')}"
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
    # Handle CLI execution for subprocess calls
    parser = create_cli_parser()
    args = parser.parse_args()

    # Create training_args from CLI arguments
    training_args = {}
    if args.dataset_limit:
        training_args["dataset_limit"] = args.dataset_limit
    if args.additional_epochs:
        training_args["epochs"] = args.additional_epochs
    if args.batch_size:
        training_args["batch_size"] = args.batch_size

    # Create default config and metadata
    config = {"emergent_training": {"cell_architecture": "gmlp"}}
    metadata = {"training_id": f"cli_run_{args.mode}"}

    # Run training
    result = main(config, metadata, training_args)

    # Save results to JSON if path provided
    if args.output_json_path and result:
        try:
            with open(args.output_json_path, "w") as f:
                json.dump(result, f)
            logger.info(f"Results saved to {args.output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    # Exit with appropriate code
    sys.exit(0 if result.get("status") != "error" else 1)
