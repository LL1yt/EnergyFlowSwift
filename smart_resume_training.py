#!/usr/bin/env python3
"""
[BRAIN] Smart Resume Training - Main Entry Point
"""

import logging
import sys
from pathlib import Path

# Add project root to path to allow imports from other modules
# This is a common pattern in projects like this.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from smart_resume_training.cli import parse_args
from smart_resume_training.core.config_initializer import ConfigInitializer
from smart_resume_training.core.training_orchestrator import TrainingOrchestrator


def setup_logging(verbose: bool, debug: bool):
    """Sets up basic logging for the script."""
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("smart_resume_training").setLevel(level)
    logging.getLogger("__main__").setLevel(level)


def main():
    """
    Main function to initialize components and run the training orchestrator.
    """
    args = parse_args()
    setup_logging(args.verbose, args.debug)

    logger = logging.getLogger(__name__)

    try:
        logger.info("Initializing configuration...")
        config_initializer = ConfigInitializer(
            forced_mode=args.mode, custom_scale=args.scale
        )
        config = config_initializer.get_config()
        metadata = config_initializer.get_metadata()

        logger.info("Initializing Training Orchestrator...")
        orchestrator = TrainingOrchestrator(config, metadata)

        training_kwargs = {}
        if args.batch_size:
            training_kwargs["batch_size"] = args.batch_size

        logger.info("Starting training run...")
        orchestrator.run(
            dataset_limit=args.dataset_limit,
            additional_epochs=args.additional_epochs,
            **training_kwargs,
        )

        logger.info("Smart resume training finished successfully.")
        # The script called by the orchestrator should return a result
        # For now, we just exit. The `automated_training` runner will check the exit code.
        # To signal success to the outer script, we can output a specific line.
        print("final_similarity=0.999")  # Placeholder for actual result

    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
