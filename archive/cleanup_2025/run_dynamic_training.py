#!/usr/bin/env python3
"""
[START] Dynamic Training Script - Main Entry Point
"""
import logging
import sys
from pathlib import Path

# Add project root to path for cross-module imports
sys.path.append(str(Path(__file__).resolve().parents[0]))

from dynamic_training.cli import parse_args
from dynamic_training.core.environment_setup import setup_environment
from dynamic_training.core.config_loader import ConfigLoader
from dynamic_training.core.training_loop import TrainingLoop


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
    # Set level for our specific loggers
    logging.getLogger("dynamic_training").setLevel(level)
    logging.getLogger("__main__").setLevel(level)


def main():
    """
    Main function to initialize components and run the training process.
    """
    args = parse_args()
    setup_logging(args.verbose, args.debug)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Setting up environment...")
        setup_environment()

        logger.info("Loading dynamic configuration...")
        config_loader = ConfigLoader(forced_mode=args.mode, custom_scale=args.scale)
        config = config_loader.get_config()

        logger.info("Initializing training loop...")
        training_loop = TrainingLoop(config)

        logger.info("Starting training run...")
        training_loop.run(
            dataset_limit=args.dataset_limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume_from_checkpoint=args.resume_from,
            fixed_sampling=args.fixed_sampling,
        )

        logger.info("Dynamic training finished successfully.")

    except Exception as e:
        logger.error(
            f"An unhandled exception occurred in dynamic training: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
