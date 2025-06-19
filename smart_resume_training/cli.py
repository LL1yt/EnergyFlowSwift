"""
CLI for Smart Resume Training
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="[BRAIN] Smart Resume Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration arguments
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Forced configuration mode (e.g., 'development', 'production'). Overrides auto-detection.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Custom scale factor for dynamic configuration.",
    )

    # Training control arguments
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit the dataset size for quick testing.",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=10,
        help="Number of epochs to train, either for a fresh start or added to a resumed session.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size from the configuration.",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        help="Path to write the final training metrics as a JSON file.",
    )

    # Logging arguments
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging (DEBUG level)."
    )

    args = parser.parse_args()
    logger.info(f"CLI arguments parsed: {args}")
    return args
