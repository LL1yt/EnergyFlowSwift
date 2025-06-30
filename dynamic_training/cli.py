"""
CLI for the Dynamic Training script.
"""

import argparse


def parse_args():
    """Parses command line arguments for dynamic training."""
    parser = argparse.ArgumentParser(
        description="🚀 Dynamic Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config args
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Force a configuration mode (e.g., 'production'). Overrides auto-detection.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Apply a custom scale factor to the configuration.",
    )

    # Training args
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Limit the number of samples from the dataset for quick tests.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from the config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size from the config.",
    )
    parser.add_argument(
        "--fixed-sampling",
        action="store_true",
        help="Use fixed sampling for the dataset for reproducible results.",
    )

    # Checkpoint args
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )

    # Logging args
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO level logging."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable DEBUG level logging."
    )

    return parser.parse_args()
