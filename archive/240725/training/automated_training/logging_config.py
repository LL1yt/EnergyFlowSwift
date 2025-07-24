"""
Centralized Logging Configuration for Automated Training (Facade)

This module is now a facade that imports from the new refactored
`training.automated_training.logging` package.
Please update your imports to point to the new submodules directly.
"""

from .logging import (
    setup_automated_training_logging,
    get_training_logger,
    get_metrics_logger,
    log_stage_start,
    log_stage_complete,
    MetricsLogger,
    AutomatedTrainingLogger,
    ColoredFormatter,
    StructuredFormatter,
)

__all__ = [
    "setup_automated_training_logging",
    "get_training_logger",
    "get_metrics_logger",
    "log_stage_start",
    "log_stage_complete",
    "MetricsLogger",
    "AutomatedTrainingLogger",
    "ColoredFormatter",
    "StructuredFormatter",
]
