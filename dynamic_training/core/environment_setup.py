"""
Sets up the environment for PyTorch training.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def setup_environment():
    """Configures the environment for optimal training."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # The memory fraction is aggressive and might be better handled by the user or a config.
        # torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.empty_cache()
        logger.info("CUDA environment configured for performance.")
    else:
        logger.info("CUDA not available. Running on CPU.")

    logger.info("Environment setup completed.")
