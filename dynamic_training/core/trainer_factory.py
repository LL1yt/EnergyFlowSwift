"""
Factory for creating the training instance.
"""

import logging
import torch
from typing import Dict, Any
from dataclasses import fields

# Import the REFACTORED trainer and its config
from emergent_training import EmergentCubeTrainer, EmergentTrainingConfig
from training.embedding_trainer.neural_cellular_automata import (
    NCAConfig,
    create_nca_config,
)

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Creates and configures the EmergentCubeTrainer."""

    def __init__(self, config: Dict):
        self.config = config

    def create_trainer(self) -> EmergentCubeTrainer:
        """
        Creates an instance of the EmergentCubeTrainer based on the
        provided dynamic configuration.
        """
        logger.info("Creating trainer instance...")

        # The new config builder method will handle everything
        emergent_config = self._create_emergent_config()
        device = self._get_device()

        try:
            # The refactored trainer only needs the config object and the device
            trainer = EmergentCubeTrainer(
                config=emergent_config,
                device=device,
            )
            logger.info("EmergentCubeTrainer created successfully.")
            return trainer
        except Exception as e:
            logger.error(f"Failed to create EmergentCubeTrainer: {e}", exc_info=True)
            raise

    def _get_device(self) -> str:
        """Determines the device for training."""
        device_config = self.config.get("device", {})
        if device_config.get("use_gpu") and torch.cuda.is_available():
            return device_config.get("gpu_device", "cuda:0")
        return "cpu"

    def _create_emergent_config(self) -> EmergentTrainingConfig:
        """
        Builds the complete EmergentTrainingConfig from the various sections
        of the main config file.
        """
        # Start with the base emergent training parameters
        emergent_params = self.config.get("emergent_training", {}).copy()

        # Get gmlp params and add them to the dict
        emergent_params["gmlp_config"] = self.config.get("gmlp", {})

        # Get NCA params and create the NCAConfig if enabled
        use_nca = self.config.get("nca", {}).get("enabled", False)
        emergent_params["enable_nca"] = use_nca
        if use_nca:
            nca_raw_params = self.config.get("nca", {}).copy()
            nca_config_fields = {f.name for f in fields(NCAConfig)}
            valid_nca_params = {
                k: v for k, v in nca_raw_params.items() if k in nca_config_fields
            }
            emergent_params["nca_config"] = create_nca_config(**valid_nca_params)
        else:
            emergent_params["nca_config"] = None

        # Filter the combined dictionary to only include valid fields for EmergentTrainingConfig
        config_fields = {f.name for f in fields(EmergentTrainingConfig)}
        final_params = {k: v for k, v in emergent_params.items() if k in config_fields}

        return EmergentTrainingConfig(**final_params)
