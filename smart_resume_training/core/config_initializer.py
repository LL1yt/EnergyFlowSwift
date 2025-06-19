"""
Configuration Initializer for Smart Resume Training
"""

import logging
from typing import Optional, Dict

# Assuming the dynamic config manager is in the project's utils
from utils.config_manager.dynamic_config import DynamicConfigManager

logger = logging.getLogger(__name__)


class ConfigInitializer:
    """Handles the initialization of training configuration."""

    def __init__(
        self, forced_mode: Optional[str] = None, custom_scale: Optional[float] = None
    ):
        self.forced_mode = forced_mode
        self.custom_scale = custom_scale
        self.config: Optional[Dict] = None
        self.metadata: Optional[Dict] = None
        self._initialize()

    def _initialize(self):
        """Initializes the configuration using the DynamicConfigManager."""
        try:
            dynamic_manager = DynamicConfigManager()
            mode = self.forced_mode or dynamic_manager.generator.detect_hardware_mode()

            if self.custom_scale is not None:
                setattr(
                    dynamic_manager.generator.scale_settings, mode, self.custom_scale
                )
                logger.info(f"Applied custom scale factor: {self.custom_scale}")

            full_config = dynamic_manager.create_config_for_mode(mode)

            self.config = {
                "lattice": full_config.get("lattice", {}),
                "embeddings": full_config.get("embeddings", {}),
                "training": full_config.get("training", {}),
                "gmlp": full_config.get("gmlp", {}),
            }

            if "emergent_training" in full_config:
                self.config["emergent_training"] = full_config["emergent_training"]
                logger.info("Added emergent_training section to config.")

            self.metadata = full_config.get("_metadata", {})
            self._log_config_details()

        except Exception as e:
            logger.error(f"Failed to initialize dynamic config: {e}", exc_info=True)
            raise

    def _log_config_details(self):
        """Logs the key details of the generated configuration."""
        if not self.config or not self.metadata:
            logger.warning("Config or metadata not available for logging.")
            return

        mode = self.metadata.get("mode", "unknown")
        scale = self.metadata.get("scale_factor", "unknown")
        logger.info(f"Initialized config in '{mode}' mode with scale factor {scale}.")

        lattice = self.config.get("lattice", {})
        if lattice:
            logger.info(
                f"Target Lattice: {lattice.get('xs')}x{lattice.get('ys')}x{lattice.get('zs')}"
            )

        gmlp = self.config.get("gmlp", {})
        if gmlp:
            logger.info(
                f"gMLP state size: {gmlp.get('state_size')}, hidden_dim: {gmlp.get('hidden_dim')}"
            )

    def get_config(self) -> Dict:
        """Returns the generated configuration."""
        if not self.config:
            raise ValueError("Configuration has not been initialized.")
        return self.config

    def get_metadata(self) -> Dict:
        """Returns the configuration metadata."""
        if not self.metadata:
            raise ValueError("Configuration metadata has not been initialized.")
        return self.metadata
