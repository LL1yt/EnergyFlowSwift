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
        self,
        forced_mode: Optional[str] = None,
        custom_scale: Optional[float] = None,
        config_path: Optional[str] = None,
    ):
        self.forced_mode = forced_mode
        self.custom_scale = custom_scale
        self.config_path = config_path
        self.config: Optional[Dict] = None
        self.metadata: Optional[Dict] = None
        self._initialize()

    def _initialize(self):
        """Initializes the configuration using DynamicConfigManager or from a file."""
        if self.config_path:
            self._load_from_file()
        else:
            self._generate_dynamically()

    def _load_from_file(self):
        """Loads configuration from a specified YAML file."""
        logger.info(f"Loading configuration from file: {self.config_path}")
        try:
            import yaml

            with open(self.config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)

            self._process_loaded_config(full_config)
            logger.info("Successfully loaded configuration from file.")

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load or parse config file {self.config_path}: {e}",
                exc_info=True,
            )
            raise

    def _generate_dynamically(self):
        """Initializes the configuration using the DynamicConfigManager."""
        logger.info("Generating configuration dynamically...")
        try:
            dynamic_manager = DynamicConfigManager()
            mode = self.forced_mode or dynamic_manager.generator.detect_hardware_mode()

            if self.custom_scale is not None:
                setattr(
                    dynamic_manager.generator.scale_settings, mode, self.custom_scale
                )
                logger.info(f"Applied custom scale factor: {self.custom_scale}")

            full_config = dynamic_manager.create_config_for_mode(mode)
            self._process_loaded_config(full_config)

        except Exception as e:
            logger.error(f"Failed to initialize dynamic config: {e}", exc_info=True)
            raise

    def _process_loaded_config(self, full_config: Dict):
        """Processes the loaded or generated config dictionary."""
        self.config = {
            "lattice": full_config.get("lattice", {}),
            "embeddings": full_config.get("embeddings", {}),
            "training": full_config.get("training", {}),
            "gmlp": full_config.get("gmlp", {}),
            "nca": full_config.get("nca", {}),
            "biological": full_config.get("biological", {}),
            "experimental": full_config.get("experimental", {}),
        }

        if "emergent_training" in full_config:
            self.config["emergent_training"] = full_config["emergent_training"]
            logger.info("Added emergent_training section to config.")

        self.metadata = full_config.get("_metadata", {})
        self._log_config_details()

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
