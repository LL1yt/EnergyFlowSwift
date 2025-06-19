"""
Configuration Loader for Dynamic Training
"""

import logging
from typing import Dict, Optional

# Assuming utils structure
from utils.config_manager.config_manager import ConfigManager, ConfigManagerSettings

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads the dynamic configuration for training."""

    def __init__(
        self,
        forced_mode: Optional[str] = None,
        custom_scale: Optional[float] = None,
        external_config: Optional[Dict] = None,
    ):
        self.forced_mode = forced_mode
        self.custom_scale = custom_scale
        self.external_config = external_config
        self.config: Optional[Dict] = None

        if self.external_config:
            logger.info("Using pre-loaded external configuration.")
            self.config = self.external_config
        else:
            self._load_from_manager()

    def _load_from_manager(self):
        """Loads configuration using the ConfigManager."""
        logger.info("Loading configuration via DynamicConfigManager.")
        try:
            settings = ConfigManagerSettings(
                enable_dynamic_config=True,
                dynamic_config_mode=self.forced_mode or "auto",
                auto_hardware_detection=not self.forced_mode,
                custom_scale_factor=self.custom_scale,
                config_search_paths=[],
            )
            manager = ConfigManager(settings)

            self.config = {
                "lattice": manager.get_config("lattice"),
                "embeddings": manager.get_config("embeddings"),
                "training": manager.get_config("training"),
                "gmlp": manager.get_config("gmlp"),
                "device": manager.get_config("device"),
            }

            # Safely add optional sections
            for section in ["emergent_training", "nca"]:
                try:
                    section_config = manager.get_config(section)
                    if section_config:
                        self.config[section] = section_config
                        logger.info(
                            f"Successfully loaded '{section}' configuration section."
                        )
                except Exception:
                    logger.warning(
                        f"Could not find or load '{section}' configuration section. It will be skipped."
                    )

            info = manager.get_dynamic_config_info()
            logger.info(
                f"Loaded config for mode '{info.get('mode')}' with scale {info.get('scale_factor')}."
            )

        except Exception as e:
            logger.error(
                f"Failed to load dynamic configuration from manager: {e}", exc_info=True
            )
            raise

    def get_config(self) -> Dict:
        """Returns the loaded configuration."""
        if not self.config:
            raise RuntimeError("Configuration has not been loaded.")
        return self.config
