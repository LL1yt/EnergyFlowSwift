#!/usr/bin/env python3
"""
Централизованная система конфигурации
====================================

Единый источник всех параметров для всех модулей системы.
Все модули должны брать значения отсюда, а не использовать хардкоды.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class CentralizedConfig:
    """
    Централизованная система конфигурации

    Обеспечивает единый источник истины для всех параметров системы.
    Все модули должны использовать этот источник вместо хардкодов.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация централизованной конфигурации

        Args:
            config_path: Путь к конфигурационному файлу (optional)
        """
        self.config_path = config_path or "config/main_config.yaml"
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Загрузка конфигурации из файла"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"[OK] Loaded config from {self.config_path}")
            else:
                logger.warning(
                    f"[WARN] Config file not found: {self.config_path}, using defaults"
                )
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"[ERROR] Failed to load config: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Дефолтная конфигурация (если файл не найден)"""
        return {
            "nca": {
                "state_size": 4,
                "hidden_dim": 3,
                "external_input_size": 1,
                "neighbor_count": 26,
                "target_params": 69,
                "activation": "tanh",
            },
            "gmlp": {
                "state_size": 8,
                "hidden_dim": 32,
                "external_input_size": 12,
                "neighbor_count": 26,
                "target_params": 23805,
                "activation": "gelu",
            },
            "lattice": {"xs": 16, "ys": 16, "zs": 16, "neighbor_count": 26},
        }

    # === NCA PARAMETERS ===
    @property
    def nca_state_size(self) -> int:
        """NCA state size - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("state_size", 4)

    @property
    def nca_hidden_dim(self) -> int:
        """NCA hidden dimension - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("hidden_dim", 3)

    @property
    def nca_external_input_size(self) -> int:
        """NCA external input size - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("external_input_size", 1)

    @property
    def nca_neighbor_count(self) -> int:
        """NCA neighbor count - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("neighbor_count", 26)

    @property
    def nca_target_params(self) -> Optional[int]:
        """NCA target parameters - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("target_params", 69)

    @property
    def nca_activation(self) -> str:
        """NCA activation function - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("nca", {}).get("activation", "tanh")

    # === GMLP PARAMETERS ===
    @property
    def gmlp_state_size(self) -> int:
        """gMLP state size - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("gmlp", {}).get("state_size", 8)

    @property
    def gmlp_neighbor_count(self) -> int:
        """gMLP neighbor count - ЕДИНСТВЕННЫЙ источник истины"""
        return self._config.get("gmlp", {}).get("neighbor_count", 26)

    # === ОБЩИЕ ПАРАМЕТРЫ ===
    @property
    def default_neighbor_count(self) -> int:
        """Дефолтное количество соседей для системы - ЕДИНСТВЕННЫЙ источник истины"""
        return 26  # 3D Moore neighborhood

    @property
    def default_state_size_nca(self) -> int:
        """Дефолтный state_size для NCA - ЕДИНСТВЕННЫЙ источник истины"""
        return 4

    @property
    def default_hidden_dim_nca(self) -> int:
        """Дефолтный hidden_dim для NCA - ЕДИНСТВЕННЫЙ источник истины"""
        return 3

    @property
    def default_external_input_size_nca(self) -> int:
        """Дефолтный external_input_size для NCA - ЕДИНСТВЕННЫЙ источник истины"""
        return 1

    # === МЕТОДЫ ДОСТУПА ===
    def get_nca_config(self) -> Dict[str, Any]:
        """Получить полную NCA конфигурацию"""
        return {
            "state_size": self.nca_state_size,
            "hidden_dim": self.nca_hidden_dim,
            "external_input_size": self.nca_external_input_size,
            "neighbor_count": self.nca_neighbor_count,
            "target_params": self.nca_target_params,
            "activation": self.nca_activation,
            "dropout": 0.0,
            "use_memory": False,
            "enable_lattice_scaling": False,
        }

    def get_gmlp_config(self) -> Dict[str, Any]:
        """Получить полную gMLP конфигурацию"""
        gmlp_section = self._config.get("gmlp", {})
        return {
            "state_size": self.gmlp_state_size,
            "neighbor_count": self.gmlp_neighbor_count,
            "hidden_dim": gmlp_section.get("hidden_dim", 32),
            "external_input_size": gmlp_section.get("external_input_size", 12),
            "target_params": gmlp_section.get("target_params", 23805),
            "activation": gmlp_section.get("activation", "gelu"),
            "dropout": gmlp_section.get("dropout", 0.1),
            "use_memory": gmlp_section.get("use_memory", False),
        }

    def get_emergent_training_config(self) -> Dict[str, Any]:
        """Получить конфигурацию для emergent training"""
        return {
            "gmlp_config": {
                "state_size": self.gmlp_state_size,
                "neighbor_count": self.gmlp_neighbor_count,
                "activation": "gelu",
                "dropout": 0.1,
                "use_memory": True,
                "spatial_connections": True,
            },
            "nca_config": self.get_nca_config(),
            "enable_nca": True,
        }

    def get_minimal_nca_cell_config(self) -> Dict[str, Any]:
        """Получить конфигурацию для minimal_nca_cell (для lattice и других модулей)"""
        return {
            "state_size": self.nca_state_size,
            "neighbor_count": self.nca_neighbor_count,
            "hidden_dim": self.nca_hidden_dim,
            "external_input_size": self.nca_external_input_size,
            "activation": self.nca_activation,
            "dropout": 0.0,
            "use_memory": False,
            "enable_lattice_scaling": False,
            "target_params": self.nca_target_params,
        }

    def get_full_config_dict(self) -> Dict[str, Any]:
        """Получить полную конфигурацию в формате, совместимом с существующими модулями"""
        return {
            "nca": self.get_nca_config(),
            "gmlp": self.get_gmlp_config(),
            "minimal_nca_cell": self.get_minimal_nca_cell_config(),
            "gmlp_config": self.get_gmlp_config(),  # Дублируем для совместимости
            "lattice": self._config.get("lattice", {}),
            "embeddings": self._config.get("embeddings", {}),
            "training": self._config.get("training", {}),
        }

    def update_config(self, section: str, key: str, value: Any):
        """Обновить параметр конфигурации"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        logger.info(f"[TOOL] Updated config: {section}.{key} = {value}")

    def log_config_summary(self):
        """Логирование сводки конфигурации"""
        logger.info("📋 CENTRALIZED CONFIG SUMMARY:")
        logger.info(
            f"   NCA: state={self.nca_state_size}, hidden={self.nca_hidden_dim}, "
            f"input={self.nca_external_input_size}, neighbors={self.nca_neighbor_count}"
        )
        logger.info(
            f"   gMLP: state={self.gmlp_state_size}, neighbors={self.gmlp_neighbor_count}"
        )
        logger.info(f"   Default neighbors: {self.default_neighbor_count}")


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===
_global_config: Optional[CentralizedConfig] = None


def get_centralized_config(config_path: Optional[str] = None) -> CentralizedConfig:
    """
    Получить глобальный экземпляр централизованной конфигурации

    Args:
        config_path: Путь к конфигурационному файлу (optional)

    Returns:
        CentralizedConfig: Экземпляр централизованной конфигурации
    """
    global _global_config

    if _global_config is None or config_path is not None:
        _global_config = CentralizedConfig(config_path)
        _global_config.log_config_summary()

    return _global_config


# === CONVENIENCE FUNCTIONS ===
def get_nca_defaults() -> Dict[str, Any]:
    """Получить NCA параметры из центральной конфигурации"""
    config = get_centralized_config()
    return config.get_nca_config()


def get_gmlp_defaults() -> Dict[str, Any]:
    """Получить gMLP параметры из центральной конфигурации"""
    config = get_centralized_config()
    return config.get_gmlp_config()


def get_default_neighbor_count() -> int:
    """Получить дефолтное количество соседей"""
    config = get_centralized_config()
    return config.default_neighbor_count


if __name__ == "__main__":
    # Тестирование
    config = get_centralized_config()
    print("[TEST] Testing Centralized Config:")
    print(f"   NCA config: {config.get_nca_config()}")
    print(f"   gMLP config: {config.get_gmlp_config()}")
