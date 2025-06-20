"""
Модуль валидации для Lattice3D
================================

Содержит функции для проверки и валидации конфигураций
и состояний решетки.
"""

from typing import Dict, Any
import torch
from .config import LatticeConfig


def validate_lattice_config(config: LatticeConfig) -> Dict[str, Any]:
    """
    Валидация конфигурации решетки.

    Args:
        config: Конфигурация для проверки

    Returns:
        Dict[str, Any]: Результаты валидации
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # Проверка размеров
    if config.total_cells > 100000:
        validation_results["warnings"].append(
            f"Large lattice size ({config.total_cells} cells) may impact performance"
        )

    # Проверка GPU
    if config.gpu_enabled and not torch.cuda.is_available():
        validation_results["warnings"].append(
            "GPU enabled but CUDA not available, falling back to CPU"
        )

    # Проверка конфигурации клеток
    if config.cell_config is None and config.auto_sync_cell_config:
        validation_results["recommendations"].append(
            "Consider loading cell_prototype configuration for better integration"
        )

    return validation_results
