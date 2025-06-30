#!/usr/bin/env python3
"""
Cells Package - клеточные процессоры
==================================

Фабрика для автоматического выбора оптимальной версии клетки.
"""

from typing import Any, Dict, Optional
from ...utils.logging import get_logger

from ...config import get_project_config
from .base_cell import BaseCell

# Импорты векторизованных компонентов
try:
    from .vectorized_gnn_cell import VectorizedGNNCell

    VECTORIZED_AVAILABLE = True
except ImportError:
    VECTORIZED_AVAILABLE = False
    get_logger(__name__).warning("[WARN]  Vectorized components not available - using legacy versions")

# Импорты legacy компонентов (DEPRECATED)
# from .gnn_cell import GNNCell  # DEPRECATED - используйте VectorizedGNNCell

logger = get_logger(__name__)


def create_cell(cell_type: Optional[str] = None, **kwargs) -> BaseCell:
    """
    Фабрика для создания векторизованной клетки

    Args:
        cell_type: Тип клетки (только 'vectorized_gnn' или 'auto')
        **kwargs: Параметры для конструктора клетки

    Returns:
        BaseCell: Экземпляр векторизованной клетки
    """
    # Всегда создаем векторизованную версию
    if cell_type is None or cell_type == "auto" or cell_type == "vectorized_gnn":
        if not VECTORIZED_AVAILABLE:
            raise ImportError(
                "VectorizedGNNCell not available. "
                "Ensure vectorized components are properly installed."
            )
        logger.info("[START] Creating VectorizedGNNCell for maximum performance")
        return VectorizedGNNCell(**kwargs)

    # Legacy версии больше не поддерживаются
    elif cell_type == "gnn":
        logger.error(
            "[ALERT] Legacy GNN Cell is DEPRECATED! Only VectorizedGNNCell is supported."
        )
        raise DeprecationWarning(
            "GNNCell is deprecated and removed. Only VectorizedGNNCell is available."
        )

    else:
        raise ValueError(
            f"Unknown cell type: {cell_type}. Only 'vectorized_gnn' is supported."
        )


def get_recommended_cell_type() -> str:
    """Возвращает рекомендуемый тип клетки (всегда vectorized_gnn)"""
    return "vectorized_gnn"


def is_vectorized_available() -> bool:
    """Проверяет доступность векторизованных компонентов"""
    return VECTORIZED_AVAILABLE


def get_performance_comparison() -> Dict[str, Any]:
    """Возвращает сравнение производительности доступных типов клеток"""
    config = get_project_config()

    comparison = {
        "recommended": get_recommended_cell_type(),
        "total_cells": config.total_cells,
        "available_types": [],
    }

    if VECTORIZED_AVAILABLE:
        comparison["available_types"].append(
            {
                "type": "vectorized_gnn",
                "description": "Vectorized GNN Cell - максимальная производительность",
                "expected_speedup": "5-800x",
                "recommended": True,
            }
        )

    comparison["available_types"].append(
        {
            "type": "gnn",
            "description": "DEPRECATED: Legacy GNN Cell - используйте VectorizedGNNCell",
            "expected_speedup": "1x (базовая)",
            "recommended": False,  # Больше не рекомендуется
            "deprecated": True,
        }
    )

    return comparison


# Экспорты
__all__ = [
    "BaseCell",
    # "GNNCell",  # DEPRECATED - удалено из экспорта
    "create_cell",
    "get_recommended_cell_type",
    "is_vectorized_available",
    "get_performance_comparison",
]

# Условный экспорт векторизованных компонентов
if VECTORIZED_AVAILABLE:
    __all__.append("VectorizedGNNCell")
