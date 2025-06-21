#!/usr/bin/env python3
"""
Utils модуль для new_rebuild архитектуры
======================================

Содержит утилиты для:
- Централизованное логирование с caller tracking
- Интеграция с legacy core/log_utils.py
- Специализированные функции для клеток
"""

from .logging import (
    # Основные функции логирования
    setup_logging,
    get_logger,
    # Специализированные функции
    log_init,
    log_function_call,
    log_performance,
    # Функции для клеток
    log_cell_init,
    log_cell_forward,
    log_cell_component_params,
    # Система предотвращения дублирования
    LogContext,
    DuplicationManager,
    AntiDuplicationFilter,
    ContextualFormatter,
    # Legacy совместимость
    _get_caller_info,
    get_caller_info,
)

__all__ = [
    # Основные функции
    "setup_logging",
    "get_logger",
    # Специализированные функции
    "log_init",
    "log_function_call",
    "log_performance",
    # Функции для клеток
    "log_cell_init",
    "log_cell_forward",
    "log_cell_component_params",
    # Система предотвращения дублирования
    "LogContext",
    "DuplicationManager",
    "AntiDuplicationFilter",
    "ContextualFormatter",
    # Legacy совместимость
    "_get_caller_info",
    "get_caller_info",
]
