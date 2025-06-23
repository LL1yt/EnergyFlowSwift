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

from .device_manager import (
    # Основные классы
    DeviceManager,
    MemoryMonitor,
    # Функции быстрого доступа
    get_device_manager,
    reset_device_manager,
    ensure_device,
    allocate_tensor,
    transfer_module,
    get_optimal_device,
    cleanup_memory,
)

__all__ = [
    # Основные функции логирования
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
    # Device Management
    "DeviceManager",
    "MemoryMonitor",
    "get_device_manager",
    "reset_device_manager",
    "ensure_device",
    "allocate_tensor",
    "transfer_module",
    "get_optimal_device",
    "cleanup_memory",
]
