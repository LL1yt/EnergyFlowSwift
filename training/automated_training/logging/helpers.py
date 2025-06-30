"""
Logging helper functions.
"""

import logging
from typing import Dict, Any

from .metrics_logger import MetricsLogger

# Global instance of metrics logger
_metrics_logger_instance = None


def get_training_logger(name: str) -> logging.Logger:
    """Получает логгер с заданным именем"""
    return logging.getLogger(name)


def get_metrics_logger(session_id: str = None) -> MetricsLogger:
    """
    Получает синглтон-экземпляр логгера метрик.
    Это гарантирует, что все метрики сессии пишутся в один файл.
    """
    global _metrics_logger_instance
    if _metrics_logger_instance is None:
        _metrics_logger_instance = MetricsLogger(session_id)
    return _metrics_logger_instance


def log_stage_start(stage: int, config: Dict[str, Any]):
    """
    Логирует начало тренировочной стадии.

    Args:
        stage (int): Номер стадии.
        config (Dict[str, Any]): Конфигурация стадии.
    """
    logger = get_training_logger("stage_runner")
    msg = (
        f"Stage {stage} starting | "
        f"Dataset Limit: {config.get('dataset_limit', 'N/A')}, "
        f"Batch Size: {config.get('batch_size', 'N/A')}, "
        f"Epochs: {config.get('epochs', 'N/A')}"
    )
    logger.warning(msg)  # Используем WARNING для высокой видимости


def log_stage_complete(stage: int, result: Dict[str, Any]):
    """
    Логирует завершение тренировочной стадии.

    Args:
        stage (int): Номер стадии.
        result (Dict[str, Any]): Результаты стадии.
    """
    logger = get_training_logger("stage_runner")
    success = result.get("success", False)
    actual_time = result.get("actual_time_minutes", 0)

    if success:
        msg = f"[OK] Stage {stage} completed successfully in {actual_time:.1f} minutes."
        logger.warning(msg)
    else:
        error_msg = result.get("error_message", "Unknown error")
        msg = f"[ERROR] Stage {stage} failed after {actual_time:.1f} minutes. Reason: {error_msg}"
        logger.error(msg)
