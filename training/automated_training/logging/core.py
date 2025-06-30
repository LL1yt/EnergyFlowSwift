"""
Core logging setup for automated training.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

from .formatters import StructuredFormatter


class AutomatedTrainingLogger:
    """Главный класс для настройки логирования"""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.log_dir = Path("logs/automated_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_loggers()

    def _setup_loggers(self):
        """Настраивает все логгеры"""
        if self.quiet:
            console_level = logging.ERROR
        elif self.verbose:
            console_level = logging.INFO
        else:
            console_level = logging.WARNING

        file_level = logging.INFO

        console_formatter = StructuredFormatter()
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        if not self.quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_file = self.log_dir / f"automated_training_{timestamp}.log"

        file_handler = logging.FileHandler(main_log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        rotating_handler = logging.handlers.RotatingFileHandler(
            "logs/main.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        rotating_handler.setLevel(logging.WARNING)
        rotating_handler.setFormatter(file_formatter)
        root_logger.addHandler(rotating_handler)

        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{timestamp}.log", encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        if self.verbose:
            logging.info(
                f"[TOOL] [LOGGING] Система логирования настроена (оптимизированная)"
            )
            logging.info(
                f"   Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)}"
            )

        self._suppress_noisy_loggers()

    def _suppress_noisy_loggers(self):
        """Подавляет избыточные логи от сторонних библиотек"""
        noisy_loggers = [
            "transformers",
            "torch",
            "matplotlib",
            "PIL",
            "urllib3",
            "requests",
            "tensorflow",
            "tensorboard",
            "wandb",
            "accelerate",
            "datasets",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Получает логгер с заданным именем"""
        return logging.getLogger(name)

    @staticmethod
    def log_with_tag(
        logger: logging.Logger, level: int, tag: str, message: str, **kwargs
    ):
        """Логирует сообщение с тегом"""
        record = logger.makeRecord(
            name=logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
            **kwargs,
        )
        record.tag = tag
        logger.handle(record)


# Global instance
_logger_instance = None


def setup_automated_training_logging(
    verbose: bool = False, quiet: bool = False
) -> AutomatedTrainingLogger:
    """
    Настраивает и возвращает синглтон-экземпляр логгера.
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AutomatedTrainingLogger(verbose=verbose, quiet=quiet)
    return _logger_instance
