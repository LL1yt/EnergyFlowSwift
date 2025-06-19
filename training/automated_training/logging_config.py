#!/usr/bin/env python3
"""
Centralized Logging Configuration for Automated Training
Централизованная конфигурация логирования для автоматизированного обучения

Предоставляет:
- Структурированное логирование с тегами
- Улучшенное форматирование для читаемости
- Разные уровни детализации
- Специализированные логгеры для разных компонентов
- Поддержка метрик и производительности

Автор: 3D Cellular Neural Network Project
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консольного вывода"""

    # ANSI цвета
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Добавляем цвет к уровню логирования
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Структурированный форматтер с тегами"""

    def format(self, record):
        # Извлекаем компонент из имени логгера
        component = self._extract_component(record.name)

        # Добавляем временную метку и компонент
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Форматируем сообщение
        if hasattr(record, "tag"):
            tag = f"[{record.tag}]"
        else:
            tag = ""

        formatted = f"{timestamp} │ {component:>12} │ {record.levelname:>7} │ {tag} {record.getMessage()}"

        # Добавляем исключение если есть
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted

    def _extract_component(self, logger_name: str) -> str:
        """Извлекает компонент из имени логгера"""
        if "automated_trainer" in logger_name:
            return "TRAINER"
        elif "session_manager" in logger_name:
            return "SESSION"
        elif "stage_runner" in logger_name:
            return "RUNNER"
        elif "progressive_config" in logger_name:
            return "CONFIG"
        elif "cli_interface" in logger_name:
            return "CLI"
        elif "cell_prototype" in logger_name:
            return "CELL"
        elif "lattice" in logger_name:
            return "LATTICE"
        else:
            # Берем последнюю часть имени
            parts = logger_name.split(".")
            return parts[-1].upper()[:12]


class AutomatedTrainingLogger:
    """Главный класс для настройки логирования"""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.log_dir = Path("logs/automated_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем основные логгеры
        self._setup_loggers()

    def _setup_loggers(self):
        """Настраивает все логгеры"""
        # Определяем уровень логирования (более консервативный подход)
        if self.quiet:
            console_level = logging.ERROR
        elif self.verbose:
            console_level = logging.INFO  # Уменьшили с DEBUG
        else:
            console_level = logging.WARNING  # По умолчанию только важное

        # Файловый уровень тоже оптимизируем
        file_level = logging.INFO  # Уменьшили с DEBUG

        # Создаем форматтеры
        console_formatter = StructuredFormatter()
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Очищаем существующие хендлеры
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Консольный хендлер
        if not self.quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # Файловый хендлер - основной лог (только для важных событий)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_file = self.log_dir / f"automated_training_{timestamp}.log"

        file_handler = logging.FileHandler(main_log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Ротирующий файловый хендлер для общих логов (только критические)
        rotating_handler = logging.handlers.RotatingFileHandler(
            "logs/main.log",
            maxBytes=5 * 1024 * 1024,  # Уменьшили до 5MB
            backupCount=3,  # Уменьшили количество backup файлов
            encoding="utf-8",
        )
        rotating_handler.setLevel(logging.WARNING)  # Только WARNING и выше
        rotating_handler.setFormatter(file_formatter)
        root_logger.addHandler(rotating_handler)

        # Логгер для ошибок остается без изменений
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{timestamp}.log", encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)

        # Минимальное логирование настройки (только если verbose)
        if self.verbose:
            logging.info(
                f"🔧 [LOGGING] Система логирования настроена (оптимизированная)"
            )
            logging.info(
                f"   Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)}"
            )

        # Подавляем избыточные логи от сторонних библиотек
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
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)  # Только предупреждения и ошибки

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


class MetricsLogger:
    """Специализированный логгер для метрик и производительности (оптимизированный)"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(f"metrics.{self.session_id}")
        self.metrics_file = (
            Path("logs/automated_training") / f"metrics_{self.session_id}.json"
        )

        # Только критические метрики в лог
        self.logger.setLevel(logging.WARNING)

    def log_stage_metrics(self, stage: int, metrics: Dict[str, Any]):
        """Логирует метрики стадии (только важные)"""
        # Логируем только успешные завершения или ошибки
        if metrics.get("status") == "completed":
            if metrics.get("success"):
                self.logger.info(
                    f"✅ Stage {stage}: {metrics.get('actual_time_minutes', 0):.1f}min"
                )
            else:
                self.logger.error(f"❌ Stage {stage}: FAILED")

        # JSON метрики сохраняем всегда для анализа
        import json

        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"session_id": self.session_id, "stages": {}}

            data["stages"][str(stage)] = {
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }

            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def log_performance(
        self, operation: str, duration: float, details: Dict[str, Any] = None
    ):
        """Логирует производительность операций (только медленные)"""
        # Логируем только операции дольше 30 секунд
        if duration > 30.0:
            details = details or {}
            self.logger.warning(f"🐌 [SLOW] {operation}: {duration:.1f}s {details}")


# Глобальные функции для удобства (оптимизированные)
_training_logger = None
_metrics_logger = None


def setup_automated_training_logging(
    verbose: bool = False, quiet: bool = False
) -> AutomatedTrainingLogger:
    """Настраивает логирование для automated training"""
    global _training_logger
    _training_logger = AutomatedTrainingLogger(verbose=verbose, quiet=quiet)
    return _training_logger


def get_training_logger(name: str) -> logging.Logger:
    """Получает логгер для automated training"""
    if _training_logger is None:
        setup_automated_training_logging()
    return _training_logger.get_logger(name)


def get_metrics_logger(session_id: str = None) -> MetricsLogger:
    """Получает логгер метрик"""
    global _metrics_logger
    if _metrics_logger is None or (
        session_id and _metrics_logger.session_id != session_id
    ):
        _metrics_logger = MetricsLogger(session_id)
    return _metrics_logger


def log_stage_start(stage: int, config: Dict[str, Any]):
    """Логирует начало стадии (минимально)"""
    logger = get_training_logger("stage")
    metrics_logger = get_metrics_logger()

    # Логируем только номер стадии
    logger.warning(f"🚀 Stage {stage} starting...")

    # Метрики сохраняем в JSON
    metrics_logger.log_stage_metrics(
        stage,
        {
            "status": "started",
            "dataset_limit": config.get("dataset_limit"),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
        },
    )


def log_stage_complete(stage: int, result: Dict[str, Any]):
    """Логирует завершение стадии (минимально)"""
    logger = get_training_logger("stage")
    metrics_logger = get_metrics_logger()

    # Логируем только результат
    if result.get("success"):
        logger.warning(
            f"✅ Stage {stage}: {result.get('actual_time_minutes', 0):.1f}min"
        )
        if result.get("final_similarity"):
            logger.warning(f"   Similarity: {result.get('final_similarity'):.3f}")
    else:
        logger.error(f"❌ Stage {stage}: FAILED")

    # Метрики в JSON
    metrics_logger.log_stage_metrics(
        stage,
        {
            "status": "completed",
            "success": result.get("success"),
            "actual_time_minutes": result.get("actual_time_minutes"),
            "final_similarity": result.get("final_similarity"),
        },
    )
