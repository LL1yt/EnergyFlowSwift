"""
Custom Log Formatters
"""

import logging
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
