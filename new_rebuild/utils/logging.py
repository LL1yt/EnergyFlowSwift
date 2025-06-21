"""
Централизованная система логирования
===================================

Обеспечивает единообразное логирование с отслеживанием:
- Какой модуль был вызван
- Кем был вызван (caller info)
- Уровень детализации в зависимости от debug_mode
"""

import logging
import inspect
import sys
from datetime import datetime
from typing import Optional, Any, Dict
from pathlib import Path


class ModuleTrackingFormatter(logging.Formatter):
    """Форматтер с отслеживанием модулей и вызывающего кода."""

    def format(self, record):
        # Добавляем информацию о caller
        caller_info = self._get_caller_info()
        record.caller_info = caller_info

        # Форматируем сообщение
        if hasattr(record, "caller_info") and record.caller_info:
            caller_str = f" [{record.caller_info}]"
        else:
            caller_str = ""

        # Базовое форматирование
        formatted = super().format(record)

        # Добавляем caller info если есть
        if caller_str:
            # Вставляем caller info после времени и уровня, но перед сообщением
            parts = formatted.split(" - ", 2)
            if len(parts) >= 3:
                formatted = f"{parts[0]} - {parts[1]}{caller_str} - {parts[2]}"

        return formatted

    def _get_caller_info(self) -> str:
        """Получает информацию о вызывающем коде."""
        try:
            # Ищем в стеке фрейм, который не относится к логированию
            frame = inspect.currentframe()
            for _ in range(10):  # Максимум 10 уровней вглубь
                if frame is None:
                    break

                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_function
                lineno = frame.f_lineno

                # Пропускаем фреймы логирования
                if not any(
                    skip in filename.lower() for skip in ["logging", "log_utils"]
                ):
                    # Извлекаем относительный путь
                    try:
                        rel_path = Path(filename).relative_to(Path.cwd())
                        return f"{rel_path}:{func_name}:{lineno}"
                    except ValueError:
                        return f"{Path(filename).name}:{func_name}:{lineno}"

                frame = frame.f_back

            return "unknown"
        except Exception:
            return "unknown"


class DebugModeFilter(logging.Filter):
    """Фильтр для контроля детализации логов в зависимости от debug_mode."""

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode

    def filter(self, record):
        # В debug_mode пропускаем все логи
        if self.debug_mode:
            return True

        # В обычном режиме фильтруем детальные логи
        # Пропускаем только INFO и выше, плюс важные DEBUG сообщения
        if record.levelno >= logging.INFO:
            return True

        # Важные DEBUG сообщения (содержат специальные маркеры)
        message = record.getMessage()
        important_markers = ["🚀 INIT", "✅", "❌", "⚠️", "ERROR", "CRITICAL"]

        return any(marker in message for marker in important_markers)


def setup_logging(debug_mode: bool = False, log_file: Optional[str] = None) -> None:
    """
    Настраивает централизованное логирование.

    Args:
        debug_mode: Включить детальное логирование
        log_file: Файл для записи логов (опционально)
    """
    # Получаем root logger
    root_logger = logging.getLogger()

    # Очищаем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Устанавливаем уровень
    if debug_mode:
        root_logger.setLevel(logging.DEBUG)
        console_level = logging.DEBUG
    else:
        root_logger.setLevel(logging.INFO)
        console_level = logging.INFO

    # Создаем форматтер
    if debug_mode:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format_str = "%(asctime)s - %(levelname)s - %(message)s"

    formatter = ModuleTrackingFormatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(DebugModeFilter(debug_mode))
    root_logger.addHandler(console_handler)

    # File handler (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # В файл пишем все
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Настраиваем логгеры для сторонних библиотек
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получает логгер с автоматическим определением модуля.

    Args:
        name: Имя логгера (если не указано, определяется автоматически)

    Returns:
        Настроенный логгер
    """
    if name is None:
        # Автоматически определяем имя модуля
        frame = inspect.currentframe().f_back
        if frame:
            module_name = frame.f_globals.get("__name__", "unknown")
            name = module_name
        else:
            name = "unknown"

    return logging.getLogger(name)


def log_init(component_name: str, **kwargs) -> None:
    """
    Специальная функция для логирования инициализации компонентов.

    Args:
        component_name: Имя компонента
        **kwargs: Дополнительная информация для логирования
    """
    logger = get_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Форматируем дополнительную информацию
    info_parts = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            info_parts.append(f"{key.upper()}: {len(value)} items")
        else:
            info_parts.append(f"{key.upper()}: {value}")

    info_str = "\n     ".join(info_parts) if info_parts else "No additional info"

    logger.info(f"🚀 INIT {component_name} @ {timestamp}\n" f"     {info_str}")


def log_function_call(func_name: str, args: Dict[str, Any] = None) -> None:
    """
    Логирует вызов функции (для debug_mode).

    Args:
        func_name: Имя функции
        args: Аргументы функции
    """
    logger = get_logger()

    if args:
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        logger.debug(f"📞 CALL {func_name}({args_str})")
    else:
        logger.debug(f"📞 CALL {func_name}()")


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Логирует информацию о производительности.

    Args:
        operation: Название операции
        duration: Время выполнения в секундах
        **kwargs: Дополнительная информация
    """
    logger = get_logger()

    # Форматируем время
    if duration < 0.001:
        time_str = f"{duration*1000000:.1f}μs"
    elif duration < 1.0:
        time_str = f"{duration*1000:.1f}ms"
    else:
        time_str = f"{duration:.2f}s"

    # Дополнительная информация
    extra_info = ""
    if kwargs:
        extra_parts = [f"{k}={v}" for k, v in kwargs.items()]
        extra_info = f" ({', '.join(extra_parts)})"

    logger.info(f"⏱️ PERF {operation}: {time_str}{extra_info}")


# Функция для совместимости с legacy кодом
def _get_caller_info() -> str:
    """Функция совместимости для получения информации о caller."""
    formatter = ModuleTrackingFormatter("")
    return formatter._get_caller_info()
