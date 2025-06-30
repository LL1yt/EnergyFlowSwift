"""
Централизованная система логирования
===================================

Обеспечивает единообразное логирование с отслеживанием:
- Какой модуль был вызван
- Кем был вызван (caller info)
- Уровень детализации в зависимости от debug_mode

ИНТЕГРАЦИЯ: Включает функционал из core/log_utils.py для unified подхода
"""

import logging
import inspect
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Set
from pathlib import Path
import threading
import hashlib


def _get_caller_info_legacy() -> str:
    """
    Интегрированная функция из core/log_utils.py
    Возвращает информацию о вызывающем коде (файл, строка, функция).

    LEGACY COMPATIBILITY: Сохраняет оригинальную логику из core/log_utils.py
    """
    try:
        # inspect.stack() is slow, but for debugging it's acceptable.
        stack = inspect.stack()
        # Ищем первый фрейм не из этого файла
        for frame_info in stack[2:]:  # Go up 2 frames to get out of this helper
            if frame_info.filename != str(Path(__file__).resolve()):
                # Return a concise string with relevant info, relative to project root 'AA'
                try:
                    # Assumes the project root is named 'AA' or is the CWD
                    project_root = Path.cwd()
                    if "AA" not in project_root.parts:
                        # Fallback if not run from a subfolder of AA
                        rel_path = frame_info.filename
                    else:
                        # Find the 'AA' part and form the path from there
                        aa_index = project_root.parts.index("AA")
                        aa_root = Path(*project_root.parts[: aa_index + 1])
                        rel_path = os.path.relpath(frame_info.filename, start=aa_root)

                except (ValueError, TypeError):
                    rel_path = frame_info.filename
                return f"{rel_path}:{frame_info.lineno} ({frame_info.function})"
        return "N/A"
    except Exception:
        # This could fail in some environments (e.g. optimized Python), so we have a fallback.
        return "N/A"


class ModuleTrackingFormatter(logging.Formatter):
    """Форматтер с отслеживанием модулей и вызывающего кода."""

    def format(self, record):
        # Добавляем информацию о caller используя legacy логику
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
        """
        Получает информацию о вызывающем коде.
        ИСПРАВЛЕНО: Убираем ошибку 'code' object has no
        """
        try:
            # Ищем в стеке фрейм, который не относится к логированию
            frame = inspect.currentframe()

            # Безопасно проходим по стеку
            for level in range(15):  # Увеличиваем глубину поиска для клеток
                if frame is None:
                    break

                try:
                    # Безопасно получаем информацию о фрейме
                    filename = getattr(frame, "f_code", None)
                    if filename is None:
                        frame = frame.f_back
                        continue

                    filename = filename.co_filename
                    func_name = frame.f_code.co_function
                    lineno = frame.f_lineno

                except (AttributeError, TypeError) as e:
                    # Если не можем получить информацию о фрейме, пропускаем
                    frame = frame.f_back
                    continue

                # Пропускаем фреймы логирования и внутренние функции
                skip_files = ["logging", "log_utils", "__init__"]
                skip_functions = [
                    "__init__",
                    "setup_logging",
                    "get_logger",
                    "_log_parameter_count",
                    "format",
                    "_get_caller_info",
                ]

                try:
                    file_basename = Path(filename).name.lower()
                except (TypeError, OSError):
                    file_basename = str(filename).lower()

                # Более строгая проверка для исключения логирующих фреймов
                is_logging_frame = (
                    any(skip in file_basename for skip in skip_files)
                    or func_name in skip_functions
                    or "logging.py" in file_basename
                )

                if not is_logging_frame:
                    # Используем legacy логику для путей с улучшениями
                    try:
                        project_root = Path.cwd()
                        if "AA" not in str(project_root):
                            rel_path = Path(filename).name
                        else:
                            # Пытаемся получить относительный путь от AA
                            try:
                                if "AA" in project_root.parts:
                                    aa_index = project_root.parts.index("AA")
                                    aa_root = Path(*project_root.parts[: aa_index + 1])
                                    rel_path = os.path.relpath(filename, start=aa_root)
                                else:
                                    # Fallback - используем имя файла
                                    rel_path = Path(filename).name
                            except (ValueError, OSError):
                                rel_path = Path(filename).name
                    except (ValueError, TypeError, OSError):
                        rel_path = Path(filename).name if filename else "unknown_file"

                    return f"{rel_path}:{func_name}:{lineno}"

                frame = frame.f_back

            return "unknown"
        except Exception as e:
            # Более детальная информация об ошибке для отладки
            return f"error:caller_info_failed"


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


def setup_logging(
    debug_mode: bool = False,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_deduplication: bool = False,
    enable_context: bool = True,
) -> None:
    """
    Настраивает централизованное логирование.

    Args:
        debug_mode: Включить детальное логирование (переопределяет level)
        level: Уровень логирования ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Файл для записи логов (опционально)
        enable_deduplication: ОТКЛЮЧЕНО - может скрыть реальные проблемы в коде
        enable_context: Включить контекстное логирование
    """
    # Получаем root logger
    root_logger = logging.getLogger()

    # Очищаем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Определяем уровень логирования
    if debug_mode:
        # debug_mode переопределяет level
        log_level = logging.DEBUG
    elif level:
        # Используем заданный уровень
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(level.upper(), logging.INFO)
    else:
        # По умолчанию INFO
        log_level = logging.INFO
    
    # Устанавливаем уровень
    root_logger.setLevel(log_level)
    console_level = log_level

    # Создаем форматтер (выбираем тип в зависимости от настроек)
    if debug_mode:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        format_str = "%(asctime)s - %(levelname)s - %(message)s"

    # Выбираем форматтер
    if enable_context:
        formatter = ContextualFormatter(format_str)
    else:
        formatter = ModuleTrackingFormatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # Добавляем фильтры (БЕЗ дедупликации)
    console_handler.addFilter(DebugModeFilter(debug_mode))
    # НЕ добавляем AntiDuplicationFilter - может скрыть реальные проблемы

    root_logger.addHandler(console_handler)

    # File handler (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # В файл пишем все
        file_handler.setFormatter(formatter)
        # В файл тоже НЕ применяем дедупликацию
        root_logger.addHandler(file_handler)

    # Настраиваем логгеры для сторонних библиотек
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Логируем успешную настройку
    logger = get_logger("logging_setup")
    level_name = logging.getLevelName(log_level)
    logger.info(
        f"Logging configured: level={level_name}, debug={debug_mode}, context={enable_context}"
    )


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


# === СПЕЦИАЛЬНЫЕ ФУНКЦИИ ДЛЯ КЛЕТОК ===


def log_cell_init(
    cell_type: str, total_params: int, target_params: int, **kwargs
) -> None:
    """
    Специальная функция для логирования инициализации клеток.

    Args:
        cell_type: Тип клетки (NCA, gMLP, Hybrid)
        total_params: Фактическое количество параметров
        target_params: Целевое количество параметров
        **kwargs: Дополнительная информация
    """
    logger = get_logger()

    # Проверяем превышение параметров (только если target_params указан)
    if target_params is not None and total_params > target_params * 1.2:
        status = "⚠️ ПРЕВЫШЕНИЕ"
        logger.warning(
            f"🚀 INIT {cell_type}Cell: {total_params:,} params (target: {target_params:,}) - {status}"
        )
    elif target_params is not None:
        status = "✅ НОРМА"
        logger.info(
            f"🚀 INIT {cell_type}Cell: {total_params:,} params (target: {target_params:,}) - {status}"
        )
    else:
        # Нет целевого количества параметров
        logger.info(f"🚀 INIT {cell_type}Cell: {total_params:,} params")

    # Дополнительная информация
    for key, value in kwargs.items():
        logger.info(f"     {key.upper()}: {value}")


def log_cell_forward(
    cell_type: str, input_shapes: Dict[str, Any], output_shape: Any = None
) -> None:
    """
    Логирует forward pass клетки (только в debug_mode).

    Args:
        cell_type: Тип клетки
        input_shapes: Формы входных тензоров
        output_shape: Форма выходного тензора
    """
    logger = get_logger()

    # Форматируем информацию о входах
    input_info = []
    for name, shape in input_shapes.items():
        input_info.append(f"{name}: {shape}")

    input_str = ", ".join(input_info)

    if output_shape:
        logger.debug(f"📞 FORWARD {cell_type}Cell: {input_str} → {output_shape}")
    else:
        logger.debug(f"📞 FORWARD {cell_type}Cell: {input_str}")


def log_cell_component_params(
    component_params: Dict[str, int], total_params: int
) -> None:
    """
    Логирует детализацию параметров по компонентам клетки.

    Args:
        component_params: Словарь {компонент: количество_параметров}
        total_params: Общее количество параметров
    """
    logger = get_logger()

    for component, count in sorted(
        component_params.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_params) * 100 if total_params > 0 else 0
        logger.info(f"     {component}: {count:,} params ({percentage:.1f}%)")


# === LEGACY COMPATIBILITY ===


def _get_caller_info() -> str:
    """
    Функция совместимости с legacy кодом.
    Использует интегрированную функцию для получения caller info.
    """
    return _get_caller_info_legacy()


# Алиас для legacy совместимости
get_caller_info = _get_caller_info


# === СИСТЕМА ПРЕДОТВРАЩЕНИЯ ДУБЛИРОВАНИЯ ЛОГОВ ===


class DuplicationManager:  # deprecated
    """
    Менеджер для предотвращения дублирования логов.

    Принципы:
    1. Event-based logging - логируем события, не действия
    2. Context tracking - каждый лог с уникальным контекстом
    3. Time-based deduplication - игнорируем повторы в коротком периоде
    4. Hierarchical logging - четкие правила кто что логирует
    """

    def __init__(self, dedup_window_seconds: int = 1):
        self.dedup_window = timedelta(seconds=dedup_window_seconds)
        self.recent_logs: Dict[str, datetime] = {}
        self.lock = threading.Lock()

    def should_log(self, message: str, logger_name: str, level: str) -> bool:
        """
        Определяет нужно ли логировать сообщение или это дубликат.

        Args:
            message: Текст сообщения
            logger_name: Имя логгера
            level: Уровень логирования

        Returns:
            True если нужно логировать, False если дубликат
        """
        # Создаем уникальный ключ для сообщения
        key = self._create_message_key(message, logger_name, level)

        with self.lock:
            now = datetime.now()

            # Проверяем был ли недавно такой же лог
            if key in self.recent_logs:
                last_time = self.recent_logs[key]
                if now - last_time < self.dedup_window:
                    return False  # Дубликат, не логируем

            # Обновляем время последнего лога
            self.recent_logs[key] = now

            # Очищаем старые записи
            self._cleanup_old_entries(now)

            return True

    def _create_message_key(self, message: str, logger_name: str, level: str) -> str:
        """Создает уникальный ключ для сообщения"""
        # Нормализуем сообщение (убираем timestamps, изменяющиеся значения)
        normalized = self._normalize_message(message)
        key_string = f"{logger_name}:{level}:{normalized}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _normalize_message(self, message: str) -> str:
        """Нормализует сообщение убирая изменяющиеся части"""
        import re

        # Убираем timestamps
        message = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+", "", message)
        # Убираем размеры тензоров (они могут меняться)
        message = re.sub(r"torch\.Size\([^)]+\)", "torch.Size(...)", message)
        # Убираем числовые значения параметров (могут слегка меняться)
        message = re.sub(r"\d+,?\d* params", "X params", message)

        return message.strip()

    def _cleanup_old_entries(self, now: datetime):
        """Очищает старые записи для экономии памяти"""
        cutoff = now - self.dedup_window * 2  # Удаляем записи старше двойного окна
        old_keys = [key for key, time in self.recent_logs.items() if time < cutoff]
        for key in old_keys:
            del self.recent_logs[key]


# Глобальный менеджер дедупликации
_deduplication_manager = DuplicationManager()


class AntiDuplicationFilter(logging.Filter):
    """Фильтр для предотвращения дублирования логов"""

    def filter(self, record):
        # Проверяем нужно ли логировать
        return _deduplication_manager.should_log(
            record.getMessage(), record.name, record.levelname
        )


# === КОНТЕКСТНОЕ ЛОГИРОВАНИЕ ===


class LogContext:
    """
    Контекст для логирования - предотвращает дублирование через иерархию.

    Пример использования:
    with LogContext("cell_creation", cell_type="NCA"):
        # Все логи в этом контексте будут помечены
        logger.info("Creating cell...")  # Автоматически: [cell_creation:NCA] Creating cell...
    """

    _context_stack: list = []
    _lock = threading.Lock()

    def __init__(self, context_name: str, **context_data):
        self.context_name = context_name
        self.context_data = context_data

    def __enter__(self):
        with self._lock:
            self._context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            if self._context_stack and self._context_stack[-1] is self:
                self._context_stack.pop()

    @classmethod
    def get_current_context(cls) -> Optional["LogContext"]:
        """Получает текущий контекст логирования"""
        with cls._lock:
            return cls._context_stack[-1] if cls._context_stack else None

    def format_context(self) -> str:
        """Форматирует контекст для добавления в лог"""
        if self.context_data:
            data_str = ":".join(f"{k}={v}" for k, v in self.context_data.items())
            return f"[{self.context_name}:{data_str}]"
        return f"[{self.context_name}]"


class ContextualFormatter(logging.Formatter):
    """Форматтер который добавляет контекст к сообщениям"""

    def format(self, record):
        # Добавляем контекст если есть
        context = LogContext.get_current_context()
        if context:
            context_str = context.format_context()
            record.msg = f"{context_str} {record.msg}"

        return super().format(record)
