#!/usr/bin/env python3
"""
Проверка на использование hardcoded значений
============================================

Этот модуль помогает отслеживать использование hardcoded значений
в коде и требует использования централизованной конфигурации.
"""

import inspect
import functools
from typing import Any, Callable, List, Set, Union, Optional
from dataclasses import fields

# Отложенный импорт для избежания циклических зависимостей
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import SimpleProjectConfig

from .logging import get_logger

# Для обратной совместимости делаем функцию доступной на уровне модуля
_original_check_hardcoded_value = None

logger = get_logger(__name__)

# Функция для отложенного получения конфига
def _get_config():
    """Отложенный импорт конфига для избежания циклических зависимостей"""
    from ..config import get_project_config
    return get_project_config()


# Список значений, которые считаются hardcoded и должны быть в конфиге
FORBIDDEN_HARDCODED_VALUES = {
    # Learning rates
    1e-4, 1e-5, 5e-5, 1e-3,
    # Batch sizes и размеры
    8, 16, 32, 64, 100, 128, 256, 512, 1000, 10000,
    # Dropout и другие коэффициенты
    0.1, 0.2, 0.5, 0.8, 0.85, 0.9,
    # Размеры архитектуры
    768, 1024, 2000, 3000, 4000, 8000, 15000,
    # Временные параметры
    10, 50, 100, 500, 1000,
    # Memory limits
    2.0, 8.0, 10.0, 20.0,
}

# Разрешенные значения (базовые константы)
ALLOWED_VALUES = {
    0, 1, 2, 3, 4, 5, 6, 7,  # Маленькие числа для индексов
    -1,  # Часто используется для "последний"
    0.0, 1.0,  # Базовые float
    True, False, None,  # Булевы и None
}


class HardcodedValueError(Exception):
    """Исключение при обнаружении hardcoded значения"""
    pass


def check_hardcoded_value(value: Any, context: str = "") -> None:
    """
    Проверяет, является ли значение hardcoded.
    
    Args:
        value: Значение для проверки
        context: Контекст где найдено значение (для сообщения об ошибке)
        
    Raises:
        HardcodedValueError: Если значение является hardcoded
    """
    if value in ALLOWED_VALUES:
        return
        
    if value in FORBIDDEN_HARDCODED_VALUES:
        config = _get_config()
        mode = config.mode.mode.value
        
        # Формируем подсказку где искать значение в конфиге
        hint = _get_config_hint(value)
        
        raise HardcodedValueError(
            f"\n❌ ОБНАРУЖЕНО HARDCODED ЗНАЧЕНИЕ: {value}\n"
            f"📍 Контекст: {context}\n"
            f"🎯 Текущий режим: {mode.upper()}\n"
            f"💡 Подсказка: {hint}\n"
            f"📝 Используйте значение из централизованного конфига!\n"
            f"   Например: config.{hint}"
        )


def _get_config_hint(value: Any) -> str:
    """Возвращает подсказку где искать значение в конфиге"""
    hints = {
        # Learning rates
        1e-4: "training_optimizer.learning_rate",
        1e-5: "training_optimizer.weight_decay",
        
        # Dropout
        0.1: "embedding_mapping.dropout_rate или architecture.cnf_dropout_rate",
        
        # Memory
        8.0: "memory_management.min_gpu_memory_gb",
        20.0: "memory_management.training_memory_reserve_gb",
        0.85: "memory_management.gpu_memory_safety_factor",
        
        # Architecture
        768: "architecture.teacher_embedding_dim",
        8000: "architecture.moe_functional_params",
        4000: "architecture.moe_distant_params",
        
        # Spatial
        1000: "architecture.spatial_max_neighbors",
        100: "architecture.max_comparison_cells или memory_management.cleanup_threshold",
        
        # Training
        10: "training_optimizer.scheduler_t0 или training_optimizer.log_batch_frequency",
        
        # Прочее
        0.8: "embedding_mapping.surface_coverage или memory_management.memory_safety_factor",
    }
    
    return hints.get(value, "Проверьте training_optimizer, architecture, memory_management или embedding_mapping")


def no_hardcoded(func: Callable) -> Callable:
    """
    Декоратор для проверки hardcoded значений в функциях.
    
    Проверяет аргументы функции на наличие hardcoded значений.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Получаем информацию о функции
        func_name = func.__name__
        module = func.__module__
        context = f"{module}.{func_name}"
        
        # Проверяем позиционные аргументы
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)) and arg not in ALLOWED_VALUES:
                check_hardcoded_value(arg, f"{context} - аргумент {i}")
                
        # Проверяем именованные аргументы
        for key, value in kwargs.items():
            if isinstance(value, (int, float)) and value not in ALLOWED_VALUES:
                check_hardcoded_value(value, f"{context} - параметр '{key}'")
                
        return func(*args, **kwargs)
    
    return wrapper


def check_class_init(cls: type) -> None:
    """
    Проверяет __init__ метод класса на hardcoded значения в дефолтах.
    
    Args:
        cls: Класс для проверки
    """
    init_signature = inspect.signature(cls.__init__)
    
    for param_name, param in init_signature.parameters.items():
        if param.default is not inspect.Parameter.empty:
            if isinstance(param.default, (int, float)):
                try:
                    check_hardcoded_value(
                        param.default, 
                        f"{cls.__module__}.{cls.__name__}.__init__ - параметр '{param_name}'"
                    )
                except HardcodedValueError:
                    # Для классов выводим предупреждение, но не останавливаем
                    logger.warning(
                        f"⚠️ Hardcoded значение {param.default} в {cls.__name__}.{param_name}. "
                        f"Рекомендуется использовать config!"
                    )


def validate_no_hardcoded_in_module(module) -> List[str]:
    """
    Проверяет весь модуль на наличие hardcoded значений.
    
    Args:
        module: Python модуль для проверки
        
    Returns:
        Список найденных проблем
    """
    issues = []
    
    # Проверяем все классы в модуле
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:  # Только классы из этого модуля
            try:
                check_class_init(obj)
            except Exception as e:
                issues.append(f"Класс {name}: {str(e)}")
                
    # Проверяем все функции
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            # Проверяем дефолтные значения параметров
            sig = inspect.signature(obj)
            for param_name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    if isinstance(param.default, (int, float)) and param.default in FORBIDDEN_HARDCODED_VALUES:
                        issues.append(
                            f"Функция {name}.{param_name}: hardcoded значение {param.default}"
                        )
                        
    return issues


def strict_no_hardcoded(value: Union[int, float], param_name: str) -> Union[int, float]:
    """
    Строгая проверка значения с автоматическим получением из конфига.
    
    Использование:
        hidden_dim = strict_no_hardcoded(64, "model.hidden_dim")
    
    Args:
        value: Значение для проверки
        param_name: Путь к параметру в конфиге (например, "architecture.spatial_max_neighbors")
        
    Returns:
        Значение из конфига
        
    Raises:
        HardcodedValueError: Если используется hardcoded значение
    """
    if value in FORBIDDEN_HARDCODED_VALUES:
        config = _get_config()
        
        # Пытаемся получить значение из конфига по пути
        try:
            parts = param_name.split('.')
            result = config
            
            for part in parts:
                result = getattr(result, part)
                
            logger.debug(f"Заменено hardcoded {value} на {result} из config.{param_name}")
            return result
            
        except AttributeError:
            raise HardcodedValueError(
                f"\n❌ ЗАПРЕЩЕНО HARDCODED ЗНАЧЕНИЕ: {value}\n"
                f"📝 Параметр '{param_name}' не найден в конфиге!\n"
                f"💡 Добавьте его в соответствующий компонент конфигурации.\n"
                f"🎯 Текущий режим: {config.mode.mode.value.upper()}"
            )
    
    # Если значение разрешено, возвращаем как есть
    return value


# Контекстный менеджер для временного отключения проверок
class allow_hardcoded:
    """
    Контекстный менеджер для временного разрешения hardcoded значений.
    
    Использовать ТОЛЬКО в тестах или при миграции!
    
    with allow_hardcoded("migration in progress"):
        # Код с hardcoded значениями
    """
    
    _disabled = False
    
    def __init__(self, reason: str):
        self.reason = reason
        
    def __enter__(self):
        logger.warning(f"⚠️ Hardcoded проверки отключены: {self.reason}")
        allow_hardcoded._disabled = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        allow_hardcoded._disabled = False
        

# Переопределяем check_hardcoded_value чтобы учитывать контекст
def _check_hardcoded_value_with_context(value: Any, context: str = "") -> None:
    """Внутренняя версия с поддержкой контекста allow_hardcoded"""
    if allow_hardcoded._disabled:
        return
    
    # Вызываем оригинальную проверку
    if value in ALLOWED_VALUES:
        return
        
    if value in FORBIDDEN_HARDCODED_VALUES:
        config = _get_config()
        mode = config.mode.mode.value
        
        # Формируем подсказку где искать значение в конфиге
        hint = _get_config_hint(value)
        
        raise HardcodedValueError(
            f"\n❌ ОБНАРУЖЕНО HARDCODED ЗНАЧЕНИЕ: {value}\n"
            f"📍 Контекст: {context}\n"
            f"🎯 Текущий режим: {mode.upper()}\n"
            f"💡 Подсказка: {hint}\n"
            f"📝 Используйте значение из централизованного конфига!\n"
            f"   Например: config.{hint}"
        )

# Сохраняем ссылку на оригинальную функцию
_original_check_hardcoded_value = check_hardcoded_value

# Заменяем на версию с контекстом
check_hardcoded_value = _check_hardcoded_value_with_context