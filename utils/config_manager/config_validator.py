"""
ConfigValidator - система валидации конфигураций

Обеспечивает проверку корректности конфигурационных данных.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod
import re


class ValidationRule(ABC):
    """Базовый класс для правил валидации"""
    
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """
        Валидация значения.
        
        Args:
            value: Значение для проверки
            context: Контекст (полная конфигурация)
            
        Returns:
            Список ошибок (пустой если все ОК)
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Описание правила"""
        pass


class TypeRule(ValidationRule):
    """Проверка типа значения"""
    
    def __init__(self, expected_type: Union[type, List[type]], allow_none: bool = True):
        self.expected_type = expected_type if isinstance(expected_type, list) else [expected_type]
        self.allow_none = allow_none
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        if value is None and self.allow_none:
            return []
        
        if not any(isinstance(value, t) for t in self.expected_type):
            type_names = [t.__name__ for t in self.expected_type]
            return [f"Expected type {' or '.join(type_names)}, got {type(value).__name__}"]
        
        return []
    
    def get_description(self) -> str:
        type_names = [t.__name__ for t in self.expected_type]
        return f"Must be of type: {' or '.join(type_names)}"


class RangeRule(ValidationRule):
    """Проверка диапазона значений (для числовых типов)"""
    
    def __init__(self, min_value: Optional[Union[int, float]] = None, 
                 max_value: Optional[Union[int, float]] = None):
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        if not isinstance(value, (int, float)):
            return []  # Проверка типа - это задача TypeRule
        
        errors = []
        
        if self.min_value is not None and value < self.min_value:
            errors.append(f"Value {value} is less than minimum {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            errors.append(f"Value {value} is greater than maximum {self.max_value}")
        
        return errors
    
    def get_description(self) -> str:
        if self.min_value is not None and self.max_value is not None:
            return f"Must be between {self.min_value} and {self.max_value}"
        elif self.min_value is not None:
            return f"Must be at least {self.min_value}"
        elif self.max_value is not None:
            return f"Must be at most {self.max_value}"
        return "No range restrictions"


class ChoicesRule(ValidationRule):
    """Проверка что значение из списка разрешенных"""
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        if value not in self.choices:
            return [f"Value '{value}' not in allowed choices: {self.choices}"]
        return []
    
    def get_description(self) -> str:
        return f"Must be one of: {self.choices}"


class RegexRule(ValidationRule):
    """Проверка соответствия регулярному выражению"""
    
    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = pattern
        self.regex = re.compile(pattern, flags)
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        if not isinstance(value, str):
            return []  # Проверка типа - это задача TypeRule
        
        if not self.regex.match(value):
            return [f"Value '{value}' does not match pattern '{self.pattern}'"]
        
        return []
    
    def get_description(self) -> str:
        return f"Must match pattern: {self.pattern}"


class RequiredRule(ValidationRule):
    """Проверка обязательности поля"""
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        if value is None:
            return ["Field is required"]
        
        # Проверка пустых строк/списков/словарей
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return ["Field cannot be empty"]
        
        return []
    
    def get_description(self) -> str:
        return "Field is required"


class CustomRule(ValidationRule):
    """Пользовательское правило валидации"""
    
    def __init__(self, validator_func: Callable[[Any, Dict[str, Any]], List[str]], 
                 description: str):
        self.validator_func = validator_func
        self.description = description
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        return self.validator_func(value, context or {})
    
    def get_description(self) -> str:
        return self.description


class FieldValidator:
    """Валидатор для отдельного поля"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
        self.rules: List[ValidationRule] = []
    
    def add_rule(self, rule: ValidationRule) -> 'FieldValidator':
        """Добавление правила валидации"""
        self.rules.append(rule)
        return self
    
    def required(self) -> 'FieldValidator':
        """Поле обязательно"""
        return self.add_rule(RequiredRule())
    
    def type_check(self, expected_type: Union[type, List[type]], allow_none: bool = True) -> 'FieldValidator':
        """Проверка типа"""
        return self.add_rule(TypeRule(expected_type, allow_none))
    
    def range_check(self, min_value: Optional[Union[int, float]] = None, 
                   max_value: Optional[Union[int, float]] = None) -> 'FieldValidator':
        """Проверка диапазона"""
        return self.add_rule(RangeRule(min_value, max_value))
    
    def choices(self, allowed_values: List[Any]) -> 'FieldValidator':
        """Проверка выбора из списка"""
        return self.add_rule(ChoicesRule(allowed_values))
    
    def regex(self, pattern: str, flags: int = 0) -> 'FieldValidator':
        """Проверка регулярным выражением"""
        return self.add_rule(RegexRule(pattern, flags))
    
    def custom(self, validator_func: Callable[[Any, Dict[str, Any]], List[str]], 
              description: str) -> 'FieldValidator':
        """Пользовательская валидация"""
        return self.add_rule(CustomRule(validator_func, description))
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """Валидация значения поля"""
        all_errors = []
        
        for rule in self.rules:
            errors = rule.validate(value, context)
            all_errors.extend(errors)
        
        return all_errors


class ConfigValidator:
    """
    Основной класс для валидации конфигураций.
    
    Позволяет определить правила валидации для различных полей
    и проверить всю конфигурацию.
    """
    
    def __init__(self, section_name: str = "unknown"):
        self.section_name = section_name
        self.field_validators: Dict[str, FieldValidator] = {}
        self.logger = logging.getLogger(__name__)
    
    def field(self, field_name: str) -> FieldValidator:
        """
        Создание валидатора для поля.
        
        Args:
            field_name: Имя поля (поддерживает dot-notation)
            
        Returns:
            FieldValidator: Валидатор поля
        """
        if field_name not in self.field_validators:
            self.field_validators[field_name] = FieldValidator(field_name)
        
        return self.field_validators[field_name]
    
    def validate(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Валидация всей конфигурации.
        
        Args:
            config_data: Данные для валидации
            
        Returns:
            Список ошибок (пустой если все ОК)
        """
        all_errors = []
        
        for field_name, field_validator in self.field_validators.items():
            try:
                value = self._get_nested_value(config_data, field_name)
                field_errors = field_validator.validate(value, config_data)
                
                # Добавляем префикс с именем поля к ошибкам
                for error in field_errors:
                    all_errors.append(f"{field_name}: {error}")
                    
            except Exception as e:
                all_errors.append(f"{field_name}: Validation failed - {str(e)}")
        
        return all_errors
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Получение вложенного значения по dot-notation"""
        if '.' not in key:
            return data.get(key)
        
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current


class ConfigValidatorBuilder:
    """Builder для упрощения создания валидаторов конфигурации"""
    
    @staticmethod
    def for_section(section_name: str) -> ConfigValidator:
        """Создание валидатора для секции"""
        return ConfigValidator(section_name)
    
    @staticmethod
    def create_lattice_validator() -> ConfigValidator:
        """Создание валидатора для секции lattice_3d"""
        validator = ConfigValidator("lattice_3d")
        
        # Валидация размеров решетки
        validator.field("dimensions.depth").required().type_check(int).range_check(1, 128)
        validator.field("dimensions.height").required().type_check(int).range_check(1, 128)
        validator.field("dimensions.width").required().type_check(int).range_check(1, 128)
        
        # Валидация типа соединений
        validator.field("connectivity.type").required().choices([
            "6-neighbors", "18-neighbors", "26-neighbors"
        ])
        
        # Валидация граничных условий
        validator.field("boundary_conditions").choices([
            "walls", "periodic", "reflective", "absorbing"
        ])
        
        return validator
    
    @staticmethod
    def create_training_validator() -> ConfigValidator:
        """Создание валидатора для секции training"""
        validator = ConfigValidator("training")
        
        # Основные параметры обучения
        validator.field("batch_size").required().type_check(int).range_check(1, 1024)
        validator.field("learning_rate").required().type_check(float).range_check(1e-6, 1.0)
        validator.field("num_epochs").required().type_check(int).range_check(1, 10000)
        
        # Временные шаги
        validator.field("time_steps").required().type_check(int).range_check(1, 1000)
        
        # Оптимизатор
        validator.field("optimizer.type").required().choices(["Adam", "SGD", "RMSprop"])
        
        return validator
    
    @staticmethod
    def create_device_validator() -> ConfigValidator:
        """Создание валидатора для секции device"""
        validator = ConfigValidator("device")
        
        validator.field("use_gpu").required().type_check(bool)
        validator.field("gpu_device").type_check(str).regex(r"cuda:\d+")
        validator.field("fallback_to_cpu").required().type_check(bool)
        
        return validator


# Предустановленные валидаторы для основных секций проекта
DEFAULT_VALIDATORS = {
    "lattice_3d": ConfigValidatorBuilder.create_lattice_validator(),
    "training": ConfigValidatorBuilder.create_training_validator(),
    "device": ConfigValidatorBuilder.create_device_validator(),
} 