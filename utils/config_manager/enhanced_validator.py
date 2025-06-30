"""
Enhanced Config Validator - продвинутая система валидации конфигураций

Расширенные возможности:
- Schema-based validation с поддержкой JSON Schema
- Валидация зависимостей между полями
- Условная валидация
- Валидация структуры данных
- Интеграция с версионированием
- Асинхронная валидация
- Кэширование результатов валидации
"""

import json
import yaml
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from enum import Enum

# Расширяем базовые классы из config_validator
from .config_validator import ValidationRule, ConfigValidator, FieldValidator
from .config_versioning import ConfigVersionManager


class ValidationSeverity(Enum):
    """Уровни серьезности ошибок валидации"""
    ERROR = "error"      # Критическая ошибка, конфигурация недействительна
    WARNING = "warning"  # Предупреждение, но конфигурация может работать
    INFO = "info"        # Информационное сообщение
    HINT = "hint"        # Рекомендация для улучшения


@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    validated_fields: Set[str] = field(default_factory=set)
    
    def add_message(self, message: str, severity: ValidationSeverity, field_path: str = ""):
        """Добавление сообщения валидации"""
        full_message = f"{field_path}: {message}" if field_path else message
        
        if severity == ValidationSeverity.ERROR:
            self.errors.append(full_message)
            self.is_valid = False
        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(full_message)
        elif severity == ValidationSeverity.INFO:
            self.info.append(full_message)
        elif severity == ValidationSeverity.HINT:
            self.hints.append(full_message)
    
    def merge(self, other: 'ValidationResult'):
        """Объединение результатов валидации"""
        if not other.is_valid:
            self.is_valid = False
        
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.hints.extend(other.hints)
        self.validated_fields.update(other.validated_fields)
        self.validation_time += other.validation_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'hints': self.hints,
            'validation_time': self.validation_time,
            'validated_fields': list(self.validated_fields),
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_info': len(self.info),
                'total_hints': len(self.hints)
            }
        }


class SchemaValidationRule(ValidationRule):
    """Валидация по JSON Schema"""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Args:
            schema: JSON Schema для валидации
        """
        self.schema = schema
        try:
            import jsonschema
            self.validator = jsonschema.Draft7Validator(schema)
            self._jsonschema_available = True
        except ImportError:
            self._jsonschema_available = False
            logging.warning("jsonschema package not available, using basic validation")
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """Валидация значения по схеме"""
        if not self._jsonschema_available:
            return self._basic_schema_validation(value)
        
        errors = []
        for error in self.validator.iter_errors(value):
            errors.append(f"Schema validation error: {error.message}")
        
        return errors
    
    def _basic_schema_validation(self, value: Any) -> List[str]:
        """Базовая валидация схемы без jsonschema"""
        errors = []
        
        # Проверяем тип
        if 'type' in self.schema:
            expected_type = self.schema['type']
            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"Expected string, got {type(value).__name__}")
            elif expected_type == 'number' and not isinstance(value, (int, float)):
                errors.append(f"Expected number, got {type(value).__name__}")
            elif expected_type == 'integer' and not isinstance(value, int):
                errors.append(f"Expected integer, got {type(value).__name__}")
            elif expected_type == 'boolean' and not isinstance(value, bool):
                errors.append(f"Expected boolean, got {type(value).__name__}")
            elif expected_type == 'array' and not isinstance(value, list):
                errors.append(f"Expected array, got {type(value).__name__}")
            elif expected_type == 'object' and not isinstance(value, dict):
                errors.append(f"Expected object, got {type(value).__name__}")
        
        return errors
    
    def get_description(self) -> str:
        return f"JSON Schema validation: {self.schema.get('title', 'Unnamed schema')}"


class DependencyValidationRule(ValidationRule):
    """Валидация зависимостей между полями"""
    
    def __init__(self, 
                 dependent_field: str,
                 dependency_condition: Callable[[Any], bool],
                 dependency_message: str = "Dependency validation failed"):
        """
        Args:
            dependent_field: Поле от которого зависим
            dependency_condition: Условие проверки зависимости
            dependency_message: Сообщение об ошибке
        """
        self.dependent_field = dependent_field
        self.dependency_condition = dependency_condition
        self.dependency_message = dependency_message
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """Валидация зависимости"""
        if context is None:
            return ["Context required for dependency validation"]
        
        dependent_value = self._get_nested_value(context, self.dependent_field)
        
        if not self.dependency_condition(dependent_value):
            return [self.dependency_message]
        
        return []
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Получение вложенного значения"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get_description(self) -> str:
        return f"Dependency on {self.dependent_field}: {self.dependency_message}"


class ConditionalValidationRule(ValidationRule):
    """Условная валидация (если-то правила)"""
    
    def __init__(self, 
                 condition: Callable[[Any, Dict[str, Any]], bool],
                 validation_rule: ValidationRule,
                 condition_description: str = ""):
        """
        Args:
            condition: Условие применения правила
            validation_rule: Правило для применения
            condition_description: Описание условия
        """
        self.condition = condition
        self.validation_rule = validation_rule
        self.condition_description = condition_description
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """Условная валидация"""
        if self.condition(value, context or {}):
            return self.validation_rule.validate(value, context)
        return []
    
    def get_description(self) -> str:
        base_desc = self.validation_rule.get_description()
        if self.condition_description:
            return f"If {self.condition_description}: {base_desc}"
        return f"Conditional: {base_desc}"


class StructureValidationRule(ValidationRule):
    """Валидация структуры сложных объектов"""
    
    def __init__(self, required_keys: List[str] = None, 
                 optional_keys: List[str] = None,
                 forbidden_keys: List[str] = None):
        """
        Args:
            required_keys: Обязательные ключи
            optional_keys: Опциональные ключи
            forbidden_keys: Запрещенные ключи
        """
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        self.forbidden_keys = forbidden_keys or []
        self.allowed_keys = set(self.required_keys + self.optional_keys)
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> List[str]:
        """Валидация структуры"""
        if not isinstance(value, dict):
            return ["Value must be a dictionary for structure validation"]
        
        errors = []
        value_keys = set(value.keys())
        
        # Проверяем обязательные ключи
        missing_required = set(self.required_keys) - value_keys
        if missing_required:
            errors.append(f"Missing required keys: {sorted(missing_required)}")
        
        # Проверяем запрещенные ключи
        forbidden_present = set(self.forbidden_keys) & value_keys
        if forbidden_present:
            errors.append(f"Forbidden keys present: {sorted(forbidden_present)}")
        
        # Проверяем неизвестные ключи (если определены allowed_keys)
        if self.allowed_keys:
            unknown_keys = value_keys - self.allowed_keys
            if unknown_keys:
                errors.append(f"Unknown keys: {sorted(unknown_keys)}")
        
        return errors
    
    def get_description(self) -> str:
        desc_parts = []
        if self.required_keys:
            desc_parts.append(f"Required: {self.required_keys}")
        if self.optional_keys:
            desc_parts.append(f"Optional: {self.optional_keys}")
        if self.forbidden_keys:
            desc_parts.append(f"Forbidden: {self.forbidden_keys}")
        
        return f"Structure validation - {', '.join(desc_parts)}"


class EnhancedFieldValidator(FieldValidator):
    """Расширенный валидатор поля с дополнительными возможностями"""
    
    def __init__(self, field_name: str):
        super().__init__(field_name)
        self._severity_rules: Dict[ValidationRule, ValidationSeverity] = {}
    
    def add_rule_with_severity(self, rule: ValidationRule, severity: ValidationSeverity = ValidationSeverity.ERROR) -> 'EnhancedFieldValidator':
        """Добавление правила с указанием серьезности"""
        self.add_rule(rule)
        self._severity_rules[rule] = severity
        return self
    
    def schema(self, schema: Dict[str, Any], severity: ValidationSeverity = ValidationSeverity.ERROR) -> 'EnhancedFieldValidator':
        """JSON Schema валидация"""
        rule = SchemaValidationRule(schema)
        return self.add_rule_with_severity(rule, severity)
    
    def depends_on(self, field_path: str, condition: Callable[[Any], bool], 
                   message: str = "Dependency not satisfied",
                   severity: ValidationSeverity = ValidationSeverity.ERROR) -> 'EnhancedFieldValidator':
        """Валидация зависимости"""
        rule = DependencyValidationRule(field_path, condition, message)
        return self.add_rule_with_severity(rule, severity)
    
    def conditional(self, condition: Callable[[Any, Dict[str, Any]], bool],
                    validation_rule: ValidationRule, 
                    condition_desc: str = "",
                    severity: ValidationSeverity = ValidationSeverity.ERROR) -> 'EnhancedFieldValidator':
        """Условная валидация"""
        rule = ConditionalValidationRule(condition, validation_rule, condition_desc)
        return self.add_rule_with_severity(rule, severity)
    
    def structure(self, required_keys: List[str] = None, 
                  optional_keys: List[str] = None,
                  forbidden_keys: List[str] = None,
                  severity: ValidationSeverity = ValidationSeverity.ERROR) -> 'EnhancedFieldValidator':
        """Валидация структуры"""
        rule = StructureValidationRule(required_keys, optional_keys, forbidden_keys)
        return self.add_rule_with_severity(rule, severity)
    
    def validate_enhanced(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Расширенная валидация с результатом"""
        result = ValidationResult()
        result.validated_fields.add(self.field_name)
        
        for rule in self.rules:
            try:
                errors = rule.validate(value, context)
                severity = self._severity_rules.get(rule, ValidationSeverity.ERROR)
                
                for error in errors:
                    result.add_message(error, severity, self.field_name)
                    
            except Exception as e:
                result.add_message(f"Validation failed: {str(e)}", ValidationSeverity.ERROR, self.field_name)
        
        return result


class EnhancedConfigValidator(ConfigValidator):
    """
    Расширенный валидатор конфигурации с поддержкой схем и версионирования.
    
    Новые возможности:
    - Schema-based validation
    - Валидация зависимостей
    - Условная валидация  
    - Интеграция с версионированием
    - Асинхронная валидация
    - Кэширование результатов
    """
    
    def __init__(self, section_name: str = "unknown", version_manager: Optional[ConfigVersionManager] = None):
        super().__init__(section_name)
        self.version_manager = version_manager
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._enable_caching = True
        
        # Переопределяем field_validators для использования Enhanced версии
        self.field_validators: Dict[str, EnhancedFieldValidator] = {}
    
    def field(self, field_name: str) -> EnhancedFieldValidator:
        """Создание расширенного валидатора поля"""
        if field_name not in self.field_validators:
            self.field_validators[field_name] = EnhancedFieldValidator(field_name)
        
        return self.field_validators[field_name]
    
    def load_schema_from_file(self, schema_file: str):
        """Загрузка схемы из файла"""
        schema_path = Path(schema_file)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        try:
            if schema_path.suffix.lower() == '.json':
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
            elif schema_path.suffix.lower() in ['.yaml', '.yml']:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported schema file format: {schema_path.suffix}")
            
            self._apply_schema(schema)
            self.logger.info(f"[OK] Loaded schema from {schema_file}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading schema from {schema_file}: {e}")
            raise
    
    def _apply_schema(self, schema: Dict[str, Any]):
        """Применение схемы к валидатору"""
        if 'properties' in schema:
            for field_name, field_schema in schema['properties'].items():
                field_validator = self.field(field_name)
                
                # Применяем JSON Schema к полю
                field_validator.schema(field_schema)
                
                # Добавляем обязательность если указано
                if field_name in schema.get('required', []):
                    field_validator.required()
    
    def validate_enhanced(self, config_data: Dict[str, Any], enable_caching: bool = None) -> ValidationResult:
        """
        Расширенная валидация с детальным результатом.
        
        Args:
            config_data: Данные для валидации
            enable_caching: Использовать кэширование (по умолчанию self._enable_caching)
            
        Returns:
            ValidationResult: Детальный результат валидации
        """
        import time
        start_time = time.time()
        
        # Проверяем кэш
        if (enable_caching if enable_caching is not None else self._enable_caching):
            config_hash = self._calculate_config_hash(config_data)
            if config_hash in self._validation_cache:
                cached_result = self._validation_cache[config_hash]
                self.logger.debug(f"[TARGET] Using cached validation result for {self.section_name}")
                return cached_result
        
        # Выполняем валидацию
        overall_result = ValidationResult()
        
        for field_name, field_validator in self.field_validators.items():
            try:
                value = self._get_nested_value(config_data, field_name)
                field_result = field_validator.validate_enhanced(value, config_data)
                overall_result.merge(field_result)
                
            except Exception as e:
                overall_result.add_message(
                    f"Validation failed: {str(e)}", 
                    ValidationSeverity.ERROR, 
                    field_name
                )
        
        # Записываем время валидации
        overall_result.validation_time = time.time() - start_time
        
        # Кэшируем результат
        if (enable_caching if enable_caching is not None else self._enable_caching):
            config_hash = self._calculate_config_hash(config_data)
            self._validation_cache[config_hash] = overall_result
        
        return overall_result
    
    async def validate_async(self, config_data: Dict[str, Any]) -> ValidationResult:
        """Асинхронная валидация"""
        # Выполняем в thread pool для CPU-intensive операций
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate_enhanced, config_data)
    
    def validate_with_version_check(self, config_data: Dict[str, Any], expected_version: str = None) -> ValidationResult:
        """
        Валидация с проверкой версии.
        
        Args:
            config_data: Данные для валидации
            expected_version: Ожидаемая версия конфигурации
            
        Returns:
            ValidationResult: Результат валидации
        """
        result = self.validate_enhanced(config_data)
        
        # Проверяем версию если указана
        if expected_version and self.version_manager:
            config_version = config_data.get('version')
            if config_version != expected_version:
                result.add_message(
                    f"Version mismatch: expected {expected_version}, got {config_version}",
                    ValidationSeverity.WARNING
                )
        
        return result
    
    def clear_cache(self):
        """Очистка кэша валидации"""
        self._validation_cache.clear()
        self.logger.info(f"[CLEAN] Cleared validation cache for {self.section_name}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Статистика валидации"""
        return {
            'section_name': self.section_name,
            'total_field_validators': len(self.field_validators),
            'cache_size': len(self._validation_cache),
            'caching_enabled': self._enable_caching,
            'field_list': list(self.field_validators.keys())
        }
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Вычисление hash конфигурации для кэширования"""
        import hashlib
        config_str = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()


class SchemaManager:
    """Менеджер схем валидации"""
    
    def __init__(self, schemas_dir: str = "config/schemas"):
        """
        Args:
            schemas_dir: Директория со схемами
        """
        self.schemas_dir = Path(schemas_dir)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Загрузка всех схем из директории"""
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                schema_name = schema_file.stem
                self._schemas[schema_name] = schema
                self.logger.info(f"[INFO] Loaded schema: {schema_name}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error loading schema {schema_file}: {e}")
        
        for schema_file in self.schemas_dir.glob("*.yaml"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema = yaml.safe_load(f)
                
                schema_name = schema_file.stem
                self._schemas[schema_name] = schema
                self.logger.info(f"[INFO] Loaded schema: {schema_name}")
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error loading schema {schema_file}: {e}")
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Получение схемы по имени"""
        return self._schemas.get(schema_name)
    
    def create_validator(self, schema_name: str, section_name: str = None) -> Optional[EnhancedConfigValidator]:
        """
        Создание валидатора на основе схемы.
        
        Args:
            schema_name: Имя схемы
            section_name: Имя секции (по умолчанию schema_name)
            
        Returns:
            EnhancedConfigValidator или None если схема не найдена
        """
        schema = self.get_schema(schema_name)
        if schema is None:
            self.logger.error(f"Schema {schema_name} not found")
            return None
        
        validator = EnhancedConfigValidator(section_name or schema_name)
        validator._apply_schema(schema)
        
        return validator
    
    def list_schemas(self) -> List[str]:
        """Список доступных схем"""
        return list(self._schemas.keys())


# =============================================================================
# ПРЕДУСТАНОВЛЕННЫЕ СХЕМЫ И ВАЛИДАТОРЫ
# =============================================================================

def create_lattice_3d_enhanced_validator() -> EnhancedConfigValidator:
    """Создание расширенного валидатора для lattice_3d"""
    validator = EnhancedConfigValidator("lattice_3d")
    
    # Размеры решетки с зависимостями
    validator.field("dimensions.depth").required().type_check(int).range_check(1, 128)
    validator.field("dimensions.height").required().type_check(int).range_check(1, 128)
    validator.field("dimensions.width").required().type_check(int).range_check(1, 128)
    
    # Проверка что общий размер не слишком большой
    def check_total_size(value, context):
        dims = context.get('dimensions', {})
        total = dims.get('depth', 1) * dims.get('height', 1) * dims.get('width', 1)
        return total <= 1000000  # Максимум 1M клеток
    
    validator.field("dimensions").custom(
        lambda v, ctx: [] if check_total_size(v, ctx) else ["Total lattice size too large (>1M cells)"],
        "Total size validation"
    )
    
    # Тип соединений с условной валидацией
    validator.field("connectivity.type").required().choices([
        "6-neighbors", "18-neighbors", "26-neighbors"
    ])
    
    # Weight sharing зависит от типа соединений
    validator.field("connectivity.weight_sharing").depends_on(
        "connectivity.type",
        lambda conn_type: conn_type in ["6-neighbors", "18-neighbors"],
        "Weight sharing only supported for 6 and 18 neighbors"
    )
    
    # Граничные условия
    validator.field("boundary_conditions").choices([
        "walls", "periodic", "reflective", "absorbing"
    ])
    
    return validator


def create_training_enhanced_validator() -> EnhancedConfigValidator:
    """Создание расширенного валидатора для training"""
    validator = EnhancedConfigValidator("training")
    
    # Основные параметры
    validator.field("batch_size").required().type_check(int).range_check(1, 1024)
    validator.field("learning_rate").required().type_check(float).range_check(1e-6, 1.0)
    validator.field("num_epochs").required().type_check(int).range_check(1, 10000)
    
    # Оптимизатор с валидацией структуры
    validator.field("optimizer").structure(
        required_keys=["type"],
        optional_keys=["lr", "weight_decay", "momentum", "betas"]
    )
    
    validator.field("optimizer.type").required().choices(["Adam", "SGD", "RMSprop", "AdamW"])
    
    # Learning rate scheduler - условная валидация
    def has_scheduler(value, context):
        return "scheduler" in context
    
    validator.field("scheduler.type").conditional(
        has_scheduler,
        validator.field("scheduler.type").choices(["StepLR", "ExponentialLR", "ReduceLROnPlateau"]).rules[0],
        "scheduler is present"
    )
    
    return validator


# Предустановленные расширенные валидаторы
ENHANCED_VALIDATORS = {
    "lattice_3d": create_lattice_3d_enhanced_validator(),
    "training": create_training_enhanced_validator(),
}