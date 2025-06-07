"""
ConfigSchema - определение схем для конфигураций

Предоставляет декларативный способ описания структуры конфигурации.
"""

from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum


class FieldType(Enum):
    """Типы полей конфигурации"""
    STRING = "string"
    INTEGER = "integer" 
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class FieldSchema:
    """Схема для отдельного поля"""
    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    description: str = ""
    
    # Валидационные параметры
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex паттерн
    
    # Для списков и словарей
    item_type: Optional[FieldType] = None
    nested_schema: Optional['ConfigSchema'] = None
    
    def __post_init__(self):
        """Валидация схемы поля"""
        if self.field_type == FieldType.LIST and self.item_type is None:
            raise ValueError(f"Field {self.name}: LIST type requires item_type")
        
        if self.field_type == FieldType.DICT and self.nested_schema is None:
            raise ValueError(f"Field {self.name}: DICT type requires nested_schema")


@dataclass 
class ConfigSchema:
    """
    Схема конфигурации.
    
    Определяет структуру, типы полей и правила валидации.
    """
    name: str
    description: str = ""
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    
    def add_field(self, field_schema: FieldSchema) -> 'ConfigSchema':
        """Добавление поля в схему"""
        self.fields[field_schema.name] = field_schema
        return self
    
    def string_field(self, name: str, required: bool = True, 
                    default: str = None, description: str = "",
                    pattern: str = None, choices: List[str] = None) -> 'ConfigSchema':
        """Добавление строкового поля"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.STRING,
            required=required,
            default=default,
            description=description,
            pattern=pattern,
            choices=choices
        )
        return self.add_field(field_schema)
    
    def int_field(self, name: str, required: bool = True,
                 default: int = None, description: str = "",
                 min_value: int = None, max_value: int = None) -> 'ConfigSchema':
        """Добавление целочисленного поля"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.INTEGER,
            required=required,
            default=default,
            description=description,
            min_value=min_value,
            max_value=max_value
        )
        return self.add_field(field_schema)
    
    def float_field(self, name: str, required: bool = True,
                   default: float = None, description: str = "",
                   min_value: float = None, max_value: float = None) -> 'ConfigSchema':
        """Добавление поля с плавающей точкой"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.FLOAT,
            required=required,
            default=default,
            description=description,
            min_value=min_value,
            max_value=max_value
        )
        return self.add_field(field_schema)
    
    def bool_field(self, name: str, required: bool = True,
                  default: bool = None, description: str = "") -> 'ConfigSchema':
        """Добавление булевого поля"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.BOOLEAN,
            required=required,
            default=default,
            description=description
        )
        return self.add_field(field_schema)
    
    def list_field(self, name: str, item_type: FieldType,
                  required: bool = True, default: List = None,
                  description: str = "") -> 'ConfigSchema':
        """Добавление поля-списка"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.LIST,
            required=required,
            default=default or [],
            description=description,
            item_type=item_type
        )
        return self.add_field(field_schema)
    
    def dict_field(self, name: str, nested_schema: 'ConfigSchema',
                  required: bool = True, default: Dict = None,
                  description: str = "") -> 'ConfigSchema':
        """Добавление поля-словаря с вложенной схемой"""
        field_schema = FieldSchema(
            name=name,
            field_type=FieldType.DICT,
            required=required,
            default=default or {},
            description=description,
            nested_schema=nested_schema
        )
        return self.add_field(field_schema)
    
    def validate(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Валидация конфигурации по схеме.
        
        Args:
            config_data: Данные для валидации
            
        Returns:
            Список ошибок (пустой если все ОК)
        """
        errors = []
        
        # Проверка обязательных полей
        for field_name, field_schema in self.fields.items():
            if field_schema.required and field_name not in config_data:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            if field_name not in config_data:
                continue  # Необязательное поле отсутствует - это нормально
            
            value = config_data[field_name]
            field_errors = self._validate_field(field_schema, value)
            errors.extend([f"{field_name}: {error}" for error in field_errors])
        
        # Проверка неизвестных полей
        for field_name in config_data:
            if field_name not in self.fields:
                errors.append(f"Unknown field '{field_name}'")
        
        return errors
    
    def apply_defaults(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Применение значений по умолчанию.
        
        Args:
            config_data: Исходные данные
            
        Returns:
            Данные с примененными defaults
        """
        result = config_data.copy()
        
        for field_name, field_schema in self.fields.items():
            if field_name not in result and field_schema.default is not None:
                result[field_name] = field_schema.default
        
        return result
    
    def get_field_info(self, field_name: str) -> Optional[FieldSchema]:
        """Получение информации о поле"""
        return self.fields.get(field_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация схемы в словарь (для сериализации)"""
        return {
            "name": self.name,
            "description": self.description,
            "fields": {
                name: {
                    "type": field.field_type.value,
                    "required": field.required,
                    "default": field.default,
                    "description": field.description,
                    "min_value": field.min_value,
                    "max_value": field.max_value,
                    "choices": field.choices,
                    "pattern": field.pattern
                }
                for name, field in self.fields.items()
            }
        }
    
    def _validate_field(self, field_schema: FieldSchema, value: Any) -> List[str]:
        """Валидация отдельного поля"""
        errors = []
        
        # Проверка типа
        if not self._check_type(value, field_schema.field_type):
            expected_type = field_schema.field_type.value
            actual_type = type(value).__name__
            errors.append(f"Expected {expected_type}, got {actual_type}")
            return errors  # Если тип неправильный, дальше проверять нет смысла
        
        # Проверка диапазона (для чисел)
        if field_schema.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
            if field_schema.min_value is not None and value < field_schema.min_value:
                errors.append(f"Value {value} is less than minimum {field_schema.min_value}")
            
            if field_schema.max_value is not None and value > field_schema.max_value:
                errors.append(f"Value {value} is greater than maximum {field_schema.max_value}")
        
        # Проверка вариантов выбора
        if field_schema.choices and value not in field_schema.choices:
            errors.append(f"Value '{value}' not in allowed choices: {field_schema.choices}")
        
        # Проверка паттерна (для строк)
        if field_schema.pattern and field_schema.field_type == FieldType.STRING:
            import re
            if not re.match(field_schema.pattern, value):
                errors.append(f"Value '{value}' does not match pattern '{field_schema.pattern}'")
        
        # Проверка элементов списка
        if field_schema.field_type == FieldType.LIST and field_schema.item_type:
            for i, item in enumerate(value):
                if not self._check_type(item, field_schema.item_type):
                    expected_type = field_schema.item_type.value
                    actual_type = type(item).__name__
                    errors.append(f"Item {i}: Expected {expected_type}, got {actual_type}")
        
        # Проверка вложенного словаря
        if field_schema.field_type == FieldType.DICT and field_schema.nested_schema:
            nested_errors = field_schema.nested_schema.validate(value)
            errors.extend(nested_errors)
        
        return errors
    
    def _check_type(self, value: Any, field_type: FieldType) -> bool:
        """Проверка соответствия типа"""
        type_mapping = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),  # int также подходит для float
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.ANY: object  # Любой тип
        }
        
        expected_types = type_mapping.get(field_type, object)
        return isinstance(value, expected_types)


class SchemaBuilder:
    """Builder для упрощения создания схем"""
    
    @staticmethod
    def create_lattice_schema() -> ConfigSchema:
        """Создание схемы для lattice_3d"""
        schema = ConfigSchema("lattice_3d", "3D решетка клеток")
        
        # Размеры решетки
        dimensions_schema = ConfigSchema("dimensions", "Размеры решетки")
        dimensions_schema.int_field("depth", description="Глубина решетки", min_value=1, max_value=128)
        dimensions_schema.int_field("height", description="Высота решетки", min_value=1, max_value=128) 
        dimensions_schema.int_field("width", description="Ширина решетки", min_value=1, max_value=128)
        
        schema.dict_field("dimensions", dimensions_schema, description="Размеры 3D решетки")
        
        # Тип связности
        connectivity_schema = ConfigSchema("connectivity", "Настройки связности")
        connectivity_schema.string_field("type", choices=["6-neighbors", "18-neighbors", "26-neighbors"])
        connectivity_schema.bool_field("include_diagonals", default=False)
        
        schema.dict_field("connectivity", connectivity_schema)
        
        # Граничные условия  
        schema.string_field("boundary_conditions", 
                           choices=["walls", "periodic", "reflective", "absorbing"],
                           default="walls")
        
        return schema
    
    @staticmethod
    def create_training_schema() -> ConfigSchema:
        """Создание схемы для training"""
        schema = ConfigSchema("training", "Настройки обучения")
        
        schema.int_field("batch_size", min_value=1, max_value=1024, default=4)
        schema.float_field("learning_rate", min_value=1e-6, max_value=1.0, default=0.001)
        schema.int_field("num_epochs", min_value=1, max_value=10000, default=100)
        schema.int_field("time_steps", min_value=1, max_value=1000, default=10)
        
        # Оптимизатор
        optimizer_schema = ConfigSchema("optimizer", "Настройки оптимизатора")
        optimizer_schema.string_field("type", choices=["Adam", "SGD", "RMSprop"], default="Adam")
        optimizer_schema.float_field("weight_decay", min_value=0.0, max_value=1.0, default=0.0001)
        
        schema.dict_field("optimizer", optimizer_schema)
        
        return schema
    
    @staticmethod
    def create_device_schema() -> ConfigSchema:
        """Создание схемы для device"""
        schema = ConfigSchema("device", "Настройки устройства")
        
        schema.bool_field("use_gpu", default=False)
        schema.string_field("gpu_device", pattern=r"cuda:\d+", default="cuda:0", required=False)
        schema.bool_field("fallback_to_cpu", default=True)
        
        return schema


# Предустановленные схемы для основных секций
DEFAULT_SCHEMAS = {
    "lattice_3d": SchemaBuilder.create_lattice_schema(),
    "training": SchemaBuilder.create_training_schema(), 
    "device": SchemaBuilder.create_device_schema(),
} 