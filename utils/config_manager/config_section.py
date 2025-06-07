"""
ConfigSection - удобный wrapper для работы с секциями конфигурации

Предоставляет объектно-ориентированный интерфейс для работы с конфигурацией.
"""

from typing import Dict, Any, Optional, Union, List
import logging


class ConfigSection:
    """
    Wrapper для работы с отдельной секцией конфигурации.
    
    Обеспечивает:
    - Dot-notation доступ к настройкам
    - Валидацию изменений
    - Автоматическое обновление родительской конфигурации
    - Type hints и автодополнение
    """
    
    def __init__(self, name: str, data: Dict[str, Any], config_manager=None):
        """
        Инициализация секции конфигурации.
        
        Args:
            name: Название секции
            data: Данные секции
            config_manager: Ссылка на родительский ConfigManager
        """
        self._name = name
        self._data = data.copy()
        self._config_manager = config_manager
        self._logger = logging.getLogger(__name__)
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения по ключу с поддержкой dot-notation.
        
        Args:
            key: Ключ (может содержать точки для вложенных значений)
            default: Значение по умолчанию
            
        Returns:
            Значение или default
            
        Example:
            section.get('database.host')
            section.get('timeout', 30)
        """
        return self._get_nested_value(self._data, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения по ключу.
        
        Args:
            key: Ключ (поддерживает dot-notation)
            value: Новое значение
            
        Example:
            section.set('database.host', 'localhost')
            section.set('timeout', 60)
        """
        self._set_nested_value(self._data, key, value)
        
        # Обновляем родительскую конфигурацию
        if self._config_manager:
            self._config_manager.set_config(self._name, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Множественное обновление значений.
        
        Args:
            updates: Словарь обновлений
            
        Example:
            section.update({
                'database.host': 'localhost',
                'database.port': 5432,
                'timeout': 30
            })
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def has(self, key: str) -> bool:
        """
        Проверка существования ключа.
        
        Args:
            key: Ключ для проверки
            
        Returns:
            True если ключ существует
        """
        return self._get_nested_value(self._data, key) is not None
    
    def keys(self) -> List[str]:
        """
        Получение списка всех ключей верхнего уровня.
        
        Returns:
            Список ключей
        """
        return list(self._data.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертация в обычный словарь.
        
        Returns:
            Копия данных секции
        """
        return self._data.copy()
    
    def validate(self) -> List[str]:
        """
        Валидация текущей конфигурации.
        
        Returns:
            Список ошибок валидации (пустой если все ОК)
        """
        if self._config_manager:
            return self._config_manager._validate_section(self._name, self._data)
        return []
    
    def reload(self) -> None:
        """Перезагрузка секции из файлов"""
        if self._config_manager:
            self._config_manager.reload_config(self._name)
            # Обновляем локальные данные
            self._data = self._config_manager.get_config(self._name, {})
    
    def export(self, file_path: str, format: str = 'yaml') -> None:
        """
        Экспорт секции в файл.
        
        Args:
            file_path: Путь для сохранения
            format: Формат файла ('yaml' или 'json')
        """
        if self._config_manager:
            self._config_manager.export_config(file_path, format, self._name)
    
    # Поддержка dict-like интерфейса
    def __getitem__(self, key: str) -> Any:
        """Поддержка section['key'] синтаксиса"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found in section '{self._name}'")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Поддержка section['key'] = value синтаксиса"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Поддержка 'key' in section синтаксиса"""
        return self.has(key)
    
    def __str__(self) -> str:
        """Строковое представление"""
        return f"ConfigSection('{self._name}', {len(self._data)} keys)"
    
    def __repr__(self) -> str:
        """Детальное представление"""
        return f"ConfigSection(name='{self._name}', keys={list(self._data.keys())})"
    
    # Методы для работы с вложенными значениями
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Получение вложенного значения по dot-notation"""
        if '.' not in key:
            return data.get(key, default)
        
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """Установка вложенного значения по dot-notation"""
        if '.' not in key:
            data[key] = value
            return
        
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value


class TypedConfigSection(ConfigSection):
    """
    Типизированная версия ConfigSection с автоматическим приведением типов.
    
    Полезно для конфигураций с известной структурой.
    """
    
    def __init__(self, name: str, data: Dict[str, Any], 
                 type_mapping: Dict[str, type] = None, 
                 config_manager=None):
        """
        Инициализация типизированной секции.
        
        Args:
            name: Название секции
            data: Данные секции
            type_mapping: Карта типов для ключей
            config_manager: Ссылка на родительский ConfigManager
        """
        super().__init__(name, data, config_manager)
        self._type_mapping = type_mapping or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получение с автоматическим приведением типа"""
        value = super().get(key, default)
        
        if key in self._type_mapping and value is not None:
            target_type = self._type_mapping[key]
            try:
                if target_type == bool and isinstance(value, str):
                    # Специальная обработка булевых значений
                    return value.lower() in ('true', 'yes', '1', 'on')
                else:
                    return target_type(value)
            except (ValueError, TypeError) as e:
                self._logger.warning(f"Failed to convert {key} to {target_type}: {e}")
                return value
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Установка с проверкой типа"""
        if key in self._type_mapping:
            target_type = self._type_mapping[key]
            if not isinstance(value, target_type):
                try:
                    value = target_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert {value} to {target_type} for key {key}: {e}")
        
        super().set(key, value) 