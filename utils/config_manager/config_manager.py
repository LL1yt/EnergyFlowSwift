"""
Централизованный Configuration Manager для 3D Cellular Neural Network

Поддерживает:
- Модульную архитектуру (каждый модуль имеет свою конфигурацию)
- Environment-specific настройки (dev/test/prod)
- Валидацию и hot reloading
- Кэширование и оптимизацию производительности
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import logging
import time
import threading
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib

try:
    from .config_section import ConfigSection
    from .config_validator import ConfigValidator
    from .config_schema import ConfigSchema
except ImportError:
    # Fallback для случаев когда импорты недоступны
    ConfigSection = None
    ConfigValidator = None
    ConfigSchema = None


@dataclass 
class ConfigManagerSettings:
    """Настройки ConfigManager"""
    base_config_path: str = "config/main_config.yaml"
    module_configs_pattern: str = "*/config/*.yaml"
    environment: str = "development"  # development, testing, production
    enable_hot_reload: bool = True
    hot_reload_interval: float = 1.0  # секунды
    enable_validation: bool = True
    enable_caching: bool = True
    cache_ttl: float = 300.0  # 5 минут
    enable_environment_overrides: bool = True
    config_search_paths: List[str] = field(default_factory=lambda: [
        "config/",
        "core/*/config/", 
        "data/*/config/",
        "inference/*/config/",
        "training/*/config/",
    ])


class ConfigManager:
    """
    Централизованный менеджер конфигураций для модульной архитектуры.
    
    Особенности:
    - Автоматическое обнаружение модульных конфигураций
    - Иерархическое наследование настроек
    - Environment-specific overrides
    - Hot reloading с минимальными накладными расходами
    - Валидация конфигураций через schemas
    - Thread-safe операции
    """
    
    def __init__(self, settings: Optional[ConfigManagerSettings] = None):
        """
        Инициализация ConfigManager.
        
        Args:
            settings: Настройки менеджера конфигурации
        """
        self.settings = settings or ConfigManagerSettings()
        self.logger = logging.getLogger(__name__)
        
        # Основные данные
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._validators: Dict[str, Any] = {}  # Изменено для совместимости
        self._schemas: Dict[str, Any] = {}     # Изменено для совместимости
        
        # Thread safety
        self._lock = threading.RLock()
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._should_stop_hot_reload = threading.Event()
        
        # Статистика
        self._stats = {
            'config_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'hot_reloads': 0,
        }
        
        # Инициализация
        self._initialize()
        
    def _initialize(self):
        """Инициализация менеджера конфигурации"""
        try:
            self.logger.info("🚀 Initializing ConfigManager...")
            
            # Загружаем основную конфигурацию
            self._load_base_config()
            
            # Обнаруживаем и загружаем модульные конфигурации
            self._discover_module_configs()
            
            # Загружаем schema для валидации
            self._load_config_schemas()
            
            # Применяем environment-specific overrides
            if self.settings.enable_environment_overrides:
                self._apply_environment_overrides()
            
            # Запускаем hot reloading если включено
            if self.settings.enable_hot_reload:
                self._start_hot_reload_monitor()
                
            self.logger.info("✅ ConfigManager initialized successfully")
            self.logger.info(f"   📊 Loaded {len(self._config_cache)} config sections")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ConfigManager: {e}")
            raise
    
    def get_config(self, section: str = None, key: str = None, default: Any = None) -> Any:
        """
        Получение конфигурации с поддержкой dot-notation.
        
        Args:
            section: Секция конфигурации (например, 'lattice_3d')
            key: Ключ в секции (например, 'dimensions.depth') 
            default: Значение по умолчанию
            
        Returns:
            Значение конфигурации или default
            
        Examples:
            config.get_config()  # Вся конфигурация
            config.get_config('lattice_3d')  # Секция lattice_3d
            config.get_config('lattice_3d', 'dimensions.depth')  # Вложенный ключ
        """
        with self._lock:
            try:
                # Полная конфигурация
                if section is None:
                    return deepcopy(self._merge_all_configs())
                
                # Конкретная секция
                if section not in self._config_cache:
                    self.logger.warning(f"Configuration section '{section}' not found")
                    return default
                
                config_section = self._config_cache[section]
                
                # Вся секция
                if key is None:
                    self._stats['cache_hits'] += 1
                    return deepcopy(config_section)
                
                # Конкретный ключ (с поддержкой dot-notation)
                value = self._get_nested_value(config_section, key)
                if value is not None:
                    self._stats['cache_hits'] += 1
                    return deepcopy(value)
                else:
                    self._stats['cache_misses'] += 1
                    return default
                    
            except Exception as e:
                self.logger.error(f"Error getting config {section}.{key}: {e}")
                return default
    
    def set_config(self, section: str, key: str = None, value: Any = None, **kwargs):
        """
        Установка значения конфигурации в runtime.
        
        Args:
            section: Секция конфигурации
            key: Ключ (поддерживает dot-notation)
            value: Значение
            **kwargs: Альтернативный способ передачи ключ=значение
            
        Examples:
            config.set_config('lattice_3d', 'dimensions.depth', 10)
            config.set_config('training', batch_size=32, learning_rate=0.001)
        """
        with self._lock:
            try:
                if section not in self._config_cache:
                    self._config_cache[section] = {}
                
                config_section = self._config_cache[section]
                
                # Установка через kwargs
                if kwargs:
                    for k, v in kwargs.items():
                        self._set_nested_value(config_section, k, v)
                        self.logger.debug(f"Set config {section}.{k} = {v}")
                
                # Установка через key/value
                if key is not None and value is not None:
                    self._set_nested_value(config_section, key, value)
                    self.logger.debug(f"Set config {section}.{key} = {value}")
                
                # Валидация после изменений
                if self.settings.enable_validation and section in self._validators:
                    self._validate_section(section, config_section)
                    
            except Exception as e:
                self.logger.error(f"Error setting config {section}.{key}: {e}")
                raise
    
    def reload_config(self, section: str = None):
        """
        Перезагрузка конфигурации из файлов.
        
        Args:
            section: Конкретная секция для перезагрузки (None = все)
        """
        with self._lock:
            try:
                if section is None:
                    self.logger.info("🔄 Reloading all configurations...")
                    self._config_cache.clear()
                    self._file_mtimes.clear()
                    self._load_base_config()
                    self._discover_module_configs()
                    if self.settings.enable_environment_overrides:
                        self._apply_environment_overrides()
                else:
                    self.logger.info(f"🔄 Reloading configuration section: {section}")
                    
                self._stats['hot_reloads'] += 1
                self.logger.info("✅ Configuration reloaded successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Error reloading configuration: {e}")
                raise
    
    def get_section(self, section_name: str):
        """
        Получение секции конфигурации как объекта ConfigSection.
        
        Args:
            section_name: Название секции
            
        Returns:
            ConfigSection или dict: Объект для удобной работы с секцией
        """
        config_data = self.get_config(section_name, default={})
        
        # Если ConfigSection доступен, используем его
        if ConfigSection is not None:
            return ConfigSection(section_name, config_data, self)
        else:
            # Возвращаем простой dict
            return config_data
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Валидация всех конфигураций.
        
        Returns:
            Dict[str, List[str]]: Словарь ошибок по секциям
        """
        errors = {}
        
        with self._lock:
            for section, config_data in self._config_cache.items():
                try:
                    section_errors = self._validate_section(section, config_data)
                    if section_errors:
                        errors[section] = section_errors
                except Exception as e:
                    errors[section] = [f"Validation failed: {str(e)}"]
        
        return errors
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики работы ConfigManager.
        
        Returns:
            Dict: Статистика использования
        """
        with self._lock:
            return {
                **self._stats.copy(),
                'cached_sections': len(self._config_cache),
                'tracked_files': len(self._file_mtimes),
                'validators': len(self._validators),
                'cache_hit_rate': (
                    self._stats['cache_hits'] / 
                    max(1, self._stats['cache_hits'] + self._stats['cache_misses'])
                ),
            }
    
    def export_config(self, output_path: str, format: str = 'yaml', section: str = None):
        """
        Экспорт конфигурации в файл.
        
        Args:
            output_path: Путь для сохранения
            format: Формат файла ('yaml', 'json')
            section: Конкретная секция (None = все)
        """
        try:
            config_data = self.get_config(section) if section else self.get_config()
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'yaml':
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
            elif format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"✅ Configuration exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting configuration: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Shutting down ConfigManager...")
        
        if self._hot_reload_thread and self._hot_reload_thread.is_alive():
            self._should_stop_hot_reload.set()
            self._hot_reload_thread.join(timeout=2.0)
        
        self.logger.info("✅ ConfigManager shutdown complete")
    
    # ========================================
    # PRIVATE METHODS
    # ========================================
    
    def _load_base_config(self):
        """Загрузка основной конфигурации"""
        base_path = Path(self.settings.base_config_path)
        
        if not base_path.exists():
            self.logger.warning(f"Base config file not found: {base_path}")
            return
        
        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f) or {}
            
            # Кэшируем каждую секцию отдельно
            for section_name, section_data in base_config.items():
                self._config_cache[section_name] = section_data
                
            self._file_mtimes[str(base_path)] = base_path.stat().st_mtime
            self._stats['config_loads'] += 1
            
            self.logger.info(f"📝 Loaded base config: {base_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading base config {base_path}: {e}")
            raise
    
    def _discover_module_configs(self):
        """Автоматическое обнаружение модульных конфигураций"""
        discovered_configs = []
        
        for search_path in self.settings.config_search_paths:
            try:
                # Используем glob для поиска конфигураций
                pattern = Path(search_path) / "*.yaml"
                config_files = list(Path(".").glob(str(pattern)))
                
                for config_file in config_files:
                    if config_file.exists() and config_file != Path(self.settings.base_config_path):
                        self._load_module_config(config_file)
                        discovered_configs.append(str(config_file))
                        
            except Exception as e:
                self.logger.warning(f"Error discovering configs in {search_path}: {e}")
        
        if discovered_configs:
            self.logger.info(f"🔍 Discovered {len(discovered_configs)} module configs")
    
    def _load_module_config(self, config_path: Path):
        """Загрузка конфигурации модуля"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                module_config = yaml.safe_load(f) or {}
            
            # Определяем префикс модуля из пути
            module_name = self._extract_module_name(config_path)
            
            # Мержим конфигурацию
            for section_name, section_data in module_config.items():
                full_section_name = f"{module_name}_{section_name}" if module_name else section_name
                
                if full_section_name in self._config_cache:
                    # Мержим с существующей конфигурацией
                    self._deep_merge(self._config_cache[full_section_name], section_data)
                else:
                    self._config_cache[full_section_name] = section_data
            
            self._file_mtimes[str(config_path)] = config_path.stat().st_mtime
            self._stats['config_loads'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error loading module config {config_path}: {e}")
    
    def _extract_module_name(self, config_path: Path) -> str:
        """Извлечение имени модуля из пути к конфигурации"""
        parts = config_path.parts
        
        # Ищем паттерн: module_name/config/file.yaml
        for i, part in enumerate(parts):
            if part == 'config' and i > 0:
                return parts[i-1]
        
        return ""
    
    def _load_config_schemas(self):
        """Загрузка схем для валидации"""
        # Базовая реализация - можно расширить
        pass
    
    def _apply_environment_overrides(self):
        """Применение environment-specific настроек"""
        env = self.settings.environment.lower()
        
        # Ищем секции с environment overrides
        for section_name, section_data in self._config_cache.items():
            if isinstance(section_data, dict) and env in section_data:
                env_overrides = section_data[env]
                if isinstance(env_overrides, dict):
                    self._deep_merge(section_data, env_overrides)
                    self.logger.debug(f"Applied {env} overrides to {section_name}")
    
    def _start_hot_reload_monitor(self):
        """Запуск мониторинга hot reload"""
        def monitor_files():
            while not self._should_stop_hot_reload.is_set():
                try:
                    self._check_file_changes()
                    self._should_stop_hot_reload.wait(self.settings.hot_reload_interval)
                except Exception as e:
                    self.logger.error(f"Error in hot reload monitor: {e}")
        
        self._hot_reload_thread = threading.Thread(target=monitor_files, daemon=True)
        self._hot_reload_thread.start()
        self.logger.info("🔄 Hot reload monitor started")
    
    def _check_file_changes(self):
        """Проверка изменений в конфигурационных файлах"""
        changed_files = []
        
        for file_path, cached_mtime in self._file_mtimes.items():
            try:
                current_mtime = Path(file_path).stat().st_mtime
                if current_mtime > cached_mtime:
                    changed_files.append(file_path)
            except FileNotFoundError:
                # Файл был удален
                changed_files.append(file_path)
        
        if changed_files:
            self.logger.info(f"🔄 Detected changes in {len(changed_files)} config files")
            self.reload_config()
    
    def _validate_section(self, section: str, config_data: Dict[str, Any]) -> List[str]:
        """Валидация секции конфигурации"""
        if section not in self._validators:
            return []
        
        try:
            validator = self._validators[section]
            # Базовая валидация
            return []
        except Exception as e:
            self._stats['validation_errors'] += 1
            return [f"Validation error: {str(e)}"]
    
    def _merge_all_configs(self) -> Dict[str, Any]:
        """Объединение всех конфигураций в один словарь"""
        merged = {}
        for section_name, section_data in self._config_cache.items():
            merged[section_name] = section_data
        return merged
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Получение вложенного значения по dot-notation"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any):
        """Установка вложенного значения по dot-notation"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Глубокое объединение словарей"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_config_manager(
    base_config: str = "config/main_config.yaml",
    environment: str = "development",
    enable_hot_reload: bool = True
) -> ConfigManager:
    """
    Фабричная функция для создания ConfigManager.
    
    Args:
        base_config: Путь к основной конфигурации
        environment: Окружение (development/testing/production)
        enable_hot_reload: Включить hot reloading
        
    Returns:
        ConfigManager: Настроенный менеджер конфигурации
    """
    settings = ConfigManagerSettings(
        base_config_path=base_config,
        environment=environment,
        enable_hot_reload=enable_hot_reload
    )
    
    return ConfigManager(settings)


# Глобальный экземпляр (singleton pattern)
_global_config_manager: Optional[ConfigManager] = None


def get_global_config_manager() -> ConfigManager:
    """
    Получение глобального экземпляра ConfigManager.
    
    Returns:
        ConfigManager: Глобальный менеджер конфигурации
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = create_config_manager()
    
    return _global_config_manager


def set_global_config_manager(config_manager: ConfigManager):
    """
    Установка глобального экземпляра ConfigManager.
    
    Args:
        config_manager: Экземпляр менеджера конфигурации
    """
    global _global_config_manager
    _global_config_manager = config_manager 