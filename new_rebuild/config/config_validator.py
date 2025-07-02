#!/usr/bin/env python3
"""
Config Validator - Строгая валидация конфигурации без fallback'ов
================================================================

Обеспечивает проверку обязательных полей конфигурации при инициализации.
В соответствии с принципом из CLAUDE.md: "лучше явная ошибка, чем скрытая проблема".
"""

from typing import Any, List, Tuple, Optional


class ConfigValidator:
    """
    Строгий валидатор конфигурации без fallback'ов
    
    Проверяет наличие всех обязательных полей и выбрасывает
    RuntimeError при их отсутствии.
    """
    
    @staticmethod
    def validate_full_config(config: Any) -> None:
        """
        Полная валидация всей конфигурации проекта
        
        Args:
            config: Объект конфигурации для проверки
            
        Raises:
            RuntimeError: При отсутствии обязательных полей
        """
        # Валидация основных компонентов
        ConfigValidator.validate_lattice_config(config)
        ConfigValidator.validate_model_config(config)
        ConfigValidator.validate_moe_config(config)
        ConfigValidator.validate_cnf_config(config)
        ConfigValidator.validate_device_config(config)
        ConfigValidator.validate_cache_config(config)
        ConfigValidator.validate_training_config(config)
        ConfigValidator.validate_logging_config(config)
    
    @staticmethod
    def validate_lattice_config(config: Any) -> None:
        """Валидация конфигурации решетки"""
        required_paths = [
            ('lattice', 'config.lattice'),
            ('lattice.dimensions', 'config.lattice.dimensions'),
            ('lattice.adaptive_radius_enabled', 'config.lattice.adaptive_radius_enabled'),
            ('lattice.adaptive_radius_ratio', 'config.lattice.adaptive_radius_ratio'),
            ('lattice.local_distance_ratio', 'config.lattice.local_distance_ratio'),
            ('lattice.functional_distance_ratio', 'config.lattice.functional_distance_ratio'),
            ('lattice.distant_distance_ratio', 'config.lattice.distant_distance_ratio'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "Lattice")
    
    @staticmethod
    def validate_model_config(config: Any) -> None:
        """Валидация конфигурации модели"""
        required_paths = [
            ('model', 'config.model'),
            ('model.state_size', 'config.model.state_size'),
            ('model.message_dim', 'config.model.message_dim'),
            ('model.hidden_dim', 'config.model.hidden_dim'),
            ('model.neighbor_count', 'config.model.neighbor_count'),
            ('model.num_heads', 'config.model.num_heads'),
            ('model.use_layer_norm', 'config.model.use_layer_norm'),
            ('model.dropout_rate', 'config.model.dropout_rate'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "Model")
    
    @staticmethod
    def validate_moe_config(config: Any) -> None:
        """Валидация MoE конфигурации"""
        required_paths = [
            # Neighbors configuration
            ('neighbors', 'config.neighbors'),
            ('neighbors.local_tier', 'config.neighbors.local_tier'),
            ('neighbors.functional_tier', 'config.neighbors.functional_tier'),
            ('neighbors.distant_tier', 'config.neighbors.distant_tier'),
            
            # Expert configuration
            ('expert', 'config.expert'),
            ('expert.functional', 'config.expert.functional'),
            ('expert.functional.params', 'config.expert.functional.params'),
            ('expert.distant', 'config.expert.distant'),
            ('expert.distant.params', 'config.expert.distant.params'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "MoE")
        
        # Проверка суммы ratios
        if hasattr(config, 'neighbors') and config.neighbors:
            total = config.neighbors.local_tier + config.neighbors.functional_tier + config.neighbors.distant_tier
            if abs(total - 1.0) > 0.001:  # Допуск на погрешность float
                raise RuntimeError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА: Сумма connection ratios должна быть 1.0, получено {total}. "
                    f"Проверьте config.neighbors (local: {config.neighbors.local_tier}, "
                    f"functional: {config.neighbors.functional_tier}, distant: {config.neighbors.distant_tier})"
                )
    
    @staticmethod
    def validate_cnf_config(config: Any) -> None:
        """Валидация CNF конфигурации"""
        required_paths = [
            ('cnf', 'config.cnf'),
            ('cnf.enabled', 'config.cnf.enabled'),
            ('cnf.integration_steps', 'config.cnf.integration_steps'),
            ('cnf.batch_processing_mode', 'config.cnf.batch_processing_mode'),
            ('cnf.max_batch_size', 'config.cnf.max_batch_size'),
            ('cnf.adaptive_method', 'config.cnf.adaptive_method'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "CNF")
        
        # Дополнительная проверка: если MoE включен, CNF должен быть включен
        if hasattr(config, 'cnf') and hasattr(config.cnf, 'enabled'):
            if not config.cnf.enabled:
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: CNF отключен (config.cnf.enabled=False), "
                    "но требуется для MoE архитектуры. Установите config.cnf.enabled = True"
                )
    
    @staticmethod
    def validate_device_config(config: Any) -> None:
        """Валидация конфигурации устройства"""
        required_paths = [
            ('device', 'config.device'),
            ('device.device', 'config.device.device'),
            ('device.dtype', 'config.device.dtype'),
            ('device.compile_model', 'config.device.compile_model'),
            ('device.memory_fraction', 'config.device.memory_fraction'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "Device")
    
    @staticmethod
    def validate_cache_config(config: Any) -> None:
        """Валидация конфигурации кэширования"""
        required_paths = [
            ('cache', 'config.cache'),
            ('cache.enabled', 'config.cache.enabled'),
            ('cache.gpu_cache_size_mb', 'config.cache.gpu_cache_size_mb'),
            ('cache.neighbor_cache_size', 'config.cache.neighbor_cache_size'),
            ('cache.connection_cache_enabled', 'config.cache.connection_cache_enabled'),
            ('cache.persistent_cache_dir', 'config.cache.persistent_cache_dir'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "Cache")
    
    @staticmethod
    def validate_training_config(config: Any) -> None:
        """Валидация конфигурации обучения"""
        required_paths = [
            ('training', 'config.training'),
            ('training.batch_size', 'config.training.batch_size'),
            ('training.learning_rate', 'config.training.learning_rate'),
            ('training.num_epochs', 'config.training.num_epochs'),
            ('training.gradient_clip_norm', 'config.training.gradient_clip_norm'),
            ('training.optimizer', 'config.training.optimizer'),  # Изменено с optimizer_type на optimizer
        ]
        ConfigValidator._validate_paths(config, required_paths, "Training")
    
    @staticmethod
    def validate_logging_config(config: Any) -> None:
        """Валидация конфигурации логирования"""
        required_paths = [
            ('logging', 'config.logging'),
            ('logging.level', 'config.logging.level'),
            ('logging.log_format', 'config.logging.log_format'),  # Изменено с format на log_format
            ('logging.enable_file_logging', 'config.logging.enable_file_logging'),
            ('logging.log_dir', 'config.logging.log_dir'),
        ]
        ConfigValidator._validate_paths(config, required_paths, "Logging")
    
    @staticmethod
    def _validate_paths(config: Any, required_paths: List[Tuple[str, str]], section_name: str) -> None:
        """
        Проверка списка обязательных путей в конфигурации
        
        Args:
            config: Объект конфигурации
            required_paths: Список кортежей (путь, описание)
            section_name: Название секции для сообщений об ошибках
        """
        for path, description in required_paths:
            if not ConfigValidator._has_nested_attr(config, path):
                raise RuntimeError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА [{section_name}]: "
                    f"Отсутствует обязательная конфигурация {description}. "
                    f"Проверьте new_rebuild\config и docs\CONFIG_MODES_SUMMARY.md"
                )
    
    @staticmethod
    def _has_nested_attr(obj: Any, path: str) -> bool:
        """
        Проверка наличия вложенных атрибутов
        
        Args:
            obj: Объект для проверки
            path: Путь в формате "attr1.attr2.attr3"
            
        Returns:
            True если атрибут существует и не None
        """
        attrs = path.split('.')
        current = obj
        for attr in attrs:
            if not hasattr(current, attr):
                return False
            current = getattr(current, attr)
            if current is None:
                return False
        return True
    
    @staticmethod
    def validate_config_mode(config: Any) -> None:
        """
        Валидация режима конфигурации
        
        Args:
            config: Объект конфигурации
        """
        if not hasattr(config, 'mode_settings'):
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.mode_settings. "
                "Добавьте ModeSettings в конфигурацию"
            )
        
        if not hasattr(config.mode_settings, 'mode'):
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: Отсутствует config.mode_settings.mode. "
                "Укажите режим работы (DEBUG/EXPERIMENT/OPTIMIZED)"
            )