"""
Config Loader для центрального конфига
Загружает настройки из config/main_config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def load_main_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Загрузка главного конфига проекта
    
    Args:
        config_path: Путь к файлу конфига (по умолчанию config/main_config.yaml)
        
    Returns:
        Dict с конфигурацией
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "main_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config

def get_teacher_models_config() -> Dict[str, Any]:
    """Получить конфигурацию teacher моделей из центрального конфига"""
    config = load_main_config()
    return config.get('teacher_models', {})

def get_available_teacher_models() -> List[str]:
    """Получить список доступных teacher моделей"""
    teacher_config = get_teacher_models_config()
    models = teacher_config.get('models', {})
    return list(models.keys())

def get_teacher_model_path(model_key: str) -> str:
    """Получить путь к teacher модели по ключу"""
    teacher_config = get_teacher_models_config()
    models = teacher_config.get('models', {})
    
    if model_key not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown teacher model: {model_key}. Available: {available}")
    
    return models[model_key]['path']

def get_multi_teacher_config() -> Dict[str, Any]:
    """Получить конфигурацию multi-teacher системы"""
    teacher_config = get_teacher_models_config()
    return teacher_config.get('multi_teacher', {})

def update_teacher_models_in_embedding_loader():
    """
    Обновить SUPPORTED_LLM_MODELS в EmbeddingLoader на основе центрального конфига
    """
    try:
        from data.embedding_loader.format_handlers import SUPPORTED_LLM_MODELS
        
        teacher_config = get_teacher_models_config()
        models = teacher_config.get('models', {})
        
        # Обновляем SUPPORTED_LLM_MODELS
        for model_key, model_config in models.items():
            SUPPORTED_LLM_MODELS[model_key] = model_config['path']
        
        logger.info(f"Updated SUPPORTED_LLM_MODELS with {len(models)} models from config")
        return True
        
    except ImportError:
        logger.warning("Could not import SUPPORTED_LLM_MODELS")
        return False

class ConfigManager:
    """Менеджер для работы с конфигурацией"""
    
    def __init__(self):
        self.config = load_main_config()
        self._update_embedding_loader()
    
    def _update_embedding_loader(self):
        """Обновить EmbeddingLoader конфигурацией"""
        update_teacher_models_in_embedding_loader()
    
    def get_teacher_models(self) -> List[str]:
        """Получить список teacher моделей для multi-teacher"""
        multi_teacher = self.config.get('teacher_models', {}).get('multi_teacher', {})
        return multi_teacher.get('models', ['distilbert'])
    
    def get_primary_teacher_model(self) -> str:
        """Получить основную teacher модель"""
        teacher_config = self.config.get('teacher_models', {})
        return teacher_config.get('primary_model', 'distilbert')
    
    def get_lattice_config(self) -> Dict[str, Any]:
        """Получить конфигурацию решетки"""
        return self.config.get('lattice', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Получить конфигурацию обучения"""
        return self.config.get('training', {})
    
    def is_debug_mode(self) -> bool:
        """Проверить, включен ли режим отладки"""
        return self.config.get('modes', {}).get('debug', False)
    
    def get_device_config(self) -> Dict[str, Any]:
        """Получить конфигурацию устройства"""
        return self.config.get('device', {'use_gpu': False, 'gpu_device': 'cuda:0'})
    
    def should_use_gpu(self) -> bool:
        """Проверить, нужно ли использовать GPU"""
        device_config = self.get_device_config()
        return device_config.get('use_gpu', False)
    
    def get_gpu_device(self) -> str:
        """Получить устройство GPU"""
        device_config = self.get_device_config()
        return device_config.get('gpu_device', 'cuda:0')

# Глобальный экземпляр для удобства
config_manager = ConfigManager() 