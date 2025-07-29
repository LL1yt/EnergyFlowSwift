"""
Dataset модуль для energy_flow архитектуры
==========================================

Оптимизированный модуль для создания и управления датасетами:
- Проверка наличия локальной модели учителя
- Загрузка и кэширование готовых эмбеддингов  
- Генерация именованных файлов датасетов из различных источников
- Интеграция с EnergyConfig и системой логирования

Основные компоненты:
- DatasetManager: центральный класс управления датасетами
- DatasetGenerator: создание именованных файлов датасетов
- TeacherModelProvider: управление моделью-учителем (DistilBERT)
- Провайдеры данных: SNLI, Precomputed, и другие источники
"""

from .config import DatasetConfig, create_dataset_config_from_energy
from .manager import DatasetManager, create_dataset_manager
from .generator import (
    GeneratorConfig, DatasetGenerator,
    create_debug_generator_config,
    create_experiment_generator_config,
    create_production_generator_config,
    create_dataset_generator
)
from .providers import (
    BaseDataProvider, 
    TeacherModelProvider,
    SNLIProvider, 
    PrecomputedProvider
)

__all__ = [
    # Основные классы конфигурации
    'DatasetConfig',
    'GeneratorConfig',
    
    # Менеджеры
    'DatasetManager', 
    'DatasetGenerator',
    
    # Провайдеры данных
    'BaseDataProvider',
    'TeacherModelProvider',
    'SNLIProvider',
    'PrecomputedProvider',
    
    # Фабричные функции
    'create_dataset_config_from_energy',
    'create_dataset_manager',
    'create_debug_generator_config',
    'create_experiment_generator_config',
    'create_production_generator_config',
    'create_dataset_generator'
]