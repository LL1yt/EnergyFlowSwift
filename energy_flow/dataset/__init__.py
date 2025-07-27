"""
Dataset модуль для energy_flow архитектуры
==========================================

Оптимизированный модуль для создания и управления датасетами:
- Проверка наличия локальной модели учителя
- Загрузка и кэширование готовых эмбеддингов  
- Генерация новых датасетов из различных источников
- Интеграция с EnergyConfig и системой логирования

Основные компоненты:
- DatasetManager: центральный класс управления датасетами
- TeacherModelProvider: управление моделью-учителем (DistilBERT)
- Провайдеры данных: SNLI, Precomputed, и другие источники
"""

from .config import DatasetConfig, create_dataset_config_from_energy
from .manager import DatasetManager, create_dataset_manager
from .providers import (
    BaseDataProvider, 
    TeacherModelProvider,
    SNLIProvider, 
    PrecomputedProvider
)

__all__ = [
    'DatasetConfig',
    'DatasetManager', 
    'BaseDataProvider',
    'TeacherModelProvider',
    'SNLIProvider',
    'PrecomputedProvider',
    'create_dataset_config_from_energy',
    'create_dataset_manager'
]