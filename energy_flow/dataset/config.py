"""
Конфигурация для dataset модуля
===============================

Интегрированная с EnergyConfig система настроек для управления датасетами:
- Настройки модели-учителя
- Параметры источников данных  
- Настройки кэширования и валидации
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch

from ..config import EnergyConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """
    Конфигурация для dataset модуля
    
    Интегрируется с EnergyConfig для получения общих настроек проекта
    """
    
    # Модель-учитель
    teacher_model: str = "distilbert-base-uncased"
    use_local_model: bool = True  # Приоритет локальной модели
    local_model_path: str = "models/local_cache/distilbert-base-uncased"
    
    # Источники данных (порядок определяет приоритет)
    dataset_sources: List[str] = field(default_factory=lambda: ["precomputed", "snli"])
    
    # Параметры загрузки данных
    batch_size: int = 32
    max_samples_per_source: Optional[int] = None  # None = без ограничений
    shuffle_data: bool = True
    normalize_embeddings: bool = True
    
    # Параметры SNLI
    snli_fraction: float = 0.2  # Какую часть SNLI использовать (0.2 = 20%)
    snli_min_text_length: int = 10  # Минимальная длина текста
    
    # Кэширование эмбеддингов
    embedding_cache_enabled: bool = True
    cache_batch_size: int = 64
    max_cache_size_mb: int = 1024  # Максимальный размер кэша в MB
    
    # Пути для данных
    data_dir: str = "data"
    embeddings_dir: str = "data/embeddings"
    cache_dir: str = "cache"
    
    # Валидация и проверки
    validate_embeddings: bool = True
    min_embedding_norm: float = 0.01
    max_embedding_norm: float = 100.0
    check_nan_inf: bool = True
    
    # GPU/CPU настройки (наследуются из EnergyConfig если не заданы)
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    
    def __post_init__(self):
        """Валидация и инициализация производных параметров"""
        
        # Валидация источников данных
        valid_sources = {"precomputed", "snli", "cache"}
        invalid = set(self.dataset_sources) - valid_sources
        if invalid:
            raise ValueError(f"Неизвестные источники данных: {invalid}. Доступны: {valid_sources}")
        
        # Валидация параметров
        assert 0 < self.snli_fraction <= 1.0, "snli_fraction должен быть в (0, 1]"
        assert self.batch_size > 0, "batch_size должен быть > 0"
        assert self.cache_batch_size > 0, "cache_batch_size должен быть > 0"
        assert self.max_cache_size_mb > 0, "max_cache_size_mb должен быть > 0"
        
        # Создание путей
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.embeddings_dir).mkdir(parents=True, exist_ok=True) 
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Лог инициализации
        logger.info(f"DatasetConfig initialized: sources={self.dataset_sources}, "
                   f"teacher_model={self.teacher_model}, use_local={self.use_local_model}")
    
    def get_absolute_local_model_path(self) -> Path:
        """Получить абсолютный путь к локальной модели"""
        return Path(self.local_model_path).resolve()
    
    def get_absolute_embeddings_dir(self) -> Path:
        """Получить абсолютный путь к директории эмбеддингов"""
        return Path(self.embeddings_dir).resolve()
    
    def get_absolute_cache_dir(self) -> Path:
        """Получить абсолютный путь к директории кэша"""
        return Path(self.cache_dir).resolve()
    
    def update_from_energy_config(self, energy_config: EnergyConfig):
        """
        Обновить настройки из EnergyConfig
        
        Args:
            energy_config: Основная конфигурация energy_flow
        """
        # Наследуем device и dtype если не заданы
        if self.device is None:
            self.device = energy_config.device
        if self.dtype is None:
            self.dtype = energy_config.dtype
            
        # Адаптируем batch_size под режим
        if hasattr(energy_config, 'batch_size'):
            # Для dataset используем тот же batch_size что и для обучения
            self.batch_size = energy_config.batch_size
            
        # Адаптируем размеры под режим (DEBUG/EXPERIMENT/OPTIMIZED)
        if energy_config.lattice_width <= 30:  # DEBUG mode
            self.max_samples_per_source = 1000
            self.cache_batch_size = 32
            self.max_cache_size_mb = 256
        elif energy_config.lattice_width <= 70:  # EXPERIMENT mode  
            self.max_samples_per_source = 5000
            self.cache_batch_size = 64
            self.max_cache_size_mb = 512
        else:  # OPTIMIZED mode
            self.max_samples_per_source = None  # Без ограничений
            self.cache_batch_size = 128
            self.max_cache_size_mb = 1024
            
        logger.info(f"Dataset config updated from EnergyConfig: "
                   f"device={self.device}, batch_size={self.batch_size}, "
                   f"max_samples={self.max_samples_per_source}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сохранения"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }


def create_dataset_config_from_energy(energy_config: EnergyConfig, **overrides) -> DatasetConfig:
    """
    Создать DatasetConfig на основе EnergyConfig с возможностью переопределить параметры
    
    Args:
        energy_config: Основная конфигурация energy_flow
        **overrides: Параметры для переопределения
        
    Returns:
        DatasetConfig адаптированная под energy_config
    """
    # Создаем базовую конфигурацию с переопределениями
    dataset_config = DatasetConfig(**overrides)
    
    # Обновляем из energy_config
    dataset_config.update_from_energy_config(energy_config)
    
    return dataset_config


def create_debug_dataset_config() -> DatasetConfig:
    """Создать конфигурацию для режима отладки"""
    return DatasetConfig(
        dataset_sources=["precomputed"],  # Только готовые эмбеддинги для скорости
        max_samples_per_source=500,
        batch_size=8,
        cache_batch_size=16,
        max_cache_size_mb=128,
        snli_fraction=0.05,  # Минимальная часть SNLI
        validate_embeddings=True
    )


def create_production_dataset_config() -> DatasetConfig:
    """Создать конфигурацию для продакшн режима"""
    return DatasetConfig(
        dataset_sources=["precomputed", "snli", "cache"],  # Все источники
        max_samples_per_source=None,  # Без ограничений
        batch_size=32,
        cache_batch_size=128,
        max_cache_size_mb=2048,
        snli_fraction=0.3,  # Больше данных из SNLI
        validate_embeddings=True
    )