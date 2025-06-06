"""
EmbeddingProcessor Configuration - Конфигурация процессора эмбедингов
====================================================================

Определяет все режимы работы и параметры EmbeddingProcessor.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import torch


class ProcessingMode(Enum):
    """Режимы работы EmbeddingProcessor"""
    AUTOENCODER = "autoencoder"    # Восстановление входного эмбединга (для обучения)
    GENERATOR = "generator"        # Семантическая трансформация эмбедингов  
    DIALOGUE = "dialogue"          # Вопрос→ответ преобразования


@dataclass
class EmbeddingConfig:
    """Конфигурация EmbeddingProcessor"""
    
    # === БАЗОВЫЕ РАЗМЕРЫ ===
    input_dim: int = 768                    # Размерность входного эмбединга
    output_dim: int = 768                   # Размерность выходного эмбединга
    cube_shape: Tuple[int, int, int] = (8, 8, 12)  # Форма 3D куба (8×8×12 = 768)
    
    # === РЕЖИМ ОБРАБОТКИ ===  
    processing_mode: ProcessingMode = ProcessingMode.AUTOENCODER
    
    # === LATTICE 3D ПАРАМЕТРЫ ===
    lattice_size: Tuple[int, int, int] = (8, 8, 8)  # Размер 3D решетки
    propagation_steps: int = 10                      # Шаги распространения сигнала
    convergence_threshold: float = 0.001             # Порог конвергенции
    
    # === EMBEDDING RESHAPER ПАРАМЕТРЫ ===
    reshaping_method: str = "adaptive"               # Метод преобразования
    preserve_semantics: bool = True                  # Сохранение семантики
    semantic_threshold: float = 0.95                 # Порог семантического сохранения
    
    # === КАЧЕСТВО И МЕТРИКИ ===
    target_similarity: float = 0.90                 # Целевая схожесть (Phase 2.5)
    quality_check_enabled: bool = True               # Проверка качества
    metrics_logging: bool = True                     # Логирование метрик
    
    # === ПРОИЗВОДИТЕЛЬНОСТЬ ===
    batch_processing: bool = True                    # Батчевая обработка
    cache_enabled: bool = True                       # Кэширование результатов
    parallel_processing: bool = False                # Параллельная обработка (пока отключено)
    
    # === УСТРОЙСТВО ===
    device: str = "cpu"                              # cpu или cuda
    dtype: torch.dtype = torch.float32               # Тип данных
    
    # === ОТЛАДКА ===
    debug_mode: bool = False                         # Режим отладки
    verbose_logging: bool = False                    # Подробное логирование
    save_intermediate_results: bool = False          # Сохранение промежуточных результатов


def create_autoencoder_config() -> EmbeddingConfig:
    """Создать конфигурацию для автоэнкодер режима"""
    return EmbeddingConfig(
        processing_mode=ProcessingMode.AUTOENCODER,
        target_similarity=0.95,  # Высокая точность восстановления
        propagation_steps=15,    # Больше шагов для точности
        debug_mode=True          # Отладка для обучения
    )


def create_generator_config() -> EmbeddingConfig:
    """Создать конфигурацию для генеративного режима"""
    return EmbeddingConfig(
        processing_mode=ProcessingMode.GENERATOR,
        target_similarity=0.85,  # Меньше точности, больше креативности
        propagation_steps=20,    # Больше шагов для трансформации
        semantic_threshold=0.90  # Сохранение общей семантики
    )


def create_dialogue_config() -> EmbeddingConfig:
    """Создать конфигурацию для диалогового режима"""
    return EmbeddingConfig(
        processing_mode=ProcessingMode.DIALOGUE,
        target_similarity=0.80,  # Баланс между точностью и релевантностью
        propagation_steps=25,    # Максимум шагов для контекста
        semantic_threshold=0.85, # Умеренное сохранение семантики
        batch_processing=True    # Оптимизация для диалогов
    )


def load_config_from_dict(config_dict: Dict[str, Any]) -> EmbeddingConfig:
    """Загрузить конфигурацию из словаря"""
    
    # Обработка режима работы
    if 'processing_mode' in config_dict:
        mode_str = config_dict['processing_mode']
        config_dict['processing_mode'] = ProcessingMode(mode_str)
    
    # Обработка формы куба
    if 'cube_shape' in config_dict and isinstance(config_dict['cube_shape'], list):
        config_dict['cube_shape'] = tuple(config_dict['cube_shape'])
    
    # Обработка размера решетки
    if 'lattice_size' in config_dict and isinstance(config_dict['lattice_size'], list):
        config_dict['lattice_size'] = tuple(config_dict['lattice_size'])
        
    # Обработка типа данных
    if 'dtype' in config_dict:
        dtype_str = config_dict['dtype'] 
        if dtype_str == "float32":
            config_dict['dtype'] = torch.float32
        elif dtype_str == "float64": 
            config_dict['dtype'] = torch.float64
    
    return EmbeddingConfig(**config_dict)


def validate_config(config: EmbeddingConfig) -> bool:
    """Валидировать конфигурацию процессора"""
    
    # Проверка соответствия размеров
    cube_volume = config.cube_shape[0] * config.cube_shape[1] * config.cube_shape[2]
    if cube_volume != config.input_dim:
        raise ValueError(f"Cube volume {cube_volume} != input_dim {config.input_dim}")
    
    if config.input_dim != config.output_dim:
        raise ValueError(f"Input dim {config.input_dim} != output dim {config.output_dim}")
    
    # Проверка пороговых значений
    if not 0.0 <= config.target_similarity <= 1.0:
        raise ValueError(f"target_similarity должен быть в [0.0, 1.0], получен {config.target_similarity}")
        
    if not 0.0 <= config.semantic_threshold <= 1.0:
        raise ValueError(f"semantic_threshold должен быть в [0.0, 1.0], получен {config.semantic_threshold}")
    
    # Проверка шагов распространения
    if config.propagation_steps <= 0:
        raise ValueError(f"propagation_steps должен быть > 0, получен {config.propagation_steps}")
    
    return True 