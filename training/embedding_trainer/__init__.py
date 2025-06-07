"""
Embedding Trainer - Модуль для обучения 3D Cubic Core

Этот модуль реализует систему обучения центрального процессора (Модуль 2)
на эмбединг→эмбединг трансформациях в рамках модульной архитектуры.

Основные компоненты:
- CubeTrainer: основной класс для обучения
- AutoencoderDataset: датасет для autoencoder задач
- DialogueDataset: датасет для диалоговых задач
- EmbeddingMetrics: система метрик качества
- TrainingLogger: логирование прогресса
- CheckpointManager: управление чекпойнтами

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0 (Phase 3.1)
Дата: 6 июня 2025
"""

# Импорты будут добавлены по мере реализации классов
# На данном этапе создаем структуру для будущих компонентов

__version__ = "1.0.0"
__author__ = "3D Cellular Neural Network Project"
__status__ = "Phase 3.1 - Active Development"

# Импорты реализованных компонентов
try:
    from .cube_trainer import CubeTrainer, TrainingConfig, EmbeddingMetrics
    CUBE_TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CubeTrainer not available: {e}")
    CUBE_TRAINER_AVAILABLE = False
    
    # Заглушка если недоступен
    class CubeTrainer:
        """Заглушка для основного тренера"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CubeTrainer dependencies not available")

# TODO: Добавить импорты после реализации:
# from .datasets import AutoencoderDataset, DialogueDataset
# from .logger import TrainingLogger
# from .checkpoint_manager import CheckpointManager
# from .utils import (
#     create_autoencoder_dataset,
#     create_dialogue_dataset,
#     calculate_embedding_similarity,
#     save_training_checkpoint,
#     load_training_checkpoint
# )

class AutoencoderDataset:
    """Заглушка для autoencoder датасета - будет реализована в Stage 1.2"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AutoencoderDataset will be implemented in Stage 1.2")

class DialogueDataset:
    """Заглушка для dialogue датасета - будет реализована в Stage 1.3"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DialogueDataset will be implemented in Stage 1.3")

# Базовая проверка зависимостей
def _check_dependencies():
    """Проверка наличия необходимых модулей"""
    try:
        import torch
        import numpy as np
        from pathlib import Path
        
        # Проверка наличия требуемых модулей проекта
        required_modules = [
            'core.embedding_processor',
            'data.embedding_reshaper',
            'data.embedding_loader',
            'utils.config_manager'
        ]
        
        missing_modules = []
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append(module_name)
        
        if missing_modules:
            print(f"⚠️  Warning: Missing required modules: {missing_modules}")
            print("   Make sure all dependencies are implemented before using EmbeddingTrainer")
        else:
            print("✅ All dependencies are available")
            
    except ImportError as e:
        print(f"❌ Critical dependency missing: {e}")
        print("   Install required packages: torch, numpy")

# Автоматическая проверка при импорте модуля
_check_dependencies()

# Информация о модуле
def get_module_info():
    """Получить информацию о модуле"""
    ready_components = []
    if CUBE_TRAINER_AVAILABLE:
        ready_components.extend(['CubeTrainer', 'TrainingConfig', 'EmbeddingMetrics'])
    
    return {
        'name': 'EmbeddingTrainer',
        'version': __version__,
        'status': __status__,
        'phase': 'Phase 3.1 - Stage 1.1',
        'description': 'Training system for 3D Cubic Core (Module 2)',
        'ready_components': ready_components,
        'in_development': ['AutoencoderDataset', 'DialogueDataset'],
        'planned': ['TrainingLogger', 'CheckpointManager']
    }

# Экспорт функции информации
__all__ = ['get_module_info', 'CubeTrainer', 'AutoencoderDataset', 'DialogueDataset'] 