"""
EmbeddingProcessor - Центральный процессор эмбедингов (Phase 2.5)
===============================================================

РЕВОЛЮЦИОННЫЙ МОДУЛЬ для завершения Модуля 2 (3D Cubic Core).
Объединяет все готовые компоненты в единую систему обработки эмбедингов.

Архитектура:
1. Входной эмбединг (768D) → EmbeddingReshaper → 3D матрица (8×8×12)
2. 3D матрица → Lattice3D → обработанная 3D матрица
3. Обработанная 3D матрица → EmbeddingReshaper → выходной эмбединг (768D)

Режимы работы:
- AUTOENCODER: вход = выход (для обучения восстановления)
- GENERATOR: обработка семантических трансформаций
- DIALOGUE: вопрос→ответ преобразования

Основные компоненты:
- EmbeddingProcessor: Главный класс процессора
- ProcessingMode: Режимы обработки  
- EmbeddingConfig: Конфигурация процессора
- ProcessingMetrics: Метрики качества

Цель Phase 2.5: Cosine similarity >90% в автоэнкодер режиме
"""

from .processor import EmbeddingProcessor
from .config import (
    EmbeddingConfig, 
    ProcessingMode,
    create_autoencoder_config,
    create_generator_config,
    create_dialogue_config,
    load_config_from_dict,
    validate_config
)
from .metrics import ProcessingMetrics, calculate_processing_quality
from .utils import (
    create_test_embedding_batch,
    validate_processor_output,
    benchmark_processing_speed,
    export_processing_results,
    run_comprehensive_test,
    create_quality_report
)

# Версия модуля Phase 2.5
__version__ = "2.5.0"

# Главные экспорты
__all__ = [
    # Основные классы
    "EmbeddingProcessor",
    "EmbeddingConfig", 
    "ProcessingMode",
    "ProcessingMetrics",
    
    # Конфигурационные функции
    "create_autoencoder_config",
    "create_generator_config", 
    "create_dialogue_config",
    "load_config_from_dict",
    "validate_config",
    
    # Утилиты метрик
    "calculate_processing_quality",
    
    # Утилиты тестирования
    "create_test_embedding_batch",
    "validate_processor_output",
    "benchmark_processing_speed",
    "run_comprehensive_test",
    "create_quality_report",
    "export_processing_results"
]

# Константы модуля
DEFAULT_INPUT_DIM = 768
DEFAULT_OUTPUT_DIM = 768
DEFAULT_CUBE_SHAPE = (8, 8, 12)
TARGET_SIMILARITY = 0.90  # Phase 2.5 цель: >90% cosine similarity 