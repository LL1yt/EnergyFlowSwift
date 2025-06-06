"""
EmbeddingReshaper - Мост между модулями системы
=====================================================

Критически важный компонент для преобразования 1D эмбедингов (768D) 
в 3D формат для куба (8×8×12) и обратно с сохранением семантической информации >95%.

Основные компоненты:
- EmbeddingReshaper: Главный класс для трансформаций 1D↔3D
- AdaptiveReshaper: Умное преобразование с оптимизацией
- SemanticPreserver: Контроль качества преобразования
- ReshapingStrategies: Три стратегии reshaping

Экспорты:
- EmbeddingReshaper: Основной класс
- AdaptiveReshaper: Адаптивная стратегия
- LinearReshaper: Простая стратегия
- SemanticReshaper: Сохранение семантики
- validate_semantic_preservation: Функция контроля качества
"""

# Основные классы
from .reshaper import EmbeddingReshaper
from .strategies import (
    AdaptiveReshaper,
    LinearReshaper, 
    SemanticReshaper
)
from .utils import (
    validate_semantic_preservation,
    calculate_similarity_metrics,
    optimize_shape_transformation,
    create_test_embeddings,
    benchmark_transformation_speed
)

# Версия модуля
__version__ = "1.0.0"

# Экспорты модуля
__all__ = [
    "EmbeddingReshaper",
    "AdaptiveReshaper", 
    "LinearReshaper",
    "SemanticReshaper",
    "validate_semantic_preservation",
    "calculate_similarity_metrics",
    "optimize_shape_transformation",
    "create_test_embeddings",
    "benchmark_transformation_speed"
]

# Константы модуля
DEFAULT_INPUT_DIM = 768
DEFAULT_CUBE_SHAPE = (8, 8, 12)
SEMANTIC_THRESHOLD = 0.95 