#!/usr/bin/env python3
"""
MoE (Mixture of Experts) Module для 3D CNN
==========================================

АРХИТЕКТУРА:
Базовая клетка: NCA Cell (локальная динамика нейрона)
Обработка связей: MoE Connection Processor с тремя экспертами

ЭКСПЕРТЫ ПО ТИПАМ СВЯЗЕЙ:
- SimpleLinearExpert (10%) - рефлексы (спинной мозг)
- HybridGNN_CNF_Expert (55%) - основная обработка (кора мозга)
- LightweightCNF (35%) - долгосрочная память (гиппокамп)

КОМПОНЕНТЫ:
- Три специализированных эксперта с точными параметрами
- GatingNetwork для адаптивного взвешивания результатов
- MoEConnectionProcessor для управления всей системой
- Полная интеграция с централизованным конфигом

БИОЛОГИЧЕСКАЯ АНАЛОГИЯ:
1. NCA Cell = Нейрон (локальная динамика, 4D состояние)
2. Local Expert = Рефлексы (быстрая реакция без сложных вычислений)
3. Functional Expert = Кора (основная обработка, GNN+CNF гибрид)
4. Distant Expert = Гиппокамп (долгосрочная память, continuous dynamics)

ПРИНЦИПЫ:
1. Централизованная конфигурация (все параметры из ProjectConfig)
2. Динамический расчет соседей (2000 для 27×27×27, до 10k для больших решеток)
3. Точное соответствие параметров спецификации
4. Биологическая правдоподобность распределения нагрузки
"""

from .simple_linear_expert import OptimizedSimpleLinearExpert, SimpleLinearExpert
from .hybrid_gnn_cnf_expert import HybridGNN_CNF_Expert, AdaptiveGatingNetwork
from .moe_processor import MoEConnectionProcessor, GatingNetwork
from .connection_classifier import UnifiedConnectionClassifier
from .connection_cache import ConnectionCacheManager
from .unified_cache_adapter import UnifiedCacheAdapter


# Фабричная функция для создания MoE Connection Processor
def create_moe_connection_processor(
    dimensions=None, state_size=None, device=None, config=None, **kwargs
):
    """
    Создает MoE Connection Processor с нужными параметрами
    """
    processor = MoEConnectionProcessor(
        state_size=state_size, lattice_dimensions=dimensions, config=config, **kwargs
    )
    if device is not None:
        processor = processor.to(device)
    return processor


# Фабричная функция для создания Connection Classifier с кэшем
def create_connection_classifier(lattice_dimensions, enable_cache=None, **kwargs):
    """
    Создает UnifiedConnectionClassifier с автоматическими настройками кэширования
    """
    return UnifiedConnectionClassifier(
        lattice_dimensions=lattice_dimensions, enable_cache=enable_cache, **kwargs
    )


__all__ = [
    # Эксперты
    "OptimizedSimpleLinearExpert",
    "SimpleLinearExpert",  # backward compatibility
    "HybridGNN_CNF_Expert",
    "AdaptiveGatingNetwork",
    # Основная MoE архитектура
    "MoEConnectionProcessor",
    "GatingNetwork",
    # Connection Classification and Caching
    "UnifiedConnectionClassifier",
    "ConnectionCacheManager",
    "UnifiedCacheAdapter",
    # Фабричные функции
    "create_moe_connection_processor",
    "create_connection_classifier",
]
