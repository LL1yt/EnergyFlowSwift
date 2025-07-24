#!/usr/bin/env python3
"""
Batch Processing Integration
===========================

Модуль для легкой интеграции batch оптимизаций в существующую систему.
Позволяет включать batch обработку через конфигурацию или runtime переключение.
"""

from typing import Optional, Tuple
import torch

from .unified_spatial_optimizer import UnifiedSpatialOptimizer
from .gpu_spatial_processor_batch import GPUSpatialProcessorBatch
from ....utils.logging import get_logger
from ....config import get_project_config

logger = get_logger(__name__)


def create_batch_optimized_spatial_optimizer(
    dimensions: Tuple[int, int, int],
    moe_processor,
    enable_batch: bool = True,
    batch_threshold: int = 4,
    **kwargs
) -> UnifiedSpatialOptimizer:
    """
    Создает UnifiedSpatialOptimizer с поддержкой batch обработки.
    
    Args:
        dimensions: размерность решетки
        moe_processor: MoE процессор
        enable_batch: включить ли batch обработку
        batch_threshold: минимальный размер chunk'а для batch обработки
        **kwargs: дополнительные параметры для UnifiedSpatialOptimizer
        
    Returns:
        UnifiedSpatialOptimizer с batch оптимизациями
    """
    config = get_project_config()
    
    # Проверяем, включена ли batch оптимизация в конфигурации
    if hasattr(config, 'performance') and hasattr(config.performance, 'enable_batch_processing'):
        enable_batch = config.performance.enable_batch_processing
        logger.info(f"Batch processing set from config: {enable_batch}")
    
    # Создаем оптимизатор
    optimizer = UnifiedSpatialOptimizer(dimensions, moe_processor, **kwargs)
    
    if enable_batch:
        # Заменяем GPU processor на batch версию
        logger.info("🚀 Enabling batch processing for spatial optimization")
        
        batch_processor = GPUSpatialProcessorBatch(
            dimensions=dimensions,
            moe_processor=moe_processor,
            enable_batch=enable_batch,
            batch_threshold=batch_threshold,
            profile_performance=True
        )
        
        # Заменяем процессор в оптимизаторе
        optimizer.gpu_processor = batch_processor
        
        # Добавляем метод для получения статистики
        def get_batch_performance_report():
            if hasattr(optimizer.gpu_processor, 'get_performance_report'):
                return optimizer.gpu_processor.get_performance_report()
            return {"status": "Batch processing not available"}
        
        optimizer.get_batch_performance_report = get_batch_performance_report
        
        # Добавляем метод для переключения batch режима
        def set_batch_enabled(enabled: bool):
            if hasattr(optimizer.gpu_processor, 'set_batch_enabled'):
                optimizer.gpu_processor.set_batch_enabled(enabled)
                logger.info(f"Batch processing {'enabled' if enabled else 'disabled'}")
            else:
                logger.warning("Batch processing control not available")
        
        optimizer.set_batch_enabled = set_batch_enabled
        
        logger.info("✅ Batch processing enabled for spatial optimizer")
    else:
        logger.info("ℹ️ Using standard per-cell processing")
    
    return optimizer


def upgrade_lattice_to_batch(lattice):
    """
    Обновляет существующую Lattice3D для использования batch обработки.
    
    Args:
        lattice: экземпляр Lattice3D
        
    Returns:
        Обновленная lattice с batch оптимизациями
    """
    if not hasattr(lattice, 'spatial_optimizer'):
        logger.error("Lattice doesn't have spatial_optimizer attribute")
        return lattice
    
    # Получаем параметры из существующего оптимизатора
    old_optimizer = lattice.spatial_optimizer
    dimensions = old_optimizer.dimensions
    moe_processor = old_optimizer.moe_processor
    
    # Создаем новый оптимизатор с batch поддержкой
    new_optimizer = create_batch_optimized_spatial_optimizer(
        dimensions=dimensions,
        moe_processor=moe_processor,
        enable_batch=True
    )
    
    # Заменяем оптимизатор
    lattice.spatial_optimizer = new_optimizer
    
    # КРИТИЧЕСКИ ВАЖНО: Принудительная инициализация кэша для batch режима
    logger.info("🔧 Принудительная инициализация кэша для batch режима...")
    cache_manager = new_optimizer.moe_processor.connection_classifier
    
    # Сбрасываем старый кэш
    cache_manager._all_neighbors_cache = None
    
    # Принудительно пересоздаем кэш
    cache_manager._all_neighbors_cache = cache_manager._compute_all_neighbors()
    
    # Валидация кэша
    expected_cells = lattice.total_cells
    cached_cells = len(cache_manager._all_neighbors_cache) if cache_manager._all_neighbors_cache else 0
    
    logger.info(f"🔍 CACHE VALIDATION:")
    logger.info(f"   Expected cells: {expected_cells}")
    logger.info(f"   Cached cells: {cached_cells}")
    
    if cached_cells != expected_cells:
        raise RuntimeError(
            f"❌ КРИТИЧЕСКАЯ ОШИБКА: Кэш содержит {cached_cells} клеток, "
            f"ожидается {expected_cells}. Batch режим невозможен."
        )
    
    # Проверяем типы ключей в кэше
    if cache_manager._all_neighbors_cache:
        sample_keys = list(cache_manager._all_neighbors_cache.keys())[:5]
        key_types = [type(k) for k in sample_keys]
        logger.info(f"   Cache key types: {key_types}")
        logger.info(f"   Key range: {min(sample_keys)} - {max(sample_keys)}")
    
    logger.info("✅ Кэш успешно инициализирован для batch режима")
    
    # Добавляем методы для управления batch режимом
    def set_batch_enabled(enabled: bool):
        if hasattr(lattice.spatial_optimizer, 'set_batch_enabled'):
            lattice.spatial_optimizer.set_batch_enabled(enabled)
    
    def get_batch_performance():
        if hasattr(lattice.spatial_optimizer, 'get_batch_performance_report'):
            return lattice.spatial_optimizer.get_batch_performance_report()
        return {"status": "Batch processing not available"}
    
    lattice.set_batch_enabled = set_batch_enabled
    lattice.get_batch_performance = get_batch_performance
    
    logger.info("✅ Lattice upgraded with batch processing support")
    
    return lattice