"""
Тест синхронизации кэша с spatial optimizer

Проверяет, что ConnectionCacheManager и UnifiedSpatialOptimizer
возвращают одинаковые наборы соседей после синхронизации.
"""
import torch
import numpy as np
from new_rebuild.config import get_project_config
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.utils import setup_logging, get_logger

setup_logging(__name__)
logger = get_logger(__name__)


def test_cache_sync():
    """Тестирует синхронизацию поиска соседей между кэшем и spatial optimizer"""
    
    # Создаем решетку (она сама получит конфигурацию)
    lattice = Lattice3D()
    
    # Получаем конфигурацию из решетки
    config = lattice.config
    
    logger.info("=== Тест синхронизации кэша начался ===")
    logger.info(f"Размеры решетки: {config.lattice.dimensions}")
    logger.info(f"Адаптивный радиус: {config.calculate_adaptive_radius()}")
    
    # Получаем компоненты
    spatial_optimizer = lattice.spatial_optimizer
    moe_processor = lattice.moe_processor
    connection_classifier = moe_processor.connection_classifier
    cache_manager = connection_classifier.cache_manager
    
    logger.info("\n=== Проверка синхронизации ===")
    
    # Выбираем несколько тестовых клеток
    total_cells = config.lattice.dimensions[0] * config.lattice.dimensions[1] * config.lattice.dimensions[2]
    test_cells = [0, total_cells // 2, total_cells - 1]  # Угловая, центральная и последняя клетки
    
    mismatches = 0
    total_checks = 0
    
    for cell_idx in test_cells:
        logger.info(f"\nПроверка клетки {cell_idx}:")
        
        # Получаем соседей через spatial optimizer
        spatial_neighbors = spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
        spatial_neighbors = [n for n in spatial_neighbors if n != cell_idx]  # Убираем саму клетку
        spatial_neighbors_set = set(spatial_neighbors)
        
        # Получаем соседей из кэша
        cache_data = cache_manager.cache.get(cell_idx, {})
        cache_neighbors = []
        for category, connections in cache_data.items():
            for conn in connections:
                cache_neighbors.append(conn["target_idx"])
        cache_neighbors_set = set(cache_neighbors)
        
        # Сравниваем результаты
        logger.info(f"  Spatial Optimizer: {len(spatial_neighbors)} соседей")
        logger.info(f"  Cache: {len(cache_neighbors)} соседей")
        
        # Находим различия
        only_spatial = spatial_neighbors_set - cache_neighbors_set
        only_cache = cache_neighbors_set - spatial_neighbors_set
        common = spatial_neighbors_set & cache_neighbors_set
        
        if only_spatial or only_cache:
            mismatches += 1
            logger.warning(f"  ❌ Несоответствие!")
            if only_spatial:
                logger.warning(f"    Только в Spatial: {sorted(list(only_spatial))[:10]}...")
            if only_cache:
                logger.warning(f"    Только в Cache: {sorted(list(only_cache))[:10]}...")
        else:
            logger.info(f"  ✅ Полное соответствие! {len(common)} общих соседей")
            
        total_checks += 1
        
        # Детальная проверка для центральной клетки
        if cell_idx == total_cells // 2:
            logger.info("\n  Детальный анализ категорий из кэша:")
            for category, connections in cache_data.items():
                logger.info(f"    {category}: {len(connections)} связей")
                
    # Итоговая статистика
    logger.info(f"\n=== Итоговая статистика ===")
    logger.info(f"Проверено клеток: {total_checks}")
    logger.info(f"Несоответствий: {mismatches}")
    logger.info(f"Процент соответствия: {(total_checks - mismatches) / total_checks * 100:.1f}%")
    
    if mismatches == 0:
        logger.info("✅ Полная синхронизация достигнута!")
    else:
        logger.warning("⚠️ Обнаружены несоответствия, требуется дополнительная отладка")
        
    # Проверяем производительность
    logger.info("\n=== Тест производительности ===")
    
    # Создаем тестовые данные
    states = torch.randn(total_cells, config.model.state_size).to(lattice.device)
    batch_size = 32
    cell_indices = torch.randint(0, total_cells, (batch_size,)).to(lattice.device)
    
    # Находим соседей для батча
    all_neighbors = []
    for idx in cell_indices:
        neighbors = spatial_optimizer.find_neighbors_by_radius_safe(idx.item())
        all_neighbors.append(neighbors[:config.model.neighbor_count])
        
    # Создаем тензор соседей
    max_neighbors = config.model.neighbor_count
    neighbor_indices = torch.full((batch_size, max_neighbors), -1, dtype=torch.long).to(lattice.device)
    for i, neighbors in enumerate(all_neighbors):
        neighbor_indices[i, :len(neighbors)] = torch.tensor(neighbors)
        
    # Классифицируем через кэш
    import time
    start_time = time.time()
    
    result = connection_classifier.classify_connections_batch(
        cell_indices=cell_indices,
        neighbor_indices=neighbor_indices,
        states=states
    )
    
    classification_time = time.time() - start_time
    
    logger.info(f"Время классификации батча из {batch_size} клеток: {classification_time*1000:.2f} мс")
    logger.info(f"Среднее время на клетку: {classification_time*1000/batch_size:.2f} мс")
    
    # Проверяем результаты классификации
    total_local = result["local_mask"].sum().item()
    total_functional = result["functional_mask"].sum().item()
    total_distant = result["distant_mask"].sum().item()
    total_connections = total_local + total_functional + total_distant
    
    logger.info(f"\nРаспределение типов связей:")
    logger.info(f"  LOCAL: {total_local} ({total_local/total_connections*100:.1f}%)")
    logger.info(f"  FUNCTIONAL: {total_functional} ({total_functional/total_connections*100:.1f}%)")
    logger.info(f"  DISTANT: {total_distant} ({total_distant/total_connections*100:.1f}%)")
    
    logger.info("\n=== Тест завершен ===")


if __name__ == "__main__":
    test_cache_sync()