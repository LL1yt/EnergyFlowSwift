"""
Адаптер для синхронизации ConnectionCacheManager с UnifiedSpatialOptimizer

Обеспечивает единую логику поиска соседей между кэшем и spatial optimizer.
"""
from typing import Dict, List, Optional, TYPE_CHECKING
import torch
import numpy as np

from new_rebuild.utils import get_logger, setup_logging
from new_rebuild.core.moe.connection_types import ConnectionCategory
from new_rebuild.core.moe.connection_cache import CachedConnectionInfo

if TYPE_CHECKING:
    from new_rebuild.core.lattice.spatial_optimization import UnifiedSpatialOptimizer
    from new_rebuild.core.moe.connection_cache import ConnectionCacheManager

setup_logging(__name__)
logger = get_logger(__name__)


class UnifiedCacheAdapter:
    """
    Адаптер для интеграции spatial optimizer с connection cache
    
    Основная задача - обеспечить, чтобы кэш использовал ту же логику 
    поиска соседей, что и spatial optimizer, для полной синхронизации.
    """
    
    def __init__(
        self, 
        cache_manager: 'ConnectionCacheManager',
        spatial_optimizer: Optional['UnifiedSpatialOptimizer'] = None
    ):
        self.cache_manager = cache_manager
        self.spatial_optimizer = spatial_optimizer
        
    def compute_neighbors_with_spatial_optimizer(
        self, 
        cell_idx: int
    ) -> List[int]:
        """
        Использует spatial optimizer для поиска соседей
        
        Args:
            cell_idx: Индекс клетки
            
        Returns:
            Список индексов соседей
        """
        if self.spatial_optimizer is None:
            logger.warning(
                "Spatial optimizer не установлен, используем встроенную логику кэша"
            )
            return self._fallback_neighbor_search(cell_idx)
            
        # Используем spatial optimizer для поиска соседей
        neighbors = self.spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
        
        logger.debug(
            f"Spatial optimizer нашел {len(neighbors)} соседей для клетки {cell_idx}"
        )
        
        return neighbors
        
    def precompute_with_spatial_optimizer(self) -> Dict[int, Dict[str, List[CachedConnectionInfo]]]:
        """
        Предвычисляет все связи используя spatial optimizer
        
        Returns:
            Словарь с классифицированными связями для каждой клетки
        """
        logger.info("Начинаем предвычисление кэша с использованием Spatial Optimizer")
        
        all_connections = {}
        
        for cell_idx in range(self.cache_manager.total_cells):
            # Получаем соседей через spatial optimizer
            neighbors = self.compute_neighbors_with_spatial_optimizer(cell_idx)
            
            if not neighbors:
                all_connections[cell_idx] = {
                    ConnectionCategory.LOCAL: [],
                    ConnectionCategory.FUNCTIONAL: [],
                    ConnectionCategory.DISTANT: []
                }
                continue
                
            # Классифицируем найденных соседей
            connections = self._classify_neighbors(cell_idx, neighbors)
            all_connections[cell_idx] = connections
            
            # Прогресс
            if cell_idx % 500 == 0 and cell_idx > 0:
                logger.debug(
                    f"Обработано {cell_idx}/{self.cache_manager.total_cells} клеток"
                )
                
        # Статистика
        total_connections = sum(
            len(conns) 
            for cell_conns in all_connections.values() 
            for conns in cell_conns.values()
        )
        logger.info(
            f"Предвычисление завершено: {total_connections} связей для "
            f"{self.cache_manager.total_cells} клеток"
        )
        
        return all_connections
        
    def _classify_neighbors(
        self, 
        cell_idx: int, 
        neighbors: List[int]
    ) -> Dict[str, List[CachedConnectionInfo]]:
        """
        Классифицирует соседей по категориям
        
        Args:
            cell_idx: Индекс исходной клетки
            neighbors: Список индексов соседей
            
        Returns:
            Словарь с классифицированными связями
        """
        connections = {
            ConnectionCategory.LOCAL: [],
            ConnectionCategory.FUNCTIONAL: [],
            ConnectionCategory.DISTANT: []
        }
        
        # Получаем координаты клетки
        cell_coords = self._idx_to_coords(cell_idx)
        
        for neighbor_idx in neighbors:
            # Получаем координаты соседа
            neighbor_coords = self._idx_to_coords(neighbor_idx)
            
            # Вычисляем расстояния
            euclidean_dist = np.sqrt(
                sum((c1 - c2) ** 2 for c1, c2 in zip(cell_coords, neighbor_coords))
            )
            manhattan_dist = sum(
                abs(c1 - c2) for c1, c2 in zip(cell_coords, neighbor_coords)
            )
            
            # Классифицируем по расстоянию
            if euclidean_dist <= self.cache_manager.local_threshold:
                category = ConnectionCategory.LOCAL
            elif euclidean_dist <= self.cache_manager.functional_threshold:
                # Для средних расстояний используем FUNCTIONAL
                category = ConnectionCategory.FUNCTIONAL
            else:
                category = ConnectionCategory.DISTANT
                
            # Создаем CachedConnectionInfo
            conn_info = CachedConnectionInfo(
                target_idx=neighbor_idx,
                euclidean_distance=euclidean_dist,
                manhattan_distance=manhattan_dist,
                category=category
            )
            
            connections[category].append(conn_info)
            
        return connections
        
    def _idx_to_coords(self, idx: int) -> tuple:
        """Преобразует линейный индекс в 3D координаты"""
        x_dim, y_dim, z_dim = self.cache_manager.lattice_dimensions
        x = idx % x_dim
        y = (idx // x_dim) % y_dim
        z = idx // (x_dim * y_dim)
        return (x, y, z)
        
    def _fallback_neighbor_search(self, cell_idx: int) -> List[int]:
        """Fallback метод для поиска соседей без spatial optimizer"""
        # Используем существующую логику из cache manager
        all_neighbors = self.cache_manager._compute_all_neighbors()
        return all_neighbors.get(cell_idx, [])
        
    def sync_cache_with_optimizer(self):
        """
        Синхронизирует кэш с текущим состоянием spatial optimizer
        
        Пересчитывает весь кэш используя логику spatial optimizer.
        """
        if self.spatial_optimizer is None:
            logger.error("Невозможно синхронизировать: spatial optimizer не установлен")
            return
            
        logger.info("Начинаем синхронизацию кэша с spatial optimizer...")
        
        # Пересчитываем все связи через spatial optimizer
        new_cache = self.precompute_with_spatial_optimizer()
        
        # Обновляем кэш
        self.cache_manager.cache = new_cache
        self.cache_manager.is_precomputed = True
        
        # Сохраняем на диск
        self.cache_manager._save_cache_to_disk()
        
        logger.info("Синхронизация завершена успешно")