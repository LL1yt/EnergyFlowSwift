#!/usr/bin/env python3
"""
Fix for the neighbor threshold issue.
The problem: adaptive_radius finds neighbors beyond distant_threshold.
The solution: Filter neighbors to only include those within distant_threshold.
"""

import torch
import numpy as np
from typing import List, Dict

def fix_connection_cache_manager():
    """
    Fix the _compute_all_neighbors_gpu method to respect distant_threshold
    """
    
    fix_code = '''
    def _compute_all_neighbors_gpu(self) -> Dict[int, List[int]]:
        """GPU-ускоренная версия вычисления соседей"""
        try:
            x_dim, y_dim, z_dim = self.lattice_dimensions
            
            # Создаем координаты всех клеток на GPU
            all_indices = torch.arange(self.total_cells, device=self.device)
            
            x_coords = all_indices % x_dim
            y_coords = (all_indices // x_dim) % y_dim
            z_coords = all_indices // (x_dim * y_dim)
            
            all_coords = torch.stack([x_coords, y_coords, z_coords], dim=1).float()
            
            logger.info(
                f"💾 GPU memory для координат: {all_coords.numel() * 4 / 1024**2:.1f}MB"
            )
            
            all_neighbors = {}
            batch_size = min(self.gpu_batch_size, self.total_cells)
            
            # Обрабатываем батчами для экономии памяти
            for start_idx in range(0, self.total_cells, batch_size):
                end_idx = min(start_idx + batch_size, self.total_cells)
                batch_coords = all_coords[start_idx:end_idx]
                
                # Вычисляем расстояния до всех других клеток
                # batch_coords: [batch_size, 3], all_coords: [total_cells, 3]
                distances = torch.cdist(
                    batch_coords, all_coords
                )  # [batch_size, total_cells]
                
                # Находим соседей в радиусе (исключая саму клетку)
                for i, cell_idx in enumerate(range(start_idx, end_idx)):
                    # ИСПРАВЛЕНИЕ: Используем distant_threshold вместо adaptive_radius
                    # Это гарантирует, что все найденные соседи попадут в одну из трех категорий
                    neighbor_mask = (distances[i] <= self.distant_threshold) & (
                        distances[i] > 0
                    )
                    neighbor_indices = torch.where(neighbor_mask)[0].cpu().tolist()
                    all_neighbors[cell_idx] = neighbor_indices
                
                # Освобождаем GPU память
                del distances
                torch.cuda.empty_cache()
                
                if start_idx % (batch_size * 10) == 0:
                    logger.info(
                        f"🚀 GPU: обработано {end_idx}/{self.total_cells} клеток"
                    )
            
            self._all_neighbors_cache = all_neighbors
            logger.info(f"✅ GPU: Вычислены соседи для {len(all_neighbors)} клеток")
            
            # Логирование для диагностики
            total_neighbors = sum(len(neighbors) for neighbors in all_neighbors.values())
            avg_neighbors = total_neighbors / len(all_neighbors) if all_neighbors else 0
            logger.info(f"   Среднее количество соседей на клетку: {avg_neighbors:.1f}")
            logger.info(f"   Используемый порог: {self.distant_threshold} (distant_threshold)")
            
            return all_neighbors
            
        except Exception as e:
            logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return self._compute_all_neighbors_cpu()
    '''
    
    print("Fix for ConnectionCacheManager:")
    print("=" * 80)
    print(fix_code)
    print("=" * 80)
    print("\nПрименение фикса:")
    print("1. В методе _compute_all_neighbors_gpu заменить self.adaptive_radius на self.distant_threshold")
    print("2. Это гарантирует, что все найденные соседи будут в пределах distant_threshold")
    print("3. Результат: никаких соседей за пределами порога!")

def fix_gpu_spatial_processor():
    """
    Alternative fix in GPUSpatialProcessor
    """
    
    print("\n\nAlternative fix for GPUSpatialProcessor:")
    print("=" * 80)
    print("""
    В методе find_neighbors можно добавить фильтрацию:
    
    # После получения neighbor_lists
    neighbor_lists = self.adaptive_hash.query_radius_batch(coords_tensor, radius)
    
    # ДОБАВИТЬ: Фильтрация по фактическому расстоянию
    if neighbor_lists and len(neighbor_lists) > 0:
        neighbors_tensor = neighbor_lists[0]
        
        # Вычисляем реальные расстояния
        if hasattr(self, 'pos_helper'):
            filtered_neighbors = []
            center_coords = coords_tensor[0]
            
            for neighbor_idx in neighbors_tensor.cpu().tolist():
                neighbor_coords = self.pos_helper.to_3d_coordinates(neighbor_idx)
                neighbor_coords_tensor = torch.tensor(neighbor_coords, device=self.device, dtype=torch.float32)
                
                # Вычисляем евклидово расстояние
                dist = torch.norm(center_coords - neighbor_coords_tensor).item()
                
                # Включаем только если расстояние <= radius
                if dist <= radius and neighbor_idx != center_idx:
                    filtered_neighbors.append(neighbor_idx)
            
            return filtered_neighbors
    """)

def show_config_adjustment():
    """
    Show how to adjust config to avoid the issue
    """
    
    print("\n\nКонфигурационное решение:")
    print("=" * 80)
    print("""
    В config_components.py можно немного уменьшить adaptive_radius_ratio:
    
    # Текущее значение
    adaptive_radius_ratio: float = 0.2  # Дает radius = 3.0 для решетки 15x15x15
    
    # Рекомендуемое значение для избежания проблемы
    adaptive_radius_ratio: float = 0.19  # Даст radius = 2.85, все соседи будут в пределах порога
    
    Это временное решение, но правильнее исправить код!
    """)

if __name__ == "__main__":
    print("АНАЛИЗ РЕШЕНИЯ ПРОБЛЕМЫ С СОСЕДЯМИ")
    print("=" * 80)
    print("\nПроблема:")
    print("- spatial_optimizer находит 191 соседа")
    print("- из них 117 находятся за пределами distant_threshold (3.0)")
    print("- эти лишние соседи классифицируются как DISTANT")
    print("- в результате получаем 197 DISTANT соединений вместо 24")
    
    print("\nРешение:")
    fix_connection_cache_manager()
    fix_gpu_spatial_processor()
    show_config_adjustment()
    
    print("\n\nРекомендация:")
    print("Применить фикс в ConnectionCacheManager._compute_all_neighbors_gpu")
    print("Это самое чистое решение, которое исправит проблему в корне.")