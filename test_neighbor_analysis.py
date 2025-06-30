#!/usr/bin/env python3
"""Анализ проблемы с отсутствующими соседями"""

import torch
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import UnifiedSpatialOptimizer

print("[SEARCH] Анализ проблемы с соседями...")

config = SimpleProjectConfig()
print(f"[RULER] Lattice dimensions: {config.lattice.dimensions}")

# Создаем spatial optimizer - используем стандартную инициализацию
optimizer = UnifiedSpatialOptimizer(
    dimensions=config.lattice.dimensions
)

# Создаем тестовые состояния
num_cells = config.lattice.total_cells
state_size = config.model.state_size
test_states = torch.randn(num_cells, state_size)

print(f"[DATA] Test states shape: {test_states.shape}")

# Проверим поиск соседей для разных клеток
test_cells = [0, 63, 255, 256, 511]  # Угловые и центральные клетки

for cell_idx in test_cells:
    # Попробуем найти соседей через spatial optimizer  
    try:
        # Конвертируем linear index в 3D координаты
        z = cell_idx // (config.lattice.dimensions[0] * config.lattice.dimensions[1])
        y = (cell_idx % (config.lattice.dimensions[0] * config.lattice.dimensions[1])) // config.lattice.dimensions[0]
        x = cell_idx % config.lattice.dimensions[0]
        coords = (x, y, z)
        
        # Попробуем найти соседей с разными радиусами
        for radius in [1.0, 1.5, 2.0]:
            neighbors = optimizer.find_neighbors_optimized(coords, radius)
            print(f"[SEARCH] Cell {cell_idx} at {coords}: radius={radius} → {len(neighbors)} neighbors")
            if len(neighbors) > 0:
                break
        
        if len(neighbors) == 0:
            print(f"[ERROR] Cell {cell_idx} has NO neighbors at any radius!")
            
    except Exception as e:
        print(f"[ERROR] Error finding neighbors for cell {cell_idx}: {e}")

print("[OK] Neighbor analysis completed")