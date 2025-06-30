#!/usr/bin/env python3
"""Тест поиска соседей для разных клеток"""

import torch
import numpy as np
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.lattice.lattice import Lattice3D

print("🔍 Тест поиска соседей...")

config = SimpleProjectConfig()
lattice = Lattice3D()

# Создаем тестовые состояния
batch_size = 1
num_cells = config.lattice.total_cells
state_size = config.model.state_size
test_states = torch.randn(batch_size, num_cells, state_size)
lattice.states = test_states

print(f"📊 Cube: {config.lattice.dimensions}")
print(f"📊 Total cells: {num_cells}")

# Проверим поиск соседей используя spatial optimizer напрямую
def get_neighbors_for_cell(cell_idx):
    """Найти соседей для клетки используя spatial optimizer"""
    # Конвертируем linear index в 3D координаты
    dims = config.lattice.dimensions
    z = cell_idx // (dims[0] * dims[1])
    y = (cell_idx % (dims[0] * dims[1])) // dims[0]
    x = cell_idx % dims[0]
    coords = [x, y, z]
    
    try:
        # Попробуем найти соседей с разными радиусами
        for radius in [1.0, 1.5, 2.0, 3.0]:
            neighbors = lattice.spatial_optimizer.find_neighbors_optimized(coords, radius)
            if len(neighbors) > 0:
                return neighbors, radius
        return [], 0.0
    except Exception as e:
        print(f"❌ Ошибка поиска соседей для клетки {cell_idx}: {e}")
        return [], 0.0

# Тестируем разные типы клеток
test_cells = [
    (0, "corner (0,0,0)"),
    (7, "corner (7,0,0)"), 
    (56, "corner (0,7,0)"),
    (63, "corner (7,7,0)"),
    (448, "corner (0,0,7)"),
    (511, "corner (7,7,7)"),
    (260, "center (4,4,4)"),
    (32, "edge"),
    (100, "interior")
]

print("\n🔍 Результаты поиска соседей:")
print("Клетка | Тип           | Соседи | Радиус")
print("-" * 45)

for cell_idx, cell_type in test_cells:
    neighbors, radius = get_neighbors_for_cell(cell_idx)
    print(f"{cell_idx:6d} | {cell_type:13s} | {len(neighbors):6d} | {radius:6.1f}")

# Попробуем также создать простых соседей по манхэттенскому расстоянию
def get_manhattan_neighbors(cell_idx, max_distance=1):
    """Найти соседей используя манхэттенское расстояние"""
    dims = config.lattice.dimensions
    z = cell_idx // (dims[0] * dims[1])
    y = (cell_idx % (dims[0] * dims[1])) // dims[0]
    x = cell_idx % dims[0]
    
    neighbors = []
    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            for dz in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Пропускаем саму клетку
                
                nx, ny, nz = x + dx, y + dy, z + dz
                
                # Проверяем границы
                if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                    neighbor_idx = nz * dims[0] * dims[1] + ny * dims[0] + nx
                    neighbors.append(neighbor_idx)
    
    return neighbors

print("\n🔍 Сравнение с манхэттенским расстоянием:")
print("Клетка | Spatial | Manhattan")
print("-" * 25)

for cell_idx, _ in test_cells[:6]:  # Только угловые клетки
    spatial_neighbors, _ = get_neighbors_for_cell(cell_idx)
    manhattan_neighbors = get_manhattan_neighbors(cell_idx, 1)
    print(f"{cell_idx:6d} | {len(spatial_neighbors):7d} | {len(manhattan_neighbors):9d}")

print("\n✅ Тест завершен")