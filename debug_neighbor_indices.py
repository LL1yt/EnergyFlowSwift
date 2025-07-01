#!/usr/bin/env python3
"""Диагностика проблемы с индексами соседей"""

from new_rebuild.config.simple_config import SimpleProjectConfig
from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import UnifiedSpatialOptimizer

print("🔍 Диагностика индексов соседей...")

config = SimpleProjectConfig()
print(f"Lattice dimensions: {config.lattice.dimensions}")
print(f"Total cells: {config.lattice.total_cells}")

# Создаем spatial optimizer 
spatial_optimizer = UnifiedSpatialOptimizer(config.lattice.dimensions)

# Проверяем несколько клеток, включая проблемные из логов
test_cells = [301, 510, 511, 0, 1, 100, 200, 400, 500]

from new_rebuild.core.lattice.position import Position3D
pos_helper = Position3D(config.lattice.dimensions)

print("\n🧪 Тестируем несколько клеток:")
total_invalid = 0
max_neighbor_idx = 0

for cell_idx in test_cells:
    if cell_idx >= config.lattice.total_cells:
        print(f"❌ Клетка {cell_idx}: вне диапазона (max={config.lattice.total_cells-1})")
        continue
        
    try:
        coords = pos_helper.to_3d_coordinates(cell_idx)
        neighbors = spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
        
        invalid_neighbors = [n for n in neighbors if n >= config.lattice.total_cells]
        max_neighbor = max(neighbors) if neighbors else 0
        max_neighbor_idx = max(max_neighbor_idx, max_neighbor)
        total_invalid += len(invalid_neighbors)
        
        print(f"📍 Клетка {cell_idx} {coords}: {len(neighbors)} соседей, max_idx={max_neighbor}, invalid={len(invalid_neighbors)}")
        
        if invalid_neighbors:
            print(f"   ❌ Невалидные: {invalid_neighbors[:5]}...")
            
    except Exception as e:
        print(f"❌ Ошибка для клетки {cell_idx}: {e}")

print(f"\n📊 Сводка:")
print(f"Общее число невалидных соседей: {total_invalid}")
print(f"Максимальный индекс соседа: {max_neighbor_idx}")
print(f"Максимальный валидный индекс: {config.lattice.total_cells - 1}")

# Проверяем дублирование компонентов
print(f"\n🔧 Проверка дублирования:")
print(f"Адрес spatial_optimizer: {id(spatial_optimizer)}")
print(f"Адрес gpu_processor: {id(spatial_optimizer.gpu_processor)}")

print("\n✅ Диагностика завершена!")