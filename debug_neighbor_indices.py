#!/usr/bin/env python3
"""Диагностика проблемы с индексами соседей"""

from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import create_unified_spatial_optimizer
from new_rebuild.config import SimpleProjectConfig

print("🔍 Диагностика индексов соседей...")

config = SimpleProjectConfig()
print(f"Lattice dimensions: {config.lattice.dimensions}")
print(f"Total cells: {config.lattice.total_cells}")

# Создаем spatial optimizer 
spatial_optimizer = create_unified_spatial_optimizer(config.lattice.dimensions)

# Тестируем клетку 301
cell_idx = 301
print(f"\n🧪 Тестируем клетку {cell_idx}:")

# Получаем соседей
neighbors = spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
print(f"Найдено соседей: {len(neighbors)}")
print(f"Диапазон индексов: {min(neighbors)} - {max(neighbors)}")
print(f"Максимальный валидный индекс: {config.lattice.total_cells - 1}")

# Проверяем валидность
invalid_neighbors = [n for n in neighbors if n >= config.lattice.total_cells]
print(f"Невалидных соседей: {len(invalid_neighbors)}")
if invalid_neighbors:
    print(f"Примеры невалидных: {invalid_neighbors[:10]}")

# Проверим координаты
from new_rebuild.core.lattice.position import Position3D
pos_helper = Position3D(config.lattice.dimensions)

print(f"\n📍 Координаты клетки {cell_idx}:")
coords_301 = pos_helper.to_3d_coordinates(cell_idx)
print(f"Клетка 301: {coords_301}")

if invalid_neighbors:
    print(f"\nКоординаты невалидных соседей:")
    for idx in invalid_neighbors[:5]:
        try:
            coords = pos_helper.to_3d_coordinates(idx)
            print(f"  Клетка {idx}: {coords}")
        except Exception as e:
            print(f"  Клетка {idx}: ОШИБКА - {e}")

print("\n✅ Диагностика завершена!")