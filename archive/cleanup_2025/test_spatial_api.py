#!/usr/bin/env python3
"""Простой тест API spatial optimizer"""

from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.lattice.lattice import Lattice3D

print("🔍 Тест API spatial optimizer...")

config = SimpleProjectConfig()
lattice = Lattice3D()

print(f"📊 Spatial optimizer type: {type(lattice.spatial_optimizer)}")
print(f"📊 Available methods:")

# Получим список всех методов
methods = [method for method in dir(lattice.spatial_optimizer) if not method.startswith('_')]
for method in methods[:10]:  # Первые 10 методов
    print(f"  - {method}")

# Попробуем вызвать find_neighbors_optimized
try:
    coords = [0, 0, 0]  # угол куба
    neighbors = lattice.spatial_optimizer.find_neighbors_optimized(coords, 1.5)
    print(f"\n✅ find_neighbors_optimized работает! Найдено {len(neighbors)} соседей для {coords}")
except Exception as e:
    print(f"\n❌ Ошибка find_neighbors_optimized: {e}")
    print(f"📊 Полный traceback:")
    import traceback
    traceback.print_exc()

print("\n✅ Тест API завершен")