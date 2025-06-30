#!/usr/bin/env python3
"""Подробный тест для debugging spatial API"""

from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.lattice.lattice import Lattice3D
import torch

print("[SEARCH] Debugging spatial API...")

config = SimpleProjectConfig()
print(f"[DATA] Размеры решетки: {config.lattice.dimensions}")

lattice = Lattice3D()
optimizer = lattice.spatial_optimizer

print(f"[DATA] Spatial optimizer: {type(optimizer)}")
print(f"[DATA] GPU processor: {type(optimizer.gpu_processor)}")

# Проверим adaptive_hash
adaptive_hash = optimizer.gpu_processor.adaptive_hash
print(f"[DATA] Adaptive hash: {type(adaptive_hash)}")

# Заполним spatial hash данными как это делается в реальной работе
print("\n[TOOL] Заполняем spatial hash...")
dims = config.lattice.dimensions
total_cells = dims[0] * dims[1] * dims[2]

# Создаем dummy states для инициализации
dummy_states = torch.randn(total_cells, 64, device='cuda' if torch.cuda.is_available() else 'cpu')
optimizer.gpu_processor._populate_spatial_hash(dummy_states)

print(f"[OK] Заполнили spatial hash для {total_cells} клеток")

# Теперь тестируем поиск соседей
test_coords = [
    [0, 0, 0],    # угол
    [1, 1, 1],    # рядом с углом  
    [5, 5, 5],    # центр (если размер > 10)
]

for coords in test_coords:
    # Проверим что координаты валидные
    if (0 <= coords[0] < dims[0] and 
        0 <= coords[1] < dims[1] and 
        0 <= coords[2] < dims[2]):
        
        print(f"\n[SEARCH] Тестируем координаты {coords}:")
        
        # Вычислим ожидаемый индекс центральной точки
        center_idx = coords[0] + coords[1] * dims[0] + coords[2] * dims[0] * dims[1]
        print(f"  📍 Центральный индекс: {center_idx}")
        
        for radius in [1.0, 1.5, 2.0, 3.0]:
            try:
                neighbors = optimizer.find_neighbors_optimized(coords, radius)
                print(f"  [RULER] Радиус {radius}: найдено {len(neighbors)} соседей")
                if neighbors and len(neighbors) < 10:
                    print(f"      Соседи: {neighbors[:5]}...")
                    # Проверим что центральная точка исключена
                    if center_idx in neighbors:
                        print(f"      [WARN] Центральная точка {center_idx} все еще в списке!")
                    else:
                        print(f"      [OK] Центральная точка {center_idx} исключена")
            except Exception as e:
                print(f"  [ERROR] Ошибка при радиусе {radius}: {e}")
    else:
        print(f"[WARN] Координаты {coords} вне решетки {dims}")

print("\n[OK] Debug завершен")