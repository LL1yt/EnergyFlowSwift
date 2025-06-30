#!/usr/bin/env python3
"""Простой анализ проблемы с соседями"""

import torch
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.lattice.lattice import Lattice3D

print("🔍 Простой анализ проблемы с соседями...")

config = SimpleProjectConfig()
print(f"📏 Lattice dimensions: {config.lattice.dimensions}")

# Создаем lattice
lattice = Lattice3D()

# Создаем тестовые состояния
batch_size = 1
num_cells = config.lattice.total_cells
state_size = config.model.state_size
test_states = torch.randn(batch_size, num_cells, state_size)

print(f"📊 Test states shape: {test_states.shape}")
print(f"📊 Total cells: {num_cells}")

# Устанавливаем состояния
lattice.states = test_states

# Анализируем spatial optimizer
print(f"🔍 Spatial optimizer: {type(lattice.spatial_optimizer).__name__}")
print(f"🔍 MoE processor set: {lattice.spatial_optimizer.moe_processor is not None}")

# Проверим несколько клеток в кубе 8x8x8
# Углы: (0,0,0)=0, (7,7,7)=511
# Грани: (0,0,4)=32, (7,7,4)=479  
# Центр: (4,4,4)=260
test_cells = [0, 32, 260, 479, 511]

print("\n🔍 Анализ клеток:")
for cell_idx in test_cells:
    # Конвертируем linear index в 3D координаты
    z = cell_idx // 64  # 8*8
    y = (cell_idx % 64) // 8
    x = cell_idx % 8
    coords = (x, y, z)
    
    # Классифицируем тип клетки
    is_corner = (x in [0, 7]) and (y in [0, 7]) and (z in [0, 7])
    is_edge = sum([x in [0, 7], y in [0, 7], z in [0, 7]]) == 2
    is_face = sum([x in [0, 7], y in [0, 7], z in [0, 7]]) == 1
    is_interior = not (is_corner or is_edge or is_face)
    
    cell_type = "corner" if is_corner else "edge" if is_edge else "face" if is_face else "interior"
    
    print(f"  Cell {cell_idx:3d} at {coords}: {cell_type}")

print("\n✅ Analysis completed")