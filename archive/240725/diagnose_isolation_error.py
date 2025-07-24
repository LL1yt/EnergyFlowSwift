#!/usr/bin/env python3
"""
Diagnostic script to identify the root cause of cell isolation errors.
"""

import torch
import numpy as np
import sys
import os

# Add the new_rebuild path
sys.path.insert(0, 'new_rebuild')

from new_rebuild.config import create_debug_config, set_project_config

def diagnose_isolation_error():
    """Diagnose why cells 106 and 331 have 0 neighbors."""
    
    print("🔍 Диагностика ошибки изоляции клеток...")
    
    # Load debug configuration
    config = create_debug_config()
    set_project_config(config)
    
    # Get lattice dimensions
    lattice_size = config.lattice.dimensions[0]  # 15 for debug mode
    total_cells = config.lattice.total_cells
    
    print(f"📊 Размер решетки: {lattice_size}x{lattice_size}x{lattice_size} = {total_cells} клеток")
    
    # Check adaptive radius from DebugPreset
    adaptive_radius = config.lattice.max_radius
    print(f"⚙️  Адаптивный радиус: {adaptive_radius} (0.4 × {lattice_size})")
    
    # Check distance thresholds
    local_threshold = config.lattice.local_distance_threshold
    functional_threshold = config.lattice.functional_distance_threshold
    distant_threshold = config.lattice.distant_distance_threshold
    
    print(f"📏 Пороги расстояний:")
    print(f"   LOCAL: ≤{local_threshold:.2f}")
    print(f"   FUNCTIONAL: ≤{functional_threshold:.2f}")
    print(f"   DISTANT: ≤{distant_threshold:.2f}")
    
    # Calculate expected neighbors for boundary cells
    max_distance = adaptive_radius
    
    # Check specific problematic cells
    problem_cells = [106, 331]
    
    # Create distance calculator
    from new_rebuild.core.moe.distance_calculator import DistanceCalculator
    distance_calc = DistanceCalculator(config.lattice.dimensions)
    
    print(f"\n🔍 Анализ проблемных клеток:")
    
    for cell_id in problem_cells:
        if cell_id >= total_cells:
            print(f"❌ Клетка {cell_id} выходит за пределы решетки (макс: {total_cells-1})")
            continue
            
        # Get cell position
        x, y, z = distance_calc.linear_to_3d(cell_id)
        print(f"\n📍 Клетка {cell_id}: позиция ({x}, {y}, {z})")
        
        # Check if it's at boundary
        is_boundary = any(p <= max_distance or p >= lattice_size - max_distance - 1 
                         for p in [x, y, z])
        print(f"   Граничная клетка: {'Да' if is_boundary else 'Нет'}")
        
        # Calculate actual possible neighbors
        possible_neighbors = 0
        actual_distances = []
        
        for dx in range(-int(max_distance)-2, int(max_distance)+3):
            for dy in range(-int(max_distance)-2, int(max_distance)+3):
                for dz in range(-int(max_distance)-2, int(max_distance)+3):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < lattice_size and 0 <= ny < lattice_size and 0 <= nz < lattice_size:
                        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                        if distance <= max_distance:
                            possible_neighbors += 1
                            actual_distances.append(distance)
        
        print(f"   Возможных соседей: {possible_neighbors}")
        
        if possible_neighbors == 0:
            print(f"   ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Даже теоретически 0 соседей!")
            print(f"   💡 Решение: увеличить adaptive_radius_ratio с 0.4 до 0.6-0.8")
        elif possible_neighbors < 5:
            print(f"   ⚠️  Мало соседей: {possible_neighbors} (нормально > 10)")
            print(f"   💡 Рекомендация: увеличить радиус или использовать торическую топологию")
        else:
            print(f"   ✅ Должно быть достаточно соседей: {possible_neighbors}")
    
    print(f"\n📋 РЕЗЮМЕ:")
    print(f"   - Адаптивный радиус: {adaptive_radius}")
    print(f"   - Это означает радиус {adaptive_radius} клеток от центра")
    print(f"   - Для 15×15×15 решетки это может быть недостаточно для граничных клеток")
    print(f"   - Рекомендуемое значение: 0.6-0.8 для debug режима")

if __name__ == "__main__":
    diagnose_isolation_error()