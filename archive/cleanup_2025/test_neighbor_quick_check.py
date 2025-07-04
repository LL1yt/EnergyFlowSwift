#!/usr/bin/env python3
"""
Quick test to check neighbor distribution issue
"""

import torch
from new_rebuild.config import create_experiment_config, set_project_config
from new_rebuild.core.moe import ConnectionCacheManager
from new_rebuild.core.moe.distance_calculator import DistanceCalculator

# Setup config
config = create_experiment_config()  # 15x15x15
set_project_config(config)

print(f"Lattice: {config.lattice.dimensions}")
print(f"Adaptive radius: {config.calculate_adaptive_radius()}")
print(f"Ratios: local={config.lattice.local_distance_ratio}, "
      f"functional={config.lattice.functional_distance_ratio}, "
      f"distant={config.lattice.distant_distance_ratio}")

# Create cache manager
cache_manager = ConnectionCacheManager(config.lattice.dimensions)

# Check if cached_data exists and if not, rebuild
print("\nChecking cache...")
if cache_manager.cache:
    print(f"Cache exists with {len(cache_manager.cache)} cells")
else:
    print("Rebuilding cache...")
    cache_manager.precompute_all_connections(force_rebuild=True)

# Check cell 677
cell_idx = 677
print(f"\nChecking cell {cell_idx}:")

# Get all neighbors from _all_neighbors_cache
all_neighbors = cache_manager._all_neighbors_cache.get(cell_idx, [])
print(f"Total neighbors found: {len(all_neighbors)}")

# Get cached classification
cached_data = cache_manager.cache.get(cell_idx, {})
if cached_data:
    local = len(cached_data.get('local', []))
    functional = len(cached_data.get('functional_candidates', []))
    distant = len(cached_data.get('distant', []))
    
    print(f"\nClassification:")
    print(f"  LOCAL: {local}")
    print(f"  FUNCTIONAL: {functional}")
    print(f"  DISTANT: {distant}")
    print(f"  TOTAL: {local + functional + distant}")
    
    # Check distances of distant connections
    dist_calc = DistanceCalculator(config.lattice.dimensions)
    distant_connections = cached_data.get('distant', [])
    
    if distant_connections:
        print(f"\nAnalyzing {len(distant_connections)} DISTANT connections:")
        distances = []
        for conn in distant_connections[:10]:  # Check first 10
            if hasattr(conn, 'euclidean_distance'):
                dist = conn.euclidean_distance
            else:
                dist = conn['euclidean_distance']
            distances.append(dist)
        
        print(f"  Sample distances: {[f'{d:.2f}' for d in distances]}")
        print(f"  Distant threshold: {cache_manager.distant_threshold:.2f}")
        
        # Count how many are beyond threshold
        beyond = 0
        for conn in distant_connections:
            if hasattr(conn, 'euclidean_distance'):
                dist = conn.euclidean_distance
            else:
                dist = conn['euclidean_distance']
            if dist > cache_manager.distant_threshold:
                beyond += 1
        
        if beyond > 0:
            print(f"\n❌ ERROR: {beyond} distant connections are beyond threshold!")
else:
    print("No cached data found for this cell")