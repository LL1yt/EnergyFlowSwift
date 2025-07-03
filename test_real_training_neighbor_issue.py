#!/usr/bin/env python3
"""
Test that reproduces the exact conditions from real training
where we see 197 DISTANT connections instead of expected ~24
"""

import torch
import numpy as np
from new_rebuild.config import create_experiment_config, set_project_config
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import OptimizationConfig

# Setup config exactly as in training
config = create_experiment_config()  # 15x15x15
set_project_config(config)

print("=== Testing Real Training Conditions ===")
print(f"Lattice: {config.lattice.dimensions}")
print(f"Adaptive radius: {config.calculate_adaptive_radius()}")

# Create lattice as in real training
lattice = Lattice3D(
    dimensions=config.lattice.dimensions,
    state_size=config.model.state_size,
    config=config
)

# Test cell 677 (from logs)
test_cell = 677
print(f"\n=== Testing cell {test_cell} ===")

# Method 1: Direct spatial optimizer call (as in moe_processor)
print("\n1. Using spatial_optimizer.find_neighbors_by_radius_safe():")
neighbors_spatial = lattice.spatial_optimizer.find_neighbors_by_radius_safe(test_cell)
print(f"   Found {len(neighbors_spatial)} neighbors")

# Method 2: Using cache (as in simple test)
print("\n2. Using ConnectionCacheManager:")
from new_rebuild.core.moe import ConnectionCacheManager
cache_manager = ConnectionCacheManager(config.lattice.dimensions)

# Get from cache
all_neighbors_cache = cache_manager._all_neighbors_cache.get(test_cell, [])
print(f"   Found {len(all_neighbors_cache)} neighbors in _all_neighbors_cache")

# Compare results
print(f"\n3. Comparison:")
print(f"   Spatial optimizer: {len(neighbors_spatial)} neighbors")
print(f"   Cache manager: {len(all_neighbors_cache)} neighbors")

if len(neighbors_spatial) != len(all_neighbors_cache):
    print(f"\n❌ MISMATCH! Difference: {abs(len(neighbors_spatial) - len(all_neighbors_cache))}")
    
    # Check what's different
    spatial_set = set(neighbors_spatial)
    cache_set = set(all_neighbors_cache)
    
    only_spatial = spatial_set - cache_set
    only_cache = cache_set - spatial_set
    
    if only_spatial:
        print(f"   Only in spatial optimizer: {len(only_spatial)} neighbors")
        print(f"   Examples: {list(only_spatial)[:5]}")
    if only_cache:
        print(f"   Only in cache: {len(only_cache)} neighbors")
        print(f"   Examples: {list(only_cache)[:5]}")

# Test classification with the spatial optimizer neighbors
print(f"\n4. Testing classification with {len(neighbors_spatial)} neighbors:")

# Create mock states
num_cells = np.prod(config.lattice.dimensions)
states = torch.randn(num_cells, config.model.state_size, device='cuda')

# Get connection classifier
classifier = lattice.moe_processor.connection_classifier

# Classify connections
classifications = classifier.classify_connections(
    cell_idx=test_cell,
    neighbor_indices=neighbors_spatial,
    cell_state=states[test_cell],
    neighbor_states=states[neighbors_spatial]
)

local_count = len(classifications.get('local', []))
functional_count = len(classifications.get('functional', []))
distant_count = len(classifications.get('distant', []))

print(f"   LOCAL: {local_count}")
print(f"   FUNCTIONAL: {functional_count}")
print(f"   DISTANT: {distant_count}")
print(f"   TOTAL: {local_count + functional_count + distant_count}")

if distant_count > 50:
    print(f"\n❌ BUG REPRODUCED! Got {distant_count} DISTANT connections (expected ~24)")
else:
    print(f"\n✅ Classification looks normal")