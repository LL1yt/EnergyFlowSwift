#!/usr/bin/env python3
"""
Test to reproduce and fix the neighbor distribution issue.

The problem: adaptive_radius is used to find ALL neighbors, but distant_threshold
should limit the DISTANT neighbors to only those within distant_threshold.
"""

import torch
import numpy as np
from new_rebuild.config import create_debug_config, set_project_config
from new_rebuild.core.moe import ConnectionCacheManager

def test_neighbor_distribution():
    """Test that neighbor distribution respects the thresholds correctly"""
    
    # Create test config
    config = create_debug_config()
    set_project_config(config)
    
    # Get parameters from config
    lattice_dimensions = config.lattice.dimensions
    adaptive_radius = config.calculate_adaptive_radius()
    local_ratio = config.lattice.local_distance_ratio
    functional_ratio = config.lattice.functional_distance_ratio
    distant_ratio = config.lattice.distant_distance_ratio
    
    # Calculate thresholds
    local_threshold = adaptive_radius * local_ratio
    functional_threshold = adaptive_radius * functional_ratio
    distant_threshold = adaptive_radius * distant_ratio
    
    print(f"=== Configuration ===")
    print(f"Lattice dimensions: {lattice_dimensions}")
    print(f"Adaptive radius: {adaptive_radius}")
    print(f"Ratios: local={local_ratio}, functional={functional_ratio}, distant={distant_ratio}")
    print(f"Thresholds: local={local_threshold}, functional={functional_threshold}, distant={distant_threshold}")
    
    # Create cache manager
    cache_manager = ConnectionCacheManager(lattice_dimensions)
    
    # Pre-compute connections
    print("\n=== Pre-computing connections ===")
    cache_manager.precompute_all_connections(force_rebuild=True)
    
    # Check a cell in the center
    center_idx = cache_manager.total_cells // 2
    print(f"\n=== Analyzing cell {center_idx} ===")
    
    # Get all neighbors found during pre-computation
    all_neighbors = cache_manager._all_neighbors_cache.get(center_idx, [])
    print(f"Total neighbors found in adaptive_radius ({adaptive_radius}): {len(all_neighbors)}")
    
    # Get cached connections
    cached_data = cache_manager.cache.get(center_idx, {})
    
    # Check distribution
    local_count = len(cached_data.get('local', []))
    functional_count = len(cached_data.get('functional_candidates', []))
    distant_count = len(cached_data.get('distant', []))
    
    print(f"\nCached distribution:")
    print(f"  LOCAL: {local_count} connections")
    print(f"  FUNCTIONAL candidates: {functional_count} connections")
    print(f"  DISTANT: {distant_count} connections")
    print(f"  TOTAL: {local_count + functional_count + distant_count}")
    
    # Calculate expected distribution based on thresholds
    distance_calc = cache_manager.distance_calculator
    
    expected_local = 0
    expected_functional = 0
    expected_distant = 0
    
    for neighbor_idx in all_neighbors:
        dist = distance_calc.euclidean_distance(center_idx, neighbor_idx)
        if dist < local_threshold:
            expected_local += 1
        elif dist <= functional_threshold:
            expected_functional += 1
        elif dist <= distant_threshold:
            expected_distant += 1
        # Note: neighbors beyond distant_threshold should NOT be included!
    
    print(f"\nExpected distribution (based on thresholds):")
    print(f"  LOCAL (0 < d < {local_threshold}): {expected_local}")
    print(f"  FUNCTIONAL ({local_threshold} <= d <= {functional_threshold}): {expected_functional}")
    print(f"  DISTANT ({functional_threshold} < d <= {distant_threshold}): {expected_distant}")
    
    # Check if DISTANT connections exceed the threshold
    print(f"\n=== Checking DISTANT connections ===")
    distant_connections = cached_data.get('distant', [])
    distances_beyond_threshold = 0
    
    for conn in distant_connections:
        # Handle both object and dict formats
        if hasattr(conn, 'euclidean_distance'):
            dist = conn.euclidean_distance
        else:
            dist = conn['euclidean_distance']
            
        if dist > distant_threshold:
            distances_beyond_threshold += 1
    
    print(f"DISTANT connections beyond distant_threshold ({distant_threshold}): {distances_beyond_threshold}")
    
    if distances_beyond_threshold > 0:
        print("\n❌ BUG CONFIRMED: DISTANT connections include neighbors beyond distant_threshold!")
        print("   This happens because adaptive_radius is used to find all neighbors,")
        print("   but the classification doesn't filter out those beyond distant_threshold.")
    else:
        print("\n✅ All DISTANT connections are within distant_threshold.")
    
    # The fix would be in _precompute_cell_connections to filter out neighbors
    # beyond distant_threshold completely instead of classifying them as DISTANT

if __name__ == "__main__":
    test_neighbor_distribution()