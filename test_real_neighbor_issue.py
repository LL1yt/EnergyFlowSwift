#!/usr/bin/env python3
"""
Test to reproduce the real neighbor distribution issue from training logs.
In logs we see: LOCAL: 6, FUNCTIONAL: 12, DISTANT: 197 (!)
"""

import torch
import numpy as np
from new_rebuild.config import create_experiment_config, set_project_config
from new_rebuild.core.moe import ConnectionCacheManager, UnifiedConnectionClassifier
from new_rebuild.core.lattice.spatial_optimization import UnifiedSpatialOptimizer

def test_real_neighbor_distribution():
    """Test that reproduces the actual issue from logs"""
    
    # Create config matching the real training scenario
    config = create_experiment_config()  # This uses 15x15x15 lattice
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
    
    print(f"=== Configuration (Matching Real Training) ===")
    print(f"Lattice dimensions: {lattice_dimensions}")
    print(f"Adaptive radius: {adaptive_radius}")
    print(f"Ratios: local={local_ratio}, functional={functional_ratio}, distant={distant_ratio}")
    print(f"Thresholds: local<{local_threshold}, functional<={functional_threshold}, distant<={distant_threshold}")
    
    # Test both with cache and with spatial optimizer
    print("\n=== Testing with ConnectionCacheManager ===")
    test_with_cache(config, lattice_dimensions, adaptive_radius, local_threshold, functional_threshold, distant_threshold)
    
    print("\n=== Testing with UnifiedSpatialOptimizer ===")
    test_with_spatial_optimizer(config, lattice_dimensions, adaptive_radius, local_threshold, functional_threshold, distant_threshold)

def test_with_cache(config, lattice_dimensions, adaptive_radius, local_threshold, functional_threshold, distant_threshold):
    """Test using ConnectionCacheManager directly"""
    # Create cache manager
    cache_manager = ConnectionCacheManager(lattice_dimensions)
    
    # Pre-compute connections
    print("Pre-computing connections...")
    cache_manager.precompute_all_connections(force_rebuild=True)
    
    # Check cell 677 (from logs)
    cell_idx = 677
    print(f"\nAnalyzing cell {cell_idx}:")
    
    # Get all neighbors
    all_neighbors = cache_manager._all_neighbors_cache.get(cell_idx, [])
    print(f"Total neighbors in adaptive_radius ({adaptive_radius}): {len(all_neighbors)}")
    
    # Get cached connections
    cached_data = cache_manager.cache.get(cell_idx, {})
    local_count = len(cached_data.get('local', []))
    functional_count = len(cached_data.get('functional_candidates', []))
    distant_count = len(cached_data.get('distant', []))
    
    print(f"Cached distribution:")
    print(f"  LOCAL: {local_count}")
    print(f"  FUNCTIONAL: {functional_count}")
    print(f"  DISTANT: {distant_count}")
    print(f"  TOTAL: {local_count + functional_count + distant_count}")
    
    # Analyze distances of DISTANT connections
    analyze_distant_connections(cache_manager, cell_idx, cached_data, distant_threshold)

def test_with_spatial_optimizer(config, lattice_dimensions, adaptive_radius, local_threshold, functional_threshold, distant_threshold):
    """Test using UnifiedSpatialOptimizer (what actually happens in training)"""
    # Create spatial optimizer
    spatial_optimizer = UnifiedSpatialOptimizer(
        lattice_dimensions=lattice_dimensions,
        config=config
    )
    
    # Find neighbors for cell 677
    cell_idx = 677
    neighbors = spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
    
    print(f"\nSpatial optimizer found {len(neighbors)} neighbors for cell {cell_idx}")
    
    # Check if spatial optimizer is finding too many neighbors
    if len(neighbors) > 50:  # Expected ~33 based on config
        print(f"⚠️ WARNING: Spatial optimizer found way more neighbors than expected!")
        print(f"   This might be the source of the problem!")
        
        # Analyze the distances
        from new_rebuild.core.moe.distance_calculator import DistanceCalculator
        dist_calc = DistanceCalculator(lattice_dimensions)
        
        distances = []
        for neighbor_idx in neighbors:
            dist = dist_calc.euclidean_distance(cell_idx, neighbor_idx)
            distances.append((neighbor_idx, dist))
        
        distances.sort(key=lambda x: x[1])
        
        print(f"\nDistance distribution of neighbors:")
        within_local = sum(1 for _, d in distances if d < local_threshold)
        within_functional = sum(1 for _, d in distances if local_threshold <= d <= functional_threshold)
        within_distant = sum(1 for _, d in distances if functional_threshold < d <= distant_threshold)
        beyond_distant = sum(1 for _, d in distances if d > distant_threshold)
        
        print(f"  Within LOCAL threshold ({local_threshold}): {within_local}")
        print(f"  Within FUNCTIONAL threshold ({functional_threshold}): {within_functional}")
        print(f"  Within DISTANT threshold ({distant_threshold}): {within_distant}")
        print(f"  BEYOND distant threshold: {beyond_distant}")
        
        if beyond_distant > 0:
            print(f"\n❌ BUG FOUND: {beyond_distant} neighbors are beyond distant_threshold!")
            print("   The spatial optimizer might be using adaptive_radius instead of distant_threshold!")

def analyze_distant_connections(cache_manager, cell_idx, cached_data, distant_threshold):
    """Analyze the actual distances of DISTANT connections"""
    distant_connections = cached_data.get('distant', [])
    
    if len(distant_connections) > 50:  # Suspiciously high number
        print(f"\n⚠️ Analyzing {len(distant_connections)} DISTANT connections:")
        
        distances = []
        for conn in distant_connections:
            if hasattr(conn, 'euclidean_distance'):
                dist = conn.euclidean_distance
            else:
                dist = conn['euclidean_distance']
            distances.append(dist)
        
        distances.sort()
        
        print(f"  Min distance: {distances[0]:.2f}")
        print(f"  Max distance: {distances[-1]:.2f}")
        print(f"  Distant threshold: {distant_threshold:.2f}")
        
        beyond_threshold = sum(1 for d in distances if d > distant_threshold)
        if beyond_threshold > 0:
            print(f"  ❌ {beyond_threshold} connections are BEYOND distant_threshold!")

if __name__ == "__main__":
    test_real_neighbor_distribution()