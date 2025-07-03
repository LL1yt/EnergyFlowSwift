#!/usr/bin/env python3
"""
Diagnostic script to trace the neighbor distribution issue.
This will help us understand why we're getting 197 DISTANT connections instead of ~14.
"""

import torch
import numpy as np
from new_rebuild.config import create_experiment_config, set_project_config, get_project_config
from new_rebuild.core.moe import ConnectionCacheManager, UnifiedConnectionClassifier, MoEConnectionProcessor
from new_rebuild.core.lattice.spatial_optimization import UnifiedSpatialOptimizer
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.moe.distance_calculator import DistanceCalculator
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def diagnose_adaptive_radius():
    """Check if adaptive_radius is consistent across components"""
    print("=== STEP 1: Checking adaptive_radius consistency ===\n")
    
    # Create config
    config = create_experiment_config()  # 15x15x15 lattice
    set_project_config(config)
    
    # Get adaptive radius from config
    config_adaptive_radius = config.calculate_adaptive_radius()
    print(f"1. Config adaptive_radius: {config_adaptive_radius}")
    print(f"   Lattice dimensions: {config.lattice.dimensions}")
    print(f"   Adaptive radius ratio: {config.lattice.adaptive_radius_ratio}")
    print(f"   Formula: max(dimensions) * ratio = {max(config.lattice.dimensions)} * {config.lattice.adaptive_radius_ratio} = {config_adaptive_radius}")
    
    # Check thresholds
    print(f"\n2. Distance thresholds from config:")
    print(f"   local_distance_ratio: {config.lattice.local_distance_ratio}")
    print(f"   functional_distance_ratio: {config.lattice.functional_distance_ratio}")
    print(f"   distant_distance_ratio: {config.lattice.distant_distance_ratio}")
    print(f"   Calculated thresholds:")
    print(f"   - LOCAL: < {config_adaptive_radius * config.lattice.local_distance_ratio}")
    print(f"   - FUNCTIONAL: <= {config_adaptive_radius * config.lattice.functional_distance_ratio}")
    print(f"   - DISTANT: <= {config_adaptive_radius * config.lattice.distant_distance_ratio}")
    
    # Create components and check their adaptive_radius
    print(f"\n3. Checking adaptive_radius in components:")
    
    # ConnectionCacheManager
    cache_manager = ConnectionCacheManager(config.lattice.dimensions)
    print(f"   ConnectionCacheManager adaptive_radius: {cache_manager.adaptive_radius}")
    print(f"   ConnectionCacheManager thresholds:")
    print(f"   - local_threshold: {cache_manager.local_threshold}")
    print(f"   - functional_threshold: {cache_manager.functional_threshold}")
    print(f"   - distant_threshold: {cache_manager.distant_threshold}")
    
    # UnifiedConnectionClassifier
    classifier = UnifiedConnectionClassifier(config.lattice.dimensions)
    print(f"\n   UnifiedConnectionClassifier adaptive_radius: {classifier.adaptive_radius}")
    print(f"   UnifiedConnectionClassifier thresholds:")
    print(f"   - local_distance_threshold: {classifier.local_distance_threshold}")
    print(f"   - functional_distance_threshold: {classifier.functional_distance_threshold}")
    print(f"   - distant_distance_threshold: {classifier.distant_distance_threshold}")
    
    # Check if values match
    print(f"\n4. Consistency check:")
    if config_adaptive_radius == cache_manager.adaptive_radius == classifier.adaptive_radius:
        print("   ✅ Adaptive radius is consistent across components")
    else:
        print("   ❌ INCONSISTENCY DETECTED in adaptive_radius values!")
    
    return config, cache_manager, classifier

def diagnose_spatial_optimizer(config):
    """Check how many neighbors spatial optimizer finds"""
    print("\n\n=== STEP 2: Checking spatial optimizer neighbor search ===\n")
    
    # Create spatial optimizer with proper config
    # UnifiedSpatialOptimizer expects OptimizationConfig, not SimpleProjectConfig
    from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import OptimizationConfig
    
    opt_config = OptimizationConfig(
        enable_morton_encoding=config.lattice.enable_morton_encoding,
        enable_adaptive_chunking=config.adaptive_chunker is not None,
        max_memory_gb=8.0,  # Default value
        target_performance_ms=config.lattice.target_performance_ms
    )
    
    spatial_optimizer = UnifiedSpatialOptimizer(
        dimensions=config.lattice.dimensions,
        config=opt_config
    )
    
    # Test cell 677 (from logs)
    test_cell = 677
    print(f"Testing cell {test_cell}:")
    
    # Find neighbors
    neighbors = spatial_optimizer.find_neighbors_by_radius_safe(test_cell)
    print(f"Spatial optimizer found {len(neighbors)} neighbors")
    
    # Get the adaptive radius used
    adaptive_radius = config.calculate_adaptive_radius()
    print(f"Using adaptive_radius: {adaptive_radius}")
    
    # Calculate distances to all neighbors
    dist_calc = DistanceCalculator(config.lattice.dimensions)
    distances = []
    for neighbor_idx in neighbors:
        dist = dist_calc.euclidean_distance(test_cell, neighbor_idx)
        distances.append((neighbor_idx, dist))
    
    distances.sort(key=lambda x: x[1])
    
    # Analyze distance distribution
    print(f"\nDistance distribution of {len(neighbors)} neighbors:")
    local_threshold = adaptive_radius * config.lattice.local_distance_ratio
    functional_threshold = adaptive_radius * config.lattice.functional_distance_ratio
    distant_threshold = adaptive_radius * config.lattice.distant_distance_ratio
    
    within_local = sum(1 for _, d in distances if d < local_threshold)
    within_functional = sum(1 for _, d in distances if local_threshold <= d <= functional_threshold)
    within_distant = sum(1 for _, d in distances if functional_threshold < d <= distant_threshold)
    beyond_distant = sum(1 for _, d in distances if d > distant_threshold)
    
    print(f"  LOCAL (d < {local_threshold:.2f}): {within_local}")
    print(f"  FUNCTIONAL ({local_threshold:.2f} <= d <= {functional_threshold:.2f}): {within_functional}")
    print(f"  DISTANT ({functional_threshold:.2f} < d <= {distant_threshold:.2f}): {within_distant}")
    print(f"  BEYOND threshold (d > {distant_threshold:.2f}): {beyond_distant}")
    
    if beyond_distant > 0:
        print(f"\n❌ ERROR: Found {beyond_distant} neighbors beyond distant_threshold!")
        print("  This explains why we get too many DISTANT connections!")
        print("\n  Examples of distances beyond threshold:")
        for idx, (neighbor, dist) in enumerate(distances):
            if dist > distant_threshold:
                print(f"    Neighbor {neighbor}: distance = {dist:.3f} (exceeds {distant_threshold:.3f} by {dist - distant_threshold:.3f})")
                if idx >= 5:  # Show only first 5
                    print(f"    ... and {beyond_distant - 5} more")
                    break
    
    return spatial_optimizer, neighbors, distances

def diagnose_cache_classification(config, cache_manager, test_cell, neighbors):
    """Check how cache classifies the neighbors"""
    print("\n\n=== STEP 3: Checking cache classification ===\n")
    
    # Force rebuild cache
    print("Rebuilding cache...")
    cache_manager.precompute_all_connections(force_rebuild=True)
    
    # Get cached data for test cell
    cached_data = cache_manager.cache.get(test_cell, {})
    
    if not cached_data:
        print(f"❌ No cached data found for cell {test_cell}")
        return
    
    local_count = len(cached_data.get('local', []))
    functional_count = len(cached_data.get('functional_candidates', []))
    distant_count = len(cached_data.get('distant', []))
    
    print(f"Cache classification for cell {test_cell}:")
    print(f"  LOCAL: {local_count}")
    print(f"  FUNCTIONAL candidates: {functional_count}")
    print(f"  DISTANT: {distant_count}")
    print(f"  TOTAL: {local_count + functional_count + distant_count}")
    
    # Check if cache has the same neighbors as spatial optimizer
    cache_neighbors = set()
    for conn_list in cached_data.values():
        for conn in conn_list:
            if hasattr(conn, 'target_idx'):
                cache_neighbors.add(conn.target_idx)
            else:
                cache_neighbors.add(conn['target_idx'])
    
    spatial_neighbors = set(neighbors)
    
    print(f"\nNeighbor count comparison:")
    print(f"  Spatial optimizer found: {len(spatial_neighbors)} neighbors")
    print(f"  Cache has: {len(cache_neighbors)} neighbors")
    
    if cache_neighbors != spatial_neighbors:
        print(f"  ❌ MISMATCH! Difference: {len(cache_neighbors.symmetric_difference(spatial_neighbors))} neighbors")
        
        only_in_cache = cache_neighbors - spatial_neighbors
        only_in_spatial = spatial_neighbors - cache_neighbors
        
        if only_in_cache:
            print(f"  Only in cache: {list(only_in_cache)[:5]}...")
        if only_in_spatial:
            print(f"  Only in spatial: {list(only_in_spatial)[:5]}...")
    else:
        print(f"  ✅ Cache and spatial optimizer have the same neighbors")

def diagnose_moe_processor(config):
    """Test the full MoE processor flow"""
    print("\n\n=== STEP 4: Testing MoE processor flow ===\n")
    
    # Create a simple lattice
    lattice = Lattice3D(
        dimensions=config.lattice.dimensions,
        state_size=config.model.state_size,
        config=config
    )
    
    # Initialize with random states
    num_cells = np.prod(config.lattice.dimensions)
    states = torch.randn(1, num_cells, config.model.state_size, device='cuda')
    
    # Test forward pass for cell 677
    test_cell = 677
    print(f"Running forward pass for cell {test_cell}...")
    
    # Get the MoE processor
    moe_processor = lattice.moe_processor
    
    # Find neighbors using spatial optimizer
    neighbors = lattice.spatial_optimizer.find_neighbors_by_radius_safe(test_cell)
    print(f"Spatial optimizer found {len(neighbors)} neighbors")
    
    # Get neighbor states
    neighbor_states = states[0, neighbors, :]
    current_state = states[0, test_cell, :]
    
    # Run forward pass
    result = moe_processor.forward(
        current_state=current_state,
        neighbor_states=neighbor_states,
        cell_idx=test_cell,
        neighbor_indices=neighbors,
        spatial_optimizer=lattice.spatial_optimizer,
        full_lattice_states=states
    )
    
    print(f"\nForward pass completed successfully")
    print(f"Result keys: {result.keys()}")

def main():
    """Run full diagnosis"""
    print("=" * 80)
    print("DIAGNOSING NEIGHBOR DISTRIBUTION ISSUE")
    print("=" * 80)
    
    # Step 1: Check adaptive radius consistency
    config, cache_manager, classifier = diagnose_adaptive_radius()
    
    # Step 2: Check spatial optimizer
    spatial_optimizer, neighbors, distances = diagnose_spatial_optimizer(config)
    
    # Step 3: Check cache classification
    test_cell = 677
    diagnose_cache_classification(config, cache_manager, test_cell, neighbors)
    
    # Step 4: Test MoE processor
    try:
        diagnose_moe_processor(config)
    except Exception as e:
        print(f"\n❌ Error in MoE processor: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()