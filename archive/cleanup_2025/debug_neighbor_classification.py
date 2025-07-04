#!/usr/bin/env python3
"""Debug script to investigate neighbor classification issues"""

import torch
import numpy as np

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.lattice import Lattice3D

logger = get_logger(__name__)

def main():
    """Debug neighbor finding and classification"""
    logger.info("🔍 Debugging neighbor classification...")
    
    # Get configuration details
    logger.info(f"📊 Lattice dimensions: {config.lattice.dimensions}")
    logger.info(f"📊 Total cells: {config.lattice.total_cells}")
    logger.info(f"📊 Adaptive radius enabled: {config.lattice.adaptive_radius_enabled}")
    logger.info(f"📊 Adaptive radius ratio: {config.lattice.adaptive_radius_ratio}")
    
    # Calculate radius
    adaptive_radius = config.calculate_adaptive_radius()
    logger.info(f"📊 Calculated adaptive radius: {adaptive_radius}")
    
    # Calculate expected neighbors
    expected_neighbors = config.estimate_neighbors_in_radius(adaptive_radius)
    logger.info(f"📊 Expected neighbors in radius: {expected_neighbors}")
    
    # Check distance thresholds
    logger.info(f"📊 Local distance threshold: {config.lattice.local_distance_threshold}")
    logger.info(f"📊 Functional distance threshold: {config.lattice.functional_distance_threshold}")
    logger.info(f"📊 Distant distance threshold: {config.lattice.distant_distance_threshold}")
    
    # Create lattice
    logger.info("\n🏗️ Creating lattice...")
    lattice = Lattice3D()
    
    # Initialize with random states
    initial_states = torch.randn(config.lattice.total_cells, config.model.state_size)
    initial_states = initial_states.to(lattice.device)
    
    # Test specific cell
    test_cell_idx = 677
    logger.info(f"\n🎯 Testing cell {test_cell_idx}...")
    
    # Get spatial optimizer from lattice
    spatial_optimizer = lattice.spatial_optimizer
    
    # Find neighbors
    neighbors = spatial_optimizer.find_neighbors_by_radius_safe(test_cell_idx)
    logger.info(f"📊 Found {len(neighbors)} neighbors")
    
    if neighbors:
        logger.info(f"📊 First 10 neighbors: {neighbors[:10]}")
        
        # Calculate distances for first few neighbors
        from new_rebuild.core.lattice.position import Position3D
        pos_helper = Position3D(config.lattice.dimensions)
        cell_coords = pos_helper.to_3d_coordinates(test_cell_idx)
        
        logger.info(f"\n📏 Distance analysis for cell {test_cell_idx} at {cell_coords}:")
        for i, neighbor_idx in enumerate(neighbors[:10]):
            neighbor_coords = pos_helper.to_3d_coordinates(neighbor_idx)
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(cell_coords, neighbor_coords)))
            logger.info(f"   Neighbor {neighbor_idx} at {neighbor_coords}: distance = {distance:.2f}")
    
    # Test MoE processor classification
    logger.info("\n🔬 Testing MoE processor...")
    
    # Get MoE processor from lattice
    moe_processor = lattice.moe_processor
    
    # Test forward pass for single cell
    current_state = initial_states[test_cell_idx].unsqueeze(0)
    neighbor_states = initial_states[neighbors] if neighbors else torch.empty(0, config.model.state_size, device=lattice.device)
    
    logger.info(f"📊 Current state shape: {current_state.shape}")
    logger.info(f"📊 Neighbor states shape: {neighbor_states.shape}")
    
    # Call MoE forward
    result = moe_processor(
        current_state=current_state,
        neighbor_states=neighbor_states,
        cell_idx=test_cell_idx,
        neighbor_indices=neighbors,
        spatial_optimizer=spatial_optimizer,
        full_lattice_states=initial_states
    )
    
    logger.info(f"\n✅ MoE forward completed")
    logger.info(f"📊 Result keys: {result.keys() if isinstance(result, dict) else 'Tensor'}")
    
    # Check connection classifier cache
    if hasattr(moe_processor, 'connection_classifier'):
        classifier = moe_processor.connection_classifier
        logger.info(f"\n📦 Connection classifier cache enabled: {classifier.cache_enabled}")
        
        # Check if classifications are cached
        if hasattr(classifier, 'connection_cache') and classifier.connection_cache:
            cache = classifier.connection_cache
            logger.info(f"📦 Cache stats: {cache.get_stats()}")

if __name__ == "__main__":
    main()