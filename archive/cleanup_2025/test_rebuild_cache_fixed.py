#!/usr/bin/env python3
"""Test cache rebuild with fixed parameters"""

import torch

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()

# Увеличим adaptive_radius_ratio чтобы distant_threshold был меньше adaptive_radius
config.lattice.adaptive_radius_ratio = 0.2  # Это даст нам radius = 2.0 для 10x10x10

set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.moe import create_moe_connection_processor

logger = get_logger(__name__)

def main():
    """Test cache rebuild with fixed parameters"""
    logger.info("🔨 Testing cache rebuild with fixed parameters...")
    
    # Print configuration
    adaptive_radius = config.calculate_adaptive_radius()
    local_threshold = adaptive_radius * config.lattice.local_distance_ratio
    functional_threshold = adaptive_radius * config.lattice.functional_distance_ratio
    distant_threshold = adaptive_radius * config.lattice.distant_distance_ratio
    
    logger.info(f"📊 Configuration:")
    logger.info(f"   Lattice dimensions: {config.lattice.dimensions}")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   Local threshold: {local_threshold} (ratio: {config.lattice.local_distance_ratio})")
    logger.info(f"   Functional threshold: {functional_threshold} (ratio: {config.lattice.functional_distance_ratio})")
    logger.info(f"   Distant threshold: {distant_threshold} (ratio: {config.lattice.distant_distance_ratio})")
    
    logger.info(f"\n📏 Ranges:")
    logger.info(f"   LOCAL: distance ≤ {local_threshold}")
    logger.info(f"   FUNCTIONAL: {local_threshold} < distance < {distant_threshold}")
    logger.info(f"   DISTANT: {distant_threshold} ≤ distance ≤ {adaptive_radius}")
    logger.info(f"   NOT NEIGHBOR: distance > {adaptive_radius}")
    
    # Create MoE processor which will trigger cache creation
    logger.info("\n🏗️ Creating MoE processor (this will rebuild cache)...")
    
    device = torch.device(config.current_device)
    moe_processor = create_moe_connection_processor(
        dimensions=config.lattice.dimensions,
        state_size=config.model.state_size,
        device=device,
    )
    
    logger.info("✅ MoE processor created successfully!")
    
    # Check cache stats
    if hasattr(moe_processor, 'connection_classifier'):
        classifier = moe_processor.connection_classifier
        if hasattr(classifier, 'cache_manager') and classifier.cache_manager:
            cache_stats = classifier.cache_manager.get_cache_stats()
            logger.info(f"\n📦 Cache statistics:")
            logger.info(f"   Status: {cache_stats['status']}")
            logger.info(f"   Cached cells: {cache_stats['cached_cells']}")
            logger.info(f"   Total connections: {cache_stats['total_connections']}")
            logger.info(f"   Local connections: {cache_stats['local_connections']}")
            logger.info(f"   Functional candidates: {cache_stats['functional_candidates']}")
            logger.info(f"   Distant connections: {cache_stats['distant_connections']}")
            logger.info(f"   Cache size: {cache_stats['cache_size_mb']:.1f} MB")

if __name__ == "__main__":
    main()