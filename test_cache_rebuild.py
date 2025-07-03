#!/usr/bin/env python3
"""Test cache rebuild with correct structure"""

import torch

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.moe import create_moe_connection_processor

logger = get_logger(__name__)

def main():
    """Test cache rebuild"""
    logger.info("🔨 Testing cache rebuild...")
    
    # Print configuration
    logger.info(f"📊 Lattice dimensions: {config.lattice.dimensions}")
    logger.info(f"📊 Adaptive radius: {config.calculate_adaptive_radius()}")
    logger.info(f"📊 Local threshold: {config.lattice.local_distance_threshold}")
    logger.info(f"📊 Functional threshold: {config.lattice.functional_distance_threshold}")
    logger.info(f"📊 Distant threshold: {config.lattice.distant_distance_threshold}")
    
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