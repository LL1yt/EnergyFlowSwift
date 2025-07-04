#!/usr/bin/env python3
"""Check cache parameters"""

import pickle
import os

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Check cache parameters"""
    logger.info("🔍 Checking cache parameters...")
    
    # Current config
    logger.info(f"\n📊 Current configuration:")
    logger.info(f"   Adaptive radius: {config.calculate_adaptive_radius()}")
    logger.info(f"   Local ratio: {config.lattice.local_distance_ratio}")
    logger.info(f"   Functional ratio: {config.lattice.functional_distance_ratio}")
    logger.info(f"   Distant ratio: {config.lattice.distant_distance_ratio}")
    
    current_local = config.calculate_adaptive_radius() * config.lattice.local_distance_ratio
    current_functional = config.calculate_adaptive_radius() * config.lattice.functional_distance_ratio
    current_distant = config.calculate_adaptive_radius() * config.lattice.distant_distance_ratio
    
    logger.info(f"\n📏 Current thresholds:")
    logger.info(f"   LOCAL ≤ {current_local}")
    logger.info(f"   FUNCTIONAL ≤ {current_functional}")
    logger.info(f"   DISTANT ≤ {current_distant}")
    
    # Load cache
    cache_file = "cache/connection_cache_0c9d4528185d935b1628be5bf2e36600.pkl"
    
    if not os.path.exists(cache_file):
        logger.error(f"Cache file not found: {cache_file}")
        return
        
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
    
    # Cache parameters
    logger.info(f"\n📦 Cache parameters:")
    logger.info(f"   Adaptive radius: {cache_data.get('adaptive_radius', 'N/A')}")
    logger.info(f"   Local threshold: {cache_data.get('local_threshold', 'N/A')}")
    logger.info(f"   Functional threshold: {cache_data.get('functional_threshold', 'N/A')}")
    logger.info(f"   Distant threshold: {cache_data.get('distant_threshold', 'N/A')}")
    logger.info(f"   Timestamp: {cache_data.get('timestamp', 'N/A')}")
    
    # Check if they match
    logger.info(f"\n⚠️ Comparison:")
    if cache_data.get('adaptive_radius') != config.calculate_adaptive_radius():
        logger.warning(f"   ❌ Adaptive radius mismatch!")
    if cache_data.get('local_threshold') != current_local:
        logger.warning(f"   ❌ Local threshold mismatch!")
    if cache_data.get('functional_threshold') != current_functional:
        logger.warning(f"   ❌ Functional threshold mismatch!")
    if cache_data.get('distant_threshold') != current_distant:
        logger.warning(f"   ❌ Distant threshold mismatch!")
        
    # Delete cache to force rebuild
    logger.info(f"\n🗑️ Deleting old cache to force rebuild...")
    os.remove(cache_file)
    logger.info("✅ Cache deleted. Next run will rebuild with correct parameters.")

if __name__ == "__main__":
    main()