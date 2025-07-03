#!/usr/bin/env python3
"""Analyze what's actually in the cache"""

import torch
import pickle
import os

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.moe.connection_types import ConnectionCategory

logger = get_logger(__name__)

def main():
    """Analyze cache contents"""
    logger.info("🔍 Analyzing cache contents...")
    
    # Load cache directly
    cache_key = f"{config.lattice.dimensions}_{config.calculate_adaptive_radius():.2f}"
    cache_file = f"cache/connection_cache_0c9d4528185d935b1628be5bf2e36600.pkl"
    
    if not os.path.exists(cache_file):
        logger.error(f"Cache file not found: {cache_file}")
        return
        
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)
    
    logger.info(f"\n📦 Cache file: {cache_file}")
    logger.info(f"   Cache keys: {list(cache_data.keys())[:10]}")
    
    # Check the actual cache structure
    actual_cache = cache_data.get("cache", {})
    logger.info(f"\n📊 Cache structure:")
    logger.info(f"   Total cells in cache: {len(actual_cache)}")
    
    # Analyze a few cells
    for cell_idx in [444, 555, 677]:
        if cell_idx not in actual_cache:
            logger.warning(f"   Cell {cell_idx} not in cache")
            continue
            
        cell_data = actual_cache[cell_idx]
        logger.info(f"\n🔍 Cell {cell_idx}:")
        logger.info(f"   Data type: {type(cell_data)}")
        logger.info(f"   Keys: {list(cell_data.keys()) if isinstance(cell_data, dict) else 'Not a dict'}")
        
        if isinstance(cell_data, dict):
            # Check for different key formats
            if ConnectionCategory.LOCAL in cell_data:
                local_key = ConnectionCategory.LOCAL
                functional_key = ConnectionCategory.FUNCTIONAL
                distant_key = ConnectionCategory.DISTANT
                logger.info("   Using enum keys")
            elif "local" in cell_data:
                local_key = "local"
                functional_key = "functional_candidates"
                distant_key = "distant"
                logger.info("   Using string keys")
            else:
                logger.error("   Unknown key format!")
                continue
                
            local_conns = cell_data.get(local_key, [])
            functional_conns = cell_data.get(functional_key, [])
            distant_conns = cell_data.get(distant_key, [])
            
            logger.info(f"   LOCAL: {len(local_conns)} connections")
            logger.info(f"   FUNCTIONAL: {len(functional_conns)} connections")
            logger.info(f"   DISTANT: {len(distant_conns)} connections")
            
            # Show some details
            if local_conns and hasattr(local_conns[0], 'euclidean_distance'):
                logger.info(f"   First LOCAL distance: {local_conns[0].euclidean_distance:.3f}")
            if functional_conns and hasattr(functional_conns[0], 'euclidean_distance'):
                logger.info(f"   First FUNCTIONAL distance: {functional_conns[0].euclidean_distance:.3f}")
            if distant_conns and hasattr(distant_conns[0], 'euclidean_distance'):
                logger.info(f"   First DISTANT distance: {distant_conns[0].euclidean_distance:.3f}")

if __name__ == "__main__":
    main()