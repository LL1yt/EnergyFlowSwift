#!/usr/bin/env python3
"""Analyze distance distribution with radius 2.0"""

import numpy as np
from collections import Counter

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
config.lattice.adaptive_radius_ratio = 0.2  # radius = 2.0
set_project_config(config)

from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Analyze distance distribution"""
    logger.info("📊 Analyzing distance distribution in 10x10x10 lattice with radius 2.0...")
    
    # All possible distances in 3D grid
    distances = []
    for dx in range(3):  # 0, 1, 2
        for dy in range(3):
            for dz in range(3):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist <= 2.0:
                    distances.append(dist)
    
    # Count unique distances
    unique_distances = sorted(set(distances))
    
    adaptive_radius = config.calculate_adaptive_radius()
    local_threshold = adaptive_radius * config.lattice.local_distance_ratio
    distant_threshold = adaptive_radius * config.lattice.distant_distance_ratio
    
    logger.info(f"\n📏 Configuration:")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   LOCAL ≤ {local_threshold}")
    logger.info(f"   FUNCTIONAL: {local_threshold} < dist < {distant_threshold}")
    logger.info(f"   DISTANT: {distant_threshold} ≤ dist ≤ {adaptive_radius}")
    
    logger.info(f"\n📊 Unique distances ≤ {adaptive_radius}:")
    for dist in unique_distances:
        count = distances.count(dist)
        if dist <= local_threshold:
            category = "LOCAL"
        elif dist >= distant_threshold:
            category = "DISTANT"
        else:
            category = "FUNCTIONAL"
        logger.info(f"   {dist:.3f}: {count} cells ({category})")
    
    # Count by category
    local_count = sum(1 for d in distances if d <= local_threshold)
    functional_count = sum(1 for d in distances if local_threshold < d < distant_threshold)
    distant_count = sum(1 for d in distances if distant_threshold <= d <= adaptive_radius)
    
    logger.info(f"\n📈 Total counts (for center cell):")
    logger.info(f"   LOCAL: {local_count}")
    logger.info(f"   FUNCTIONAL: {functional_count}")
    logger.info(f"   DISTANT: {distant_count}")
    logger.info(f"   TOTAL: {len(distances)}")
    
    # Suggest better ratios
    logger.info(f"\n💡 Suggestion: To get more FUNCTIONAL connections, try:")
    logger.info(f"   local_distance_ratio: 0.5 (threshold = {adaptive_radius * 0.5})")
    logger.info(f"   distant_distance_ratio: 0.95 (threshold = {adaptive_radius * 0.95})")

if __name__ == "__main__":
    main()