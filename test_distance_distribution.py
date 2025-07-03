#!/usr/bin/env python3
"""Analyze distance distribution in the lattice"""

import torch
import numpy as np
from collections import Counter

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.moe.distance_calculator import DistanceCalculator

logger = get_logger(__name__)

def main():
    """Analyze distance distribution"""
    logger.info("📊 Analyzing distance distribution in 10x10x10 lattice...")
    
    dimensions = (10, 10, 10)
    calc = DistanceCalculator(dimensions)
    
    # Get thresholds
    adaptive_radius = config.calculate_adaptive_radius()
    local_threshold = adaptive_radius * config.lattice.local_distance_ratio
    functional_threshold = adaptive_radius * config.lattice.functional_distance_ratio
    distant_threshold = adaptive_radius * config.lattice.distant_distance_ratio
    
    logger.info(f"\n📏 Thresholds:")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   LOCAL ≤ {local_threshold}")
    logger.info(f"   FUNCTIONAL: {local_threshold} < dist < {distant_threshold}")
    logger.info(f"   DISTANT ≥ {distant_threshold}")
    
    # Analyze distances from center cell (position 5,5,5)
    center_idx = 5 + 5 * 10 + 5 * 100  # = 555
    
    distances = []
    distance_counts = Counter()
    
    # Check all possible neighbors within adaptive radius
    for idx in range(1000):
        if idx == center_idx:
            continue
            
        dist = calc.euclidean_distance(center_idx, idx)
        if dist <= adaptive_radius:
            distances.append(dist)
            
            # Classify
            if dist <= local_threshold:
                distance_counts['LOCAL'] += 1
            elif dist >= distant_threshold:
                distance_counts['DISTANT'] += 1
            else:
                distance_counts['FUNCTIONAL'] += 1
    
    logger.info(f"\n📊 Distance distribution for cell {center_idx}:")
    logger.info(f"   Total neighbors in radius: {len(distances)}")
    logger.info(f"   LOCAL: {distance_counts['LOCAL']}")
    logger.info(f"   FUNCTIONAL: {distance_counts['FUNCTIONAL']}")
    logger.info(f"   DISTANT: {distance_counts['DISTANT']}")
    
    # Show some example distances
    if distances:
        distances.sort()
        logger.info(f"\n📏 Example distances:")
        logger.info(f"   Min: {distances[0]:.3f}")
        logger.info(f"   Max: {distances[-1]:.3f}")
        logger.info(f"   First 10: {[f'{d:.3f}' for d in distances[:10]]}")
        
        # Show distances in functional range
        functional_distances = [d for d in distances if local_threshold < d < distant_threshold]
        logger.info(f"\n🔍 Distances in FUNCTIONAL range ({local_threshold:.3f} - {distant_threshold:.3f}):")
        if functional_distances:
            logger.info(f"   Count: {len(functional_distances)}")
            logger.info(f"   Examples: {[f'{d:.3f}' for d in functional_distances[:10]]}")
        else:
            logger.info("   ❌ No distances in functional range!")

if __name__ == "__main__":
    main()