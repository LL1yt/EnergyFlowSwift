#!/usr/bin/env python3
"""Test distance thresholds to understand why connections are not being classified"""

import torch
import numpy as np

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Test distance thresholds"""
    logger.info("🔍 Testing distance thresholds...")
    
    # Get configuration
    dimensions = config.lattice.dimensions
    adaptive_radius = config.calculate_adaptive_radius()
    
    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Lattice dimensions: {dimensions}")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   Local threshold: {config.lattice.local_distance_threshold}")
    logger.info(f"   Functional threshold: {config.lattice.functional_distance_threshold}")
    logger.info(f"   Distant threshold: {config.lattice.distant_distance_threshold}")
    
    # Test some example distances
    logger.info(f"\n📏 Distance examples in a {dimensions} lattice:")
    
    # Adjacent cells (distance = 1)
    logger.info(f"   Adjacent cells (1 unit away): distance = 1.0")
    logger.info(f"      Classification: {'LOCAL' if 1.0 <= config.lattice.local_distance_threshold else 'NOT LOCAL'}")
    
    # Diagonal neighbors (distance = sqrt(3) ≈ 1.732)
    diagonal_dist = np.sqrt(3)
    logger.info(f"   Diagonal neighbors: distance = {diagonal_dist:.3f}")
    logger.info(f"      Classification: {'LOCAL' if diagonal_dist <= config.lattice.local_distance_threshold else 'NOT LOCAL'}")
    
    # Test various distances
    test_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    logger.info(f"\n📊 Classification for various distances:")
    for dist in test_distances:
        if dist <= config.lattice.local_distance_threshold:
            classification = "LOCAL"
        elif dist >= config.lattice.distant_distance_threshold:
            classification = "DISTANT"
        elif dist <= config.lattice.functional_distance_threshold:
            classification = "FUNCTIONAL (candidate)"
        else:
            classification = "FUNCTIONAL (after similarity check)"
        logger.info(f"   Distance {dist}: {classification}")
    
    # Check if thresholds make sense
    logger.info(f"\n⚠️ Threshold analysis:")
    if config.lattice.local_distance_threshold < 1.0:
        logger.warning(f"   Local threshold ({config.lattice.local_distance_threshold}) is less than 1.0 - no adjacent cells will be LOCAL!")
    if config.lattice.distant_distance_threshold <= 1.0:
        logger.warning(f"   Distant threshold ({config.lattice.distant_distance_threshold}) is <= 1.0 - all non-adjacent cells will be DISTANT!")
    if config.lattice.functional_distance_threshold <= config.lattice.local_distance_threshold:
        logger.warning(f"   Functional threshold <= local threshold - no FUNCTIONAL connections possible!")

if __name__ == "__main__":
    main()