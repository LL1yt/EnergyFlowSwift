#!/usr/bin/env python3
"""Simple forward pass test with fixed radius"""

import torch
import time

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
config.lattice.adaptive_radius_ratio = 0.2  # radius = 2.0 для 10x10x10
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core import create_lattice

logger = get_logger(__name__)

def main():
    """Test forward pass"""
    logger.info("🧪 Testing forward pass with fixed radius...")
    
    device = torch.device(config.current_device)
    
    # Log configuration
    adaptive_radius = config.calculate_adaptive_radius()
    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Lattice: {config.lattice.dimensions}")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   Local threshold: {adaptive_radius * config.lattice.local_distance_ratio}")
    logger.info(f"   Functional threshold: {adaptive_radius * config.lattice.functional_distance_ratio}")
    logger.info(f"   Distant threshold: {adaptive_radius * config.lattice.distant_distance_ratio}")
    
    # Create lattice (использует конфигурацию из ProjectConfig)
    logger.info("\n🏗️ Creating lattice...")
    lattice = create_lattice()
    
    # Create random input
    batch_size = 1
    total_cells = config.lattice.dimensions[0] * config.lattice.dimensions[1] * config.lattice.dimensions[2]
    input_states = torch.randn(batch_size, total_cells, config.model.state_size, device=device)
    
    logger.info(f"\n📊 Input shape: {input_states.shape}")
    
    # Multiple forward passes
    logger.info("\n🔄 Running multiple forward passes...")
    num_passes = 3
    outputs = []
    
    for i in range(num_passes):
        logger.info(f"\n--- Forward pass {i+1}/{num_passes} ---")
        start_time = time.time()
        
        # Forward pass
        output = lattice.forward(input_states)
        outputs.append(output)
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Forward pass time: {elapsed_time:.3f}s")
        logger.info(f"📊 Output shape: {output.shape}")
        logger.info(f"📊 Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
        
        # Check consistency
        if i > 0:
            diff = torch.abs(outputs[i] - outputs[0]).max().item()
            logger.info(f"🔍 Max difference from first pass: {diff:.6e}")
            if diff < 1e-6:
                logger.info("✅ Output is consistent!")
            else:
                logger.warning(f"⚠️ Output differs from first pass by {diff}")
    
    logger.info("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()