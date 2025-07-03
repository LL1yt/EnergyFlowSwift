#!/usr/bin/env python3
"""Test multiple forward passes with proper radius - FIXED VERSION"""

import torch
import time
import numpy as np

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()

# Увеличим adaptive_radius_ratio чтобы distant_threshold был меньше adaptive_radius
config.lattice.adaptive_radius_ratio = 0.2  # Это даст нам radius = 2.0 для 10x10x10

set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core import create_lattice, create_cell
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.core.training.utils import create_simple_embeddings

logger = get_logger(__name__)

def main():
    """Test multiple forward passes"""
    logger.info("🧪 Testing multiple forward passes with fixed parameters...")
    
    device = torch.device(config.current_device)
    
    # Log configuration
    adaptive_radius = config.calculate_adaptive_radius()
    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Lattice: {config.lattice.dimensions}")
    logger.info(f"   Adaptive radius: {adaptive_radius}")
    logger.info(f"   Local threshold: {adaptive_radius * config.lattice.local_distance_ratio}")
    logger.info(f"   Distant threshold: {adaptive_radius * config.lattice.distant_distance_ratio}")
    
    # Create simple training data
    logger.info("\n📝 Creating simple training data...")
    input_text = "hello world"
    output_text = "world hello"
    
    # Create embeddings (batch_size=1, embedding_dim=768)
    embeddings_input = create_simple_embeddings(input_text, config.embedding.model_name)
    embeddings_output = create_simple_embeddings(output_text, config.embedding.model_name)
    
    # Create trainer
    logger.info("\n🏗️ Creating trainer...")
    trainer = EmbeddingTrainer(config, device)
    
    # Multiple forward passes
    logger.info("\n🔄 Running multiple forward passes...")
    num_passes = 3
    outputs = []
    
    for i in range(num_passes):
        logger.info(f"\n--- Forward pass {i+1}/{num_passes} ---")
        start_time = time.time()
        
        # Forward pass
        output = trainer.lattice.forward(embeddings_input.to(device))
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
    
    # Compute loss
    logger.info("\n📊 Computing loss...")
    loss = trainer.compute_loss(embeddings_input, embeddings_output)
    logger.info(f"📉 Loss: {loss.item():.6f}")
    
    logger.info("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()