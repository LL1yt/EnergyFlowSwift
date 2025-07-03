#!/usr/bin/env python3
"""Test script to verify neighbor classification fix"""

import torch

# Configure before imports
from new_rebuild.config import set_project_config, create_debug_config

config = create_debug_config()
set_project_config(config)

from new_rebuild.utils.logging import get_logger
from new_rebuild.core.training import EmbeddingTrainer

logger = get_logger(__name__)

def main():
    """Test neighbor classification"""
    logger.info("🔍 Testing neighbor classification fix...")
    
    # Create trainer
    trainer = EmbeddingTrainer(config)
    
    # Create test batch
    batch_size = 1
    embedding_dim = config.embedding.teacher_dim
    test_embeddings = torch.randn(batch_size, embedding_dim).to(trainer.device)
    
    logger.info(f"📊 Running single forward pass...")
    logger.info(f"📊 Batch size: {batch_size}, Embedding dim: {embedding_dim}")
    logger.info(f"📊 Lattice: {config.lattice.dimensions}, Adaptive radius: {config.calculate_adaptive_radius()}")
    
    try:
        # Single forward pass
        losses = trainer._forward_pass(
            input_embeddings=test_embeddings,
            target_embeddings=test_embeddings,
            texts=None
        )
        
        logger.info(f"✅ Forward pass completed successfully!")
        logger.info(f"📈 Total loss: {losses['total'].item():.6f}")
        logger.info(f"📈 Reconstruction loss: {losses['reconstruction'].item():.6f}")
        
        # Check if losses are reasonable (not zero)
        if losses['total'].item() == 0:
            logger.warning("⚠️ Total loss is zero - this might indicate a problem!")
        else:
            logger.info("✅ Loss values look reasonable")
            
    except Exception as e:
        logger.error(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()