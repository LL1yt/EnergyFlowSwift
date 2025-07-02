#!/usr/bin/env python3
"""
Test custom debug levels functionality
"""

import sys
sys.path.append('.')

from new_rebuild.config import ProjectConfig
from new_rebuild.utils.logging import get_logger


def test_custom_debug_levels():
    """Test different debug level configurations"""
    
    print("=== Testing Custom Debug Levels ===\n")
    
    # Test 1: Default configuration (INFO level)
    print("1. Default configuration (INFO level):")
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = []
    config.__post_init__()
    
    logger = get_logger("test_default")
    logger.debug("This debug message should NOT appear")
    logger.debug_cache("This cache debug should NOT appear")
    logger.info("This info message SHOULD appear")
    print()
    
    # Test 2: Enable only cache debug
    print("\n2. Enable only CACHE debug category:")
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = ['cache']
    config.__post_init__()
    
    logger = get_logger("test_cache")
    logger.debug("This regular debug should NOT appear")
    logger.debug_cache("🔍 Cache lookup performed")
    logger.debug_spatial("This spatial debug should NOT appear")
    logger.info("Regular info message")
    print()
    
    # Test 3: Enable spatial and memory debug
    print("\n3. Enable SPATIAL and MEMORY debug categories:")
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = ['spatial', 'memory']
    config.__post_init__()
    
    logger = get_logger("test_spatial")
    logger.debug_spatial("📍 Finding neighbors in 3D space")
    logger.debug_memory("💾 GPU memory allocated: 1.2GB")
    logger.debug_cache("This cache debug should NOT appear")
    logger.info("Regular info message")
    print()
    
    # Test 4: Custom DEBUG_CACHE level
    print("\n4. Set logging level to DEBUG_CACHE:")
    config = ProjectConfig()
    config.logging.level = "DEBUG_CACHE"
    config.__post_init__()
    
    logger = get_logger("test_level")
    logger.debug_verbose("This verbose debug should NOT appear (lower level)")
    logger.debug("This regular debug should NOT appear (lower level)")
    logger.debug_cache("Cache operations are visible at this level")
    logger.debug_spatial("Spatial operations visible (higher level)")
    logger.info("Info messages always visible")
    print()
    
    # Test 5: All debug categories
    print("\n5. Enable ALL debug categories:")
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = config.logging.ALL_DEBUG
    config.__post_init__()
    
    logger = get_logger("test_all")
    logger.debug_cache("✅ Cache hit for key: model_123")
    logger.debug_spatial("🗺️ Spatial chunking: 8x8x8 blocks")
    logger.debug_forward("➡️ Forward pass: batch_size=32")
    logger.debug_memory("🧠 Memory usage: 4.5GB/32GB")
    logger.debug_training("📊 Training loss: 0.0234")
    logger.debug_init("🚀 Initializing model components")
    logger.debug_verbose("🔍 Very detailed debug info")
    print()
    
    # Test 6: Using preset debug categories
    print("\n6. Using preset TRAINING_DEBUG categories:")
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = config.logging.TRAINING_DEBUG
    config.__post_init__()
    
    logger = get_logger("test_training")
    logger.debug_training("📈 Epoch 1/10, Loss: 0.543")
    logger.debug_forward("🔄 Processing batch 1/100")
    logger.debug_cache("This cache debug should NOT appear")
    logger.info("Training started")
    print()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_custom_debug_levels()