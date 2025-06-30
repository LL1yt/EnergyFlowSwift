#!/usr/bin/env python3
"""
Simple test for UTF8StreamHandler
"""

import logging
import sys
from new_rebuild.utils.logging import UTF8StreamHandler

def test_utf8_handler():
    """Test the custom UTF8StreamHandler"""
    
    # Create logger
    logger = logging.getLogger("test_utf8")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add our custom handler
    handler = UTF8StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    print("Testing UTF8StreamHandler...")
    
    # Test ASCII text
    logger.info("Basic ASCII text")
    
    # Test Russian text
    logger.info("Инициализация EmbeddingTrainer на устройстве")
    
    # Test emojis
    logger.info("✅ CUDA устройство выбрано")
    
    # Test mixed
    logger.info("🚀 Starting Russian текст with emojis ✨")
    
    print("Test completed!")

if __name__ == "__main__":
    test_utf8_handler()