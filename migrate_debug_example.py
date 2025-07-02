#!/usr/bin/env python3
"""
Example showing how to migrate existing debug calls to use custom debug levels
"""

import sys
sys.path.append('.')

from new_rebuild.utils.logging import get_logger

# Example of migrating connection_classifier.py debug calls
def migrate_connection_classifier_example():
    logger = get_logger("connection_classifier")
    
    print("=== Migrating connection_classifier.py debug calls ===\n")
    
    # BEFORE:
    # logger.debug(f"🔍 classify_connections для клетки {cell_idx}: cache_manager={self.cache_manager is not None}")
    # logger.debug(f"🔍 Attempting cache lookup for cell {cell_idx}")
    # logger.debug(f"✅ Cache hit для клетки {cell_idx}")
    
    # AFTER:
    cell_idx = 42
    cache_manager = True
    
    logger.debug_cache(f"🔍 classify_connections для клетки {cell_idx}: cache_manager={cache_manager is not None}")
    logger.debug_cache(f"🔍 Attempting cache lookup for cell {cell_idx}")
    logger.debug_cache(f"✅ Cache hit для клетки {cell_idx}")
    

# Example of migrating spatial operations
def migrate_spatial_operations_example():
    logger = get_logger("spatial_optimizer")
    
    print("\n=== Migrating spatial optimizer debug calls ===\n")
    
    # BEFORE:
    # logger.debug(f"Processing chunk {chunk_idx}/{total_chunks}")
    # logger.debug(f"Chunk memory requirement: {memory_mb}MB")
    
    # AFTER:
    chunk_idx = 3
    total_chunks = 8
    memory_mb = 256
    
    logger.debug_spatial(f"Processing chunk {chunk_idx}/{total_chunks}")
    logger.debug_memory(f"Chunk memory requirement: {memory_mb}MB")


# Example of migrating forward pass logging
def migrate_forward_pass_example():
    logger = get_logger("gnn_cell")
    
    print("\n=== Migrating forward pass debug calls ===\n")
    
    # BEFORE:
    # logger.debug(f"Forward pass: input_shape={input_shape}, neighbor_states={neighbor_states.shape}")
    
    # AFTER:
    input_shape = (32, 64)
    neighbor_states_shape = (32, 20, 64)
    
    logger.debug_forward(f"Forward pass: input_shape={input_shape}, neighbor_states={neighbor_states_shape}")


# Example of migrating training logs
def migrate_training_logs_example():
    logger = get_logger("trainer")
    
    print("\n=== Migrating training debug calls ===\n")
    
    # BEFORE:
    # logger.debug(f"Epoch {epoch}: loss={loss:.4f}, lr={lr}")
    # logger.debug(f"Gradient norm: {grad_norm:.4f}")
    
    # AFTER:
    epoch = 5
    loss = 0.0234
    lr = 0.001
    grad_norm = 1.234
    
    logger.debug_training(f"Epoch {epoch}: loss={loss:.4f}, lr={lr}")
    logger.debug_training(f"Gradient norm: {grad_norm:.4f}")


# Show how to configure for different scenarios
def show_configuration_examples():
    print("\n=== Configuration Examples ===\n")
    
    print("1. Debug only cache operations:")
    print("   config.logging.level = 'INFO'")
    print("   config.logging.debug_categories = ['cache']")
    print()
    
    print("2. Debug spatial and memory issues:")
    print("   config.logging.level = 'INFO'")
    print("   config.logging.debug_categories = ['spatial', 'memory']")
    print()
    
    print("3. Full training debug:")
    print("   config.logging.level = 'INFO'")
    print("   config.logging.debug_categories = config.logging.TRAINING_DEBUG")
    print()
    
    print("4. Everything at DEBUG_CACHE level and above:")
    print("   config.logging.level = 'DEBUG_CACHE'")
    print()


if __name__ == "__main__":
    # Set up config to show some of the messages
    from new_rebuild.config import ProjectConfig
    
    config = ProjectConfig()
    config.logging.level = "INFO"
    config.logging.debug_categories = ['cache', 'spatial', 'memory', 'forward', 'training']
    config.__post_init__()
    
    # Run examples
    migrate_connection_classifier_example()
    migrate_spatial_operations_example()
    migrate_forward_pass_example()
    migrate_training_logs_example()
    show_configuration_examples()