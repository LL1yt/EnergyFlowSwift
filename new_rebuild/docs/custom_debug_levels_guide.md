# Custom Debug Levels Guide

This guide explains how to use the new custom debug levels in the 3D Cellular Neural Network project.

## Overview

We've implemented custom debug levels that sit between `DEBUG` (10) and `INFO` (20) to provide fine-grained control over logging output. This allows you to enable specific categories of debug messages without getting overwhelmed by all debug output.

## Available Debug Levels

| Level Name      | Value | Purpose                                      |
|----------------|-------|----------------------------------------------|
| DEBUG_VERBOSE  | 11    | Most verbose debug output                    |
| DEBUG_CACHE    | 12    | Cache operations and lookups                 |
| DEBUG_SPATIAL  | 13    | Spatial optimization and neighbor finding    |
| DEBUG_FORWARD  | 14    | Forward pass details                         |
| DEBUG_MEMORY   | 15    | Memory management and GPU operations         |
| DEBUG_TRAINING | 16    | Training progress and metrics                |
| DEBUG_INIT     | 17    | Initialization and setup                     |

## Configuration

### 1. Via ProjectConfig

```python
from new_rebuild.config import ProjectConfig

# Enable specific debug categories
config = ProjectConfig()
config.logging.level = "INFO"  # Base level
config.logging.debug_categories = ['cache', 'spatial']  # Only show cache and spatial debug

# Use preset categories
config.logging.debug_categories = config.logging.CACHE_DEBUG      # ['cache']
config.logging.debug_categories = config.logging.SPATIAL_DEBUG    # ['spatial', 'memory']
config.logging.debug_categories = config.logging.TRAINING_DEBUG   # ['training', 'forward']
config.logging.debug_categories = config.logging.ALL_DEBUG        # All categories
```

### 2. Setting a specific debug level

```python
# Show all messages at DEBUG_CACHE level and above
config.logging.level = "DEBUG_CACHE"
```

## Usage in Code

### Getting a logger

```python
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)  # Auto-detects module name
# or
logger = get_logger("my_module")
```

### Using custom debug methods

```python
# Cache operations
logger.debug_cache(f"Cache hit for cell {cell_idx}")
logger.debug_cache(f"Cache miss, computing connections for {len(indices)} cells")

# Spatial operations
logger.debug_spatial(f"Finding neighbors within radius {radius}")
logger.debug_spatial(f"Spatial chunking: {chunk_size}x{chunk_size}x{chunk_size}")

# Forward pass
logger.debug_forward(f"Forward pass: batch_size={batch_size}, device={device}")
logger.debug_forward(f"Cell states shape: {states.shape}")

# Memory operations
logger.debug_memory(f"GPU memory allocated: {allocated_gb:.2f}GB")
logger.debug_memory(f"Chunking to fit in {available_memory}MB")

# Training progress
logger.debug_training(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
logger.debug_training(f"Learning rate: {lr}, Gradient norm: {grad_norm:.4f}")

# Initialization
logger.debug_init(f"Initializing {component_name} with params: {params}")
logger.debug_init(f"Model loaded from checkpoint: {checkpoint_path}")

# Most verbose
logger.debug_verbose(f"Detailed tensor values: {tensor[:5]}")
```

## Best Practices

### 1. Use appropriate categories

```python
# In connection_classifier.py
logger.debug_cache(f"🔍 Cache lookup for cell {cell_idx}")

# In spatial_optimizer.py  
logger.debug_spatial(f"📍 Processing chunk {chunk_idx}/{total_chunks}")

# In training loop
logger.debug_training(f"📊 Batch {batch_idx}: loss={loss:.4f}")
```

### 2. Include relevant context

```python
# Good: includes useful information
logger.debug_memory(f"GPU allocation failed: {allocated}MB/{required}MB, falling back to CPU")

# Bad: too vague
logger.debug_memory("Memory error")
```

### 3. Use emojis sparingly for important messages

```python
logger.debug_cache("✅ Cache hit rate: 95.3%")
logger.debug_init("🚀 Model initialization complete")
logger.debug_memory("⚠️ High memory usage detected")
```

## Examples

### Example 1: Debugging cache issues

```python
# Enable only cache debug
config.logging.debug_categories = ['cache']

# In your code
logger.debug_cache(f"Cache key: {cache_key}")
logger.debug_cache(f"Cache size: {cache_size}MB")
logger.debug_spatial("This won't appear")  # Different category
```

### Example 2: Debugging training

```python
# Enable training-related debug
config.logging.debug_categories = config.logging.TRAINING_DEBUG  # ['training', 'forward']

# In training loop
logger.debug_training(f"Starting epoch {epoch}")
logger.debug_forward(f"Processing batch: {batch.shape}")
logger.debug_training(f"Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
```

### Example 3: Debugging memory issues

```python
# Enable memory and spatial debug
config.logging.debug_categories = ['memory', 'spatial']

# In memory-intensive operations
logger.debug_memory(f"Available GPU memory: {torch.cuda.memory_available()}")
logger.debug_spatial(f"Chunk requires {chunk_memory}MB")
logger.debug_memory(f"Splitting into {num_chunks} smaller chunks")
```

## Migration Guide

### Before (using regular debug)

```python
logger.debug(f"Cache lookup for cell {cell_idx}")
logger.debug(f"Finding neighbors in radius {radius}")
logger.debug(f"Forward pass with batch size {batch_size}")
```

### After (using specific debug levels)

```python
logger.debug_cache(f"Cache lookup for cell {cell_idx}")
logger.debug_spatial(f"Finding neighbors in radius {radius}")  
logger.debug_forward(f"Forward pass with batch size {batch_size}")
```

## Configuration Examples

### For development/debugging

```python
# See everything
config.logging.level = "DEBUG"
config.logging.debug_mode = True
```

### For testing specific features

```python
# Test cache system
config.logging.level = "INFO"
config.logging.debug_categories = ['cache']

# Test spatial optimization
config.logging.level = "INFO"
config.logging.debug_categories = ['spatial', 'memory']
```

### For production

```python
# Minimal logging
config.logging.level = "WARNING"
config.logging.debug_categories = []
```

## Tips

1. **Start specific**: Enable only the debug categories you need
2. **Use presets**: Take advantage of preset category groups like `SPATIAL_DEBUG`
3. **Combine with grep**: Filter logs further with grep if needed
4. **Performance**: Custom debug levels have minimal overhead when disabled
5. **Consistency**: Use the same debug level for related operations