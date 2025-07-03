# Debug Findings Summary

## Issues Found

### 1. Too Many Neighbors (999 out of 1000 cells)
- **Problem**: With a 10x10x10 lattice (1000 cells) and adaptive_radius_ratio=0.6, the calculated radius is 6.0
- **Impact**: This radius includes almost all cells in the lattice as neighbors
- **Fix**: Reduced adaptive_radius_ratio to 0.3 for DEBUG mode in config_components.py

### 2. Zero Neighbors for Each Expert
- **Root Cause**: Connection classifier was using local indices (1, 2, 3...) but cache contains global indices
- **Impact**: No connections were matching between classifier and cache
- **Fix**: Changed connection_classifier.py to pass global neighbor indices directly to cache

### 3. Same Cell Processed Multiple Times
- **Likely Cause**: Batch processing or chunking in spatial optimizer
- **Added**: Debug tracking to detect duplicate processing

## Code Changes Made

### 1. config_components.py
```python
# Added to DebugPreset class
lattice_adaptive_radius_ratio: float = 0.3  # Smaller ratio for debug mode
```

### 2. simple_config.py
```python
# Added to _apply_debug_mode()
if hasattr(preset, 'lattice_adaptive_radius_ratio'):
    self.lattice.adaptive_radius_ratio = preset.lattice_adaptive_radius_ratio
```

### 3. connection_classifier.py
```python
# Changed from local indices to global indices
result = self.cache_manager.get_cached_connections(
    cell_idx=cell_idx,
    neighbor_indices=neighbor_indices_list,  # Now using global indices
    states=all_states,
    functional_similarity_threshold=self.functional_similarity_threshold,
)
```

### 4. Added Debug Logging
- Enhanced logging in moe_processor.py to show classification results
- Added debug logging in connection_cache.py to show cache lookups
- Added cell processing tracking in unified_spatial_optimizer.py

## Expected Results

With these fixes:
1. Adaptive radius of 3.0 for 10x10x10 lattice should give ~100-200 neighbors
2. Connections should be properly classified into LOCAL/FUNCTIONAL/DISTANT
3. Each cell should be processed only once per forward pass

## Test Command

Run the test script to verify fixes:
```bash
python test_neighbor_fix.py
```