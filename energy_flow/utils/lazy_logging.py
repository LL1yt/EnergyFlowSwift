"""
Lazy logging utilities for deferring expensive computations until needed.

This module provides utilities to avoid computing expensive statistics when
they won't be logged due to log level settings.
"""

import logging
from typing import Callable, Any, Dict, List, Optional
import torch
from functools import wraps
import time


class LazyLogMessage:
    """
    A lazy log message that defers computation until str() is called.
    
    This is useful for expensive computations that should only run
    if the log message will actually be displayed.
    """
    
    def __init__(self, compute_fn: Callable[[], str], cache: bool = True):
        """
        Args:
            compute_fn: Function that computes the log message
            cache: Whether to cache the result after first computation
        """
        self.compute_fn = compute_fn
        self.cache = cache
        self._cached_result = None
        self._computed = False
    
    def __str__(self) -> str:
        """Compute the message when needed"""
        if self.cache and self._computed:
            return self._cached_result
        
        result = self.compute_fn()
        
        if self.cache:
            self._cached_result = result
            self._computed = True
            
        return result


class LazyStatisticsComputer:
    """
    Computes statistics lazily only when needed based on log level.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Args:
            logger: The logger to check for level
        """
        self.logger = logger
        self._stats_cache = {}
        self._cache_expiry_steps = 5  # Cache statistics for N steps
        self._current_step = 0
    
    def should_compute(self, level: int) -> bool:
        """Check if we should compute statistics for given log level"""
        return self.logger.isEnabledFor(level)
    
    def compute_if_needed(self, level: int, compute_fn: Callable[[], Any], 
                         cache_key: Optional[str] = None) -> Optional[Any]:
        """
        Compute statistics only if logging level requires it.
        
        Args:
            level: Logging level (e.g., logging.DEBUG)
            compute_fn: Function that computes the statistics
            cache_key: Optional key for caching results
            
        Returns:
            Computed statistics or None if not needed
        """
        if not self.should_compute(level):
            return None
        
        # Check cache if key provided
        if cache_key and cache_key in self._stats_cache:
            cached_data, cached_step = self._stats_cache[cache_key]
            if self._current_step - cached_step < self._cache_expiry_steps:
                return cached_data
        
        # Compute the statistics
        result = compute_fn()
        
        # Cache if key provided
        if cache_key:
            self._stats_cache[cache_key] = (result, self._current_step)
        
        return result
    
    def increment_step(self):
        """Increment the current step counter"""
        self._current_step += 1
        
        # Clean old cache entries
        if self._current_step % 10 == 0:
            self._clean_cache()
    
    def _clean_cache(self):
        """Remove old cache entries"""
        keys_to_remove = []
        for key, (_, step) in self._stats_cache.items():
            if self._current_step - step > self._cache_expiry_steps * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._stats_cache[key]


class FlowStatisticsComputer:
    """
    Specialized lazy computer for flow-related statistics.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Args:
            logger: The logger to use
        """
        self.logger = logger
        self.lazy_computer = LazyStatisticsComputer(logger)
    
    def compute_z_distribution(self, active_flows: List[Any], step: int) -> Optional[Dict]:
        """
        Compute Z-distribution statistics lazily.
        
        Args:
            active_flows: List of active flows
            step: Current step number
            
        Returns:
            Dictionary with statistics or None if not needed
        """
        def _compute():
            if not active_flows:
                return None
            
            # Collect Z-coordinates
            z_positions = torch.stack([flow.position[2] for flow in active_flows])
            
            # Basic statistics
            stats = {
                'min': z_positions.min().item(),
                'max': z_positions.max().item(),
                'mean': z_positions.mean().item(),
                'std': z_positions.std().item() if len(z_positions) > 1 else 0.0,
                'count': len(active_flows)
            }
            
            # Detailed boundary statistics (expensive)
            if self.logger.isEnabledFor(logging.DEBUG):
                stats['boundary'] = {
                    'z_min_boundary': (z_positions <= -0.95).sum().item(),
                    'z_max_boundary': (z_positions >= 0.95).sum().item(),
                    'z_center': ((z_positions > -0.2) & (z_positions < 0.2)).sum().item(),
                }
                
                # Z-layer distribution (very expensive for many flows)
                z_int = z_positions.int()
                unique_z, counts = torch.unique(z_int, return_counts=True)
                stats['layer_distribution'] = {
                    int(z.item()): int(count.item()) 
                    for z, count in zip(unique_z, counts)
                }
            
            return stats
        
        # Only compute for early steps or debug level
        if step <= 5 or self.logger.isEnabledFor(logging.DEBUG):
            return self.lazy_computer.compute_if_needed(
                logging.DEBUG, 
                _compute,
                cache_key=f"z_dist_step_{step}"
            )
        return None
    
    def compute_flow_aggregation_stats(self, flows: List[Any], 
                                      aggregation_type: str) -> Optional[Dict]:
        """
        Compute statistics for flow aggregation lazily.
        
        Args:
            flows: List of flows to aggregate
            aggregation_type: Type of aggregation ('surface', 'buffer', etc.)
            
        Returns:
            Dictionary with aggregation statistics or None
        """
        def _compute():
            if not flows:
                return None
            
            stats = {
                'count': len(flows),
                'aggregation_type': aggregation_type
            }
            
            # Only compute detailed stats for debug level
            if self.logger.isEnabledFor(logging.DEBUG):
                # Energy statistics
                energies = torch.stack([flow.energy for flow in flows])
                energy_norms = torch.norm(energies, dim=-1)
                
                stats['energy'] = {
                    'mean_norm': energy_norms.mean().item(),
                    'std_norm': energy_norms.std().item() if len(energies) > 1 else 0.0,
                    'min_norm': energy_norms.min().item(),
                    'max_norm': energy_norms.max().item()
                }
                
                # Age and step statistics
                ages = [flow.age for flow in flows]
                steps = [flow.steps_taken for flow in flows]
                
                stats['age'] = {
                    'mean': sum(ages) / len(ages) if ages else 0,
                    'min': min(ages) if ages else 0,
                    'max': max(ages) if ages else 0
                }
                
                stats['steps'] = {
                    'mean': sum(steps) / len(steps) if steps else 0,
                    'min': min(steps) if steps else 0,
                    'max': max(steps) if steps else 0
                }
            
            return stats
        
        return self.lazy_computer.compute_if_needed(
            logging.DEBUG,
            _compute,
            cache_key=f"agg_stats_{aggregation_type}_{len(flows)}"
        )
    
    def format_statistics(self, stats: Optional[Dict], prefix: str = "") -> str:
        """
        Format statistics dictionary into a readable string.
        
        Args:
            stats: Statistics dictionary or None
            prefix: Prefix for the message
            
        Returns:
            Formatted string
        """
        if stats is None:
            return f"{prefix}[statistics not computed - increase log level for details]"
        
        parts = [prefix]
        
        # Basic stats
        if 'count' in stats:
            parts.append(f"count={stats['count']}")
        
        if 'min' in stats and 'max' in stats:
            parts.append(f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        
        if 'mean' in stats:
            parts.append(f"mean={stats['mean']:.2f}")
            
        if 'std' in stats and stats['std'] > 0:
            parts.append(f"std={stats['std']:.2f}")
        
        # Boundary stats if available
        if 'boundary' in stats:
            b = stats['boundary']
            parts.append(f"boundaries={{z0:{b['z_min_boundary']}, "
                        f"zdepth:{b['z_max_boundary']}, "
                        f"center:{b['z_center']}}}")
        
        # Layer distribution if available (compact format)
        if 'layer_distribution' in stats:
            ld = stats['layer_distribution']
            if len(ld) <= 10:
                parts.append(f"layers={ld}")
            else:
                # Show only summary for many layers
                parts.append(f"layers=[{len(ld)} unique]")
        
        return " ".join(parts)


def lazy_log(logger: logging.Logger, level: int, 
             message_fn: Callable[[], str], *args, **kwargs):
    """
    Log a message lazily - only compute if level is enabled.
    
    Args:
        logger: Logger instance
        level: Log level
        message_fn: Function that computes the message
        *args, **kwargs: Additional arguments for logger
    """
    if logger.isEnabledFor(level):
        message = message_fn()
        logger.log(level, message, *args, **kwargs)


def timed_lazy_computation(compute_fn: Callable[[], Any], 
                          threshold_ms: float = 1.0) -> Callable[[], Any]:
    """
    Wrap a computation function to track timing.
    
    Args:
        compute_fn: Function to wrap
        threshold_ms: Log warning if computation takes longer than this (milliseconds)
        
    Returns:
        Wrapped function that tracks timing
    """
    @wraps(compute_fn)
    def wrapper():
        start_time = time.perf_counter()
        result = compute_fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if elapsed_ms > threshold_ms:
            logging.getLogger(__name__).debug(
                f"Expensive computation took {elapsed_ms:.2f}ms "
                f"(threshold: {threshold_ms}ms)"
            )
        
        return result
    
    return wrapper


class LazyLoggerAdapter:
    """
    Adapter that wraps a logger to provide lazy logging methods.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Args:
            logger: The logger to wrap
        """
        self.logger = logger
        self.stats_computer = LazyStatisticsComputer(logger)
        self.flow_stats = FlowStatisticsComputer(logger)
    
    def debug_lazy(self, message_fn: Callable[[], str]):
        """Log debug message lazily"""
        lazy_log(self.logger, logging.DEBUG, message_fn)
    
    def info_lazy(self, message_fn: Callable[[], str]):
        """Log info message lazily"""
        lazy_log(self.logger, logging.INFO, message_fn)
    
    def warning_lazy(self, message_fn: Callable[[], str]):
        """Log warning message lazily"""
        lazy_log(self.logger, logging.WARNING, message_fn)
    
    def log_statistics(self, level: int, prefix: str, 
                       compute_fn: Callable[[], Dict], 
                       cache_key: Optional[str] = None):
        """
        Log statistics with lazy computation.
        
        Args:
            level: Log level
            prefix: Message prefix
            compute_fn: Function to compute statistics
            cache_key: Optional cache key
        """
        if self.logger.isEnabledFor(level):
            stats = self.stats_computer.compute_if_needed(
                level, compute_fn, cache_key
            )
            if stats:
                message = self.flow_stats.format_statistics(stats, prefix)
                self.logger.log(level, message)
    
    def __getattr__(self, name):
        """Forward other method calls to the wrapped logger"""
        return getattr(self.logger, name)


# Example usage functions
def create_lazy_logger(name: str) -> LazyLoggerAdapter:
    """
    Create a logger with lazy logging capabilities.
    
    Args:
        name: Logger name
        
    Returns:
        LazyLoggerAdapter instance
    """
    logger = logging.getLogger(name)
    return LazyLoggerAdapter(logger)


def benchmark_statistics_computation():
    """
    Benchmark function to test the performance gain from lazy logging.
    """
    import random
    
    # Create mock flows
    class MockFlow:
        def __init__(self):
            self.position = torch.randn(3)
            self.energy = torch.randn(768)
            self.age = random.randint(0, 100)
            self.steps_taken = random.randint(0, 50)
    
    # Create many flows
    flows = [MockFlow() for _ in range(1000)]
    
    # Create logger
    logger = logging.getLogger("benchmark")
    lazy_logger = LazyLoggerAdapter(logger)
    
    # Test with different log levels
    for level_name, level in [("DEBUG", logging.DEBUG), 
                              ("INFO", logging.INFO), 
                              ("WARNING", logging.WARNING)]:
        
        logger.setLevel(level)
        
        start_time = time.perf_counter()
        
        # Simulate 100 logging calls
        for i in range(100):
            # This will only compute if level is enabled
            lazy_logger.log_statistics(
                logging.DEBUG,
                f"Step {i} statistics: ",
                lambda: {
                    'flow_count': len(flows),
                    'z_positions': [f.position[2].item() for f in flows[:10]],
                    'energy_norms': [torch.norm(f.energy).item() for f in flows[:10]]
                },
                cache_key=f"step_{i}"
            )
        
        elapsed = time.perf_counter() - start_time
        print(f"Level {level_name}: {elapsed:.4f} seconds")


if __name__ == "__main__":
    # Run benchmark
    print("Running lazy logging benchmark...")
    benchmark_statistics_computation()
