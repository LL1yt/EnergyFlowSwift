"""
Metrics & Profiling utilities (lightweight scaffolding)
======================================================

Zero-overhead by default. All components are safe no-ops unless explicitly enabled
via MetricsConfig flags.
"""

from .config import MetricsConfig
from .collector import MetricsCollector
from .gpu_monitor import GPUMonitor
from .profiler import ProfilerManager

__all__ = [
    "MetricsConfig",
    "MetricsCollector",
    "GPUMonitor",
    "ProfilerManager",
]
