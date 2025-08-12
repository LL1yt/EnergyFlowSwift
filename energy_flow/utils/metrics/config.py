from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsConfig:
    """Central configuration for metrics & profiling.

    Defaults are conservative to avoid overhead unless explicitly enabled.
    """

    # Master switches
    enable_metrics: bool = True
    enable_profiler: bool = True
    enable_gpu_monitoring: bool = True

    # Lightweight metrics
    collect_timing: bool = True
    collect_throughput: bool = True
    collect_memory_basic: bool = True

    # Advanced (higher overhead)
    collect_gpu_utilization: bool = False
    collect_memory_detailed: bool = False
    collect_component_profiling: bool = False

    # Profiler settings
    profiler_record_shapes: bool = False
    profiler_profile_memory: bool = False
    profiler_with_stack: bool = False
    profiler_export_interval: int = 1000
    profiler_trace_dir: str = "traces"

    # Collection intervals
    metrics_log_interval: int = 10
    gpu_monitor_interval: float = 1.0
    memory_cleanup_check_interval: int = 50

    # Automatic controls
    auto_disable_on_slow: bool = True
    performance_threshold: float = 0.05  # 5%

    # Optional tag to group runs (not used internally)
    run_tag: Optional[str] = None
