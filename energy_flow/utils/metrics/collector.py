from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Deque, Optional, Tuple

try:
    import torch  # type: ignore
except Exception:  # torch is optional
    torch = None  # type: ignore

from ..logging import get_logger, DEBUG_PERFORMANCE
from .config import MetricsConfig


@dataclass
class TimingRecord:
    name: str
    elapsed_s: float
    t_wall: float


class MetricsCollector:
    """Lightweight central metrics collector.

    All paths are guarded by config flags to keep overhead negligible.
    Storage is bounded to avoid memory growth.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.enabled = bool(config.enable_metrics)
        self.logger = get_logger(__name__)

        self.timing_metrics: Optional[Deque[TimingRecord]] = (
            deque(maxlen=1000) if (self.enabled and config.collect_timing) else None
        )
        self.throughput_metrics: Optional[Deque[Tuple[str, float, float]]] = (
            deque(maxlen=1000) if (self.enabled and config.collect_throughput) else None
        )
        self.memory_metrics: Optional[Deque[Tuple[str, int, float]]] = (
            deque(maxlen=500) if (self.enabled and config.collect_memory_basic) else None
        )

    @contextmanager
    def time_component(self, component_name: str):
        """Context manager for timing a component.

        If disabled, acts as a no-op with near-zero overhead.
        """
        if not (self.enabled and self.config.collect_timing):
            yield
            return

        start_time = time.perf_counter()
        start_wall = time.time()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if self.timing_metrics is not None and elapsed >= 0.0005:  # ignore <0.5ms
                self.timing_metrics.append(
                    TimingRecord(name=component_name, elapsed_s=elapsed, t_wall=start_wall)
                )

    def record_throughput(self, name: str, items: int, elapsed_s: float) -> None:
        if not (self.enabled and self.config.collect_throughput):
            return
        if elapsed_s <= 0:
            return
        rate = items / elapsed_s
        if self.throughput_metrics is not None:
            self.throughput_metrics.append((name, rate, time.time()))
        # Optional log at performance level
        self.logger.log(
            DEBUG_PERFORMANCE, f"Throughput[{name}]: {rate:.2f} items/s (items={items}, t={elapsed_s:.3f}s)"
        )

    def snapshot_gpu_memory(self, label: str = "snapshot") -> None:
        if not (self.enabled and self.config.collect_memory_basic):
            return
        if torch is None:
            return
        try:
            allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
            reserved = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        except Exception:
            allocated = 0
            reserved = 0
        if self.memory_metrics is not None:
            self.memory_metrics.append((label, allocated, time.time()))
        # Lightweight, log only allocated to avoid noise
        self.logger.log(
            DEBUG_PERFORMANCE,
            f"GPU mem[{label}]: allocated={allocated/1024**3:.2f}GB, reserved={reserved/1024**3:.2f}GB",
        )
