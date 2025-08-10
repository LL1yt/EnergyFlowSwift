import time
from typing import Optional

from ..logging import get_logger, DEBUG_PERFORMANCE
from .config import MetricsConfig


class GPUMonitor:
    """Low-overhead GPU utilization monitor with graceful fallbacks.

    If NVML is unavailable, remains a safe stub. Polling is rate-limited.
    """

    def __init__(self, config: MetricsConfig):
        self.logger = get_logger(__name__)
        self.enabled = bool(config.enable_gpu_monitoring)
        self.interval = float(config.gpu_monitor_interval)
        self._last_poll: float = 0.0
        self._cached_utilization: float = 0.0
        self._nvml = None

        if self.enabled:
            self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.logger.log(DEBUG_PERFORMANCE, "GPU monitoring: NVML initialized")
        except Exception:
            self._nvml = None
            self.logger.log(
                DEBUG_PERFORMANCE,
                "GPU monitoring: NVML not available, falling back to torch (utilization will be 0)",
            )

    def _poll_gpu_utilization(self) -> float:
        if not self._nvml:
            return 0.0
        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    @property
    def gpu_utilization(self) -> float:
        if not self.enabled:
            return 0.0
        now = time.time()
        if (now - self._last_poll) < self.interval:
            return self._cached_utilization
        self._last_poll = now
        self._cached_utilization = self._poll_gpu_utilization()
        return self._cached_utilization
