from contextlib import contextmanager
from typing import Generator, Optional

try:
    import torch  # type: ignore
except Exception:  # torch optional
    torch = None  # type: ignore

from ..logging import get_logger, DEBUG_PROFILING
from .config import MetricsConfig


class ProfilerManager:
    """Thin wrapper over torch.profiler with safe no-op behavior when disabled.

    Exports Chrome traces periodically if enabled.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self.enabled = bool(config.enable_profiler and torch is not None)
        self.step_count: int = 0
        self.logger = get_logger(__name__)

    @contextmanager
    def profile_step(self, step_name: str = "train_step") -> Generator[Optional[object], None, None]:
        if not self.enabled or torch is None:
            yield None
            return
        activities = [
            torch.profiler.ProfilerActivity.CPU,
        ]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=self.config.profiler_record_shapes,
            profile_memory=self.config.profiler_profile_memory,
            with_stack=self.config.profiler_with_stack,
        ) as prof:
            yield prof
        self.step_count += 1
        if self.step_count % max(1, int(self.config.profiler_export_interval)) == 0:
            self._export_trace(prof, step_name)

    def _export_trace(self, prof, step_name: str) -> None:
        try:
            import os
            os.makedirs(self.config.profiler_trace_dir, exist_ok=True)
            trace_path = f"{self.config.profiler_trace_dir}/{step_name}_step_{self.step_count}.json"
            prof.export_chrome_trace(trace_path)
            self.logger.log(DEBUG_PROFILING, f"Exported profiler trace: {trace_path}")
        except Exception as e:
            self.logger.warning(f"Profiler export failed: {e}")
