"""
Simple and safe memory cleanup helpers
=====================================

- Minimalistic utilities to reduce GPU/CPU memory pressure
- No heavy heuristics; designed for research iteration speed
- Integrates with centralized logging (DEBUG_MEMORY)
"""

from contextlib import contextmanager
import gc
from typing import Iterable, Optional

try:
    import torch  # type: ignore
except Exception:  # torch optional
    torch = None  # type: ignore

from .logging import get_logger, DEBUG_MEMORY

logger = get_logger(__name__)


def free_tensor(t) -> None:
    """Best-effort free of a tensor: detach, move to CPU (if large), and drop references.

    Accepts anything; silently ignores non-tensors.
    """
    if torch is None:
        return
    try:
        if isinstance(t, torch.Tensor):
            # Detach from graph
            if t.grad is not None:
                try:
                    t.grad = None
                except Exception:
                    pass
            # Move tiny tensors to CPU is unnecessary; avoid churn. Only large ones.
            try:
                if t.is_cuda and t.numel() >= 1_000_000:  # ~4MB for float32
                    t.data = t.detach().to("cpu", copy=False)
                else:
                    t.data = t.detach()
            except Exception:
                pass
    except Exception:
        pass


def clear_cuda_cache() -> None:
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Some backends may not support ipc_collect; ignore
        pass


def collect_garbage() -> None:
    try:
        gc.collect()
    except Exception:
        pass


@contextmanager
def memory_guard(threshold_gb: float = 28.0) -> None:
    """Context that checks GPU memory after a block and performs cleanup if above threshold.

    - Logs before and after in DEBUG_MEMORY
    - Does not raise; best-effort cleanup only
    """
    if torch is None or not torch.cuda.is_available():
        yield
        return

    try:
        before_alloc = torch.cuda.memory_reserved() / 1024**3
        logger.log(DEBUG_MEMORY, f"🧠 mem-guard start: reserved={before_alloc:.2f}GB")
    except Exception:
        before_alloc = 0.0
    try:
        yield
    finally:
        try:
            after_alloc = torch.cuda.memory_reserved() / 1024**3
        except Exception:
            after_alloc = 0.0
        if after_alloc >= threshold_gb:
            logger.log(
                DEBUG_MEMORY, f"🧹 mem-guard threshold hit ({after_alloc:.2f}GB >= {threshold_gb:.2f}GB) → cleanup"
            )
            clear_cuda_cache()
            collect_garbage()
            try:
                post = torch.cuda.memory_reserved() / 1024**3
            except Exception:
                post = after_alloc
            logger.log(DEBUG_MEMORY, f"🧹 mem-guard after cleanup: reserved={post:.2f}GB")
        else:
            logger.log(DEBUG_MEMORY, f"🧠 mem-guard end: reserved={after_alloc:.2f}GB (ok)")


def bulk_free(items: Iterable[object]) -> None:
    """Free a collection of tensors/objects safely and clear caches.

    Designed to be used at safe boundaries (end of step/epoch).
    """
    for obj in items:
        free_tensor(obj)
    clear_cuda_cache()
    collect_garbage()
