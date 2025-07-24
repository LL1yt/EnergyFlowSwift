"""
Debugging utilities for emergent training.
"""

import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def debug_computational_graph(
    outputs: Dict[str, torch.Tensor],
    losses: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
):
    """Logs the state of the computational graph for debugging."""
    logger.debug("--- Computational Graph Debug ---")
    for name, tensor in outputs.items():
        if tensor is not None and tensor.requires_grad:
            logger.debug(
                f"Output '{name}': grad_fn={tensor.grad_fn.__class__.__name__}"
            )
    for name, tensor in losses.items():
        if tensor is not None and tensor.requires_grad:
            logger.debug(f"Loss '{name}': grad_fn={tensor.grad_fn.__class__.__name__}")
    logger.debug("---------------------------------")


def debug_tensor_versions(trainer_instance, step_name: str):
    """Logs the _version_ of key tensors to track in-place modifications."""
    # This requires the trainer to expose its tensors.
    # We will assume the trainer has a method `get_debug_tensors` for this.
    logger.debug(f"--- Tensor Versions at: {step_name} ---")
    tensors_to_track = trainer_instance.get_debug_tensors()
    for name, tensor in tensors_to_track.items():
        if tensor is not None:
            logger.debug(f"Tensor '{name}': version={tensor._version}")
    logger.debug("-------------------------------------")
