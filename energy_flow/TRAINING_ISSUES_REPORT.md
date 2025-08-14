# Energy Flow Training Issues Report (Actionable)

This report reviews three training problems in the energy_flow project, consolidating guidance from energy_flow/CLAUDE.md and code inspection. For each problem: symptoms, likely root causes, diagnostics, concrete fixes, and verification steps.

Repository context reviewed:
- full_energy_trainer.py
- energy_flow/config/energy_config.py
- energy_flow/training/energy_trainer.py
- energy_flow/core/flow_processor.py
- energy_flow/core/energy_carrier.py
- energy_flow/dataset/manager.py

Note: The project sets CUDA as default device when available (energy_flow/config/energy_config.py), so any CPU device assumptions must be explicit.

---

## Problem 1 — GPU OOM on the second batch of the first epoch

Symptoms
- Training stops on batch 2 with CUDA OOM (28+ GB allocated) and fragmentation hints.
- Expectation that memory is freed at the end of each batch is not met.

Likely root causes
- Batch assembly on CUDA due to default_device=“cuda” leaking memory between batches.
  - Any tensor creation without explicit device can land on GPU (e.g., collate_fn, temporary tensors in forward/metrics).
- DataLoader shuffle generator device mismatch previously caused errors; now fixed, but still ensure tensors stay on CPU until needed.
- Gradient accumulation retains graphs or references:
  - retain_graph=True anywhere in training will keep graphs alive.
  - Storing large CUDA tensors (intermediates, predictions, or debug features) on self.* keeps them alive across steps.
- Aggressive metric/logging collection forces synchronizations and may keep references.
- cuDNN autotune/benchmark switching and AMP scaling can increase transient peak allocation.
- CUDA memory fragmentation over large allocations (e.g., GRU activations) on batch boundaries.

Diagnostics
- Verify where tensors are created and on which device:
  - Ensure collate_fn stacks CPU tensors (already done in full_energy_trainer.py) and only moves to CUDA inside EnergyTrainer.train_step.
- Track CUDA memory over steps (peak and current) in DEBUG_MEMORY with minimal syncs.
- Temporarily run with:
  - batch_size reduced by 2–4x to confirm scaling behavior.
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (Windows supported) to mitigate fragmentation.
- Search for retain_graph=True and for long-lived CUDA tensors assigned to self.* in flow_processor/energy_carrier/energy_trainer.

Concrete fixes
- Keep batch tensors on CPU until the very moment of compute in train_step:
  - Move teacher_input_embeddings/teacher_target_embeddings to device inside train_step, not in DataLoader/collate.
- Eliminate retain_graph in training unless strictly required. Rebuild the graph each accumulation step.
- Ensure no step-scoped CUDA tensors are cached in self.* beyond what’s necessary for model state.
- Reduce intermediate retention:
  - Wrap validation and heavy metrics in torch.no_grad().
  - Avoid storing per-step CUDA tensors in history structures.
- Control memory sync-heavy calls:
  - Call torch.cuda.reset_peak_memory_stats() only at epoch boundaries or in DEBUG_MEMORY blocks.
  - Avoid frequent torch.cuda.synchronize() except when measuring critical timings.
- Use fragmentation mitigations:
  - Set env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - Prefer pinned-memory DataLoader (pin_memory=True when CUDA available) with CPU tensors.
- If GRU activation footprint is high:
  - Try smaller hidden sizes or sequence chunking in energy_carrier.

Verification
- With DEBUG_MEMORY on, confirm: peak memory stabilizes across batches; no monotonic growth.
- Confirm no CUDA OOM at batch 2 for original batch size; if needed, step back up from reduced batch size.
- Nsight Systems/torch profiler (short window) to confirm no ever-growing allocation sites.

---

## Problem 2 — Performance plummets after the first batch

Symptoms
- First batch is comparatively fast; subsequent batches are significantly slower.

Likely root causes
- Forced CUDA synchronizations introduced by metrics and memory stats:
  - torch.cuda.reset_peak_memory_stats() at per-step cadence forces syncs.
  - Accessing .item() on CUDA tensors for logging forces device sync.
- DataLoader multiprocessing on Windows causes extra overhead if objects are not picklable or workers thrash; now fixed by moving wrapper/collate to top level, but keep an eye on worker count and pinned memory.
- Autocast/GradScaler warmup: first steps build kernels; later steps add syncs due to stats collection.
- Default device CUDA in config may push DataLoader or dataset objects to GPU inadvertently, causing contention.

Diagnostics
- Measure per-step timings with minimal logging: isolate time in
  - data loading (next(dataloader)),
  - train_step total,
  - model forward (flow_processor),
  - optimizer.step/scaler.
- Temporarily disable DEBUG_* logs and any metric snapshots; compare timings.
- Compare runs with torch.backends.cudnn.benchmark=True vs deterministic paths.

Concrete fixes
- Rate-limit or guard expensive metrics/logs:
  - Use gated_log with a larger interval.
  - Avoid .item() in hot loops; aggregate on GPU or move to CPU after detach().
- Move torch.cuda.reset_peak_memory_stats() to
  - epoch boundaries, or
  - only when DEBUG_MEMORY enabled.
- Keep DataLoader efficient and CPU-bound:
  - pin_memory=True when CUDA available.
  - num_workers tuned (start with 2; adjust up/down based on CPU and disk IO).
  - persistent_workers=True to avoid worker respawn cost.
- Ensure no unnecessary torch.cuda.synchronize() except for precise timing in profiling mode.
- Consider fusing small ops or using torch.compile (per plan in TORCH_COMPILE_PLAN.md) for FlowProcessor, EnergyCarrier, SimpleNeuron when the training is stable.

Verification
- Step time after batch 1 should remain within 10–20% of batch 1 under comparable conditions.
- GPU utilization should be steady; spikes only when logging snapshots occur at a low rate.

---

## Problem 3 — NaNs in AddmmBackward0 (and occasional OOM in GRU)

Symptoms
- RuntimeError: Function 'AddmmBackward0' returned nan values.
- Warnings about missing forward pass info (suggest enabling detect_anomaly during forward).
- Later, GRU forward triggers CUDA OOM (~492 MiB additional alloc requested) when GPU is already saturated.

Likely root causes
- Non-finite values entering linear layers or GRU due to numerical spikes in energy computations or text bridge.
- Too high learning rate and insufficient gradient clipping when using AMP.
- Hidden state blow-up in EnergyCarrier GRU with certain input ranges.
- Mixed precision scaling edge cases causing inf grads that propagate to NaNs.

Diagnostics
- Enable torch.autograd.set_detect_anomaly(True) for first 1–2 steps only (EnergyTrainer already plans anomaly_steps_remaining) to localize the op producing NaN in forward.
- Insert finite checks:
  - torch.isfinite on inputs/outputs around: FlowProcessor.forward outputs, EnergyCarrier inputs/hidden, and loss components.
  - Use torch.nan_to_num at carefully chosen boundaries to sanitize inputs.
- Log min/max/rms of critical tensors at a low frequency (e.g., every 50 steps) to detect drift.

Concrete fixes
- Stability tweaks (already partially present) and extend:
  - Reduce LR (already scaled to 0.25x once); consider another 0.5x if NaNs persist.
  - Enforce gradient clipping (e.g., clip_grad_norm_ or value clip at 0.1–0.5) every optimizer step.
  - Ensure GradScaler is enabled with AMP; if NaNs persist, try disabling AMP temporarily to isolate.
- Sanitize inputs to GRU and linear layers:
  - Apply clamp or layernorm on FlowProcessor outputs before feeding into EnergyCarrier GRU.
  - Use torch.nan_to_num on suspect tensors (only as a stopgap when non-finites detected).
- Determinism for debugging sessions:
  - torch.backends.cudnn.deterministic = True (only during debug runs; revert to benchmark=True for speed after stabilization).
- Add protective checks in training:
  - If not torch.isfinite(loss), skip optimizer.step, log context, and continue to next batch.

Verification
- No NaNs reported in first N steps with anomaly detection disabled after initial check.
- Loss curves remain finite and smooth; gradient norms bounded by clip threshold.
- GRU input/hidden ranges remain within expected bounds; no OOM during GRU call when overall memory is within budget.

---

## Prioritized Action Plan

1) Stop the memory leak/fragmentation across batches
- Keep batch tensors on CPU until train_step; confirm all temporary tensors in collate/metrics are on CPU.
- Reduce sync-heavy metrics; move peak memory reset to epoch boundary.
- Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and verify stabilization.

2) Stabilize numerical behavior
- Add finite checks around FlowProcessor outputs and before GRU.
- Lower LR (another 0.5x if needed) and enforce gradient clipping at 0.1–0.2.
- Temporarily disable AMP to test; if stable, re-enable with tuned scaler.

3) Restore sustained performance
- Rate-limit logs; remove .item() in hot loops.
- Tune DataLoader workers and persist workers; ensure pin_memory=True.
- Avoid unnecessary torch.cuda.synchronize except for profiling.

---

## Suggested Code Changes (summary)

- full_energy_trainer.py
  - Ensure DataLoader generator device matches CUDA when shuffle (done), collate stacks on CPU (done), pin_memory depends on CUDA (done).
- energy_flow/training/energy_trainer.py
  - Move reset_peak_memory_stats to epoch-level or behind DEBUG_MEMORY flag.
  - Guard metric/logging calls to avoid frequent .item() and device syncs.
  - Add optional finite checks and early skip on non-finite losses.
  - Ensure no retain_graph=True in backward paths during accumulation.
  - Apply gradient clipping consistently after unscale_ when using GradScaler.
- energy_flow/core/energy_carrier.py
  - Optionally clamp/layernorm inputs to GRU; add finite guards in debug mode.
- Global
  - Consider enabling PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True during training sessions to reduce fragmentation.

---

## Verification Checklist

- [ ] No OOM on batch 2 at original batch size.
- [ ] Peak CUDA memory stabilizes across batches and epochs.
- [ ] No NaNs with anomaly detection off after initial debug.
- [ ] Per-batch time remains stable after batch 1 (±10–20%).
- [ ] GRU input/hidden ranges stay bounded; no runaway values.

If you’d like, I can implement the specific guarded logging/memory tweaks in energy_trainer.py and add finite checks around GRU inputs as a follow-up.

