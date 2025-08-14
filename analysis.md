Issue 1 — OOM on 2nd batch

- Text cache on GPU: Cached embeddings are stored/cloned on the same device as p
  roduced (GPU), growing per unique text and not freed between batches.
  - Evidence: `text_bridge/text_cache.py` → `put_text_to_surface()` stores `surf
ace_embedding.clone().detach()` without moving to CPU; `get_surface_from_text()`
    returns clones on same device; `load_cache()` moves to default device (could be
    CUDA).
  - Fix: Store cache on CPU and move on demand. - In `put_text_to_surface()`: `cloned = surface_embedding.detach().to('cpu',
 copy=True)`; store CPU tensors. - In `get_surface_from_text()`: return `.clone().to(request_device, non_bloc
king=True)` if caller is on GPU. - Optionally quantize cache (e.g., `to(torch.float16)` on CPU) and cap `max_
size`. - Easiest: disable cache during training (`config.text_cache_enabled=False`)
    and enable only for eval.
- Unnecessary grads for target surface: Target tensor for encoder loss is create
  d with `requires_grad_(True)`, which builds extra graph/grad storage and increas
  es memory.
  - Evidence: `energy_trainer.train_step()` sets `target_surface_input_grad = ta
rget_surface_input.clone().detach().requires_grad_(True)`.
  - Fix: Do not require grads on the target. Use the `no_grad`-computed target d
    irectly (detached): - Replace with `target_surface_input_detached = target_surface_input` (alrea
    dy no_grad) and compute `encoder_loss = mse(encoder_outputs, target_surface_inpu
t_detached)`.
- Memory fragmentation and aggressive cleanup:
  - Symptom: Error recommends `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    .
  - Fixes: - Set env var: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_col
lection_threshold:0.8`. - Reduce cleanup frequency and avoid calling `empty_cache()` too often; rais
    e `memory_threshold_gb` to ~24 on a 32 GB GPU; ensure only one cleanup block run
    s per step (the trainer has two similar cleanup sections — keep one).
- Tensorized storage size: Preallocated `TensorizedFlowStorage` keeps big tensor
  s resident for `max_active_flows`.
  - Fix: Verify `config.max_active_flows` is realistic for your batch/width/heig
    ht; lower it if not needed.
- Sanity: You already call `optimizer.zero_grad(set_to_none=True)` at the start
  of accumulation and `lattice.reset()` per forward—both good.

Issue 2 — Training slows down after first batch

- Decoder cost dominates: `CubeToTextDecoder.decode_surface()` is heavy (T5 gene
  ration) and logged ~1458 ms per batch.
  - Evidence: DEBUG_TRAINING log: “Text bridge ... time=1458.6ms”.
  - Fixes: - Don’t run decoder every step. Options: - Compute decoder loss every N steps (e.g., N=50) or on a small sub-batch
    (e.g., 2–4 samples). - Set its contribution very small or 0 during early training; keep only en
    coder MSE for now. - Run decoder in eval-only checkpoints, not inside the main training loop.
- Debug memory syncs: `_snapshot_memory()` calls `torch.cuda.synchronize()` when
  `DEBUG_MEMORY` enabled; called multiple times per step.
  - Fix: Keep DEBUG_MEMORY off in normal training, or gate snapshots more sparse
    ly; remove duplicated cleanup/snapshot block near end of `train_step`.
- Cache effects: Moving the text cache to CPU (Issue 1 fix) will also reduce dev
  ice allocation churn and implicit syncs.
- Anomaly mode and diagnostics: You enable anomaly detection for first 2 steps (
  fine). Ensure other diagnostics guarded tightly; gradient monitoring loops are a
  lready gated.
- Mixed precision stability can change kernel choices after warmup; see Issue 3
  bfloat16 recommendation for more stable/fast kernels.

- FP16 matmul instability in linear/GRU layers under autocast:
  - Likely sites: linear blocks in `core/embedding_mapper.py` (projection/recons
    truction with LayerNorm/Dropout), `core/energy_carrier.py` (projection heads), a
    nd possibly GRU internals.
  - Fixes (pick 1–2 high-impact first): - Use bfloat16 instead of float16: set `config.mixed_precision_dtype = torch
.bfloat16`. 5090 supports bf16 well and it greatly reduces NaNs with LayerNorm/L
    inear. - Keep LayerNorm in fp32: wrap mapper projections with `with torch.autocast(
device_type='cuda', enabled=False):` only around LayerNorm layers or set those s
    ubmodules to `.float()`. - Keep GRU in fp32 if necessary: either exclude GRU forward from autocast or
    set `dtype=torch.float32` for GRU weights and inputs.
- Target requiring grad for encoder loss (same as Issue 1) can exacerbate instab
  ility by creating unnecessary gradient paths.
  - Fix: Remove `requires_grad_(True)` from targets.
- Guard rails already present:
  - You clip gradients (config clamps to 0.1), reduce LR by 0.25 on init, and ha
    ve non-finite loss checks. Keep those.
  - Add pre-loss sanitization if needed: `cube_output_surface = torch.nan_to_num
(cube_output_surface, nan=0.0, posinf=1.0, neginf=-1.0)` and same for targets, b
    ut prefer finding the root cause first.

Quick, code-level changes to try (ordered)

- Disable GPU caching during training:
  - Config: `text_cache_enabled=False`.
  - Or modify `text_cache.py` to store on CPU (`to('cpu')`) and move back on dem
    and.
- Switch autocast to bf16:
  - Config: `use_mixed_precision=True`, `mixed_precision_dtype=torch.bfloat16`.
- Stop training the decoder each step:
  - In `train_step`, guard decoder work with `(self.global_step % N == 0)` or sa
    mple subset; or skip entirely until base is stable.
- Remove grads on targets:
  - Replace `target_surface_input.clone().detach().requires_grad_(True)` with a
    plain detached tensor.
- Reduce cleanup overhead:
  - Keep only one memory cleanup block in `train_step` and raise `memory_thresho
ld_gb` to ~24. Increase `memory_cleanup_interval` to 50–100.
  - Export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_
threshold:0.8`.
- If NaNs persist:
  - Exclude mapper’s LayerNorms from autocast (keep them fp32) and, if needed, G
    RU as well.
  - Temporarily set `gradient_scale_init` lower if using GradScaler (e.g., 2^8)
    to reduce overflow risk.
