# TCN/Decoder Deferred Readbacks — Implementation Plan and Fixes

Owner: EnergyFlowSwift
Status: Ready to implement (post Phase 6a)
Scope: Finish deferred readbacks across TCN/Decoder, align trainers/tests, and fix current compile-time issues.

## Context and goals

- Project principles: GPU-first, no silent CPU fallbacks, single GPU actor owning state, deterministic sync via batch-level fences.
- Current state (from docs/Async_GPU_Actor_Refactor_Plan.md):
  - GPUActor + GPUReadback + batch sync are in place.
  - Elementwise/Linear/Conv1D have Deferred APIs; KD/CE metrics run on GPU with Deferred readbacks and a guard against early value().
  - Trainers (CombinedTrainer/DecoderTrainer) partially use Deferred.
  - Remaining gap: TCN/Decoder paths still mix immediate ops, reducing batch-level deferral benefits.

## Immediate blocking fixes (compile and runtime correctness)

1) Fix actor-isolation error in GPUActor

- Error:
  - "Actor-isolated instance method 'enqueueHostReadback(label:execute:)' cannot be called from outside of the actor"
- File: Sources/EFCore/GPU/GPUActor.swift
- Issue: scheduleCommandBuffer’s completed-handler runs in a nonisolated closure and calls an actor-isolated method without awaiting a hop to the actor.
- Change: add `await` when calling `enqueueHostReadback` inside the Task body. No signature changes required.

Pseudo-diff:
```
// inside GPUActor.scheduleCommandBuffer(...)
commandBuffer.addCompletedHandler { [weak self] _ in
    guard let self else { return }
    Task { [weak self] in
        guard let self else { return }
-       self.enqueueHostReadback(label: label) {
+       await self.enqueueHostReadback(label: label) {
            if let error = commandBuffer.error {
                state.resolve(.failure(GPUActorError.commandBufferFailed(label: label, underlying: error)))
                return
            }
            do {
                let value = try produce()
                state.resolve(.success(value))
            } catch {
                state.resolve(.failure(error))
            }
        }
    }
}
```

2) Fix immediate Conv1D forward to prevent batch-guard fatalError

- File: Sources/EFCore/GPU/GPUActor+Conv1D.swift
- Issue: `conv1DForward(...)` calls the Deferred variant and then awaits `value()` while a batch may be active; the readback requires `syncBatch()` first and will fatalError.
- Change: pass `deferUntilSync: false` from the immediate wrapper so that the readback does not require a batch sync.

Pseudo-diff:
```
public func conv1DForward(..., x: Tensor) async throws -> Tensor {
-    let readback = try await conv1DForwardDeferred(..., x: x)
+    let readback = try await conv1DForwardDeferred(..., x: x, deferUntilSync: false)
     return try await readback.value()
}
```

3) Resolve EFTextEval @main vs top-level code conflict

- Error:
  - "'main' attribute cannot be used in a module that contains top-level code"
- File: Sources/EFTextEval/main.swift
- Likely cause: The EFTextEval target is correct, but Swift complains if any file in the EFTextEval module contains top-level executable statements when `@main` is present. Our EFTextEval looks clean, but this error can appear if:
  - Another source file is (incorrectly) included in the EFTextEval target and contains top-level statements; or
  - The compiler treats a file named `main.swift` with `@main` as conflicting under some configurations.
- Actions:
  - Ensure EFTextEval target includes only Sources/EFTextEval/* (Package.swift already does this) and that no file in that target has top-level executable statements.
  - As a robust workaround, rename `Sources/EFTextEval/main.swift` → `Sources/EFTextEval/AppMain.swift` (keeping the `@main` entry type). This avoids any confusion about implicit main stemming from the filename.
  - Rebuild (manually by user) to confirm the error is gone.

## Complete deferred coverage in core ops (Phase A)

- Add Deferred APIs for GELU and LayerNorm (parity with Elementwise/Linear/Conv1D):
  - `geluForwardDeferred(x:deferUntilSync:) -> GPUReadback<Tensor>`
  - `layerNormForwardDeferred(x:gamma:beta:eps:deferUntilSync:) -> GPUReadback<Tensor>`
  - Keep existing immediate methods as thin wrappers calling Deferred with `deferUntilSync: false` and awaiting value().

## Propagate deferred readbacks through TCN/Decoder (Phase B)

- TCNBlock.forward:
  - Switch to `conv1.forwardDeferred`, `geluForwardDeferred`, `conv2.forwardDeferred`.
  - Use `residualAddDeferred` and `maskZeroDeferred`.
  - Collect all GPUReadback<Tensor> and await after trainer-level `syncBatch()`.
- TCNStack.forward:
  - Thread through Deferred results block-by-block, returning a readback or deferring readback resolution until higher-level sync.
- TextDecoder.forward / forwardForTraining:
  - `condProj.forwardDeferred`, `addBroadcast2DInto3DDeferred`, TCN path uses Deferred, `outProj.forwardDeferred`.
  - Return readbacks or await only after an external `syncBatch()` depending on call site.
- TextToCubeEncoder.encodeTokens / forwardForTraining:
  - Keep projection Deferred path; switch maskedMean to `maskedMeanDeferred` and defer resolution.

Note: To simplify API surface, we can keep model-level forward as async returning a Tensor, but internally push the awaits after a `syncBatch()` done by the caller (e.g., trainers). Alternatively, add explicit `forwardDeferred` variants at model level.

## Trainers alignment (Phase C)

- CombinedTrainer.stepA/stepB:
  - Use `enc.projectionInputGradientsGPUDeferred` (already available) rather than the immediate method; await after `syncBatch()`.
  - Use `gpu.maskedMeanBackwardDeferred` via encoder helper where applicable; await after `syncBatch()`.
  - Already good: KD/CE metrics are Deferred and awaited post-`syncBatch`.

## Tests update (Phase D)

- Remove or rewrite legacy LNExecCacheTests (ExecutableCache is gone).
- Update GPU tests to call GPUActor APIs (prefer Deferred) instead of static EFCore/Metal wrappers (ElementwiseGPU, GELUGPU, LayerNormGPU).
- Convert affected tests to async and use `beginBatch()` / `syncBatch()`.
- Add a guard test: calling `value()` on a Deferred readback before `syncBatch()` should trap (use an Expect-crash style test if supported).

## Docs update (Phase E)

- docs/Async_GPU_Actor_Refactor_Plan.md: mark Phase 6b items as addressed as work lands (TCN/decoder deferred; tests updated).
- docs/Deferred_Readbacks_TCN.md: reflect the concrete API changes (new Deferred for GELU/LN, model-level changes).
- docs/EFTrain_Optimization_Plan.md: mark CombinedTrainer unification done; prune the old text-mode lastGrads bug note.

## Optional instrumentation (Phase F)

- Under DEBUG, annotate per-op timings inside GPUActor (e.g., timestamps around command buffer commit/complete) and include batch labels in logs to localize stalls that moved to syncBatch.

## Rollout sequence (small PRs)

1) Immediate fixes (compilation + guard):
   - GPUActor await hop in scheduleCommandBuffer.
   - Conv1D immediate wrapper `deferUntilSync: false`.
   - EFTextEval rename to AppMain.swift if @main error persists.
2) Add Deferred APIs for GELU and LayerNorm (no call-site changes yet).
3) TCNBlock/TCNStack switch to Deferred ops internally; keep public signature stable (await near the call site after syncBatch).
4) TextDecoder and TextToCubeEncoder switch to Deferred for elementwise/LN/GELU/conv.
5) Trainers: exclusive Deferred usage for projection input grads and maskedMeanBackward; centralize sync points.
6) Tests: remove legacy LNExecCache test, update others to actor APIs and async.
7) Docs: mark progress and finalize Phase 6b.

## Risks and mitigations

- Hidden stalls concentrated at syncBatch: add optional DEBUG per-op timing.
- Buffer lifetimes and reuse: rely on GPUActor buffers cache; label buffers consistently; consider generation tokens to bound cache.
- API churn: add Deferred variants first, then migrate call sites incrementally to keep builds green.

## Acceptance criteria

- No compiler errors, including:
  - No actor isolation violations in GPUActor.
  - EFTextEval builds clean with @main and no top-level code conflicts.
- TCN/Decoder forward paths run without per-op waits; awaits happen after a single `syncBatch()` per micro-batch.
- Trainers use Deferred for projection input grads and maskedMeanBackward, awaiting after sync.
- Tests pass; legacy LNExecCache test removed or updated; add a guard test for early readback trap.
- Throughput improvement observed on TCN-heavy batches (expect better GPU utilization by deferring readbacks).
