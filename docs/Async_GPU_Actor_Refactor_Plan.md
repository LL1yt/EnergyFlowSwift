# Async/Actor GPU Refactor Plan (EnergyFlowSwift)

Owner: EnergyFlowSwift
Status: Draft
Scope: Make GPU execution actor-isolated and async, remove nonisolated hacks, keep deterministic waits by default, and set foundations for future overlap.

Goals
- Correct Swift Concurrency model for GPU code: no main-actor globals, no nonisolated(unsafe) statics for mutable GPU state.
- Centralize device/queue/pipeline state in a single GPU actor.
- Propagate async only where necessary, with minimal ceremony, but consistently across hot paths.
- Preserve current numerical behavior and determinism: keep waitUntilCompleted by default inside GPU calls.
- Avoid feature flags and fallbacks; optimized path becomes the only path.
- Stay aligned with project rules: simple/efficient, MPS/Metal first, no MPSGraph on hot paths.

Non-Goals (for this iteration)
- Overlapped micro-batch execution (no-wait between ops) — can be a follow-up after functional migration.
- Changing math or training semantics.
- Introducing runtime flags for toggling implementations.

Design Overview
- Introduce a GPUActor that owns:
  - MTLDevice and MTLCommandQueue
  - Lazy-compiled Metal pipelines (LayerNorm, GELU, Elementwise, Im2Col/Col2Im, etc.)
  - MPSMatrix helper allocation utilities
  - A BufferPool replacement with actor-owned cache (label → MTLBuffer)
  - Optional weight/buffer caches keyed by stable identifiers
- Move existing GPU helpers (ElementwiseGPU, GELUGPU, LayerNormGPU, EmbeddingGPU, Im2Col/ConvPack) into methods on GPUActor.
- Refactor MPSMatrix wrappers (GraphLinear, GraphConv1D) so that any GPU buffers and encodes flow through GPUActor. The structs retain CPU-side weights; the actor manages MTLBuffers pinned/cached as needed.
- Make forward/grad methods async across layers that touch GPU; await at callsites.
- Keep waitUntilCompleted before returning from each actor GPU op to preserve determinism.

Step-by-step Plan

Phase 0 — Stabilize build (short-lived)
- Revert/avoid nonisolated(unsafe) where it clashes with compiler actor rules.
- Keep temporary build hacks only until phases below land; remove them as we move functions under GPUActor.

Phase 1 — Introduce GPUActor
- New file: Sources/EFCore/GPU/GPUActor.swift
  - class GPUActor: actor
    - let device: MTLDevice
    - let commandQueue: MTLCommandQueue
    - private lazy var pipelines: structs holding MTLComputePipelineStates per domain (elementwise, LN, GELU, im2col, etc.) compiled on demand
    - private var buffers: [String: MTLBuffer] buffer pool (rounded lengths)
    - func buffer(length: Int, label: String) async -> MTLBuffer
    - Helpers to ensure pipelines (compile once)
- Provide a global entry point:
  - nonisolated(unsafe) enum GPU { static let shared = GPUActor() }
  - Note: GPU.shared is a reference to the actor; methods must be awaited.

Phase 2 — Port Metal helpers to GPUActor
- ElementwiseGPU → methods on GPUActor:
  - residualAdd(y: Tensor, x: Tensor) async -> Tensor
  - maskZero(y: Tensor, mask: [[Int]]) async -> Tensor
  - maskedMean(x: Tensor, mask: [[Int]]) async -> Tensor
  - maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) async -> Tensor
  - addBroadcast2DInto3D(y: Tensor, addBD: Tensor, L: Int) async -> Tensor
- GELUGPU → GPUActor.geluForward(x: Tensor) async, geluBackward(x: Tensor, dY: Tensor) async
- LayerNormGPU → GPUActor.layerNormForward(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float) async; backward async
- Im2Col/ConvPack kernels → GPUActor.im2col/col2im/fillBias + pack/unpack helpers (async)
- EmbeddingGPU → GPUActor.embeddingForward(ids: [[Int]], weight: Tensor) async
- Delete/empty original static singletons and move code under actor; fix imports accordingly.

Phase 3 — Refactor MPSMatrix wrappers via GPUActor
- GraphLinear
  - Remove internal MTLBuffer fields (wBufFP16, bBufFP16) from the struct or make them opaque tokens.
  - Actor owns weight buffers keyed by ObjectIdentifier(self) + version.
  - Public API:
    - mutating func forward(_ x: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor
    - func gradientsGPUAsync(X: Tensor, dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> (dW: Tensor, dB: Tensor)
    - func inputGradientsGPUAsync(dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor
  - Invalidate cache: bump version token so actor recreates buffers next call.
- GraphConv1D
  - Actor manages Wcol FP16 buffers and im2col intermediates.
  - public func forward(_ x: Tensor, on gpu: GPUActor = GPU.shared) async -> Tensor

Phase 4 — Propagate async across model stack
- TCNBlock.forward → async, awaiting LN, convs, GELU, elementwise ops.
- TCNStack.forward → async.
- TextDecoder.forward/forwardForTraining → async; condition-add via actor method.
- TextToCubeEncoder.encodeTokens/forwardForTraining/forwardForTrainingWithLastBlockCache → async.
- LastTCNBackward and DecoderLastTCNBackward that use GPU ops → async helpers or ensure GPU calls are awaited.

Phase 5 — Trainers and EFTrain async
- CombinedTrainer.stepA/stepB → async; await encoder/decoder calls and GPU grad ops.
- DecoderTrainer.step/stepScaled → async.
- EFTrain.run → async (convert to @main async entry or wrap with Task and RunLoop).
- Update tests to async XCTest (use async test functions and await calls).

Phase 6 — Deterministic waits (default behavior)
- In every GPUActor method, commit command buffer and waitUntilCompleted before returning.
- Provide internal helper encodeAndWait(label:) to unify command buffer lifecycle and status checks.
- Optional future step (not now): batched encode without waits and a sync fence at batch end — gated by internal design, not flags.

Phase 6b — Batched waits and GPU metrics (remove local waits)
Pre-req: Phase 6 is complete and stable
Goals:
- Remove per-op waitUntilCompleted within GPUActor methods; synchronize once per micro-batch/batch.
- Move runtime metrics to GPU (return scalars):
  - Mode A: MSE and cosine similarity means.
  - Mode B: CE mean over [B,L].
- Avoid CPU readbacks of large tensors mid-graph; only read back tiny scalars or finalize at barriers.

Tasks:
- GPUActor synchronization API
  - func beginBatch() async: optional no-op marker (for future multi-buffer).
  - func syncBatch(label: String? = nil) async: waits for all enqueued command buffers since beginBatch (or last sync) to complete.
  - Internals: keep a list of pending command buffers (weak), clear list on sync.
- GPU metrics kernels/APIs
  - func kdMetricsMean(y: Tensor, t: Tensor) async -> (mse: Float, cos: Float)
    - Implement pairwise rowwise ops + reductions on GPU; return host scalars.
  - func ceMean(logits: Tensor, targets: [[Int]]) async -> Float
    - Softmax+NLL on GPU; return host scalar.
  - Phase 6b keeps training grads implementation unchanged (dLogits on CPU as is). Optional future: dLogits GPU.
- Call-site refactors (no per-op waits)
  - Replace local waits in GELU/LN/Conv/Linear helpers with simple commit (no wait).
  - CombinedTrainer/DecoderTrainer/Encoder paths:
    - Within micro-batch loop: only enqueue GPU ops; compute metrics via GPUActor.*Metrics*.
    - At batch end: await gpu.syncBatch(); then log and update running stats.
  - Ensure no CPU tensor reads before syncBatch.
- Safety & validation
  - Add asserts to detect CPU readbacks of GPU-produced buffers prior to sync.
  - Compare metrics to prior CPU versions within tolerance; keep CPU fallback test harness during migration only.
- Logging
  - Throughput should be measured around sync points (true effective rate), not per-op.

Risks & mitigations
- Hidden stalls move to syncBatch: harder to locate slow ops → add optional per-op timestamps (cmdBuf GPUStart/GPUEnd) under DEBUG.
- Buffer lifetime: ensure actor retains buffers until sync; use label-based cache with generation tokens.

API sketches (illustrative)
```swift path=null start=null
extension GPUActor {
    func beginBatch() async {}
    func syncBatch(label: String? = nil) async {
        // wait on pending command buffers, clear list
    }
    func kdMetricsMean(y: Tensor, t: Tensor) async -> (Float, Float) { /* GPU reductions */ }
    func ceMean(logits: Tensor, targets: [[Int]]) async -> Float { /* GPU softmax+NLL */ }
}

// Trainer pseudo-flow
for batch in batches {
    await gpu.beginBatch()
    // enqueue forwards/backwards (no waits)
    let (mse, cos) = await gpu.kdMetricsMean(y: out, t: zt)
    let ce = args.enableAB ? await gpu.ceMean(logits: logits, targets: targets) : 0
    await gpu.syncBatch(label: "train-batch")
    log(mse, cos, ce)
}
```

Phase 7 — Cleanup deprecated pieces
- Remove EFCore/MPSGraph/ExecutableCache.swift and GraphContext if no longer used.
- Remove BufferPool.swift; rehome logic in GPUActor.
- Remove nonisolated(unsafe) hacks from earlier stopgaps.

Compatibility & Migration Notes
- This is a source-breaking change: forward methods become async across the stack.
- We will stage changes in small PRs, each updating call sites and tests.
- Keep numerical parity by asserting shapes and running existing unit tests after each phase.

Testing Strategy
- Convert affected unit tests to async and verify:
  - EFDecoderModeBMiniEpochTests
  - EFTrainMiniEpochTests and CombinedABAlternationTests
  - LNGeLUGemmCacheTests, LNExecCacheTests — will be removed or replaced with actor-based LN/GELU tests
  - GPUKernelsAndStrideTests, Im2ColCol2ImParityTests — rewire to GPUActor APIs
- Add microbench tests (optional) to confirm no regression in throughput; still using waits.

API Sketches (illustrative)
```swift path=null start=null
actor GPUActor {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    // lazy pipelines, buffers cache

    func residualAdd(y: Tensor, x: Tensor) async -> Tensor {
        let (cmd, yBuf, xBuf) = try await encodeResidualAdd(y: y, x: x)
        cmd.commit(); cmd.waitUntilCompleted()
        return readTensor(shape: y.shape, from: yBuf)
    }
}

struct GraphLinear {
    var inFeatures: Int
    var outFeatures: Int
    var weight: Tensor
    var bias: Tensor?
    private var version: UInt64 = 0

    mutating func invalidateCache() { version &+= 1 }

    mutating func forward(_ x: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        try await gpu.linearForward(self, x: x, version: version)
    }
}
```

Risk & Mitigation
- Actor reentrancy: keep GPUActor methods short and non-blocking except for final waits; avoid calling back into actor from callbacks.
- Memory pressure: reuse buffers in actor buffer cache; round sizes; monitor alignment.
- Large PR risk: split work by domains (Elementwise/LN/GELU → Conv → Linear → Models → Trainers → EFTrain).

Rollout Sequence (PR-size friendly)
1) Add GPUActor + buffer cache + minimal residualAdd (wire one call site) — ensure build.
2) Port Elementwise entirely; update TCN/TextDecoder calls; tests.
3) Port GELU and LayerNorm; update TCN; tests.
4) Port Im2Col/Col2Im; update GraphConv1D; tests.
5) Port Embedding; update Embedding.forward; tests.
6) Refactor GraphLinear; update encoder/decoder proj paths; tests.
7) Propagate async through models/trainers; convert EFTrain to async entry; tests.
8) Phase 6b: Introduce GPU metrics + batched waits; remove per-op waits; add trainer sync points; tests for parity.
9) Cleanup MPSGraph, BufferPool; remove hacks; green build.

Performance Notes (with waits)
- Even with waitUntilCompleted, we remove CPU loops and centralize GPU resource reuse; expect net improvements vs current baseline.
- Future: after functional parity, introduce overlapped micro-batch execution by deferring waits and syncing at batch end.

Post-migration Enhancements (later)
- Pin embedding weights in GPUActor to avoid re-upload per batch when not trainable.
- Add fused LN→GELU→GEMM kernels where dimensions are static.
- Introduce command buffer batching per micro-batch with a single fence.

Checklist
- [ ] GPUActor.swift landed
- [ ] Elementwise ported & call sites updated
- [ ] GELU/LN ported
- [ ] Im2Col/Col2Im ported
- [ ] Embedding ported
- [ ] GraphLinear refactored
- [ ] GraphConv1D refactored
- [ ] Models async
- [ ] Trainers async
- [ ] EFTrain async main
- [ ] Tests async + green
- [ ] Phase 6b: GPU metrics + batched waits (no per-op waits)
- [ ] MPSGraph/BufferPool removed
- [ ] Nonisolated hacks removed

Timeline (rough)
- Phases 1–2: 1–2 sessions
- Phases 3–4: 1–2 sessions
- Phases 5–7: 1–2 sessions

Notes
- We will keep semantics identical (waits) to simplify verification.
- All public APIs affected will be updated in one direction (no flags).
