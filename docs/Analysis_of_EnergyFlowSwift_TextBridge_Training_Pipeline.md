Analysis of EnergyFlowSwift TextBridge & Training
Pipeline
Below is a prioritized list of recommended improvements, fixes, and optimizations for the TextBridge and
EFTrain pipeline. The list is sorted by highest impact vs. lowest implementation effort:

1.  Fix Gradient Application in Text-Mode Training – In the current training loop, the text-mode
    branch does not apply accumulated gradients for the last TCN block, even when
    unfreezeLastTCN is enabled. In the token-mode branch, accumulated gradients for the last block
    are packaged and passed into optimizerStepProjectionAndLastBlock , but in text-mode the
    call passes lastGrads: nil (meaning last-block updates are skipped) 1 2
    . This is likely an
    oversight from duplicating code. Fix: unify the logic so that if unfreezeLastTCN is true, the last
    block’s gradients are accumulated and applied in text-mode as well. This one-line change ensures
    the last TCN block is updated consistently, preventing a silent bug where text-mode training never
    updates the last block weights.
2.  Optimize the Embedding Lookup – The Embedding.forward method currently uses nested
    loops to copy each token’s embedding vector element-by-element 3
    . This is a performance hotspot
    for larger batch sizes or sequence lengths. Fix: replace the inner loop with a vectorized copy. For
    example, use memcpy or Swift’s withUnsafeBytes to copy an entire embedding row in one go
    (256 floats) instead of 256 scalar assignments. This alone can drastically speed up embedding on
    CPU with minimal code changes. In the longer term, consider moving the embedding lookup to GPU
    (e.g. using an MPSGraph matrix multiply of one-hot token indicators with the embedding matrix, or
    a custom Metal kernel) for further speedup, but a simple memory copy optimization yields a big win
    for little effort.
3.  Refactor Duplicate Training Loop Code – The training loop has two largely duplicate code paths for
    tokenized vs raw-text datasets 4 5
    . This duplication not only led to the bug above, but also
    increases maintenance effort. Improvement: refactor these into a single function or loop that
    handles both cases. For example, perform tokenization (if needed) up front and then run a unified
    batch-processing loop. This guarantees both modes stay in sync feature-wise and fixes like the last-
    block gradient update apply to both. It reduces code size and the chance of future inconsistencies.
4.  Increase GPU Utilization by Fusing Operations – Currently, many operations in the forward pass
    are offloaded to Metal/MPS, but intermediate results are copied back to CPU between steps. For
    example, after 1D convolution, the code calls cmd.waitUntilCompleted() and reads the output
    into a CPU Tensor 6
    , then immediately sends it back to GPU for the next step (GELU, residual
    add, etc.). These host transfers add overhead. Improvement: keep data on the GPU through the
    whole forward pass by fusing operations or using MPSGraph to create a single graph. In fact, a
    7 8
    cached MPSGraph executable for LayerNorm→GELU→MatMul has been introduced .
    Leverage this to perform layer norm, activation, and linear projection in one GPU graph call.
    Similarly, consider fusing the last TCN block’s LN + conv1 + GELU + conv2 into one MPSGraph or
    1
5.
6.
7.
8.
9.  Metal kernel. By reducing GPU<->CPU handoffs, you’ll maximize M4 GPU throughput and lower
    training time.
    Use Asynchronous GPU Calls and Concurrency – The current implementation often performs GPU
    work sequentially, waiting for each command buffer to complete before preparing the next batch.
    The Apple M-series GPUs support asynchronous command execution. Improvement: dispatch GPU
    operations without immediate waitUntilCompleted() , and prepare the next micro-batch on the
    CPU in parallel. For example, you can enqueue the Metal command buffer for one micro-batch and
    then start assembling the next batch’s data/host tensors while the GPU runs in background. Using
    Swift Concurrency or GCD to overlap CPU and GPU work can hide latency. Similarly, dataset loading
    and shuffling (which currently happens in the main thread) could be done in the background. The
    JSONL parsing loop could be parallelized or replaced with a faster parser 9
    . These changes are
    more involved but can yield substantial speedups by utilizing all CPU cores and the GPU concurrently
    (especially important as the CPU still handles tokenization and some prep).
    Extend GPU Offloading to Remaining Ops – A few operations are still on CPU that could be moved
    to GPU with minimal risk. For instance, embedding gradient update and layer norm backward are
    currently done on CPU. Apple’s MPSGraph or Metal kernels could compute these gradients on GPU
    (the session notes suggest exploring MPSGraph autodiff for conv/linear) 10 11
    . Also, bias addition
    for linear layers is noted to occur on CPU in some paths 12
    – folding the bias add into the GPU
    matmul would eliminate that overhead. These tweaks would push nearly the entire training math
    onto the GPU. The benefit is faster training and simpler code (no special-case CPU math), at the cost
    of writing or integrating a few additional GPU kernels or MPSGraph ops.
    Memory Pooling and Buffer Reuse – The project already introduced a BufferPool for Metal
    buffers. To further reduce overhead and memory fragmentation, consider using MTLHeaps or a
    single MPSGraph MPSNDArray buffer reused across iterations 13
    . Since tensor shapes (B, L, D,
    etc.) are mostly fixed during training, you can allocate GPU buffers once and reuse them for each
    batch, rather than allocating/freeing every time. This also ensures better cache locality. Similarly,
    caching the FP16 weight buffers (as is done in GraphLinear and GraphConv1D ) should be
    extended so they persist between training steps. The session summary confirms strides were made
    here (aligned row bytes, persistent MPSNDArray for cached executables) 14 15
    . Continuing this
    work – e.g., never deallocating and reallocating large buffers inside the training loop – will improve
    speed and lower peak memory use.
    Improved Logging and Profiling – The current logger uses Apple’s unified os.Logger and
    supports log levels 16 17
    . To aid debugging and optimization, consider enhancing the training log
    output. For example, log the time per epoch or per 100 batches, and the throughput (samples/
    second), to identify performance bottlenecks over time. You might also log GPU memory usage if
    possible, or cache hits for the new MPSGraph executables. This can be done at info level so as not to
    flood debug logs. Additionally, adding a lightweight metrics logger (even just writing CSV of epoch,
    train/val loss, etc.) can help in analyzing training curves outside of Console. These changes are low-
    effort and can greatly improve developer insight into the model’s behavior (the roadmap even calls
    18 19
    for “improved logging” and validation metrics in Phase 6) .
    Tuning Hyperparameters for Stable Initial Training – To get the best results on the first runs,
    some configuration defaults could be adjusted:
    2
10. Learning Rate & Warmup: Ensure a proper LR warmup is used. The config supports warmupSteps
    and cosine decay; a warmup of around 500–1000 steps is often helpful to avoid unstable initial
    updates 20
    . Verify that the default config sets this (e.g. 0.0 to some value); if not, adjusting it can
    improve stability.
11. 21
    Gradient Clipping: You have global L2 grad clipping ( clipNorm ) implemented – using a
    modest clip (e.g. 1.0 or 2.0) can prevent rare spikes when training the projection. It’s good to keep
    18
    this on by default (the roadmap also noted this) .
12. 22 23
    Batch Sizes: The default micro-batch of 32 is already chosen for GPU efficiency . You might
    experiment with higher if memory permits, since Apple GPUs often benefit from larger matmul
    workloads.
13. Loss Weighting: Monitor the α (cosine) vs β (MSE) weighting. A high α (e.g. 1.0) accelerates cosine
    similarity gain, but if MSE is also important, you might start with a balanced ratio (say 0.5 and 0.5) or
    gradually increase α. The current default of α=1, β=1 (if that’s the case) is reasonable, but any
    extreme imbalance could slow improvement of the other metric.
    These adjustments don’t require code changes, just config tweaks or smarter defaults in
    TrainConfig . They can make initial training more forgiving and faster to converge.
14. Tokenizer and Data Pipeline Enhancements (Longer-term) – The SimpleTokenizer in use is basic.
    For better text coverage and consistency with Python prototypes, integrating a more advanced
    tokenizer is worthwhile (e.g. HuggingFace’s swift-tokenizers or Apple’s new TokenizerKit, as
    noted in the TextBridge plan 24
    ). A trained BPE or SentencePiece model would ensure stable vocab
    and let you reuse Python tokenization logic. This improves the reversibility of Text→Cube→Text
    (one of the project goals) 25
    . It’s a moderate effort (adding a dependency or bridging a vocab), but
    not urgent for performance. Additionally, consider streaming the dataset from disk or using lazy
    loading if memory becomes a concern. The current SimpleJSONLDataset loads everything into
    RAM upfront 9 26
    ; for very large datasets, a lazy iterator could save memory. These changes will
    future-proof the pipeline as you scale up data and integrate with external NLP tooling.
    Each of these suggestions aims to either boost performance (through greater GPU usage and parallelism)
    or improve training reliability and maintainability with minimal changes. Implementing the top items
    (bug fix, embedding optimization, code unification, and GPU fusions) should yield the most immediate
    benefit for the least effort. Subsequent items like fully GPU-driven training and hyperparameter tuning will
    further streamline the training pipeline on macOS M4. By addressing these, the project will train faster, use
    resources more efficiently, and be easier to extend with new features.
    Sources: The recommendations are based on analysis of the EnergyFlowSwift code and project docs,
    including the training loop in EFTrain/main.swift 1 2 3
    , the Embedding implementation , GPU kernel
    usage in GraphConv1D and GraphLinear 6 14 8 11
    , and the project’s own roadmap discussions
    which highlight ongoing efforts in GPU optimization and training loop improvements. These references
    show the current behavior and planned direction, supporting the changes suggested above.
    1 2 4 5 /EnergyFlowSwift/Sources/EFTrain/main.swift
    3 /EnergyFlowSwift/Sources/PyTorchSwift/Embedding.swift
    6 /EnergyFlowSwift/Sources/EFCore/MPSGraph/GraphConv1D.swift
    7 8 10 12 13 14 15 /docs/Session_Summary_2025-10-07.md
    9 26 /EnergyFlowSwift/Sources/EnergyFlow/Dataset/SimpleJSONLDataset.swift
    11 18 19 20 22 23 /docs/Session_Summary_2025-10-03.md
    16 17 /EnergyFlowSwift/Sources/EFCore/Logging/Logger.swift
    21 /EnergyFlowSwift/Sources/EnergyFlow/Training/OptimStep.swift
    24 25 /docs/TextBridge_MinimalPipeline_Plan.md
