# Swift Port — Phase 1 Summary and Next Steps (Beginner-friendly)

This note summarizes what we’ve done so far and lays out a small-steps plan for the next phase. It’s written to be explicit and easy to follow if you’re new to Swift and ML infra.

—

## What we accomplished in Phase 1

1) Project scaffolding
- Created SwiftPM package EnergyFlowSwift with targets:
  - EFCore: base tensor type and low-level utilities (CPU-only for now)
  - PyTorchSwift: small PyTorch-like building blocks (Embedding, Linear, LayerNorm, Activations)
  - EnergyFlow: higher-level architecture modules starting with TextBridge

2) Minimal Tensor (EFCore/Tensor/Tensor.swift)
- Shape + flat data storage
- Utilities: zeros, randomUniform, strides, flattenIndex

3) PyTorch-like modules (PyTorchSwift/*.swift)
- Embedding: lookup [vocab, d] -> [B, L, d]
- Linear: naive CPU matmul + bias for 2D inputs [B, in] -> [B, out]
- LayerNorm: per-last-dim normalization for [B, dim]
- Activations: GELU (approx), Tanh (scalar)

4) Text Bridge: minimal forward
- Simple whitespace tokenizer that returns ids + attentionMask with padding
- TextToCubeEncoder: tokenize -> Embedding(256) -> add sinusoidal positional encoding -> (Transformer placeholder) -> masked-avg -> MLP(256->256->surface_dim) + tanh
- Config: DEBUG/EXPERIMENT/OPTIMIZED lattice sizes so surfaceDim = width * height

5) Tests
- A unit test validates output shape [B, surfaceDim] and value range [-1, 1]

Result: `swift build` and `swift test` pass on CPU. We now have a working, modular skeleton that’s easy to extend.

—

## Key ideas to remember (mental model)

- Tensor: a multi-dimensional view over a flat array of Float. Indexing by shape + strides.
- Embedding: a table that maps token ids to learned vectors (one row per token).
- Positional encoding: adds “token position” information to embeddings so the model can learn order.
- Masked mean: average only over real tokens (mask = 1) and ignore padding (mask = 0).
- MLP projection to surface: takes a sentence embedding [B,256] and maps to [B, surfaceDim] then tanh.
- Modular design: each block (Tokenizer, Embedding, LayerNorm, Linear) is a small building unit that we can test and swap.

—

## Next Phase: small, modular steps

Goal: prepare the encoder for training on a text->embedding dataset, keep everything clear and incremental.

Phase 2 outline (small steps):

Step 2.1 — Add shape helpers and simple logging
- Add EFCore helpers for common reshapes/checks (assertShape, prettyShape)
- Add a very simple Logger (print-based) that can be turned on/off
- Why: makes debugging shapes and data flow much easier

Step 2.2 — Stabilize initialization and configs
- Add explicit seed control for all layers
- Move hiddenDim, maxLength, and surfaceDim access behind a Config struct used by TextToCubeEncoder
- Why: reproducible results and clear knobs

Step 2.3 — Dataset I/O (Option A: Swift-native, Option B: Python-precomputed)
- A. Swift-native minimal dataset:
  - A small CSV/JSON reader that yields: {text: String, target: [Float] (embedding)}
  - Tokenize on the fly in Swift
- B. Python-precomputed dataset:
  - Use your Python tools to pre-generate token ids AND/OR target embeddings into .npz/.jsonl files (see discussion below)
  - Load them in Swift (no tokenization required here)
- Why: training needs (text, target). In B, we keep Swift simpler at first.

Step 2.4 — Training utilities (CPU first)
- Add a minimal loss (MSE) and an optimizer (AdamW)
- A very small training loop: for batch in dataset -> forward -> loss -> backward (placeholder) -> update
- Note: before autodiff, we can simulate training via numeric gradients or stub a no-op “training loop” to test I/O and batching
- Why: wiring the loop clarifies interfaces we need before adding true gradients/MPSGraph

Step 2.5 — Real TransformerEncoder (CPU reference)
- Implement a simple Multi-Head Attention and FFN with residual + LayerNorm
- Keep batch_first = true semantics
- Hook into TextToCubeEncoder (replacing the placeholder)
- Why: correctness and ready-to-port structure for the GPU path

Step 2.6 — MPSGraph path (inference)
- Wrap matmul/gelu/layernorm/softmax with MPSGraph
- Build an MPSGraph-based encoder forward
- Keep shapes fixed via padding for graph caching
- Why: get the speed path ready, then add autodiff later

Step 2.7 — Autodiff and optimizer on GPU (later)
- Use MPSGraph gradients(of:with:) where possible
- Keep CPU fallback for debugging

You can stop at any checkpoint (e.g., after Step 2.3) and still have a usable system for testing data plumbing.

—

## Tokenizer strategy — Do we need Swift WordPiece?

You have two pragmatic options:

- Option A: Implement WordPiece in Swift
  - Pros: Truly standalone Swift pipeline, identical input specs (if you load the same vocab.txt)
  - Cons: More code to write and test now; higher complexity if you’re still new to Swift

- Option B: Precompute in Python (recommended to start)
  - Use your existing Python code (e.g., a script like `generate_energy_dataset.py`) to output either:
    1) token ids + attention masks (so Swift Embedding sees the same inputs as PyTorch); or
    2) even better for “text->embedding” training: precomputed target embeddings per text (e.g., from a reference encoder)
  - Store as JSONL/CSV/NPZ, e.g.:
    {"text": "...", "input_ids": [...], "attention_mask": [...], "target": [ ... 256 floats ... ]}
  - Pros: Faster to start training; aligns with your idea to train TextToCubeEncoder on (text, embedding) pairs
  - Cons: You rely on a Python preprocessing step (which is fine in research)

Given your note, I recommend Option B for now: precompute a dataset with (text, target_embedding) via Python. Then, in Swift, you train the projection (and later the full encoder) to match these targets. The “projection to cube” can wait until the lattice is ready.

—

## Concrete next steps (actionable)

1) Decide dataset path: B (Python-precomputed)
   - Extend your Python side to export a small demo dataset: a JSONL with fields text, target embedding (256 floats). If you want identical tokenization, also export input_ids and attention_mask.

2) Add a Swift dataset loader
   - EnergyFlow/Sources/EnergyFlow/Dataset/SimpleJSONLDataset.swift
   - Reads N lines, each with {text, target: [Float]} (and optionally input_ids, attention_mask)
   - Returns batches of (texts or ids/mask, targets)

3) Wire a stub training loop (CPU)
   - Implement MSE loss (no autograd yet): compute loss value only to test batching and forward piping
   - Print mean/std of outputs and loss to validate signal flow

4) Implement a minimal TransformerEncoder (CPU)
   - MHA (Q/K/V linears, attention weights via softmax, value aggregation)
   - FFN (Linear -> GELU -> Linear)
   - Residual + LayerNorm
   - Replace the placeholder in TextToCubeEncoder

5) Once stable, add an MPSGraph forward path for inference
   - Keep shapes fixed with padding
   - Compare speed vs CPU on small inputs

This sequence keeps steps small, testable, and beginner-friendly.

—

## FAQ

- Why not training right now?
  - We need gradients. We can either do MPSGraph autodiff or write a tiny autograd. Both are a bit of work. It’s fine to validate the data path and model forward first.

- Can we train only the projection while keeping embeddings frozen?
  - Yes. If your dataset provides a 256-dim target per text, you can train only the final MLP to match it first, then “unfreeze” earlier layers later.

- When to implement WordPiece?
  - After you have a training loop running with precomputed inputs. Then you can decide if the benefits outweigh the complexity.

—

If you agree, I’ll proceed with Step 2.3 (Swift dataset loader for JSONL), then a stub training loop (loss-only, no gradients), keeping changes very small and self-contained.
