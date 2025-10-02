# TCN + FP16 GPU Port: Status and Next Steps

Date: 2025-10-01

This document summarizes the status we want to carry forward into a fresh session and the concrete next steps to implement.

## Confirmed working in repo

- FP16 GraphLinear (GPU) with cached weights/bias
  - Unit test comparing CPU Linear vs GPU GraphLinear with relaxed tolerance (FP16) passes.
  - Opt-in micro-benchmark shows strong FP16 speedup on M4.

## Planned changes to apply next (clean, TCN-only stack)

1. Чистка legacy

- Полностью “обнулить” файлы TransformerEncoder.swift и Attention.swift (раньше в них оставались выражения на верхнем уровне, из-за чего сборка падала).
- Упростить TextToCubeEncoderConfig под TCN-only (убрал numLayers/numHeads/dropout/maxPosition).
- TextToCubeEncoder перевести на TCN-only + FP16 GraphLinear (без фолбэков), positional encoding удалён.
- EFTextEval очистить от параметров трансформера (layers/heads), логика конфигурации TCN-only.

2. Conv1D на GPU (FP16) в TCN

- Добавить GraphConv1D (EFCore/MPSGraph/GraphConv1D.swift): 1D-каузальная свёртка через im2col + GEMM (MPSMatrixMultiplication) в FP16:
  - Xcol [BL, CinK] собирается с каузальным паддингом (t - k\*dilation < 0 → 0).
  - Wcol [Cout, Cin*K] (перепаковка весов) кешируется в MTLBuffer как Float16 один раз.
  - Матмул: Y = Xcol · Wcol^T → [B*L, Cout], reshape → [B,L,Cout], bias добавляется на CPU.
- TCNBlock будет использовать GraphConv1D для conv1 (k,d) и conv2 (1×1), активность и residual/маскирование сохранены.
- Кэш весов (FP16) для Conv1D живёт на GPU, вход x конвертируется в FP16 на лету.

3. Smoke-тесты

- Тест формы TCN-энкодера (TCNEncoderSmokeTests) — проверка [B, out] на encode(texts).
- Тест формы end-to-end encodeTokens (TextToCubeEncoderTCNEncodeTokensTests) — проверка [B, out] на pretokenized входах.
- FP16 GraphLinear-тесты оставлены (с допуском 3e-4), микробенч — по флагу.

## Notes

- We will prioritize minimal code paths (no fallbacks) to keep the code lean for research iteration. If a fallback or legacy path is ever needed, Git history can be used to restore it.
