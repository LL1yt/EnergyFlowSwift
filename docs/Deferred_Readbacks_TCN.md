# План: Deferred Readbacks для TCN/Decoder и тестов

## Цели

- Перевести TCN/decoder стек на `GPUReadback`, чтобы `syncBatch` управлял всеми CPU-ридбэками.
- Убрать локальные `await` с немедленным CPU копированием из Conv/Elementwise/Tensor ops.
- Обновить тесты, чтобы они работали с асинхронными readbacks и batch guard.

## Этапы

### 1. Conv1D + Elementwise в TCN
- `TCNBlock.forward`: использовать `forwardDeferred` для `conv1/conv2`.
- `residualAdd`, `maskZero`, `maskedMean` → вызывать `*_Deferred`, собирать readbacks.
- Ввести локальные массивы readbacks и вызывать `await .value()` после `syncBatch`.

### 2. TextToCubeEncoder / Decoder
- `TextToCubeEncoder.forwardForTrainingWithLastBlockCache`: собрать все readbacks (conv, elementwise, maskedMean, projection) и дождаться их после `syncBatch`.
- `Decoder.forwardForTraining`: аналогично для conv/elem шагов.
- Обновить `DecoderTrainer.step/stepScaled` — все `GraphLinear`, `conv`, `metrics` должны использовать deferred API.

### 3. Trainers / Evaluation
- `CombinedTrainer`: накопить readbacks (KD metrics уже deferred). Добавить deferred для maskedMeanBackward/ residualAdd если вызываются напрямую.
- `EFTextEval`: в батчах собирать readback из `encodeTokens` (если encoder возвращает readback) либо оставить как сейчас, если `encodeTokens` внутри возвращает готовый `Tensor`.

### 4. Тесты
- Обновить `EnergyFlowSwiftTests`:
  - `GPUKernelsAndStrideTests`, `GraphLinearTrainingStepTests`, мини-эпохи decoder/encoder.
  - Использовать `await` и `syncBatch` в тестах; следить за `GPU.ensureBatchSynced` guard.
- Добавить regression-тест: deferred conv + syncBatch → `fatalError` если `value()` вызвать без sync.

### 5. Документация/чистка
- После этапов обновить `docs/Async_GPU_Actor_Refactor_Plan.md` (Phase 6b progress).
- Удалить/обновить устаревшие комментарии в GPUActor+Elementwise/Conv1D.

## Проверка

- `swift build` (пользователь вручную).
- `swift test --parallel` (пользователь вручную).
- Сравнить логи тренера до/после (должны совпасть).
