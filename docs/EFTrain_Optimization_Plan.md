# План улучшений и оптимизаций EFTrain (EnergyFlowSwift)

Документ описывает конкретные улучшения и оптимизации для тренировочного пайплайна EFTrain, основанные на анализе docs/Analysis_of_EnergyFlowSwift_TextBridge_Training_Pipeline.md и ревью текущего кода:

- EFTrain: EnergyFlowSwift/EnergyFlowSwift/Sources/EFTrain/{main.swift, Evaluation.swift, CheckpointIO.swift, MaskUtils.swift}
- Зависимости ядра тренировки:
  - EnergyFlow/TextBridge: TextToCubeEncoder.swift (+ Config)
  - EnergyFlow/Training: {OptimStep.swift, LastTCNBackward.swift, Gradients.swift, CombinedTrainer.swift}
  - EnergyFlow/Decoder: {DecoderTrainer.swift, TextDecoder.swift, TextDecoderConfig.swift}
  - EFCore/MPSGraph: {GraphLinear.swift, GraphConv1D.swift, ExecutableCache.swift, GraphContext.swift}
  - EFCore/Metal: {LayerNormGPU.swift, GELUGPU.swift, Im2ColCol2ImGPU.swift, ConvPackGPU.swift, ElementwiseGPU.swift, BufferPool.swift}
  - PyTorchSwift/Embedding.swift
  - Dataset: EnergyFlow/Dataset/SimpleJSONLDataset.swift
  - Optim: EFCore/Optim/{AdamW.swift, LossScaler.swift}

Цель: устранить функциональные несоответствия между режимами, убрать дублирование, повысить загрузку GPU (MPS/MPSGraph), убрать лишние CPU↔GPU переходы, сделать конвейер асинхронным и устойчивым, и заложить основу для масштабирования данных и токенизатора.

---

## Краткий обзор текущего пайплайна

- EFTrain/main.swift: два пути тренировки внутри одного цикла по батчам
  - Mode A (tokens): KD по z_teacher (MSE + 1 − cosine), проекция (GraphLinear) с FP16 + optional unfreeze последнего TCN блока; микробатчи + накопление градиента; динамический loss scaling; LR warmup + cosine; grad clipping.
  - Mode B (decoder CE): teacher forcing, CE по logits, тренируется DecoderTrainer; при unfreeze синхронизируются веса последнего TCN блока между encoder/decoder.
- Evaluation.swift: оценка в батчах с микро-батчами, для token/text.
- CheckpointIO.swift: бинарные чекпоинты только для проекции энкодера (W, optional b). Есть отдельное сохранение состояния оптимизатора (EFOP1).
- TextToCubeEncoder: CPU токенизация → Embedding (CPU) → TCN (GPU: LN/conv/GELU/conv/residual/mask) → maskedMean (GPU) → GraphLinear (GPU) → optional tanh. Для unfreeze предоставляет кеш последнего блока и функции градиентов (через вспомогательные GPU-классы).
- EnergyFlow/Training/\*: OptimStep объединяет шаг оптимизации над проекцией и опционально последним блоком; LastTCNBackward реализует бэквард последнего блока (GPU path с MPSMatrix, ElementwiseGPU и т.п.).
- Механизм MPSGraph: подготовлены кэши LNExecCache и LNGeLUGemmCache для фьюзинга LN→GELU→MatMul, но в энкодере ещё напрямую не применяются для последнего блока.

---

## Выявленные проблемы и быстрые исправления (высокая ценность; низкие затраты)

1. Баг: text-mode не применяет lastGrads при unfreezeLastTCN

- Где: EnergyFlowSwift/EnergyFlowSwift/Sources/EFTrain/main.swift
- Token-mode правильно накапливает и передаёт lastPack в optimizerStepProjectionAndLastBlock(..., lastGrads: lastPack, ...), а text-mode передаёт lastGrads: nil.
- Эффект: в text-mode последний TCN блок никогда не обучается, даже при включённом unfreezeLastTCN.
- Исправление: собрать accLast\* и собрать lastPack аналогично token-mode, передать lastGrads: lastPack. Также убедиться, что маска и cache используются симметрично (уже есть forwardForTrainingWithLastBlockCache(texts:)).

2. Дублирование утилиты padOrTruncateMask

- Где: main.swift локальная функция и EFTrain/MaskUtils.swift.
- Действие: использовать общую MaskUtils.padOrTruncateMask и удалить локальный дубликат, чтобы избежать расхождений.

3. Двойной сброс аккумуляторов после шага

- Где: main.swift (после optimizerStep повторный accW/accB reset). Не критично, но лишнее.
- Действие: оставить один сброс.

4. Улучшение логов по эпохам/батчам

- Добавить вывод throughput (samples/sec), шаг LR, долю Mode A/B в батче, и счётчики overflow/scale в LossScaler. Это поможет быстрее ловить регрессии производительности.

---

## Архитектурные улучшения (убрать дублирование, упростить сопровождение)

5. Единый тренировочный путь для tokens/text

- Сейчас: два разветвлённых пути с дублирующейся логикой (подготовка градиентов, накопление, шаг оптимизатора, метрики), что уже привело к багу с lastGrads.
- Варианты:
  a) Претокенизировать текстовые батчи заранее и всегда работать в token-mode. SimpleTokenizer уже есть, а Dataset умеет в JSONL/EFB с опциональными токенами. Если у sample нет токенов — токенизировать на лету и кешировать.
  b) Использовать CombinedTrainer (EnergyFlow/Training/CombinedTrainer.swift) как единую точку Mode A/B, а EFTrain/main.swift оставить тонким оркестратором.
- Рекомендация: внедрить CombinedTrainer в EFTrain и удалить дублирование в main.swift. Это выровняет логику Mode A/B и сократит объем кода.

6. Централизация LR/Clip/Schedule и шагов оптимизатора

- Сейчас в main.swift вручную собираются градиенты/масштабирование, затем вызывается optimizerStepProjectionAndLastBlock. В CombinedTrainer код ещё ближе к модульному использованию. Консолидировать вызовы и добавить тесты на эквивалентность суммарного шага при accumSteps>1.

7. Конфигурации

- TrainConfig уже централизован, но стоит добавить флаги: useCombinedTrainer, useFusedLNGeLUGemm (для последнего блока, см. ниже), useAsyncGPU, pretokenizeText. Эти флаги позволят поэтапно включать улучшения без ломки.

---

## Производительность: эмбеддинги и GPU фьюзинг

8. Оптимизировать Embedding.forward (CPU)

- Где: PyTorchSwift/Embedding.swift — сейчас копирование построчно по элементам (вложенные циклы 256 операций на токен). Это узкое место при больших B×L.
- Быстрый win: заменить внутренний цикл на memcpy/withUnsafeBytes для копирования целой строки (row) весов (embeddingDim элементов) за один вызов, вместо скалярных присваиваний.
- Дальше: рассмотреть GPU gather/one-hot @ W через MPSGraph (для батча), но сначала сделать memcpy-версию (минимальные изменения, ожидаемый большой выигрыш на CPU).

9. Фьюзинг LN→GELU→MatMul на GPU через MPSGraph

- Уже есть LNGeLUGemmCache (EFCore/MPSGraph/ExecutableCache.swift). Применение:
  - Для последнего блока TCN, когда второй conv — точечный (kernel=1), часть пути после LN и GELU можно свести к матмулу. Можно сформулировать как: LN (на [B*L,D]) → GELU → MatMul([B*L,H]×[Out,H]^T) со смещением. Для K=1 это даст эквивалент conv2.
  - В проекте уже используются специализированные GPU-компоненты (LayerNormGPU, GELUGPU, GraphLinear). Перенос этой последовательности в один MPSGraph-экзекьютабл снизит число CPU↔GPU переходов и уменьшит накладные расходы.
- План:
  1. В последнем блоке: ветка «fused» под флагом useFusedLNGeLUGemm;
  2. Кэшировать и пинить веса/гамма/бета между шагами;
  3. Отпинить после шага оптимизатора.

10. Снижение CPU↔GPU round-trip

- Сейчас GraphConv1D/GraphLinear читают результат в CPU Tensor после выполнения и cmd.waitUntilCompleted(). При последовательном вызове таких узлов получаются лишние sync-точки.
- Направление: переход на MPSNDArray/MTLBuffer как внутреннее представление между узлами для горячего пути (без немедленного чтения на CPU), и чтение на CPU только для логов/метрик/записи чекпоинтов. Это более крупная рефакторинга (в несколько фаз), см. раздел «Асинхронность».

---

## Асинхронность и конвейерность (так же подумать, можем ли мы тут реализовать многопоточность на несколько ядер cpu или в нашем случае это слишком сложно будет, а прирост производительности незначительный будет?)

11. Удаление жёстких ожиданий и перекрытие CPU/GPU

- Избегать cmd.waitUntilCompleted() там, где не требуется немедленного чтения на CPU — готовить следующий микробатч, пока GPU считает текущий.
- Использовать MTLCommandBuffer.addCompletedHandler, Swift Concurrency или GCD для конвейеризации: CPU (подготовка данных/токенизация) ↔ GPU (вычисление) работают одновременно.
- Dataset:
  - JSONL парсить в фоне и/или заменить на более быстрый парсер;
  - Для больших файлов — ленивые итераторы/streaming. Для .efb добавить mmap-путь или страничное чтение, если файл велик.

12. Потокобезопасность буфер-пула

- BufferPool сегодня nonisolated(unsafe). При введении параллелизма ограничить использование буферов одним потоком или сделать простой синхронизатор на горячие лейблы. Ввести слой Allocator с раздельными пулами по стадиям конвейера, чтобы избежать гонок.

---

## Полный GPU-бэквард, память и кэширование

13. Уточнить, что ещё на CPU и можно ли перенести

- В текущей реализации многие backward-пути уже GPU (LN backward через MPSGraph, matmul-град через MPSMatrix и т.д.). Остаётся проверка: bias добавление, некоторые редкие ветки, сборки тензоров (host). Цель — убрать CPU-ветви из горячего пути.

14. Управление памятью GPU

- Расширить BufferPool/MTLHeaps для крупных стабильных буферов (Xcol/Y, FP16 веса, временные). Предварительные размеры фиксированы (B,L,D) — выгодно аллоцировать единожды и переиспользовать.
- В кэшах ExecutableCache (LNGeLUGemmCache) уже есть persistent MPSNDArray — продолжить линию «pin/unpin»: пинить веса/гамма/бета до шага, отпинивать сразу после обновления параметров.

---

## Логирование/метрики, гиперпараметры, данные

15. Логирование и профилирование

- Добавить: throughput (samples/sec), время на 100 микробатчей, текущий LR, текущий loss scale, счётчики overflow, доля Mode A/B фактически, кэш-хиты MPSGraph exec.
- При необходимости — os_signpost для измерений GPU/CPU этапов.

16. Гиперпараметры по умолчанию

- В соответствии с Analysis\*: warmupSteps ~ 500–1000, clipNorm ~ 1–2, microBatch максимально возможный по памяти GPU. у нас унифицированная память на 128gb - какой оптимальный размера microBatch при такой памяти?
- В TrainConfig уже есть поля; обновить sample-конфиг и README-заметки.

17. Токенизатор и данные

- Для тренировки: требовать, чтобы датасет уже содержал input_ids/attention_mask, полученные токенайзером учителя. фолбеки убрать.
- Для инференса/валидации в Swift (и чтобы не зависеть от внешних скриптов) подготовить совместимый токенайзер(использовать vocab.txt и реализовать BasicTokenizer + WordPiece вручную в Swift с теми же правилами, что и у DistilBERT):
  парсить tokenizer.json DistilBERT и реализовать в Swift минимальные компоненты:
  • нормализатор (lowercase/strip accents для uncased)
  • BertPreTokenizer (split on punctuation/whitespace)
  • WordPiece декодер с тем же vocab и unk_token="[UNK]"
- SimpleJSONLDataset: для очень больших наборов — ленивые итераторы и/или предварительная конверсия в .efb; для .efb — добавить версионирование и метаданные (длина макс. L, vocab info).

---

## Чекпоинты и совместимость

18. Расширить CheckpointIO

- Сейчас сохраняется только проекция энкодера. При активном unfreeze полезно сохранять ещё и параметры последнего TCN блока (w1,b1,w2,b2, gamma, beta) и (опционально) состояние оптимизатора по этим параметрам.
- Добавить версию формата, шапку с dims, и безопасный отказ при несовпадении.

---

## Тестовое покрытие (минимум для регрессий)

19. Юнит-тесты

- EFTrain: тест на исправление text-mode lastGrads (валидировать, что веса последнего блока меняются при unfreezeLastTCN=true).
- Embedding: тест производительности/функционального эквивалента memcpy-версии.
- CombinedTrainer vs текущий main: тест эквивалентности loss/grad для одного шага на одинаковом батче.
- MPSGraph fused путь: паритет выходов и обратного прохода (где применимо) с нефьюзенной версией; допуск по числовой погрешности.
- Accumulation correctness: сравнить N микробатчей с accumSteps=N vs один батч.

---

## Пошаговый план внедрения (фазы)

Фаза 0 — Быстрые фиксы (0.5 дня)

- [ ] Исправить text-mode lastGrads в main.swift: собрать lastPack и передавать в optimizerStepProjectionAndLastBlock(..., lastGrads: lastPack,...).
- [ ] Убрать дублирующуюся padOrTruncateMask (использовать EFTrain/MaskUtils.swift).
- [ ] Убрать двойной reset аккумуляторов.
- [ ] Добавить базовые метрики в логи: throughput, LR, loss scale, overflow count.
- [ ] Тест: EFTrainMiniEpochTests дополнить проверкой обновления последнего блока в text-mode.

Фаза 1 — Embedding memcpy и микро-оптимизации (0.5–1 день)

- [ ] Переписать Embedding.forward на блочное копирование строки весов.
- [ ] Микробенчмарк (B×L×D) до/после.
- [ ] Тесты корректности значений.

Фаза 2 — Консолидация тренировки через CombinedTrainer (1–2 дня)

- [ ] Перенести тренировочную логику EFTrain/main.swift на CombinedTrainer с общей обработкой tokens/text (предтокенизация при необходимости).
- [ ] Оставить EFTrain как тонкий CLI-оркестратор (чтение конфигов, сплит train/val, A/B расписание).
- [ ] Юнит-тест на эквивалентность результатов одного шага (до/после).

Фаза 3 — Фьюзинг MPSGraph (2–3 дня)

- [ ] Добавить флаг useFusedLNGeLUGemm.
- [ ] Реализовать fused-путь для последнего блока, когда conv2 — pointwise (kernel=1): LN→GELU→MatMul(+bias) через LNGeLUGemmCache.
- [ ] Пинning gamma/beta/W/b до шага, unpin после.
- [ ] Тесты на паритет и прирост скорости.

Фаза 4 — Асинхронность и конвейер (2–4 дня)

- [ ] Убрать лишние waitUntilCompleted, где это безопасно.
- [ ] Задизайнить «GPU tensor» протокол (MPSNDArray/MTLBuffer) для горячего пути и «холодный» CPU Tensor только на границах.
- [ ] Подготовка следующего микробатча в фоне (Swift Concurrency/GCD).
- [ ] Развести BufferPool по стадиям/лейблам для потокобезопасности.

Фаза 5 — Данные и токенизация (2–4 дня)

- [ ] Добавить ленивый режим чтения JSONL/streaming, ускоренный парсер.
- [ ] .efb: опционально mmap/страничное чтение; версия/метаданные формата.
- [ ] Интеграция токенизатора (BPE/SentencePiece/TokenizerKit), совместимость с Python-прототипами.

Фаза 6 — Логи/метрики/приёмочные тесты (0.5–1 день)

- [ ] Добавить метрики (throughput, GPU cache hits, mem usage, CE в валидации при A/B).
- [ ] Обновить sample-конфиги и README.
- [ ] Финальные юнит/интеграционные тесты.

---

## Риски и практики

- MPSGraph/FP16 числовые отличия: держать возможность отладки через CPU fallback (ENV EF_USE_CPU_LN) и тесты на допуск.
- Параллелизм: упереться в потокобезопасность буферов — изолировать пулы, простая синхронизация, избегать общий пул в нескольких потоках одновременно.
- Совместимость чекпоинтов: ввести версию формата и проверки размеров.

---

## Критерии приёмки

- Баг text-mode unfreeze исправлен (тест фиксирует изменение весов последнего блока в text-mode).
- Embedding.forward быстрее не менее чем в 3–5× на CPU при типичном B×L.
- При включённом useFusedLNGeLUGemm прирост скорости последнего блока (вперед) не менее 1.2× (при kernel=1) при равносильной точности.
- Throughput вырос не менее чем на 20% на тестовом сценарии (Configs/train_debug.json) после Фаз 0–3.
- EFTrain использует единый тренировочный путь (через CombinedTrainer), без дублирования кода.

---

## Приложение: ключевые места в коде

- Баг lastGrads (text-mode): EnergyFlowSwift/EnergyFlowSwift/Sources/EFTrain/main.swift (ветка text-mode: сейчас lastGrads=nil — нужно собрать accLast\* и передать lastPack, как в token-mode).
- Embedding memcpy: EnergyFlowSwift/EnergyFlowSwift/Sources/PyTorchSwift/Embedding.swift — заменить внутренний цикл на memcpy.
- Фьюзинг LN→GELU→MatMul: EnergyFlowSwift/EnergyFlowSwift/Sources/EFCore/MPSGraph/ExecutableCache.swift (LNGeLUGemmCache) и точки вызова в TextToCubeEncoder/TCN-блоке.
- Асинхронность: GraphLinear/GraphConv1D — места с cmd.waitUntilCompleted; переход на deferred readback и/или MPSNDArray.
- Централизация тренировки: EnergyFlowSwift/EnergyFlowSwift/Sources/EnergyFlow/Training/CombinedTrainer.swift — использовать из EFTrain.

---

Готово к итеративной реализации. Рекомендуемый старт — Фаза 0 и 1 (быстрые исправления + memcpy в Embedding), затем переход к консолидации тренировки (Фаза 2) и фьюзингу (Фаза 3).
