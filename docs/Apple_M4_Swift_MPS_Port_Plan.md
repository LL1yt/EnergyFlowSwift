# Energy Flow → Swift/MPS/Metal (Apple M4) — Port Plan

Цель: поэтапно переписать ключевые части energy_flow на Swift с опорой на MPS/MPSGraph/Metal, сохранив архитектурные инварианты и получив прототип тренируемой модели на Apple Silicon (M4).

—

## 1) Вывод требований из текущей архитектуры (PyTorch)

- Инварианты координат:
  - Нормализация позиций в диапазоне [-1, 1]; обязательный clamp после любых смещений.
  - Движение вдоль оси Z к выходным поверхностям, сбор на Z=0 и Z=depth (dual planes опционально).
- Размерности и формы:
  - surface_dim = lattice_width × lattice_height; согласована с EmbeddingMapper и Text Bridge.
  - EnergyCarrier: вход `[B, 65]` (64 — выход SimpleNeuron, 1 — скаляр энергии), GRU с общими весами.
  - SimpleNeuron: вход `position[B,3] + energy[B,1] → out[B,64]`.
  - EmbeddingMapper: `[B,768] → [B,H,W]` через проекцию + Tanh.
  - OutputCollector: `{(x,y): energy}` агрегируется в `[B,H,W] → [B,768]`.
  - TextToCubeEncoder: токенизация → Transformer(2 слоя, d=256, nhead=8, ff=512) → Linear → `[B,surface_dim]` (Tanh).
- Операции/примитивы (минимальный набор):
  - Линейные слои (GEMM + bias), LayerNorm, GELU, Tanh, Add/Mul, Masked mean, Clamp, Reshape/View.
  - Графовые: GRU cell, Multi-Head Attention (MHA), Softmax, Dropout (обучение), LayerNorm fwd/bwd.
  - Индексация/квантование: арифметическая квантизация индексов поверхности, без переборов.
- Точки интеграции GPU:
  - GEMM/Attention/Norm/Act — MPSGraph (предпочтительно) или BNNS/Accelerate (CPU ref).
  - GRU — либо MPSRNN (GRUDescriptor), либо матричная реализация через MPSGraph.
  - Горячие пути под Metal kernels: masked mean, квантование координат, fused Linear+GELU(+LN).

—

## 2) Архитектура Swift-пакета

Структура SwiftPM:

- `EnergyFlowSwift/` (package)
  - `Sources/EFCore/Tensor/` — базовый тензор и память
    - `Tensor.swift` — shape, dtype, device, strides
    - `Storage.swift` — CPU (UnsafeMutable/Accelerate), GPU (MTLBuffer/Heap)
    - `DType.swift` — `.float32`, `.float16`, `.bfloat16`
    - `Device.swift` — `.cpu`, `.gpu(queue: MTLCommandQueue)`
  - `Sources/EFCore/Autograd/`
    - `Tape.swift` — узлы вычислительного графа, `backward` замыкания
    - `GradOps.swift` — градиенты для Linear, LayerNorm, GELU, Tanh, Matmul, Reshape
  - `Sources/EFCore/Ops/`
    - `MathOps.swift` — elementwise (add, mul, clamp, tanh)
    - `BLAS.swift` — матмулы (Accelerate/BNNS)
    - `MPSGraphOps.swift` — врапперы MPSGraph (matmul, ln, gelu, softmax, mha под капотом)
    - `MetalKernels/` — кастомные шейдеры (masked mean, квантование, fusion)
  - `Sources/EFCore/NN/`
    - `Linear.swift`, `LayerNorm.swift`, `Activations.swift`
    - `Attention.swift` (MHA), `TransformerEncoder.swift`
    - `GRU.swift` (матричная реализация) + опционально `GRU_MPSRNN.swift`
  - `Sources/EnergyFlow/Models/`
    - `TextToCubeEncoder.swift`
    - `EmbeddingMapper.swift`, `OutputCollector.swift`
    - `SimpleNeuron.swift`, `EnergyCarrier.swift`
    - `EnergyLattice.swift`, `FlowProcessor.swift`
  - `Sources/EnergyFlow/Utils/`
    - `Config.swift`, `Logger.swift`, `MemoryPool.swift` (MTLHeaps), `Profiler.swift`
    - `Normalization.swift` (нормализация координат и квантование индексов)
  - `Sources/EnergyFlow/TextBridge/`
    - `Tokenizer.swift` (BPE/WordPiece — JSON-конфиг), кэш
  - `Tests/` — юниты на корректность и сравнение с PyTorch-эталоном (CPU ref)

Ключевые принципы:
- Единое API `Tensor` для CPU и GPU, переключение backend — прозрачно.
- Минимальный autograd для первых итераций; точечная интеграция MPSGraph autodiff, где это проще.
- Пул памяти: MTLHeaps + переиспользование MTLBuffer для промежуточных тензоров.
- Стабильные формы: фиксированные графы и кэш MPSGraph по сигнатурам `[shape, dtype]`.

—

## 3) Дорожная карта (по этапам)

Этап 0 — Инициализация проекта
- Создать SwiftPM пакет `EnergyFlowSwift`, настроить Metal и MPSGraph.
- Реализовать `Tensor` (CPU только), базовые ops (add, mul, matmul via Accelerate, tanh, clamp).
- Минимальный autograd (Tape): Linear, GELU, LayerNorm (fwd/bwd на CPU).
- Тесты: сверка простых сеток с PyTorch (разница < 1e-4 FP32).

Этап 1 — GPU-движок (базовые операции)
- Враппер `MPSGraphEngine` с кэшем графов по форме/типу.
- Реализовать GPU-версии: Matmul+Bias, GELU, LayerNorm, Softmax, ReduceMean.
- Выбрать размещение: веса — `MTLStorageMode.private` (одна загрузка на старте), входы/выходы — `shared`.
- Добавить `MemoryPool` на Heaps и слотах предварительно выделенных буферов.

Этап 2 — TransformerEncoder и TextToCubeEncoder (forward)
- `TransformerEncoder` (2 слоя, d=256, nhead=8, ff=512, dropout 0.1) на MPSGraph.
- `PositionalEncoding` (синусоидальная, 512×256) на GPU буфере.
- `TextToCubeEncoder.swift`: токенизация → эмбеддинги → энкодер → Linear → Tanh.
- Временная токенизация: простой WordPiece/BPE из экспортированного JSON (DistilBERT), либо временно whitespace для smoke-тестов.
- Валидировать выходные формы и диапазоны; сравнить статистики с PyTorch (mean/std в пределах допуска).

Этап 3 — Обратное распространение для TextToCubeEncoder
- Реализовать градиенты для Linear, LayerNorm, GELU, Matmul в autograd CPU и MPSGraph (предпочтительно MPSGraph autodiff).
- Поддержать `noGrad` скоупы; аккумуляцию градиентов; оптимизатор AdamW.
- Обучить простую задачу (регрессия на surface embedding) и сверить динамику loss CPU vs GPU.

Этап 4 — EmbeddingMapper и OutputCollector
- `EmbeddingMapper.swift`: `[B,768] → [B,H,W]` (Tanh, позиционное кодирование, масштаб/смещение).
- `OutputCollector.swift`: сбор `{(x,y): energy}` в поверхность `[B,H,W]` и реконструкция `[B,768]`.
- GPU-варианты: матмулы (MPSGraph), редукции/суммирование, op для позиционного кода.

Этап 5 — SimpleNeuron и EnergyCarrier (GRU)
- `SimpleNeuron.swift`: позиция `[B,3]` нормализуется → линейные блоки → `[B,64]`.
- `GRU.swift`: матричная реализация с инициализацией весов (Xavier/Orthogonal), вычисление `Δx,Δy,Δz`, Tanh/Clamp.
- Альтернатива: `MPSRNN` (GRUDescriptor) для ускорения — провести A/B сравнение.
- Соблюсти инварианты: clamp смещений до применения, масштабирование displacement, nan-protection.

Этап 6 — EnergyLattice и FlowProcessor
- Векторизованная арифметическая квантизация координат: без переборов, O(1) преобразование.
- Буферизация выходов: сопоставление нормализованных `(x,y)` к индексам поверхности.
- Массовое создание/удаление потоков, batched обновления позиций, сбор выходов на двух плоскостях.
- Критичные части перенести в Metal kernels (квантование, masked scatter/add, clamp + reflect по X/Y).

Этап 7 — Оптимизации и профилирование
- FP16/BF16 пути, gradient scaling; проверка численной стабильности.
- Fusion-проходы: Linear+GELU(+LayerNorm), batched LN, MHA fusions (где поддерживает MPSGraph).
- Пайплайнинг командных буферов, параллельные энкодеры, тройная буферизация.
- Профилирование через Xcode GPU Capture + Metrics; устранение лишних копий.

Этап 8 — Бенчмарки и демо
- Сценарии: линейная регрессия, MLP, затем интеграция `TextToCubeEncoder` в упрощенный тренинг.
- Сравнение CPU (Accelerate) vs GPU (MPSGraph), профили памяти/скорости.

—

## 4) API-шаблоны (Swift)

- `Tensor`
  - `init(shape:[Int], dtype:DType, device:Device, storage: …)`
  - `requiresGrad: Bool`, `grad: Tensor?`, `backward()`
  - `to(device:)`, `astype(_:)`, `reshape(_:)`, `view(_:)`, `contiguous()`

- `Module`
  - `func forward(_ x: Tensor) -> Tensor`
  - `var parameters: [Tensor] { get }`
  - `func zeroGrad()`

- `Linear(in: Int, out: Int, bias: Bool)`
  - веса в `.private` буферах, биасы — там же; вход/выход — `.shared`.

- `LayerNorm(dim: Int, eps: Float)`
- `GELU()`, `Tanh()`
- `MultiHeadAttention(dModel: Int, nHead: Int, drop: Float)`
- `TransformerEncoderLayer(dModel: Int, nHead: Int, ff: Int)`
- `GRUCell(inputSize: Int, hiddenSize: Int)`

- `TextToCubeEncoder.encode(texts:[String], maxLength:Int=128) -> Tensor // [B, surface_dim]`
- `EmbeddingMapper.mapToSurface(_ x: Tensor) -> Tensor // [B,H,W]`
- `OutputCollector.collect(_ surface: Tensor) -> Tensor // [B,768]`
- `SimpleNeuron.forward(position: Tensor[B,3], energy: Tensor[B,1]) -> Tensor[B,64]`
- `EnergyCarrier.forward(neuronOut: Tensor[B,64], energy: Tensor[B,1], hidden: Tensor) -> (energy: Tensor[B,1], nextPos: Tensor[B,3], …)`
- `EnergyLattice.placeInitialEnergy(emb: Tensor[B,768], mapper: EmbeddingMapper) -> [FlowID]`

—

## 5) Детализация первых шагов (начальная часть)

Шаг 1 — CPU reference backend
- Реализовать `Tensor` (CPU), `Linear`, `LayerNorm`, `GELU`, `Tanh`, `matmul` на Accelerate.
- Мини-автоград: tape, `requiresGrad`, bwd для перечисленных ops.
- Тест: `Linear→GELU→LayerNorm→Linear` (потеря MSE), сверка градиентов с PyTorch.

Шаг 2 — MPSGraph обвязка
- Инициализировать `MPSGraphEngine` с кэшем графов по `[op, shape, dtype]`.
- Обеспечить zero-copy путь для входов/выходов (`shared`), веса — `private` (перенос один раз).
- Опробовать GEMM+Bias, GELU, LN на батче `[B,256]`.

Шаг 3 — TransformerEncoder (2 слоя)
- Реализовать MHA, FFN, LN, residual; собрать в энкодер.
- Сгенерировать синусoidal PE (512×256) и добавить к токен-эмбеддингам.
- Прогнать фейковые токены (словарь ~30k, d=256), сверить форму и статистики.

Шаг 4 — TextToCubeEncoder forward
- Имплементировать токенизацию (временно whitespace или экспорт DistilBERT токенайзера в JSON).
- Собрать пайплайн: токен→embed→PE→Transformer→Linear→Tanh→`[B, surface_dim]`.
- Проверка диапазонов, логирование mean/std; smoke-тест на 2–3 коротких строки.

Шаг 5 — Backprop для TextToCubeEncoder
- Включить автоград (предпочтительно через MPSGraph autodiff); оптимизатор AdamW.
- Небольшая задача: подгон `Text→surface` к зафиксированным таргетам (sanity-check тренировки).

—

## 6) Использование unified memory и производительность (M4)

- Размещение данных:
  - Веса и промежуточные буферы графа — `MTLStorageMode.private` (макс пропускная способность GPU).
  - Входы/выходы и редкие обновления — `shared` для минимизации копий.
  - Предзагрузка весов в `private` через blit один раз на старте.
- Пул памяти: `MTLHeap` + аренда/возврат буферов по размеру/типу; избегать частых alloc/free.
- Параллелизм: независимые графы/командные буферы для потоков данных; параллельные энкодеры.
- FP16/BF16: включить по умолчанию для матмулов/активаций; градиент скейлинг.
- Кэш графов: MPSGraph компилировать 1 раз на форму; биндинг тензоров между итерациями.
- Профили: Xcode GPU Capture, Counters; отслеживать горячие точки — attention, layernorm, scatter-операции решетки.

—

## 7) Риски и обходные пути

- Токенизация DistilBERT: либо экспорт JSON из HF `tokenizers`, либо простой whitespace для старта, затем заменить.
- GRU производительность: MPSRNN иногда быстрее матричной реализации — держать оба варианта.
- BF16 поддержка: проверить доступность на конкретной версии macOS/MPS; при отсутствии — FP16.
- Autograd сложность: для начала — смешанный подход (MPSGraph для autodiff, ручные градиенты — точечно).
- Вариативность форм: держать фиксированные max shapes (padding) для кэша графов.

—

## 8) Мини-план портирования TextToCubeEncoder

1) Бэкенды и ops (Шаги 1–2 выше) — готовность к матмулам/нормам/активациям.
2) Реализовать `Embedding`, `PositionalEncoding`, `TransformerEncoderLayer` на MPSGraph.
3) Собрать `TextToCubeEncoder.swift` с адаптивной `surface_dim` из `Config`.
4) Тесты: сравнение PyTorch vs Swift по статистикам и форме на одинаковом сидировании.
5) Подключить тренинг (MSE к таргетным surface embeddings) и убедиться в снижении loss.

—

## 9) Чек-лист соответствия инвариантам

- Координаты всегда clamp в [-1,1] после смещений.
- Квантование индексов поверхности — только арифметическое O(1).
- Общие веса для SimpleNeuron и EnergyCarrier по решетке.
- Согласованность `surface_dim` по всем модулям и мостам.
- Новые тензоры по умолчанию — на GPU (если доступен), dtype = mixed precision, без лишних `.copy`.

—

## 10) Быстрый старт по шагам (резюме)

- S0: SwiftPM + CPU Tensor + autograd mini + unit tests.
- S1: MPSGraph backend (matmul/gelu/ln/softmax) + memory pool.
- S2: TransformerEncoder(2) + PE + TextToCubeEncoder forward.
- S3: Backprop через MPSGraph + AdamW + sanity тренировка.
- S4: EmbeddingMapper/OutputCollector на GPU.
- S5: SimpleNeuron/GRU(EnergyCarrier) + валидация смещений/клампов.
- S6: EnergyLattice/FlowProcessor + Metal kernels для горячих путей.
- S7: FP16/BF16, fusion, профилирование, бенчмарки.

—

Если нужно — добавлю шаблоны исходников (`Tensor.swift`, `MPSGraphEngine.swift`, `TransformerEncoder.swift`) и минимальные тесты для старта.

