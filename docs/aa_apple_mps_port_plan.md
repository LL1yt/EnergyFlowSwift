
# AA / energy_flow → Swift/MPS/Metal (Apple Silicon M4) — План миграции и оптимизаций

_Стиль: скептический и проверяемый. Я буду отмечать уверенность_ (Факты ≤ 95%, Теории ≤ 85%, Гипотезы ≤ 70%, Мнения ≤ 60%).

## 0) Что уже есть в вашем репозитории (краткая инвентаризация)

С определённой долей вероятности (≈ 90%):

- `energy_flow/text_bridge/text_to_cube_encoder.py` — **лёгкий текстовый энкодер**: токенизация (DistilBERT tokenizer), `nn.Embedding` (вместо внешней LM), 2 слоя `nn.TransformerEncoder`, агрегирование masked‑avg, проекция в `surface_dim = lattice_width*lattice_height` через MLP + `tanh` (≈5М параметров). Формы: токены → `[B, L]`, эмбеддинги `[B, L, 256]`, агрегат `[B, 256]`, поверхность `[B, surface_dim]`. [Источник кода].  
- `energy_flow/core/embedding_mapper.py` — мэппер **768D ↔ surface_dim** (MLP + GELU + LayerNorm); позиционное скалярное поле для поверхности (параметр) (Факт ≈ 95%). [Источник кода].  
- `energy_flow/core/energy_lattice.py` — **решётка 3D** с нормированными координатами в `[-1, 1]`, размещение входной энергии на `Z = depth/2`, быстрые предвычисления, собиратель статистики; опциональное тензорное хранилище активных потоков (Факт ≈ 95%). [Источник кода].  
- `energy_flow/core/simple_neuron.py` — небольшой MLP (`Linear`+`LayerNorm`+`GELU`) над `(x,y,z)` и скалярной энергией; выдаёт признаки для GRU‑носителя (Факт ≈ 95%). [Источник кода].  
- `energy_flow/training/energy_trainer.py` — обучающий цикл с **AMP**, аккумулированием, замером памяти/времени, комбинированным лоссом _energy_ + _text_ (Факт ≈ 95%). [Источник кода].  
- `energy_flow/text_bridge/cube_to_text_decoder.py` — «инверсия эмбеддингов поверхности» через **T5‑small** (Core: `T5ForConditionalGeneration`), плюс адаптер `surface_dim → 512` и итеративная коррекция; для Swift-порта рационально заменить на Core ML‑версию T5 или временно отключить (Факт ≈ 95%). [Источник кода].

> Прим.: точные ссылки и обоснования приведены в основном ответе рядом с цитатами. Здесь не дублирую, чтобы сохранить читаемость.

---

## 1) Требования к API из текущей архитектуры (минимально достаточные)

Уверенность (≈ 85%):
1. **Тензорная абстракция**: `Tensor(shape, dtype, device)`, представление буфера (CPU/Metal), страйды, просмотр (`view/reshape`), типы: `float32`, `float16`, `bfloat16`, `int32`.
2. **Базовые операторы** (должны иметь и CPU‑референс, и GPU‑вариант):
   - алгебра: `add`, `mul`, `addmm`/`matmul`, `bias_add`;
   - нелинейности: `gelu`, `tanh`;
   - нормализации: `layer_norm`;
   - редукции/формы: `mean(sum)`, `masked_mean`, `transpose`, `reshape`, `concat`, `clip`;
   - **embedding (gather)**: индексация `W[idx]` для токенов;
   - **softmax** (для attention), **dropout** (только в train).
3. **Модули**:
   - `Embedding(vocab, d)`, `Linear(in, out)`, `LayerNorm(d)`;
   - `TransformerEncoderLayer(d_model=256, nhead=8, ff=512, dropout=0.1, activation=gelu, batch_first=True)`;
   - `MultiHeadAttention` (QKV‑проекции, батч‑матрицы, softmax, V‑агрегация).
4. **Автодифференцирование**:
   - autodiff граф на уровне операторов **или** делегирование подграфов в **MPSGraph** с `gradients(of:with:)`.
5. **Память/буфера**:
   - пул MTLHeap/MTLBuffer, `storageModeShared`, опционально aliasing/перепользование;
   - смешанная точность (`fp16/bf16`) и квантизация для «холодных» узлов (опционально).
6. **I/O интерфейсы**:
   - загрузка/сохранение весов (CPU и GPU‑буферы);
   - минимальный текстовый токенизатор **WordPiece** (BERT/DistilBERT) и словарь.

---

## 2) Дизайн фреймворка под Apple Silicon (Swift + Metal/MPS)

Уверенность (≈ 80%):

### 2.1 Слои
- **Core (Swift)**: `Tensor`, память, автодифф-ядро (минимум для CPU‑референса).
- **CPU backend**: BLAS из Accelerate (cBLAS) + чистый Swift (fallback); BNNS — опционально для `LayerNorm`/активаций.
- **GPU backend**:
  - **MPSGraph** — быстрый путь для линейной алгебры/нормализаций/softmax/attention; есть `bfloat16` (см. WWDC’23/24).
  - **Metal compute** — горячие пути: attention (в т.ч. flash‑подобный tiled SDPA), masked‑avg, gather/embedding (при необходимости).
- **Autodiff**: гибрид — приоритет MPSGraph‑дифференцирования; вручную: простые операторы (mask‑avg, reshape/no‑grad).
- **Токенизация**: чистый Swift WordPiece (на основе словаря DistilBERT); изначально CPU.

### 2.2 Память и планировщик
- **Unified memory** (`storageModeShared`) → нулевые копии CPU↔GPU, общее адресное пространство; **MTLHeaps** для пула буферов, aliasing и быстрых аллокаций; `MTLFence`/hazard‑tracking по умолчанию.
- **Командные очереди**: один `MTLCommandQueue`, несколько `MTLCommandBuffer`/`MTLComputeCommandEncoder` на батч, пайплайн overlap (encode следующего шага пока GPU исполняет текущий).
- **Mixed precision**: веса/активации в `bf16` или `fp16`, аккуратность градиентов (fp32 master weights или loss‑скейлинг).
- **Operator fusion**: `linear+bias+gelu(+dropout)`, `ln+mul+add`, SDPA fused kernel.
- **KV‑cache/append‑friendly layout** (на будущее для LLM): предвыделение кругового буфера в shared‑memory, индексируемое дописывание «в конец», чтобы _минимизировать движения памяти_ при добавлении контекста.

---

## 3) Дорожная карта (пошагово, с критериями готовности)

_Сроки ориентировочные; разумно идти итерациями._

### Этап A — База (CPU‑референс + каркас) — **1–2 недели**
- [ ] `Tensor` (CPU): буфер + shape + dtype + элементарные операции.
- [ ] Операторы: `linear`, `gelu`, `layer_norm`, `softmax`, `embedding`, `masked_mean`.
- [ ] Мини‑autograd (reverse‑mode) на CPU для перечисленных операторов.
- [ ] Токенизация WordPiece (Swift) + загрузка `vocab.txt` DistilBERT.
- [ ] Юнит‑тесты: сверка с PyTorch на случайных входах (допуск по `1e-4`/`1e-3`).  
**DoD**: на CPU обучается маленькая MLP и **воспроизводится** `TextToCubeEncoder` forward (без обучения).

### Этап B — MPSGraph (GPU быстрый путь) — **1–2 недели**
- [ ] Обёртка `GraphModule`: сборка подграфов (Linear→GELU→…→Projection).
- [ ] Реализация `MultiHeadAttention` на MPSGraph (QKV‑matmul, scale, softmax, V‑matmul, merge).
- [ ] Поддержка `fp16/bf16` (перевод весов/активаций, loss‑скейлинг для тренировки).
- [ ] Маппинг `TextToCubeEncoder` полностью на MPSGraph (кроме токенизации).  
**DoD**: инференс и **обучение** `TextToCubeEncoder` на небольшом датасете; ускорение ≥ **3–6×** vs CPU‑референс на M4 (ожидание, не гарантия).

### Этап C — Energy‑модули на Swift — **1–2 недели**
- [ ] `EnergyEmbeddingMapper` (MLP) на MPSGraph + CPU fallback.
- [ ] `EnergyLattice` (нормализации, квантование позиций, батч‑создание потоков) — CPU‑ориентированно (много ветвлений) + мелкие Metal‑ядра для арифметических O(N) мест (квантование координат, формирование поверхностей).
- [ ] `SimpleNeuron` (MLP) на MPSGraph.  
**DoD**: `FlowProcessor.forward(…)` в Swift с входом `[B,768]` и выходом поверхности / 768 (в зависимости от ветки сбора).

### Этап D — Text Bridge (декодер) — **1–2 недели**
- [ ] Вариант 1 (быстрее): **отключить** T5‑decoder, оставить только encoder‑loss на первых итерациях.
- [ ] Вариант 2 (полноценный): **конвертировать T5‑small в Core ML** (coremltools), вызывать из Swift; адаптер `surface_dim→512` оставить собственный.
- [ ] LRU‑кэш (CPU) — прямой порт.  
**DoD**: end‑to‑end: текст → поверхность → решётка → поверхность → (опц.) текст.

### Этап E — Память и производительность — **1–2 недели**
- [ ] Пул ресурсов: `MTLHeap` + переиспользование временных буферов.
- [ ] Фьюзинг подграфов (см. выше) и смешанная точность по умолчанию.
- [ ] Параллельная токенизация (GCD), overlap CPU‑подготовки и GPU‑вычислений.
- [ ] Бенчмарк: CPU vs GPU (B=1/8/32/128), профилировка (`MTLCommandBuffer` timestamps).  
**DoD**: стабильный тренинг малых сетей и демонстрация выигрыша на M4.

---

## 4) Конкретная цель первой инкрементации: `TextToCubeEncoder` на Swift

### 4.1 Мини‑спецификация
- Вход: `List<String>` → токены → `[B, L] (int32)`.
- Слои: `Embedding(vocab, 256)` → add `pos[512,256]` → `2× TransformerEncoderLayer(256, 8, 512, GELU, dropout)` → masked‑avg → `MLP(256→256→surface_dim)` + `tanh`.
- Выход: `[B, surface_dim]`; функции `reshape_to_surface([B,H*W]) → [B,H,W]`.

### 4.2 Отображение на MPSGraph
- `Embedding` — gather (таблица `vocab×256` в `fp16/bf16`).
- `Attention` — QKV‑линейки + `matmul` + `softmax` + `matmul` + сшивка голов.
- `LayerNorm` — штатный op MPSGraph.
- `Masked‑avg` — `reduceSum(emb * mask) / clamp(sum(mask))`.
- `Projection` — 2× `linear + GELU + LayerNorm (+ Dropout)` + `linear + tanh`.

### 4.3 Проверки эквивалентности
- Сверка форм (assert), статистики (mean/std), и векторные тесты на 5–10 фиксированных seed.
- Допуск по точности: `fp32` ≤ 1e‑4, `bf16/fp16` ≤ 3e‑3 (эмпирически).

---

## 5) Где именно будет «железоотдача» на M4

С определённой долей вероятности (≈ 80%):

- **Unified memory** (`storageModeShared`) убирает копирования CPU↔GPU; держим веса/активности в общедоступных `MTLBuffer` и кормили их напрямую в графы; heap‑пул снижает стоимость аллокаций и уменьшает фрагментацию.
- **Dynamic Caching** в M4 GPU улучшает загрузку локальной памяти → выше утилизация в матмуль/attention.
- **bfloat16** в MPSGraph/Metal даёт баланс скорость/стабильность для тренировки на GPU.
- **Фьюзинг SDPA** (attention) + тайлинг по threadgroup memory уменьшит проходы по DRAM.

---

## 6) Риски/неизвестности и обходные пути

- MPSGraph не содержит «готового» MHA — придётся собирать из матмулов и softmax; на горячем пути лучше кастомный Metal‑kernel (Теория, ≈ 80%).
- Конвертация T5 в Core ML: качество/скорость зависят от версий `coremltools` и поддержанных операторов; fallback — отключить decoder на первых шагах (Мнение, ≈ 60%).
- Порт токенизатора WordPiece: стоит начать с чистого Swift; позже — оптимизировать (SIMD, mmap словаря).

---

## 7) Мини‑скелеты (ориентиры)

```swift
// Tensor.swift (набросок)
public struct Tensor {
    public let shape: [Int]
    public let dtype: DType
    public let device: Device

    // CPU storage
    var cpu: UnsafeMutableRawPointer?
    // GPU storage
    var mtlBuffer: MTLBuffer?

    // init / view / to(device:) / cast / etc.
}
```

```swift
// GraphModules.swift (набросок)
final class TextToCubeEncoderG {
    let graph = MPSGraph()
    // nodes: embeddings, pos, mha blocks, mlp projection
    // build(inputs: tokenIDs, mask, weights) -> MPSGraphTensor
}
```

```swift
// Tokenizer.swift (минимальный WordPiece)
struct WordPieceTokenizer {
    let vocab: [String: Int]
    func encode(_ text: String, maxLen: Int) -> [Int] { /* ... */ }
}
```

---

## 8) Бенчмарк‑план

- Наборы: `B ∈ {1, 8, 32, 128}`, `L ∈ {64, 128, 256}`; dtype: `fp32`, `fp16`, `bf16`.
- Метрики: время шага (forward/backward), утилизация GPU (через timestamps), потребление памяти, точность лосса.
- Сценарии: CPU‑реф, GPU‑MPSGraph, GPU‑MPSGraph(+fused attention).

---

## 9) Следующие шаги (сейчас)

1) Создать каркас Swift‑проекта, добавить зависимости (Metal, Accelerate, MPSGraph).  
2) Реализовать CPU‑операторы и WordPiece.  
3) Собрать граф `TextToCubeEncoder` на MPSGraph (inference), затем добавить backward.  
4) Подключить `EnergyEmbeddingMapper` и провести сквозной тест до поверхности.  
5) Подготовить дорожку для Core ML‑T5 (или временно отключить декодер).

---

## Приложение: соответствие форм/операций

- `TextToCubeEncoder`: токены `[B,L]` → `[B,L,256]` → `[B,L,256]` (2×EncoderLayer) → masked‑avg `[B,256]` → MLP `[B,surface_dim]`.  
- `EnergyEmbeddingMapper`: `[B,768]` → `[B,H*W]` + позиционное поле `[H,W]`.  
- `EnergyLattice`: работает с нормализованными координатами; создаёт потоки из `[B,H*W]` в центре `Z=depth/2`, затем собирает поверхность/эмбеддинг.  
- `SimpleNeuron`: `(x,y,z)+energy_scalar` → `[hidden]` → `[out]` (для GRU‑carrier).

