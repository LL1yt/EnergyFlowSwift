# Экспорт DistilBERT → Core ML + Swift токенизация (общий план)

> Цель: получить **DistilBERT** в формате **Core ML** (ML Program, затем `.mlmodelc`) с выходом `last_hidden_state`, а в **Swift** обеспечить корректную **WordPiece**‑токенизацию и воспроизводимость эмбеддингов. План ориентирован на экспорт через **Apple coremltools** и дальнейшую интеграцию в iOS/macOS (SPM/Xcode).

---

## 0) Предварительные требования

- macOS + Xcode (Command Line Tools).
- Python 3.10+ с `coremltools`, `transformers`, а также PyTorch **или** TensorFlow 2 (для выбора пути экспорта).
- Доступ к весам и токенайзеру **DistilBERT** (напр. `distilbert-base-uncased`). Файлы: `config.json`, `vocab.txt`, `tokenizer.json`, веса (`pytorch_model.bin` или `tf_model.h5`).

Практика: фиксируйте **максимальную длину** (`max_seq_length`, например 128/192/256) — это упрощает конвертацию/оптимизацию и повышает стабильность рантайма Core ML.

---

## 1) Получение артефактов DistilBERT

- Скачайте модель (например, `distilbert-base-uncased`) и сопутствующие файлы (`vocab.txt`, `tokenizer.json`).  
- Выберите «базовую» (не узкоспециализированную) контрольную точку, если вам нужны **скрытые состояния** для эмбеддингов/пулинга, а не только готовый классфикатор.

**Важно:** Имена токенов/вокаб должны строго соответствовать используемому токенизатору в Swift — это ключ к一致ности эмбеддингов между Python/Swift.

---

## 2) Варианты экспорта в Core ML (через coremltools)

### Вариант A — из TensorFlow 2
1. Загрузите `TFDistilBertModel` или `TFDistilBertForMaskedLM`.
2. Постройте Keras‑модель, у которой **выход** — `last_hidden_state` (тензор формы `[batch, seq_len, hidden]`, как правило `hidden=768`):
   - либо используйте голову `TFDistilBertModel` (без классификатора), 
   - либо оберните `TFDistilBertForMaskedLM` и выберите нужный выход (скрытые состояния).
3. Конвертируйте в Core ML:
   - `ct.convert(tf_model, convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL)`
4. Сохраните результат как `.mlpackage` (`mlmodel.save(...)`).

### Вариант B — из PyTorch (удобно, если стартуете с HF-весов)
1. Загрузите `AutoModel`/`DistilBertModel` и выведите `last_hidden_state`.
2. (Опционально) Трассируйте модель (`torch.jit.trace`) на фиктивных входах (`input_ids`, `attention_mask`) с выбранной длиной последовательности.
3. Конвертируйте через `coremltools` (или через экспортёр Hugging Face), аналогично Варианту A, с `convert_to="mlprogram"` и фиксированными входами `int32` формы `[1, max_seq_length]`.

**Рекомендации по форме входов:**
- Входы: `input_ids: int32[1, L]`, `attention_mask: int32[1, L]` (опционально `token_type_ids`, если нужна совместимость с BERT‑стеком).  
- Жёстко фиксируйте `L=max_seq_length`. Динамические формы часто усложняют жизнь на рантайме Core ML.

---

## 3) Компиляция и подготовка `.mlmodelc`

У вас будет `.mlpackage` или `.mlmodel`. Для быстрой загрузки в приложении:
- Либо соберите `.mlmodelc` заранее из `.mlmodel`:
  ```bash
  xcrun coremlcompiler compile DistilBERT.mlmodel .
  xcrun coremlcompiler generate DistilBERT.mlmodel . --language Swift
  ```
  Это создаст папку `DistilBERT.mlmodelc` и Swift‑обёртку класса модели.
- Либо используйте `.mlpackage` напрямую в Xcode (скомпилируется при первой инициализации), а затем кэшируйте **скомпилированный** путь; для крупных моделей это ускоряет повторные загрузки.

**Формат**: предпочтителен **ML Program** (по умолчанию для целевых версий iOS 15+/macOS 12+).

---

## 4) Интеграция в Swift (SPM/Xcode)

### 4.1 Добавление ресурса
- Если это **Swift Package**: положите `DistilBERT.mlmodelc` в `Sources/YourTarget/` и пропишите в `Package.swift` как ресурс:  
  ```swift
  .target(
    name: "YourTarget",
    dependencies: [],
    resources: [.process("DistilBERT.mlmodelc")]
  )
  ```
  В коде загрузите модель:
  ```swift
  let url = Bundle.module.url(forResource: "DistilBERT", withExtension: "mlmodelc")!
  let model = try MLModel(contentsOf: url, configuration: MLModelConfiguration())
  ```

- Если это **Xcode‑проектор**: добавьте `.mlpackage`/`.mlmodel` в проект, Xcode сам сгенерирует класс и скомпилирует в `.mlmodelc` при сборке.

### 4.2 Выбор вычислительных устройств
```swift
var config = MLModelConfiguration()
config.computeUnits = .all   // CPU + GPU + ANE
// ...
let model = try DistilBERT(configuration: config) // или через MLModel(contentsOf:)
```

---

## 5) Токенизация в Swift (WordPiece) — 2 проверенных подхода

### Подход 1 — современный: `swift-transformers` (SPM)
- Подтянуть пакет `huggingface/swift-transformers`, модуль **Tokenizers** (есть готовый `BertTokenizer`).
- Использовать `AutoTokenizer.from(pretrained: ...)` **или** загрузить локально `vocab.txt`/`tokenizer.json` и инициализировать BERT‑совместимый токенизатор.
- Преимущества: поддержка разных семейств токенизаторов, обновляемая экосистема, удобные API и документация.

### Подход 2 — исторический/минимальный: `swift-coreml-transformers` (архив)
- В репозитории есть чистые реализации `BasicTokenizer` и `WordpieceTokenizer` на Swift; можно прямо встроить их в проект, если нужна «тонкая» ручная совместимость с BERT‑WordPiece и максимально лёгкие зависимости.
- Минус: репозиторий **архивирован**, обновлений нет; используйте как эталонный код.

**Общая проверка корректности:**
1. Сформируйте набор тестовых строк (20–50 примеров, включая Unicode/пунктуацию).
2. Сравните результат токенизации Swift и Python/HuggingFace (ID и `attention_mask` должны совпасть).  
3. Зафиксируйте те же `vocab.txt` и правила нормализации (cased/uncased, lowercasing, strip accents и т.д.).

---

## 6) Прогон и пулинг эмбеддингов

1. Препроцессинг (Swift): токенизируете текст → `input_ids`, `attention_mask` (фикс. длина, паддинг).  
2. Прямой проход Core ML‑модели → получите `last_hidden_state` (`[1, L, 768]`).  
3. Пулинг: **masked mean pooling** по временной оси (учитывая `attention_mask`) или `[CLS]`‑пулинг — в зависимости от вашего пайплайна.  
4. (Опционально) L2‑нормализация эмбеддинга.

---

## 7) Производительность и стабильность (ANE/on‑device best practices)

- **Фиксируйте длину** `L` (128/192/256) и последовательности ввода одинаковой формы — это повышает шанс полной/частичной выгрузки на NE/GPU и снижает накладывание перегонов между устройствами.  
- Выбирайте `convert_to="mlprogram"` и `compute_units=ALL` (ANE + GPU + CPU).  
- При чувствительности к точности можно сохранить модель c `compute_precision=FLOAT32` (в ML Program) — но учтите возможные ограничения на NE.  
- Для быстрых повторных запусков кешируйте `.mlmodelc` (путь компиляции) и используйте «compiled model» API для мгновенной инициализации.  
- Профилируйте в Xcode (вкладка **Performance** у модели).

---

## 8) Частые ошибки и как их избежать

- **Несовпадение токенизации** (разные `vocab.txt`/правила): держите юнит‑тест, сверяющий Swift vs Python.
- **Динамические формы** → краши/деградация: фиксируйте `seq_len`, не полагайтесь на произвольные длины.  
- **Неверные типы входов**: используйте `int32` для `input_ids`/`attention_mask` и корректные имена входов, соответствующие сохранённой спецификации модели.  
- **Компиляция в SPM**: для пакетов чаще требуется заранее скомпилированный `.mlmodelc` и явное объявление ресурса в `Package.swift`.

---

## 9) Быстрый чек‑лист команд

```bash
# (A) Экспорт (пример из TF2; адаптируйте под TFDistilBertModel c last_hidden_state)
python - <<'PY'
import numpy as np, coremltools as ct, tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
max_len = 128
tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
inp = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
out = distil({"input_ids": inp, "attention_mask": mask}).last_hidden_state
tf_model = tf.keras.Model(inputs=[inp, mask], outputs=out)
mlmodel = ct.convert(tf_model, convert_to="mlprogram")
mlmodel.save("DistilBERT_LastHidden.mlpackage")
PY

# (B) Компиляция (если нужен .mlmodelc и Swift-класс)
xcrun coremlcompiler compile DistilBERT_LastHidden.mlmodel ./
xcrun coremlcompiler generate DistilBERT_LastHidden.mlmodel . --language Swift
```

---

## 10) Где взять готовые куски кода (для копипаста)

- **Токенизация в Swift (актуальный пакет)**: `huggingface/swift-transformers`, модуль **Tokenizers** — используйте `BertTokenizer`/`AutoTokenizer`.
- **Исторические реализации WordPiece/BasicTokenizer (Swift)**: `huggingface/swift-coreml-transformers` — можно заимствовать минимальные реализации.
- **Оптимизированный конвейер под Apple Neural Engine (DistilBERT)**: пример пошаговой оптимизации и экспорта (включая `ct.convert` → `.mlpackage`) — пригоден как референс для стабильных форм и настройки конвертера.
- **Интеграция модели в Swift Package**: примеры добавления `.mlmodelc` как ресурса и генерации Swift‑обёртки через `coremlcompiler`.

---

### Итог

- Экспортируйте **базовый DistilBERT** с выходом `last_hidden_state` → **ML Program** → соберите `.mlmodelc`.  
- В Swift используйте **WordPiece**‑токенизацию, совместимую с `vocab.txt` модели (через `swift-transformers` либо лёгкие реализации BERT‑токенайзера).  
- Фиксируйте формы, профилируйте на устройстве, при необходимости используйте ANE‑ориентированные практики.
