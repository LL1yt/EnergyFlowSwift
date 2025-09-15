# AA → Swift/MPS/Metal (Apple Silicon) — Объединённый план порта

Стратегия по переносу ключевых частей energy_flow на Swift с использованием MPSGraph/Metal, сохраняя архитектурные инварианты и добавляя пошаговую, понятную даже новичку, дорожную карту.

—

## 0) Контекст и цель

- Исходный проект: Python/PyTorch архитектура energy_flow (см. AA/energy_flow)
- Цель: получить прототип тренируемой версии на Apple Silicon (M4), с модульной структурой и возможностью «собирать по кубикам»
- Подход: начать с минимально работающего CPU-варианта, затем переносить узлы на MPSGraph/Metal

—

## 1) Инварианты и требования (из текущей архитектуры)

- Формы и размеры
  - surface_dim = lattice_width × lattice_height
  - Нормализация координат решётки в диапазон [-1, 1]
  - TextToCubeEncoder: токены → эмбеддинги (256) → 2×TransformerEncoder → агрегирование (masked-avg) → MLP → Tanh → [B, surface_dim]
- Минимальный набор операций
  - Линейные слои (GEMM + bias), LayerNorm, GELU, Tanh, Add/Mul
  - Softmax, Multi-Head Attention (MHA), Dropout (train), embedding (gather)
  - Reduce/reshape: mean/sum, masked mean, reshape/view/concat/transpose
- Автодифференцирование
  - Цель: использовать autodiff MPSGraph, плюс точечные ручные градиенты для простых op при необходимости
- Память
  - Unified memory (shared/private), MTLHeaps для пула, смешанная точность (fp16/bf16) позже

—

## 2) Архитектура Swift-проекта

SwiftPM пакет: EnergyFlowSwift

- Sources/EFCore/
  - Tensor/: базовый тензор и операции (CPU-референс)
  - (дальше) Ops/: обёртки под Accelerate/MPSGraph/Metal (постепенно)
  - Autograd/: mini-tape (позже)
- Sources/PyTorchSwift/
  - Аналоги PyTorch модулей: Embedding, Linear, LayerNorm, Activations, (позже) Attention/TransformerEncoder
- Sources/EnergyFlow/
  - Utils/: Config (DEBUG/EXPERIMENT/OPTIMIZED), Logger/Profiler (позже)
  - TextBridge/: Tokenizer, TextToCubeEncoder (первый модуль)
  - Core/: SimpleNeuron, EnergyLattice, FlowProcessor, EmbeddingMapper (позже)
- Tests/
  - Юнит-тесты на форму и простую корректность

Принципы
- Единый Tensor API, поначалу CPU-only, затем точечно подключаем GPU
- Фиксированные формы для кэша графов (padding в токенах)
- Модульность и «сборка по кубикам»: каждый модуль независим и тестируем

—

## 3) Дорожная карта (интегрированная)

Этап A — База (CPU-референс + каркас)
- Tensor (CPU): shape, dtype, data; элементарные операции
- Ops: linear, gelu, layer_norm, tanh, masked_mean, embedding (gather) — CPU
- Мини-autograd (опционально позже) — для первых опытов можно без обучения
- Tokenizer: упрощённый WordPiece/BPE (пока whitespace)
- DoD: воспроизвести forward TextToCubeEncoder на CPU

Этап B — MPSGraph (GPU быстрый путь)
- Обёртка GraphModule: сборка подграфов (GEMM, GELU, LN, Softmax)
- MultiHeadAttention на MPSGraph, смешанная точность fp16/bf16
- Полное отображение TextToCubeEncoder на MPSGraph

Этап C — Energy-модули
- EmbeddingMapper (MLP), SimpleNeuron (MLP), EnergyLattice/FlowProcessor (частично CPU, горячие места — Metal)

Этап D — Text Bridge (decoder)
- Временно отключить или конвертировать T5-small в Core ML (позже)

Этап E — Память и производительность
- Пул ресурсов (MTLHeap), фьюзинг, смешанная точность, бенчмарки

—

## 4) Первая фаза: подробный пошаговый план (для старта и понимания)

Шаг 1 — Структура проекта и базовые типы
- Создать SwiftPM-пакет EnergyFlowSwift с таргетами EFCore, PyTorchSwift, EnergyFlow
- В EFCore/Tensor реализовать:
  - Tensor(shape:[Int], data:[Float]) с помощниками: zeros, randomUniform, count
  - Простую адресацию по индексам (flattenIndex/strides)
- Результат: можно хранить и передавать тензоры нужной формы

Шаг 2 — Аналоги PyTorch (минимум)
- PyTorchSwift/Embedding: таблица весов [vocab, d], forward(ids[B,L]) → [B,L,d]
- PyTorchSwift/Linear: веса [out,in], биас [out], forward([B,in]) → [B,out]
- PyTorchSwift/LayerNorm: по последней размерности, eps=1e-5
- PyTorchSwift/Activations: GELU (аппроксимация), Tanh
- Результат: можно строить простые сети из знакомых блоков

Шаг 3 — Токенизация (минимальная)
- TextBridge/Tokenizer: whitespace токенизация + словарь с [PAD]=0, [UNK]=1
- encodeBatch: возвращает ids[B,L], attentionMask[B,L] с паддингом
- Результат: входы для энкодера готовы, даже без внешних зависимостей

Шаг 4 — TextToCubeEncoder (CPU)
- Архитектура на старте (упростим):
  - Embedding(vocab, 256) → синусоидальный positional encoding → (пока заглушка Transformer = identity)
  - Masked-avg → Linear(256→256) → GELU → LayerNorm → Linear(256→surface_dim) → Tanh
- Проверка: для B=2, width×height=20×20, выход shape = [2, 400]
- Результат: рабочий forward, легко расширяемый до полноценного Transformer

Шаг 5 — Тесты
- Тест на форму: encode(["hello", "swift mps"]) → [2, surface_dim]
- Тест на диапазон: значения после Tanh ∈ [-1, 1]
- Результат: базовая уверенность в корректности пайплайна

Примечания для новичка
- Embedding — это просто «таблица» вещественных векторов, индексируемая токенами
- Masked-avg — среднее по непустым токенам, чтобы паддинги не влияли
- LayerNorm — стабилизация распределения признаков по последней размерности
- Transformer можно добавить позже: наш интерфейс уже готов, сейчас это «кубики», которые легко заменить на реализацию MHA+FFN

—

## 5) Верификация с Python-эталоном

- Сравнить формы и базовые статистики (mean/std) на одинаковых сидированиях и входе
- Допуски: fp32 ≤ 1e-4; для fp16/bf16 позднее — ≤ 3e-3
- Для строгой численной сверки потребуется одинаковая токенизация и совпадение инициализаций

—

## 6) Производительность и память (куда «поедет» на M4)

- Unified memory: shared для входов/выходов, private — для весов
- MPSGraph: ускорение GEMM/Attention/Norm/Act; смешанная точность
- Metal kernels: для горячих мест (masked mean, квантование координат) на этапе C/E

—

## 7) Риски/обходные пути

- Реализация MHA в MPSGraph потребует аккуратной сборки из matmul+softmax
- Токенизатор DistilBERT: начнём с простого; полноценный WordPiece подключим позднее
- Автоград: начнём без тренировки; затем подключим MPSGraph autodiff

—

## 8) Чек-лист инвариантов

- Координаты решётки всегда clamp в [-1,1]
- surface_dim согласован между модулями
- Переиспользуем буферы (когда появится GPU-ветка)
- Новые тензоры по умолчанию соответствуют текущему режиму (DEBUG/EXPERIMENT/OPTIMIZED)

—

## 9) Быстрый старт

- Собрать и запустить тесты: `swift build` / `swift test`
- Запуск первого модуля: TextToCubeEncoder.encode(["hello world"]) → Tensor[B, surface_dim]
- Дальше: заменить заглушку Transformer на реальную реализацию, перенести на MPSGraph