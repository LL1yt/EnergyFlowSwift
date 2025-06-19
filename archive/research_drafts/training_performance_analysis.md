# 📊 Training Performance Analysis

Детальный анализ влияния параметров на производительность и качество обучения 3D клеточной нейронной сети

## 🎯 Основные факторы влияния

### 1. 🚀 Batch Size

**Влияние на скорость:**

- ✅ **Больший batch = быстрее обучение** (меньше итераций на эпоху)
- ✅ Лучшее использование GPU/векторизации
- ❌ Но есть предел после которого прирост минимален

**Влияние на память:**

- ❌ **Квадратичный рост памяти** с увеличением batch size
- GPU memory = batch_size × embedding_dim × lattice_neurons
- Пример: batch=128 vs batch=32 = 4x больше памяти

**Влияние на качество:**

- 📈 Малые batch (16-32): лучше генерализация, более шумный градиент
- 📉 Большие batch (128+): может застревать в локальных минимумах
- 🎯 **Оптимум: 32-64** для большинства задач

### 2. 💾 Dataset Size

**Память во время обучения:**

- ✅ **Почти не влияет** - DataLoader загружает по batch-ам
- Только активный batch в памяти

**Память на диске:**

- ❌ **Прямо пропорциональна** размеру датасета
- SNLI 109k примеров = 643 MB precomputed embeddings
- Полный SNLI 549k примеров ≈ 3.2 GB

**Время обучения:**

- ❌ **Линейно растет** с размером датасета
- 1000 примеров ≈ 2 минуты
- 10000 примеров ≈ 20 минут
- 50000 примеров ≈ 100 минут

### 3. 🧠 Lattice Dimensions

**Влияние на память:**

- ❌ **Кубический рост**: xs × ys × zs
- Small (10×10×10) = 1,000 neurons
- Medium (20×15×15) = 4,500 neurons
- Large (30×20×20) = 12,000 neurons

**Влияние на качество:**

- 📈 Больше нейронов = больше параметров = больше возможностей
- 📉 Но риск переобучения на малых датасетах
- 🎯 **Правило**: neurons ≈ dataset_size / 10

### 4. 📐 Embedding Dimensions

**Стандартные размеры:**

- sentence-transformers/all-MiniLM-L6-v2: **384** (легкий)
- sentence-transformers/all-mpnet-base-v2: **768** (стандарт)
- text-embedding-ada-002: **1536** (тяжелый)

**Влияние:**

- 💾 Память: линейно растет с размером
- 🎯 Качество: больше dimension = лучше представление
- ⚡ Скорость: больше dimension = медленнее

### 5. 🌡️ Learning Rate & Warm-up

**Learning Rate:**

- 🎯 **Критически важен** для сходимости
- Слишком большой: divergence, NaN loss
- Слишком малый: очень медленное обучение
- **Оптимум**: 1e-4 to 1e-3 для AdamW

**Warm-up (новая функция):**

- ✅ **Автоматически применяется** при resume
- 📈 Эпохи 1-3: постепенно увеличиваем LR от 20% до 100%
- 🛡️ Защищает от "забывания" весов при resume

## 📈 Практические рекомендации

### 🎮 Для быстрого тестирования:

```bash
# Оптимальные параметры для скорости
--mode development      # Маленькая решетка
--dataset-limit 1000    # Быстрый датасет
--batch-size 64         # Хороший баланс
--additional-epochs 5   # Короткое обучение
```

### 🧪 Для экспериментов:

```bash
# Сбалансированные параметры
--mode research         # Средняя решетка
--dataset-limit 5000    # Достаточно данных
--batch-size 32         # Лучшее качество
--additional-epochs 15  # Достаточно эпох
```

### 🚀 Для продакшена:

```bash
# Максимальное качество
--mode production       # Большая решетка
--dataset-limit 50000   # Много данных
--batch-size 64         # Компромисс
--additional-epochs 30  # Долгое обучение
```

## ⚖️ Memory vs Speed vs Quality

### Memory-Constrained (мало RAM/GPU):

```
Lattice: development mode (small)
Batch Size: 16-32
Dataset: 1000-5000 examples
Embedding: 384-dim (MiniLM)
```

### Speed-Focused (быстрое обучение):

```
Lattice: development mode
Batch Size: 128 (если влезает)
Dataset: 2000-5000 examples
Embedding: 384-dim
Epochs: 5-10
```

### Quality-Focused (лучший результат):

```
Lattice: research/production mode
Batch Size: 32-64
Dataset: 20000-50000 examples
Embedding: 768-dim (mpnet)
Epochs: 20-50 with automated training
```

## 🔧 Автоматическая оптимизация

### 🤖 Automated Training:

Автоматически выбирает оптимальную стратегию:

**Stage 1: Foundation** (быстрое изучение основ)

- Small dataset, many epochs
- Focus на скорость обучения основных паттернов

**Stage 2-3: Consolidation** (консолидация знаний)

- Medium dataset, balanced epochs
- Focus на стабильность и качество

**Stage 4-5: Mastery** (финальная полировка)

- Large dataset, few epochs
- Focus на генерализацию и производительность

### 📊 Smart Resume:

- ✅ **Автоматический warm-up** при загрузке checkpoint
- 🎯 **Совместимость моделей** на основе архитектуры
- 🔄 **Адаптивный learning rate** в зависимости от прогресса

## 🚨 Предупреждения и лимиты

### Memory Limits:

```
Windows 32GB RAM:
- Max batch size ≈ 128-256 (зависит от lattice)
- Max dataset ≈ 100k examples (в памяти по batch-ам)
- Max lattice ≈ 50×30×30 neurons

GPU 8GB VRAM:
- Max batch size ≈ 64-128
- Embedding dim ≤ 768 recommended
```

### Performance Bottlenecks:

```
Главные узкие места:
1. Lattice size (cubic growth)
2. Embedding forward pass
3. Surface projection operations
4. Loss computation and backprop

Оптимизации:
- Gradient clipping (max_norm=1.0)
- Memory cleanup каждые 10 эпох
- Mixed precision (будущая фича)
```

## 📋 Checklist для оптимального обучения

### Перед запуском:

- [ ] Проверить доступную память (RAM/GPU)
- [ ] Выбрать режим под размер решетки
- [ ] Настроить batch size под память
- [ ] Оценить время обучения

### Во время обучения:

- [ ] Мониторить loss и similarity
- [ ] Следить за использованием памяти
- [ ] Проверять warm-up логи при resume
- [ ] Сохранять промежуточные результаты

### После обучения:

- [ ] Анализировать final vs best similarity
- [ ] Проверять сохраненные checkpoints
- [ ] Планировать следующие эксперименты
- [ ] Документировать найденные оптимальные параметры

---

**🎯 Основной принцип:** Начинать с малого, постепенно увеличивать сложность, использовать автоматизацию для долгого обучения!
