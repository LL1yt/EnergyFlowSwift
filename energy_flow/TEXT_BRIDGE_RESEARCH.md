# Text Bridge Research для Energy Flow Architecture

## 🔍 Анализ задачи

**Цель**: Создать двунаправленный преобразователь между эмбеддингами куба и текстом для архитектуры energy_flow, позволяющий:
- Преобразовывать текст в эмбеддинги поверхности куба
- Восстанавливать текст из эмбеддингов поверхности куба
- Кэшировать известные пары для оптимизации
- Участвовать в процессе обучения для контроля качества
- Обеспечить возможность общения с моделью на естественном языке

---

## ❌ Ошибки первоначального подхода

### 1. Избыточная сложность архитектуры
- **Проблема**: Использование полноценного DistilBERT (66M параметров) для простой задачи инверсии эмбеддингов
- **Последствие**: Неоправданная вычислительная нагрузка и сложность интеграции

### 2. Неправильные размерности
- **Проблема**: Фиксация на 768D (размер BERT эмбеддингов) вместо размеров поверхности куба
- **Реальность**: Для `create_debug_config()` размер поверхности = 20×20 = 400D
- **Последствие**: Несовместимость с `EnergyEmbeddingMapper` и `EnergyOutputCollector`

### 3. Устаревший подход к управлению устройствами
- **Проблема**: Ручное перемещение тензоров на GPU через `.to(device)`
- **Реальность**: В energy_flow уже настроен `torch.set_default_device('cuda')`
- **Последствие**: Дублирование логики и потенциальные проблемы с device mismatch

### 4. Архитектурная несовместимость
- **Проблема**: Игнорирование существующих компонентов `embedding_mapper.py`
- **Последствие**: Необходимость дублирования функциональности

---

## 🔬 Исследование современных решений (2024)

### Vec2Text: State-of-the-Art Embedding Inversion

**Ключевые характеристики**:
- 92% exact recovery для 32-токенных текстов
- BLEU score 97.3 на in-domain данных
- Cosine similarity > 0.99 между оригинальными и восстановленными эмбеддингами

**Архитектура**:
```
Эмбеддинг → T5-base (220M) → Iterative Correction → Текст
```

**Процесс обучения**:
1. Zero-step модель: генерация текста из эмбеддингов
2. Hypothesis generation: создание гипотез для training data
3. Correction model: обучение на тройках (true_embedding, hypothesis, hypothesis_embedding)

**GitHub**: `vec2text/vec2text` - основная реализация

### InvBERT: Lightweight Alternative

**Характеристики**:
- InvBERT Classify: 24M параметров
- InvBERT Seq2Seq: 93M параметров
- Обучение на одной Tesla V100-PCIe-32GB

**Преимущества**:
- Значительно меньше параметров чем vec2text
- Специализация на BERT-like embeddings

### MultiVec2Text (ACL 2024)

**Новизна**: Multilingual extension с поддержкой множественных языков
**Особенность**: Embedding space alignment - перенос обученной модели на новые энкодеры через affine mapping

---

## ✅ Исправленная архитектура

### Концепция

Вместо полноценной LLM используем **lightweight embedding inversion** с адаптацией под размеры поверхности куба energy_flow.

### Ключевые принципы

1. **Surface-aware dimensions**: Работа с размерами поверхности куба (width × height)
2. **Lightweight models**: Использование T5-small (~60M параметров) вместо больших моделей
3. **Default device compatibility**: Использование настроенного `torch.set_default_device('cuda')`
4. **Existing components integration**: Совместимость с `EnergyEmbeddingMapper`/`EnergyOutputCollector`

### Архитектурная схема

```
┌─────────────────────────────────────────────────────────────┐
│                     Energy Flow Cube                        │
│  Input Surface (400D) ←→ 3D Lattice ←→ Output Surface (400D) │
└─────────────────────────────────────────────────────────────┘
           ↑                                        ↓
    ┌─────────────────┐                    ┌─────────────────┐
    │TextToCubeEncoder│                    │CubeToTextDecoder│
    │  Text → 400D    │                    │  400D → Text    │
    │   (~5M params)  │                    │  (~60M params)  │
    └─────────────────┘                    └─────────────────┘
           ↑                                        ↑
      ┌─────────┐                              ┌─────────┐
      │ TextCache │←──────── Caching ──────────→│ TextCache │
      └─────────┘                              └─────────┘
```

---

## 🤔 Анализ архитектурных решений: Единая vs Раздельная модель

### Вопрос: Нужны ли отдельные модели или можно сделать одну двунаправленную?

#### Подход 1: Единая двунаправленная модель

**Потенциальные преимущества:**
- Меньше параметров (одна модель вместо двух)
- Общие представления для текста и эмбеддингов куба
- Проще в управлении и обслуживании
- Естественная симметрия: text ↔ cube_embedding

**Архитектура единой модели:**
```python
class BidirectionalTextCubeConverter(nn.Module):
    def __init__(self, config):
        self.surface_dim = config.lattice_width * config.lattice_height
        
        # Общий transformer backbone
        self.shared_transformer = T5Model.from_pretrained('t5-small')
        
        # Модальностные адаптеры
        self.text_adapter = nn.Linear(512, 512)
        self.surface_adapter = nn.Linear(self.surface_dim, 512)
        
        # Проекционные головы
        self.to_surface_head = nn.Linear(512, self.surface_dim)
        self.to_text_head = nn.Linear(512, vocab_size)
    
    def forward(self, input_data, mode='text_to_cube'):
        if mode == 'text_to_cube':
            return self.encode_text_to_surface(input_data)
        else:
            return self.decode_surface_to_text(input_data)
```

#### Подход 2: Раздельные специализированные модели ✅ **РЕКОМЕНДУЕТСЯ**

**Почему раздельные модели лучше:**

1. **Асимметричность задач**:
   - **Text → Surface**: понимание смысла + пространственная проекция (2D сетка)
   - **Surface → Text**: интерпретация пространственных паттернов + генерация последовательности

2. **Разная сложность задач**:
   - **Кодирование**: относительно простая задача, много данных для supervised learning
   - **Декодирование**: сложная задача инверсии эмбеддингов, требует специальных техник (итеративная коррекция)

3. **Исследования 2024 года подтверждают**:
   - Vec2text использует итеративную коррекцию для инверсии
   - InvBERT показывает, что embedding inversion - это отдельная специализированная задача
   - Прямое кодирование и инверсия требуют разных подходов

4. **Независимое обучение**:
   - **TextToCubeEncoder**: обучается в основном цикле с EnergyTrainer (supervised)
   - **CubeToTextDecoder**: pre-training на synthetic данных + валидация

5. **Модульность для исследований**:
   - Можно экспериментировать с разными архитектурами для каждой задачи
   - Легче отладка и анализ каждого компонента
   - Возможность независимого fine-tuning

### Итоговое решение: Гибридный подход с улучшенными названиями

**Переименование для лучшего понимания:**
- ~~SurfaceTokenizer~~ → **TextToCubeEncoder** 
- ~~LightweightInverter~~ → **CubeToTextDecoder**

**Обоснование названий:**
- `TextToCubeEncoder`: четко показывает направление и назначение
- `CubeToTextDecoder`: подчеркивает специализацию на декодировании

---

## 🏗️ Детальный план реализации

### 1. Модульная структура (обновленная)

```
energy_flow/text_bridge/
├── __init__.py                    # ✅ Создан
├── text_to_cube_encoder.py        # TextToCubeEncoder - текст в эмбеддинги поверхности
├── cube_to_text_decoder.py        # CubeToTextDecoder - эмбеддинги поверхности в текст  
├── text_cache.py                  # LRU кэш для surface_embeddings ↔ text
├── bridge_integration.py          # Интеграция с EnergyTrainer
└── bridge_trainer.py              # Отдельный тренер для text bridge компонентов
```

### 2. TextToCubeEncoder

**Назначение**: Преобразование текста в эмбеддинги поверхности куба (400D/2500D/10000D)

**Архитектура** (lightweight, ~5M параметров):
```python
class TextToCubeEncoder(nn.Module):
    def __init__(self, config):
        self.surface_dim = config.lattice_width * config.lattice_height
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Lightweight но эффективная архитектура
        self.text_encoder = nn.Sequential(
            nn.Embedding(self.tokenizer.vocab_size, 256),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(256, nhead=8, dim_feedforward=512),
                num_layers=2
            ),
            nn.Linear(256, self.surface_dim),
            nn.Tanh()  # Нормализация в [-1, 1] для совместимости с энергией
        )
```

**Особенности**:
- Динамическая адаптация к размерам куба из конфига  
- Обучается на реальных парах (text, surface_embedding) в основном цикле
- Совместимость с `EnergyEmbeddingMapper.forward()`

### 3. CubeToTextDecoder

**Назначение**: Восстановление текста из эмбеддингов поверхности куба (специализированная инверсия)

**Базовая архитектура** (на основе vec2text, ~60M параметров):
```python
class CubeToTextDecoder(nn.Module):
    def __init__(self, config):
        self.surface_dim = config.lattice_width * config.lattice_height
        
        # T5-small backbone для качественной инверсии
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # Адаптер surface_dim → T5 hidden size
        self.surface_adapter = nn.Linear(self.surface_dim, 512)
        
        # Итеративная коррекция (из vec2text research)
        self.correction_steps = config.iterative_correction_steps  # 3-5 шагов
```

**Стратегия обучения**:
1. **Pre-training**: обучение на synthetic surface embeddings → простые тексты
2. **Transfer learning**: адаптация vec2text моделей к нашим размерностям
3. **Validation-only**: не обучается в основном цикле, только валидация

### 4. TextCache

**LRU кэш** для часто используемых пар:

```python
from functools import lru_cache
import pickle

class SurfaceTextCache:
    def __init__(self, max_size=10000, cache_file="surface_text_cache.pkl"):
        self.cache = {}
        self.max_size = max_size
        self.cache_file = cache_file
        self.enabled = False
        
    def get(self, surface_embedding_hash) -> Optional[str]:
        # LRU логика + персистентное хранение
        
    def set(self, surface_embedding_hash, text):
        # Обновление кэша с LRU eviction
```

### 5. Интеграция с EnergyConfig

**Новые параметры**:

```python
@dataclass
class EnergyConfig:
    # ... существующие параметры ...
    
    # Text Bridge параметры
    enable_surface_text_bridge: bool = False
    surface_text_cache_enabled: bool = False
    surface_text_validation_interval: int = 10
    bridge_model_size: str = "small"  # small/base/large
    iterative_correction_steps: int = 3
    bridge_learning_rate: float = 1e-4
    text_reconstruction_weight: float = 0.1  # Вес в общем loss
```

### 6. Уточненная интеграция с EnergyTrainer

#### Ключевое понимание процесса обучения

**Входная сторона (полноценное обучение)**:
- Используем пары (оригинальный_текст, surface_embedding_через_EnergyMapper)
- Обучаем SurfaceTokenizer в рамках основного training loop
- Много качественных supervised данных

**Выходная сторона (только валидация)**:
- Выходные эмбеддинги куба без ground truth текста
- Только мониторинг качества через LightweightInverter
- Периодическая проверка осмысленности

#### Практическая реализация

**Обучение SurfaceTokenizer (входная сторона)**:

```python
def train_step_with_text_bridge(self, batch):
    # Основное обучение energy_flow
    main_loss = self.compute_main_loss(batch)
    
    # Дополнительно: обучение SurfaceTokenizer
    if self.config.enable_surface_text_bridge:
        original_texts = batch['texts']  # Исходные фразы
        teacher_embeddings = batch['teacher_embeddings']  # 768D
        
        # Получаем целевые surface embeddings через existing mapper
        with torch.no_grad():
            target_surface = self.energy_mapper(teacher_embeddings)  # 400D
        
        # Обучаем encoder предсказывать эти embeddings из текста
        predicted_surface = self.text_to_cube_encoder(original_texts)
        encoder_loss = F.mse_loss(predicted_surface, target_surface)
        
        # Добавляем к общему loss с небольшим весом
        total_loss = main_loss + self.config.text_reconstruction_weight * encoder_loss
    else:
        total_loss = main_loss
    
    return total_loss
```

**Валидация CubeToTextDecoder (выходная сторона)**:

```python
def validate_text_quality(self, epoch):
    if not self.config.enable_surface_text_bridge:
        return
        
    if epoch % self.config.surface_text_validation_interval == 0:
        # Собираем текущие выходы куба (если есть)
        output_embeddings = self.collect_output_surface_embeddings()
        
        if output_embeddings is not None and len(output_embeddings) > 0:
            # Пытаемся восстановить текст из выходных эмбеддингов
            predicted_texts = self.cube_to_text_decoder(output_embeddings)
            
            # Только логирование для контроля качества
            logger.info(f"=== Text Bridge Validation (Epoch {epoch}) ===")
            for i, text in enumerate(predicted_texts[:5]):
                logger.info(f"Output {i}: {text}")
            
            # Автоматические метрики качества (если доступны)
            self.log_text_metrics(predicted_texts)
        else:
            logger.info("No output embeddings available for text validation")
```

---

## 🔄 Двухэтапная стратегия обучения

### Предварительная подготовка CubeToTextDecoder

**Проблема**: CubeToTextDecoder нужен для валидации, но у нас нет данных для его обучения на выходных эмбеддингах куба.

**Решение**: Pre-training на синтетических или внешних данных:

1. **Synthetic data generation**:
   - Генерируем случайные surface embeddings в диапазоне [-1, 1]
   - Обучаем базовую способность преобразования embeddings → meaningful text

2. **Transfer learning**:
   - Используем pre-trained vec2text модели
   - Fine-tuning на наших размерностях (400D вместо стандартных)

3. **Progressive training**:
   - Начинаем с простых фраз
   - Постепенно увеличиваем сложность

### Основная интеграция с EnergyTrainer

**TextToCubeEncoder**:
- Обучается параллельно с основной моделью
- Использует реальные пары (text, surface_embedding) из training pipeline
- Минимальная дополнительная нагрузка на обучение (~5M параметров)

**CubeToTextDecoder**:
- Pre-trained модель для валидации (~60M параметров)
- Не обучается в основном цикле
- Только мониторинг качества выходов

### Преимущества подхода

✅ **Реалистичные данные**: входные пары текст↔surface_embedding из реального training pipeline  
✅ **Эффективность**: обучение только одного компонента (TextToCubeEncoder)  
✅ **Прогрессивная валидация**: качество инверсии растет по мере обучения куба  
✅ **Минимальная интеграция**: не нарушает существующий training loop

---

## ⚠️ Важные технические детали

### Device Management

Используем существующую настройку default device:
```python
# energy_config.py уже содержит:
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)
```

**Что это означает**:
- Все новые тензоры автоматически создаются на GPU
- НЕ нужно явно вызывать `.to(device)` для новых тензоров
- Модели все равно нужно перемещать через `.to(device)`

### Размерности и совместимость

**Критически важно**:
- surface_dim = lattice_width × lattice_height
- Debug config: 20×20 = 400D
- Experiment config: 50×50 = 2500D  
- Optimized config: 100×100 = 10000D

**SurfaceTokenizer должен адаптироваться**:
```python
def __init__(self, config):
    self.surface_dim = config.lattice_width * config.lattice_height
    # Архитектура зависит от surface_dim
```

### Интеграция с существующими компонентами

**Обязательно использовать**:
- `EnergyEmbeddingMapper.forward()` для получения целевых surface embeddings
- `EnergyOutputCollector` для сбора выходных embeddings
- Существующую структуру batching и device management

---

## 📊 Технические характеристики

### Производительность

**Модель размеры**:
- SurfaceTokenizer: ~5M параметров (зависит от vocab size)
- LightweightInverter: ~60M параметров (T5-small)
- **Общий размер**: ~65M параметров (vs 768M+ для полноценной LLM)

**Точность** (ожидаемая на основе vec2text research):
- BLEU score: 85-95% на surface-adapted данных
- Exact recovery: 70-85% для коротких текстов
- Cosine similarity: >0.95 между cyclical embeddings

### Память и вычисления

**GPU память** (RTX 5090 32GB):
- Модель: ~250MB
- Batch processing: зависит от batch_size, ~50MB per batch
- **Остается свободным**: >30GB для основного обучения energy_flow

**Скорость**:
- Text → Surface: ~1ms per sample
- Surface → Text: ~10ms per sample (iterative correction)
- Cache hit: ~0.1ms per sample

---

## 🔗 Ссылки и ресурсы

### GitHub репозитории

1. **vec2text/vec2text** - основная библиотека для embedding inversion
   - https://github.com/vec2text/vec2text
   - Содержит pre-trained модели и примеры использования

2. **siebeniris/MultiVec2Text** - multilingual extension (ACL 2024)
   - https://github.com/siebeniris/MultiVec2Text
   - Embedding space alignment техники

3. **google/embedding-tests** - BERT inversion examples
   - https://github.com/google/embedding-tests/blob/master/inversion_bert.py

### Научные статьи

1. **"Text Embeddings Reveal (Almost) As Much As Text"** (2024)
   - 92% exact recovery результаты
   - Vec2text methodology

2. **"InvBERT: Reconstructing Text from Contextualized Word Embeddings"**
   - Lightweight архитектура (24M-93M параметров)
   - BERT-specific optimizations

3. **"Improving Text Embeddings with Large Language Models"** (ACL 2024)
   - Современные подходы к text embeddings

### Команды для установки

```bash
# Vec2text library
pip install vec2text

# Основные зависимости
pip install transformers torch datasets evaluate

# Для кэширования
pip install diskcache
```

### Примеры использования vec2text

```python
import vec2text

# Загрузка предобученной модели
corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

# Инверсия эмбеддингов
embeddings = model.encode(["Hello world", "How are you?"])
reconstructed = corrector.correct(embeddings)
```

---

## 📅 План реализации (поэтапный)

### Этап 0: Предварительная подготовка (0.5 дня)
- [ ] Pre-training LightweightInverter на synthetic данных
- [ ] Адаптация размерностей к surface_dim (400D → 2500D → 10000D)
- [ ] Базовая валидация качества инверсии на тестовых данных

### Этап 1: Базовая функциональность (1-2 дня)
- [ ] SurfaceTokenizer с адаптивными размерностями  
- [ ] Интеграция с EnergyEmbeddingMapper для получения целевых embeddings
- [ ] Базовое тестирование text → surface_embedding
- [ ] LightweightInverter integration для валидации

### Этап 2: Кэширование и оптимизация (1 день)
- [ ] TextCache с LRU и персистентным хранением для surface embeddings
- [ ] Оптимизация производительности batch processing
- [ ] Hash-based кэширование для известных пар

### Этап 3: Интеграция с обучением (1-2 дня)
- [ ] Дополнительные loss функции в EnergyTrainer
- [ ] Периодическая валидация качества
- [ ] Конфигурационные параметры

### Этап 4: Финальная интеграция и тестирование (1 день)
- [ ] BridgeTrainer для отдельного обучения
- [ ] Comprehensive testing
- [ ] Документация и примеры использования

---

## 💡 Дополнительные соображения

### Качество данных для обучения

Для эффективного обучения text bridge нужны качественные пары (text, surface_embedding):
- Использовать существующие датасеты из energy_flow обучения
- Генерировать synthetic data через existing EnergyEmbeddingMapper
- Применять data augmentation для разнообразия

### Мониторинг качества

Встроенные метрики для отслеживания качества инверсии:
- BLEU score между оригинальным и восстановленным текстом
- Cosine similarity между cyclical embeddings
- Perplexity восстановленного текста
- Human evaluation на sample данных

### Возможные улучшения

1. **Adaptive correction steps**: динамическое количество итераций в зависимости от сложности
2. **Multi-scale training**: обучение на разных размерах поверхности куба
3. **Domain adaptation**: fine-tuning на специфичных для задачи данных
4. **Compression techniques**: quantization для уменьшения размера модели

---

**Статус**: Исследование завершено, готов к реализации  
**Дата**: 2025-01-25  
**Автор**: Claude Code + Energy Flow Research Team