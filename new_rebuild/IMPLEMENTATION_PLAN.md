# 🎯 ПЛАН ПЕРЕХОДА К CLEAN АРХИТЕКТУРЕ

## 3D Клеточная Нейронная Сеть - Clean Implementation

> **СТАТУС**: Переход от сложной legacy архитектуры к оптимизированной и модульной  
> **ЦЕЛЬ**: Максимальная эмерджентность при минимальных вычислениях  
> **РЕЗУЛЬТАТ**: удобно тестировать для исследовательских целей. никакого продакшена

сначала смотрим на реализацию из корневого проекта(если не находим реализацию, то спрашиваем у пользователя уточнений) и стараемся скопировать с минимальными изменениями принципов реализации, но конечно меняя так, что бы работало в новых условиях - в том смысле, что с оглядкой на уже реализованное.(хотя если есть простая и эффективная оптимизация, то делаем запрос у ползователя на подтверждение)

1. **MinimalNCACell** - готовая архитектура (~69 параметров)
2. **GMLPOptConnections** - готовая архитектура (~80000 параметров)
3. **3D Lattice** - концепцию структуры решетки
4. **Hybrid подход** - NCA нейроны + gMLP связи
5. **Конфигурации** - удачные параметры из экспериментов
6. держим план урхитектуры актуальным - напротив каждого файла в схеме можно указывать его актуальность и описание, что делает краткое - ждя этого можно иметь отдельный файл с визуальным представлением структуры.
7. **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
8. Использовать централизованное логирование, что бы можно было отследить какой модуль был вызван и кем. debug_mode: true - максимум логов, но можено поменять на более компактный режим.

Реализация processing концепции:
TRAINING MODE: 4096D LLaMA → 225D Surface → FULL CUBE INFLUENCE → 225D Surface → Learning
INFERENCE MODE: Question → 225D Front → [EMERGENT PROCESSING] → 225D Back → Answer

### ❌ ЧТО ИСКЛЮЧАЕМ:

1. **CLI автоматизация** - только Python API
2. **Множественные конфигурации** - только один centralized_config.py
3. **Legacy совместимость** - чистый код без адаптеров
4. **Динамические конфигурации** - статичные конфигурации
5. **Bottleneck архитектура** - убираем ограничения, берем полноценную архитектуру

---

### ✅ ГОТОВЫЕ КОМПОНЕНТЫ (найдены в проекте):

## 🔍 АНАЛИЗ СУЩЕСТВУЮЩИХ КОМПОНЕНТОВ

Рабочие рещения из Legacy проекта, которые можно использовать:
core/cell_prototype/architectures/minimal_nca_cell.py
core/cell_prototype/architectures/gmlp_opt_connections.py
core\lattice_3d
data\embedding_adapter\universal_adapter.py
core\cell_prototype
core\signal_propagation
emergent_training переименовать в -> training (так же Реализация processing концепции: TRAINING MODE: размер эмбединга обучающей llm → размер входного эмбединга поверхности куба → FULL CUBE INFLUENCE → размер выходного эмбединга поверхности куба → Learning; INFERENCE MODE: Question → размер входного эмбединга поверхности куба Front → [PROCESSING] → размер выходного эмбединга поверхности куба Back → Answer)
training\automated_training - можно использовать, как основу, но убрать все CLI и постараться реализовать обучными средствами атоматизацию.
production_training
inference\lightweight_decoder Компактный декодер для преобразования эмбедингов в текст
data\embedding_loader - Модуль для загрузки и предобработки векторных представлений (эмбедингов) различных типов. Обеспечивает унифицированный интерфейс для работы с популярными форматами эмбедингов в контексте 3D клеточной нейронной сети.
data\embeddings - готовые эмбединги от DistilBERT
training\embedding_trainer\dialogue_dataset.py - DialogueDataset - Класс для подготовки данных к обучению куба в dialogue режиме Этот модуль реализует специализированный dataset для обучения 3D Cubic Core на задачах диалога (question_embedding → answer_embedding).
training\embedding_trainer\autoencoder_dataset.py - AutoencoderDataset - Класс для подготовки данных к обучению куба в autoencoder режиме Этот модуль реализует специализированный dataset для обучения 3D Cubic Core на задачах реконструкции эмбедингов (autoencoder mode).
training\embedding_trainer\advanced_loss_functions.py - Продвинутая система loss functions для Stage 2.3 Включает: - Curriculum learning loss (easy→hard progression) - Triplet loss для enhanced semantic alignment - Contrastive learning approaches - Multi-objective optimization (similarity + diversity)
training\embedding_trainer\neural_cellular_automata.py - Реализация emergent behavior preservation во время GPU-optimized training. Ключевые принципы NCA для 3D Cellular Neural Network: 1. Stochastic Cell Updates - избежание глобальной синхронизации 2. Residual Update Rules - маленькие, стабильные модификации 3. Pattern Formation Metrics - количественная оценка emergence 4. Emergent Behavior Preservation - сохранение паттернов при оптимизации
download_distilbert.py - Скрипт для предварительной загрузки DistilBERT в локальную папку проекта models/local_cache.
generate_large_embedding_dataset.py - Генератор большого датасета эмбеддингов для обучения 3D куба Создает тысячи пар question-answer и сохраняет готовые эмбеддинги
generate_snli_embedding_dataset.py - Генератор эмбеддингов из SNLI датасета для обучения 3D куба Использует 1/5 часть SNLI (Stanford Natural Language Inference) датасета
precomputed_embedding_loader.py - Загрузчик готовых эмбеддингов из предварительно сгенерированного файла Используется для быстрого обучения без пересчета эмбеддингов
study_plan реализации архитектуры на основе локальных правил и эмерджентной связности.md - это последняя попытка интегрировать новую архитектуру в проект, но она завершилась неудачей.
Современные методы динамической связности для крупномасштабных 3D клеточных нейронных сетей.md - понимание оптимизации новой архитектуры
PHASE_5_PLUS_ROADMAP.md

## 🚀 ПЛАН РЕАЛИЗАЦИИ new_rebuild

### **ЭТАП 1: БАЗОВАЯ ИНФРАСТРУКТУРА** ⏱️ 1-2 часа

#### 1.1 Создать централизованную конфигурацию

```python
# new_rebuild/config/project_config.py
@dataclass
class ProjectConfig:
    """Единственный источник истины для всей архитектуры"""

    # === АРХИТЕКТУРА ===
    architecture_type: str = "hybrid"  # nca | gmlp | hybrid

    # === 3D РЕШЕТКА ===
    # Начинаем с малой для тестов, масштабируем до цели
    lattice_dimensions: Tuple[int, int, int] = (6, 6, 6)  # отладка
    # lattice_dimensions: Tuple[int, int, int] = (16, 16, 16)  # test
    # target_dimensions: Tuple[int, int, int] = (666, 666, 333)  # научные опыты

    # === NCA НЕЙРОНЫ (биологический аналог) ===
    nca_state_size: int = 4      # состояние нейрона
    nca_hidden_dim: int = 3      # внутренняя обработка
    nca_neighbor_count: int = 26 # 3D Moore neighborhood
    nca_target_params: int = 69  # ~60 параметров как в биологии
    nca_activation: str = "tanh" # стабильная для NCA

    # === gMLP СВЯЗИ (межнейронные соединения) - БЕЗ BOTTLENECK ===
    gmlp_state_size: int = 32       # полноценная архитектура
    gmlp_hidden_dim: int = 64       # Динамически: 32-332
    gmlp_neighbor_count: int = 26   # синхронизация с NCA
    gmlp_external_input_size: int = 8  # полноценный external input
    gmlp_target_params: int = 50000 # ~10k связей как в биологии, но так как размер сложно подобрать, 50k-100k параметров полноценной архитектуры
    gmlp_activation: str = "gelu"   # современная активация
    gmlp_use_memory: bool = False   # память отключена (shared weights)

    # === HYBRID ИНТЕГРАЦИЯ ===
    hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    hybrid_gmlp_weight: float = 0.9 # 90% влияние связей

    # === ОБУЧЕНИЕ ===
    learning_rate: float = 0.001
    batch_size: int = 4
    device: str = "cuda"  # auto-detect cuda/cpu

    # === ЭМБЕДДИНГИ ===
    embedding_dim: int = 768     # from DistilBERT
    phrase_based_training: bool = True  # целые фразы, не токены

    # === BIOLOGICAL PRINCIPLES ===
    biological:
    # Клетки как нейроны с общими весами
    shared_weights: true

    # Решетка как нервная ткань
    tissue_simulation: true

    # Рецепторная стратегия (100% покрытия)
    receptor_coverage: 1

    # Сигналы как нервные импульсы
    signal_propagation: true

    # === PHASE 4 ADDITIONS ===
    # Новые секции для поддержки clean конфигураций

    # Пластичность (из clean конфигураций)
    plasticity:
    enable_plasticity: true
    plasticity_rule: "combined" # STDP + BCM + competitive
    enable_competitive_learning: true
    enable_metaplasticity: true
    enable_clustering: false # Пока отключено

    # Оптимизация памяти
    optimization:
    memory_efficient: true
    use_checkpointing: true
    mixed_precision: true
```

#### 1.2 для начала Перенести и оптимизировать клетки

**Задачи:**

1. ✅ Скопировать `MinimalNCACell` → `new_rebuild/core/cells/nca_cell.py`
2. ✅ Скопировать `GMLPOptConnections` → `new_rebuild/core/cells/gmlp_cell.py`
   - **ВАЖНО**: Убрать bottleneck архитектуру для полноценной производительности
   - Увеличить `hidden_dim` с 32 до 64
   - Убрать `bottleneck_dim` ограничения
   - Оставить оптимизированную SGU архитектуру
3. ✅ Создать базовый интерфейс `BaseCell`

#### 1.3 Создать 3D решетку

**Задачи:**

1. ✅ Создать `Lattice3D`
2. ✅ Реализовать простое 6-соседство или 26-соседство с потенциалом расширения. в идеале нам нужно подумать о 10000 соседях
3. ✅ Граничные условия: periodic
4. ✅ Методы: `get_neighbors()`, `forward_pass()`, `get_states()`

## 📋 ДЕТАЛЬНЫЙ ЧЕКЛИСТ РЕАЛИЗАЦИИ

### **PHASE 1: Foundation**

- [ ] **1.1** Создать `new_rebuild/config/project_config.py`
- [ ] **1.2** Перенести `MinimalNCACell` → `core/cells/nca_cell.py`
- [ ] **1.3** Перенести `GMLPOptConnections` → `core/cells/gmlp_cell.py`
  - **УБРАТЬ** bottleneck архитектуру
  - **УВЕЛИЧИТЬ** hidden_dim до 64
  - **УВЕЛИЧИТЬ** external_input_size до 8
  - **УБРАТЬ** bottleneck_dim полностью
- [ ] **1.4** Создать `BaseCell` интерфейс
- [ ] **1.5** Создать `Lattice3D`
- [ ] **1.** Тест: каждый компонент создаются без ошибок

### **PHASE 2: Hybrid Architecture**

- [ ] **2.1** Реализовать `HybridCell` с полноценной архитектурой
- [ ] **2.2** Реализовать `HybridLattice3D` с расширяемой топологией
- [ ] **2.3** Тест: hybrid forward pass работает
- [ ] **2.4** Валидация: количество параметров корректное (~50k+ для gMLP)

### **PHASE 3: Training System**

- [ ] **3.1** Создать `Trainer`
- [ ] **3.2** Создать `Processor` с полноценной обработкой
- [ ] **3.3** Тест: один epoch обучения проходит
- [ ] **3.4** Валидация: loss уменьшается
- иметь возможеность проверять эмбединг насколько он уже генерирует осознанные фразы - например раз за эпоху.

---
