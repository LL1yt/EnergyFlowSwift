# 🎯 ПЛАН ПЕРЕХОДА К CLEAN АРХИТЕКТУРЕ

## 3D Клеточная Нейронная Сеть - Clean Implementation

> **СТАТУС**: Переход от сложной legacy архитектуры к оптимизированной и модульной  
> **ЦЕЛЬ**: Максимальная эмерджентность при минимальных вычислениях  
> **РЕЗУЛЬТАТ**: удобно тестировать для исследовательских целей. никакого продакшена

сначала смотрим на реализацию из корневого проекта(если не находим реализацию, то спрашиваем у пользователя уточнений) и стараемся скопировать с минимальными изменениями принципов реализации, но конечно меняя так, что бы работало в новых условиях - в том смысле, что с оглядкой на уже реализованное в new_rebuild.(хотя если есть простая и эффективная оптимизация, то делаем запрос у ползователя на подтверждение)

1. **MinimalNCACell** - готовая архитектура (~69 параметров)
2. **GMLPOptConnections** - готовая архитектура (~80000 параметров)
3. **3D Lattice** - концепцию структуры решетки
4. **Hybrid подход** - NCA нейроны + gMLP связи
5. **Конфигурации** - удачные параметры из экспериментов
6. держим план урхитектуры актуальным - напротив каждого файла в схеме можно указывать его актуальность и описание, что делает краткое - ждя этого можно иметь отдельный файл с визуальным представлением структуры.
7. **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
8. Использовать централизованное логирование, что бы можно было отследить какой модуль был вызван и кем. debug_mode: true - максимум логов, но можено поменять на более компактный режим с меньшим количеством логов.

9. **ИСКЛЮЧАЕМ CLI автоматизация** - оставляем только Python API
10. **ИСКЛЮЧАЕМ Множественные конфигурации** - оставляем только один centralized_config.py
11. **ИСКЛЮЧАЕМ Legacy совместимость** - оставляем чистый код
12. **ИСКЛЮЧАЕМ Динамические конфигурации** - оставляем статичные конфигурации
13. **ИСКЛЮЧАЕМ Bottleneck архитектура** - оставляем полноценную архитектуру

---

### ✅ ГОТОВЫЕ КОМПОНЕНТЫ (найдены в проекте):

## 🔍 АНАЛИЗ СУЩЕСТВУЮЩИХ КОМПОНЕНТОВ

Рабочие рещения из Legacy проекта, которые можно использовать:
core/cell_prototype/architectures/minimal_nca_cell.py
core/cell_prototype/architectures/gmlp_opt_connections.py
core\lattice_3d - готовое решение для 3д решетки с оптимизированным подходом.
data\embedding_adapter\universal_adapter.py - Универсальная система конвертации эмбедингов между любыми размерностями
core\cell_prototype - Этот модуль содержит прототип "умной клетки" - базовый строительный блок для построения 3D решетки нейронов.
core\signal_propagation - Этот модуль управляет временной динамикой сигналов в 3D решетке клеток.
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
core\log_utils.py - возвращает информацию о вызывающем коде (файл, строка, функция).

## 🚀 ПЛАН РЕАЛИЗАЦИИ new_rebuild

### **ЭТАП 1: БАЗОВАЯ ИНФРАСТРУКТУРА** ⏱️ 1-2 часа

#### 1.1 Создать централизованную конфигурацию ✅ ВЫПОЛНЕНО

```python
# new_rebuild/config/project_config.py - СОЗДАН
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

    # Топология соседства - нам нужно имитировать связи с 10000 других нейронов
    neighbors: 26 # 3D соседство (биологически правдоподобно)
    neighbor_finding_strategy: "tiered"
    neighbor_strategy_config:
        local_tier: 0.2 # 20% локальные соседи
        functional_tier: 0.6 # 50% функциональные (spatial hashing)
        random_tier: 0.2 # 20% дальние стохастические
        local_grid_cell_size: 8 # Размер spatial hash bins

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

#### 1.2 Создать 3D решетку 🚧 СЛЕДУЮЩИЙ ЭТАП

**Задачи:**

1. ⏭️ Создать `Lattice3D` из Legacy core/lattice_3d/
2. ⏭️ Реализовать простое 26-соседство с потенциалом расширения. в идеале нам нужно подумать 10000 соседей - возможно это реализовано уже через Morton и Сверхлегкие Spatial Hashing методы
3. ⏭️ Граничные условия: periodic
4. ⏭️ Методы: `get_neighbors()`, `forward_pass()`, `get_states()`

#### 1.3 Перенести и оптимизировать клетки ✅ ВЫПОЛНЕНО

**Задачи:**

1. ✅ Скопировать `MinimalNCACell` → `new_rebuild/core/cells/nca_cell.py` - СОЗДАН (55 параметров)
2. ✅ Скопировать `GMLPOptConnections` → `new_rebuild/core/cells/gmlp_cell.py` - СОЗДАН (113k параметров ⚠️)
   - ✅ **УБРАНА bottleneck архитектура** для полноценной производительности
   - ✅ Увеличен `hidden_dim` с 32 до 64
   - ✅ Убраны `bottleneck_dim` ограничения
   - ✅ Оставлена оптимизированная SGU архитектура
3. ✅ Создан базовый интерфейс `BaseCell` + `CellFactory`

### **PHASE 2: Training System**

### **PHASE X: дополнительные моменты, которые пришли в процессе реализации и которые можно будет реализовать после успешного тестирования всей реализации(что бы не забыть):**

⚠️ **КРИТИЧНО:** gMLP превышает целевые параметры!

- **Текущее:** 113,161 параметров (цель: 50,000)
- **Превышение:** ~2.26x от цели
- **TODO:** подумать о том, какие параметры можно уменьшить в gMLP без потери информации, что бы как можно приблизить к 50000
- **Возможные решения:**

  - Уменьшить FFN expansion с `hidden_dim * 2` до `hidden_dim * 1.5`
  - Оптимизировать Spatial Gating Unit
  - Более эффективное использование shared weights
  - Возможно небольшое уменьшение hidden_dim с 64 до 48-56

- нужно подумать о том, по какому параметру нам синхронизировать NCA/gMLP клетками, так как изначально была идея деалть это по state_size, но мы не можем этого сделать, так как nca должен быть не более 100 параметров, а gmlp не менее 10000

---
