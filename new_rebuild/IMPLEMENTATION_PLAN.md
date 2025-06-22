# 🎯 ПЛАН ПЕРЕХОДА К CLEAN АРХИТЕКТУРЕ

## 3D Клеточная Нейронная Сеть - Clean Implementation

> **СТАТУС**: Phase 3.5 завершена! Переход к Phase 4: **Lightweight CNF**  
> **ЦЕЛЬ**: Максимальная эмерджентность при минимальных вычислениях  
> **РЕЗУЛЬТАТ**: удобно тестировать для исследовательских целей. никакого продакшена

сначала смотрим на реализацию из корневого проекта(если не находим реализацию, то спрашиваем у пользователя уточнений) и стараемся скопировать с минимальными изменениями принципов реализации, но конечно меняя так, что бы работало в новых условиях - в том смысле, что с оглядкой на уже реализованное в new_rebuild.(хотя если есть простая и эффективная оптимизация, то делаем запрос у ползователя на подтверждение)

1. **MinimalNCACell** - готовая архитектура (~69 параметров)
2. **GMLPOptConnections** → **GNNCell** - готовая архитектура (~8k параметров) ✅
3. **3D Lattice** - концепцию структуры решетки ✅
4. **Hybrid подход** - NCA нейроны + GNN связи ✅
5. **Пластичность** - BCM + STDP + Competitive Learning ✅
6. **Lightweight CNF** - Neural ODE для functional + distant connections (следующий этап)
7. **Конфигурации** - удачные параметры из экспериментов
8. держим план урхитектуры актуальным - напротив каждого файла в схеме можно указывать его актуальность и описание, что делает краткое - ждя этого можно иметь отдельный файл с визуальным представлением структуры.
9. **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
10. Использовать централизованное логирование, что бы можно было отследить какой модуль был вызван и кем. debug_mode: true - максимум логов, но можено поменять на более компактный режим с меньшим количеством логов.

11. **ИСКЛЮЧАЕМ CLI автоматизация** - оставляем только Python API
12. **ИСКЛЮЧАЕМ Множественные конфигурации** - оставляем только один centralized_config.py
13. **ИСКЛЮЧАЕМ Legacy совместимость** - оставляем чистый код
14. **ИСКЛЮЧАЕМ Динамические конфигурации** - оставляем статичные конфигурации
15. **ИСКЛЮЧАЕМ Bottleneck архитектура** - оставляем полноценную архитектуру

---

### ✅ ГОТОВЫЕ КОМПОНЕНТЫ (найдены в проекте):

## 🔍 АНАЛИЗ СУЩЕСТВУЮЩИХ КОМПОНЕНТОВ

Рабочие рещения из Legacy проекта, которые можно использовать:
core/cell_prototype/architectures/minimal_nca_cell.py ✅
core/cell_prototype/architectures/gmlp_opt_connections.py → GNNCell ✅
core\lattice_3d → new_rebuild/core/lattice ✅
data\embedding_adapter\universal_adapter.py - Универсальная система конвертации эмбедингов между любыми размерностями
core\cell_prototype - ✅ Этот модуль содержит прототип "умной клетки" - базовый строительный блок для построения 3D решетки нейронов.
~~core\signal_propagation~~ - **ПРОПУСКАЕМ** - переходим сразу к CNF
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
core\log_utils.py → new_rebuild/utils/logging.py ✅

## 🚀 ПЛАН РЕАЛИЗАЦИИ new_rebuild

### ✅ **ЭТАП 1: БАЗОВАЯ ИНФРАСТРУКТУРА** ⏱️ 1-2 часа - **ЗАВЕРШЕН**

#### ✅ 1.1 Создать централизованную конфигурацию - **ВЫПОЛНЕНО**

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

    # === GNN СВЯЗИ (межнейронные соединения) - ОБНОВЛЕНО ===
    gnn_state_size: int = 32       # полноценная архитектура
    gnn_message_dim: int = 16      # размер сообщений между клетками
    gnn_hidden_dim: int = 24       # скрытый слой для обработки
    gnn_neighbor_count: int = 26   # синхронизация с NCA
    gnn_external_input_size: int = 8  # полноценный external input
    gnn_target_params: int = 8000  # ~8k параметров оптимизированная архитектура
    gnn_activation: str = "gelu"   # современная активация
    gnn_use_attention: bool = True # attention mechanism для селективности

    # === LEGACY gMLP SUPPORT (автоматически маппится на GNN) ===
    gmlp_state_size: int = 32       # автоматически = gnn_state_size
    gmlp_hidden_dim: int = 64       # автоматически адаптируется
    gmlp_neighbor_count: int = 26   # синхронизация
    gmlp_external_input_size: int = 8
    gmlp_target_params: int = 8000  # теперь реальные 8k вместо 50k
    gmlp_activation: str = "gelu"
    gmlp_use_memory: bool = False   # память отключена (shared weights)

    # === HYBRID ИНТЕГРАЦИЯ ===
    hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    hybrid_gnn_weight: float = 0.9  # 90% влияние связей (обновлено с gmlp)

    # === ОБУЧЕНИЕ ===
    learning_rate: float = 0.001
    batch_size: int = 4
    device: str = "cuda"  # auto-detect cuda/cpu

    # === ЭМБЕДДИНГИ ===
    embedding_dim: int = 768     # from DistilBERT
    phrase_based_training: bool = True  # целые фразы, не токены

    # === ПЛАСТИЧНОСТЬ (Phase 3.5 завершена) ===
    enable_plasticity: bool = True
    plasticity_rule: str = "combined" # STDP + BCM + competitive
    enable_competitive_learning: bool = True
    enable_metaplasticity: bool = True
    enable_stdp: bool = True

    # === PHASE 4: LIGHTWEIGHT CNF ===
    enable_cnf: bool = False  # Пока отключено
    cnf_functional_connections: bool = True   # CNF для functional (60%)
    cnf_distant_connections: bool = True      # CNF для distant (30%)
    cnf_integration_steps: int = 3            # 3-step Euler (вместо 10 RK4)
    cnf_adaptive_step_size: bool = True       # Адаптивный шаг интеграции
    cnf_target_params_per_connection: int = 500  # ~500 параметров на связь

    # === ТОПОЛОГИЯ СОСЕДСТВА (биологическое 10k соседей) ===
    neighbors: int = 26 # 3D соседство (биологически правдоподобно)
    neighbor_finding_strategy: str = "tiered"
    neighbor_strategy_config: Dict = field(default_factory=lambda: {
        "local_tier": 0.1,      # 10% локальные соседи → SimpleLinear/GNN
        "functional_tier": 0.6,  # 60% функциональные → CNF (Phase 4)
        "distant_tier": 0.3,     # 30% дальние → CNF (Phase 4)
        "local_grid_cell_size": 8
    })

    # === ОПТИМИЗАЦИЯ ПАМЯТИ ===
    memory_efficient: bool = True
    use_checkpointing: bool = True
    mixed_precision: bool = True
```

#### ✅ 1.2 Создать 3D решетку - **ЗАВЕРШЕНО**

**Компоненты:**

1. ✅ Создан полный модуль `new_rebuild/core/lattice/` из Legacy core/lattice_3d/
2. ✅ Реализовано Morton encoding + Spatial Hashing для 10k соседей
3. ✅ Граничные условия: periodic, walls
4. ✅ Методы: `get_neighbors()`, `forward_pass()`, `get_states()`
5. ✅ I/O система с IOPointPlacer (все стратегии размещения)

#### ✅ 1.3 Перенести и оптимизировать клетки - **ЗАВЕРШЕНО**

**Компоненты:**

1. ✅ `MinimalNCACell` → `new_rebuild/core/cells/nca_cell.py` - **55 параметров**
2. ✅ `GMLPOptConnections` → `new_rebuild/core/cells/gnn_cell.py` - **8,881 параметров** ⭐ КАРДИНАЛЬНОЕ УЛУЧШЕНИЕ
3. ✅ `HybridCellV2` - NCA модулирует GNN operations - **9,163 параметров**
4. ✅ Создан базовый интерфейс `BaseCell` + `CellFactory`

### ✅ **PHASE 2: 3D Решетка + Централизованное логирование** - **ЗАВЕРШЕНО**

- ✅ Полная интеграция Legacy lattice_3d с чистой архитектурой
- ✅ Централизованное логирование с caller tracking
- ✅ Тестирование всех компонентов

### ✅ **PHASE 3: Биологическая пластичность + архитектурные улучшения** - **ЗАВЕРШЕНО**

- ✅ **Phase 3.1:** Интеграция централизованного логирования ✅
- ✅ **Phase 3.2:** GNN замена gMLP (113k → 8.8k параметров) ⭐
- ✅ **Phase 3.3:** HybridCellV2 с биологической модуляцией ⭐
- ✅ **Phase 3.4:** Рефакторинг HybridCellV2 на подмодули ✅
- ✅ **Phase 3.5:** Полная биологическая пластичность (BCM + STDP + Competitive Learning) ⭐

### 🔮 **PHASE 4: Lightweight CNF Integration** - **ТЕКУЩИЙ ЭТАП**

> **НОВОЕ РЕШЕНИЕ:** Пропускаем Signal Propagation, переходим сразу к CNF  
> **РАСШИРЕННЫЙ ПОДХОД:** CNF для functional (60%) + distant (30%) connections  
> **ЦЕЛЬ:** Continuous dynamics для 90% связей с минимальными вычислениями

#### 4.1 Создать базовый Lightweight CNF модуль

**Создать:** `new_rebuild/core/cnf/`

- `lightweight_cnf.py` - основной CNF класс с Neural ODE
- `euler_solver.py` - 3-step Euler integration (вместо 10 RK4)
- `connection_classifier.py` - классификация связей: local/functional/distant
- `cnf_config.py` - конфигурация CNF параметров

**Архитектура:**

```python
class LightweightCNF(nn.Module):
    """
    Continuous Normalizing Flow для functional + distant connections

    ПРИНЦИПЫ:
    1. Neural ODE только для non-local connections (90% связей)
    2. 3-step Euler solver (7x быстрее чем 10-step RK4)
    3. Адаптивный размер шага на основе активности
    4. ~500 параметров на связь
    """

    def __init__(self, state_size: int, connection_type: str):
        # connection_type: "functional" или "distant"
        self.ode_network = nn.Sequential(
            nn.Linear(state_size * 2, state_size),  # neighbor влияние
            nn.GELU(),
            nn.Linear(state_size, state_size)
        )

    def compute_derivative(self, t, current_state, neighbors):
        # dx/dt = f(x, соседи, время)
        return self.ode_network(torch.cat([current_state, neighbors], dim=-1))

    def evolve_state(self, initial_state, neighbors, dt=0.1, steps=3):
        # 3-step Euler integration
        current_state = initial_state
        for _ in range(steps):
            derivative = self.compute_derivative(0, current_state, neighbors)
            current_state = current_state + dt * derivative
        return current_state
```

#### 4.2 Интегрировать CNF с топологией соседства

**Обновить:** `new_rebuild/core/lattice/topology.py`

```python
class CNFTopologyMixin:
    """Расширение топологии для CNF connections"""

    def classify_connections(self, cell_idx: int) -> Dict[str, List[int]]:
        """
        Классификация соседей по типам связей:
        - local (10%): обычная GNN обработка
        - functional (60%): CNF эволюция
        - distant (30%): CNF эволюция
        """
        neighbors = self.get_neighbors(cell_idx)

        return {
            "local": neighbors[:int(len(neighbors) * 0.1)],
            "functional": neighbors[int(len(neighbors) * 0.1):int(len(neighbors) * 0.7)],
            "distant": neighbors[int(len(neighbors) * 0.7):]
        }
```

#### 4.3 Создать Hybrid Connection Processor

**Создать:** `new_rebuild/core/cells/hybrid_connection_processor.py`

```python
class HybridConnectionProcessor(nn.Module):
    """
    MoE архитектура для разных типов связей:

    local connections (10%) → GNN обработка
    functional connections (60%) → Lightweight CNF
    distant connections (30%) → Lightweight CNF
    """

    def __init__(self, config):
        self.gnn_processor = GNNCell(config)  # Для local connections
        self.functional_cnf = LightweightCNF(config, "functional")
        self.distant_cnf = LightweightCNF(config, "distant")

    def forward(self, states, neighbor_classification):
        results = {}

        # Local connections через GNN
        if neighbor_classification["local"]:
            results["local"] = self.gnn_processor(
                states, neighbor_classification["local"]
            )

        # Functional connections через CNF
        if neighbor_classification["functional"]:
            results["functional"] = self.functional_cnf.evolve_state(
                states, neighbor_classification["functional"]
            )

        # Distant connections через CNF
        if neighbor_classification["distant"]:
            results["distant"] = self.distant_cnf.evolve_state(
                states, neighbor_classification["distant"]
            )

        return self.merge_results(results)
```

#### 4.4 Тестирование CNF интеграции

**Создать:** `test_phase4_cnf_integration.py`

- Проверка CNF на малых решетках (6x6x6)
- Сравнение производительности CNF vs GNN
- Эмерджентные паттерны от continuous dynamics
- Стабильность интеграции

### 🌟 **PHASE 5+: Масштабирование и продакшн** (после Phase 4)

- Масштабирование до 50×50×50 решеток
- Memory management с chunking
- Оптимизация для RTX 5090
- Full training pipeline интеграция

---

## 📊 **ТЕКУЩИЙ СТАТУС АРХИТЕКТУРЫ**

### ✅ Реализованные компоненты:

- **NCA:** 55 параметров (биологическая точность)
- **GNN:** 8,881 параметров (замена gMLP, 12x меньше)
- **HybridCellV2:** 9,163 параметров (NCA модулирует GNN)
- **3D Lattice:** Полностью функциональная с spatial hashing
- **Пластичность:** BCM + STDP + Competitive Learning
- **Логирование:** Централизованное с caller tracking

### 🎯 Следующие шаги (Phase 4):

1. **Lightweight CNF** для functional + distant connections
2. **Hybrid Connection Processor** с тремя экспертами
3. **Continuous dynamics** для 90% связей
4. **Оптимизация производительности** для средних решеток

### 🔮 Долгосрочные цели:

- **Эмерджентность:** Максимальная через CNF continuous dynamics
- **Масштабируемость:** До 100×100×100 клеток на RTX 5090
- **Биологическая точность:** Полная интеграция нейробиологических принципов
