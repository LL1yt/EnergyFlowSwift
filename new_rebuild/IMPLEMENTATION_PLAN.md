# 🎯 ПЛАН ПЕРЕХОДА К CLEAN АРХИТЕКТУРЕ

## 3D Клеточная Нейронная Сеть - Clean Implementation

> **СТАТУС**: Переход от сложной legacy архитектуры к оптимизированной и модульной  
> **ЦЕЛЬ**: Максимальная эмерджентность при минимальных вычислениях  
> **РЕЗУЛЬТАТ**: удобно тестировать для исследовательских целей. никакого продакшена

1. **MinimalNCACell** - готовая архитектура (~69 параметров)
2. **EmergentGMLPCell** - готовая архитектура (~23000 параметров)
3. **3D Lattice** - концепцию структуры решетки
4. **Hybrid подход** - NCA нейроны + gMLP связи
5. **Конфигурации** - удачные параметры из экспериментов
6. держим план урхитектуры актуальным - напротив каждого файла в схеме можно указывать его актуальность и описание, что делает краткое - ждя этого можно иметь отдельный файл с визуальным представлением структуры.
7. **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
   @PHASE_4_PLAN.md - это последняя попутка интегрировать новую архитектуру в проект, но она завершилась неудачей.

### ❌ ЧТО ИСКЛЮЧАЕМ:

1. **CLI автоматизация** - только Python API
2. **Множественные конфигурации** - только один centralized_config.py
3. **Legacy совместимость** - чистый код без адаптеров
4. **Динамические конфигурации** - статичные конфигурации

---

## 🔍 АНАЛИЗ СУЩЕСТВУЮЩИХ КОМПОНЕНТОВ

### ✅ ГОТОВЫЕ КОМПОНЕНТЫ (найдены в проекте):

#### **1. MinimalNCACell** (~69 параметров) ✅

- **Расположение**: `core/cell_prototype/architectures/minimal_nca_cell.py`
- **Параметры**: state_size=4, hidden_dim=3, neighbor_count=26
- **Статус**: ✅ Готов к переносу
- **Особенности**: NCA динамика, learnable weights для соседей

#### **2. gmlp** (~50000 параметров) ✅

- **Расположение**: `core\cell_prototype\architectures\gmlp_opt_connections.py`
- **Параметры**: state_size=32, hidden_dim=64, spatial_connections=True
- **Статус**: ✅ Готов к переносу
- **Особенности**: Пространственные связи, память отключаема

#### **3. 3D Lattice структура** ✅

- **Расположение**: `core/lattice_3d/lattice.py`
- **Особенности**: Topology, neighbor finding, boundary conditions
- **Статус**: ✅ перенести
- **Проблема**:

#### **4. Hybrid архитектура** ⚠️ (частично готова)

- **Расположение**: `FINAL_HYBRID_ARCHITECTURE_IMPLEMENTATION_PLAN.md`
- **Статус**: ⚠️ Концепция готова, но не реализована
- **Задача**: Реализовать HybridNCAGMLPCell

---

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
    lattice_dimensions: Tuple[int, int, int] = (16, 16, 16)  # test
    target_dimensions: Tuple[int, int, int] = (666, 666, 333)  # научные опыты

    # === NCA НЕЙРОНЫ (биологический аналог) ===
    nca_state_size: int = 4      # состояние нейрона
    nca_hidden_dim: int = 3      # внутренняя обработка
    nca_neighbor_count: int = 26 # 3D Moore neighborhood
    nca_target_params: int = 69  # ~60 параметров как в биологии
    nca_activation: str = "tanh" # стабильная для NCA

    # === gMLP СВЯЗИ (межнейронные соединения) ===
    gmlp_state_size: int = 32       # совместимость с NCA
    gmlp_hidden_dim: int = 364      # обработка связей
    gmlp_target_params: int = 80000 # ~10k связей как в биологии, но так как размер сложно подобрать, то можно и больше взять 50k-100k
    gmlp_activation: str = "gelu"   # современная активация
    gmlp_use_memory: bool = False   # память отключена (shared weights)

    # === HYBRID ИНТЕГРАЦИЯ ===
    hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    hybrid_gmlp_weight: float = 0.9 # 90% влияние связей

    # === ОБУЧЕНИЕ ===
    learning_rate: float = 0.001
    batch_size: int = 4
    device: str = "cuda"  # auto-detect cuda/cpu

    # === ЭМБЕДДИНГИ === data\embedding_adapter\universal_adapter.py
    embedding_dim: int = 768     # from DistilBERT
    phrase_based_training: bool = True  # целые фразы, не токены
```

#### 1.2 Перенести и упростить клетки

**Задачи:**

1. ✅ Скопировать `MinimalNCACell` → `new_rebuild/core/cells/nca_cell.py`
2. ✅ Скопировать `gmlp` → `new_rebuild/core/cells/gmlp_connections.py`
3. ✅ Создать базовый интерфейс `BaseCell`

#### 1.3 Создать упрощенную 3D решетку

**Задачи:**

1. ✅ Создать `Lattice3D`
2. ✅ Реализовать простое 6-соседство или 26-соседство с потенциалом расширения. в идеале нам нужно подумать о 10000 соседях
3. ✅ Граничные условия: periodic
4. ✅ Методы: `get_neighbors()`, `forward_pass()`, `get_states()`

### **ЭТАП 2: HYBRID АРХИТЕКТУРА** ⏱️ 2-3 часа

#### 2.1 Реализовать HybridCell

```python
# new_rebuild/core/hybrid/hybrid_cell.py
class HybridCell(BaseCell):
    """
    Биологически корректная композиция:
    - NCA нейрон обрабатывает локальную динамику
    - gMLP связи обрабатывают межнейронные соединения
    - Learnable веса для интеграции
    """

    def __init__(self, config: ProjectConfig):
        super().__init__()

        # NCA компонент (нейронная динамика)
        self.nca_neuron = NCACell(
            state_size=config.nca_state_size,
            hidden_dim=config.nca_hidden_dim,
            neighbor_count=config.nca_neighbor_count,
            activation=config.nca_activation,
            target_params=config.nca_target_params
        )

        # gMLP компонент (связи между нейронами)
        self.gmlp_connections = GMLPCell(
            state_size=config.gmlp_state_size,
            hidden_dim=config.gmlp_hidden_dim,
            use_memory=config.gmlp_use_memory,
            activation=config.gmlp_activation,
            target_params=config.gmlp_target_params
        )

        # Learnable веса интеграции
        self.integration_weights = nn.Parameter(
            torch.tensor([config.hybrid_nca_weight, config.hybrid_gmlp_weight])
        )

        # Проекции для совместимости размерностей
        self.nca_projection = nn.Linear(config.nca_state_size, config.nca_state_size)
        self.gmlp_projection = nn.Linear(config.gmlp_state_size, config.nca_state_size)

    def forward(self, own_state, neighbor_states, external_input=None):
        """
        Hybrid forward: NCA обрабатывает состояние, gMLP - связи
        """
        # NCA: локальная динамика нейрона
        nca_output = self.nca_neuron(neighbor_states, own_state, external_input)

        # gMLP: обработка межнейронных связей
        gmlp_output = self.gmlp_connections(own_state, neighbor_states, external_input)

        # Проекция к единой размерности
        nca_proj = self.nca_projection(nca_output)
        gmlp_proj = self.gmlp_projection(gmlp_output)

        # Learnable интеграция
        weights = torch.softmax(self.integration_weights, dim=0)
        integrated = weights[0] * nca_proj + weights[1] * gmlp_proj

        return integrated
```

#### 2.2 Создать HybridLattice

```python
# new_rebuild/core/hybrid/hybrid_lattice.py
class HybridLattice3D:
    """3D решетка с Hybrid клетками"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.dimensions = config.lattice_dimensions
        self.total_cells = np.prod(self.dimensions)

        # Создаем Hybrid клетки
        self.cells = nn.ModuleList([
            HybridCell(config) for _ in range(self.total_cells)
        ])

        # Простая топология соседства
        self.topology = self._build_topology()

    def forward(self, embeddings):
        """
        Параллельный forward pass через все клетки
        """
        batch_size = embeddings.size(0)
        outputs = []

        for cell_idx in range(self.total_cells):
            # Получаем состояние клетки и соседей
            own_state = self.get_cell_state(cell_idx)
            neighbor_states = self.get_neighbor_states(cell_idx)
            external_input = self.get_external_input(cell_idx, embeddings)

            # Forward через Hybrid клетку
            cell_output = self.cells[cell_idx](own_state, neighbor_states, external_input)
            outputs.append(cell_output)

        return torch.stack(outputs, dim=1)
```

### **ЭТАП 3: СИСТЕМА ОБУЧЕНИЯ** ⏱️ 1-2 часа

#### 3.1 Создать простой trainer

```python
# new_rebuild/training/trainer.py
class SimpleTrainer:
    """Упрощенный trainer без CLI и сложной автоматизации"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = torch.device(config.device if config.device != "auto"
                                 else "cuda" if torch.cuda.is_available() else "cpu")

        # Создаем модель
        self.model = HybridLattice3D(config).to(self.device)

        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Loss функция
        self.criterion = nn.MSELoss()

    def train_step(self, input_embeddings, target_embeddings):
        """Один шаг обучения"""
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(input_embeddings)

        # Loss (reconstruction)
        loss = self.criterion(outputs, target_embeddings)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """Один epoch обучения"""
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            loss = self.train_step(inputs, targets)
            total_loss += loss

        return total_loss / len(dataloader)
```

### **ЭТАП 4: DATA PIPELINE** ⏱️ 1 час

#### 4.1 Работа с embeddings

```python
# new_rebuild/data/embeddings.py (data\embedding_adapter\universal_adapter.py)
class EmbeddingProcessor:


    def __init__(self, config: ProjectConfig):
        self.config = config
        self.embedding_dim = config.embedding_dim

    def process_phrases(self, phrases: List[str], model_name="distilbert"):
        """
        Обрабатывает целые фразы в embeddings
        """
        # Используем простую модель для получения embeddings
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        embeddings = []
        for phrase in phrases:
            tokens = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**tokens)
                # Берем [CLS] token или mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

        return torch.stack(embeddings)

    def create_phrase_pairs(self, questions: List[str], answers: List[str]):
        """
        Создает пары вопрос-ответ для обучения
        """
        question_embeddings = self.process_phrases(questions)
        answer_embeddings = self.process_phrases(answers)

        return list(zip(question_embeddings, answer_embeddings))
```

### **ЭТАП 5: ТЕСТИРОВАНИЕ И ВАЛИДАЦИЯ** ⏱️ 1-2 часа

#### 5.1 Поэтапные тесты

```python
# new_rebuild/tests/test_step_by_step.py

def test_nca_cell():
    """Тест NCA клетки"""
    config = ProjectConfig()
    cell = NCACell(config)

    # Тестовые данные
    own_state = torch.randn(1, config.nca_state_size)
    neighbor_states = torch.randn(1, config.nca_neighbor_count, config.nca_state_size)

    output = cell(neighbor_states, own_state)

    assert output.shape == own_state.shape
    assert cell.count_parameters() <= config.nca_target_params * 1.1  # 10% tolerance
    print(f"✅ NCA Cell: {cell.count_parameters()} params")

def test_gmlp_cell():
    """Тест gMLP клетки"""
    config = ProjectConfig()
    cell = GMLPCell(config)

    # Тестовые данные
    own_state = torch.randn(1, config.gmlp_state_size)
    neighbor_states = torch.randn(1, config.nca_neighbor_count, config.gmlp_state_size)

    output = cell(own_state, neighbor_states)

    assert output.shape == own_state.shape
    print(f"✅ gMLP Cell: {cell.count_parameters()} params")

def test_hybrid_cell():
    """Тест Hybrid клетки"""
    config = ProjectConfig()
    cell = HybridCell(config)

    # Тестовые данные
    own_state = torch.randn(1, config.nca_state_size)
    neighbor_states = torch.randn(1, config.nca_neighbor_count, config.nca_state_size)

    output = cell(own_state, neighbor_states)

    total_params = cell.count_parameters()
    expected_params = config.nca_target_params + config.gmlp_target_params

    print(f"✅ Hybrid Cell: {total_params} params (expected ~{expected_params})")

def test_lattice_3d():
    """Тест 3D решетки"""
    config = ProjectConfig()
    lattice = HybridLattice3D(config)

    # Тестовые embeddings
    batch_size = 2
    embeddings = torch.randn(batch_size, config.embedding_dim)

    output = lattice(embeddings)

    expected_shape = (batch_size, lattice.total_cells, config.nca_state_size)
    assert output.shape == expected_shape

    print(f"✅ Lattice 3D: {lattice.dimensions} = {lattice.total_cells} cells")

def test_full_pipeline():
    """Тест полного pipeline"""
    config = ProjectConfig()

    # Данные
    questions = ["What is AI?", "How does neural network work?"]
    answers = ["AI is artificial intelligence", "Neural networks learn patterns"]

    # Обработка
    processor = EmbeddingProcessor(config)
    pairs = processor.create_phrase_pairs(questions, answers)

    # Тренировка
    trainer = SimpleTrainer(config)

    for i, (input_emb, target_emb) in enumerate(pairs):
        loss = trainer.train_step(input_emb.unsqueeze(0), target_emb.unsqueeze(0))
        print(f"✅ Training step {i}: loss = {loss:.4f}")

if __name__ == "__main__":
    test_nca_cell()
    test_gmlp_cell()
    test_hybrid_cell()
    test_lattice_3d()
    test_full_pipeline()
    print("🎉 ALL TESTS PASSED!")
```

---

## 📋 ДЕТАЛЬНЫЙ ЧЕКЛИСТ РЕАЛИЗАЦИИ

### **PHASE 1: Foundation** ⏱️ 1-2 часа

- [ ] **1.1** Создать `new_rebuild/config/project_config.py`
- [ ] **1.2** Перенести `MinimalNCACell` → `core/cells/nca_cell.py`
- [ ] **1.3** Перенести `EmergentGMLPCell` → `core/cells/gmlp_cell.py`
- [ ] **1.4** Создать `BaseCell` интерфейс
- [ ] **1.5** Создать `SimpleLattice3D`
- [ ] **1.6** Тест: все компоненты создаются без ошибок

### **PHASE 2: Hybrid Architecture** ⏱️ 2-3 часа

- [ ] **2.1** Реализовать `HybridCell`
- [ ] **2.2** Реализовать `HybridLattice3D`
- [ ] **2.3** Тест: hybrid forward pass работает
- [ ] **2.4** Валидация: количество параметров корректное

### **PHASE 3: Training System** ⏱️ 1-2 часа

- [ ] **3.1** Создать `SimpleTrainer`
- [ ] **3.2** Создать `EmbeddingProcessor`
- [ ] **3.3** Тест: один epoch обучения проходит
- [ ] **3.4** Валидация: loss уменьшается

### **PHASE 4: Integration & Testing** ⏱️ 1-2 часа

- [ ] **4.1** Полные end-to-end тесты
- [ ] **4.2** Документация usage examples
- [ ] **4.3** Performance benchmarks
- [ ] **4.4** Масштабирование: 16³ → 32³ → 64³

### **PHASE 5: Production Ready** ⏱️ 1 час

- [ ] **5.1** Создать `main.py` entry point
- [ ] **5.2** Простое логирование и мониторинг
- [ ] **5.3** Save/load functionality
- [ ] **5.4** README с quick start

---

## 🎯 EXPECTED RESULTS

### **Архитектурные цели:**

- ✅ **Параметры**: NCA ~69, gMLP ~23k, Hybrid ~23k
- ✅ **Решетка**: Начало 16³ (4k клеток) → цель 666×666×333 (148M клеток)
- ✅ **Эмерджентность**: Максимальная за счет biological правил
- ✅ **Простота**: Один config файл, Python API, без CLI

### **Производительность:**

- ✅ **Memory**: Efficient для 16³ решетки
- ✅ **Speed**: Forward pass < 100ms на GPU
- ✅ **Scalability**: Linear scaling до 64³+

### **Качество кода:**

- ✅ **Модульность**: Каждый компонент независим
- ✅ **Testability**: 100% покрытие тестами
- ✅ **Maintainability**: Простая архитектура без magic

---

## 🚀 НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ

**Сегодня (2-3 часа работы):**

1. 📁 Создать базовую структуру папок
2. ⚙️ Реализовать `ProjectConfig`
3. 🧠 Перенести `MinimalNCACell` и `EmergentGMLPCell`
4. 🔗 Создать `HybridCell`
5. ✅ Первые тесты

**Завтра:**

1. 🕸️ Реализовать `HybridLattice3D`
2. 🎓 Создать `SimpleTrainer`
3. 📊 Полные end-to-end тесты
4. 📈 Benchmark на малой решетке

**Эта неделя:**

1. 🎯 Production готовый код
2. 📖 Документация и примеры
3. 🔄 Масштабирование до 64³
4. 🚀 Готовность к реальному обучению

---

**STATUS**: 🎯 READY TO START  
**CONFIDENCE**: 🔥 HIGH (все компоненты найдены)  
**TIMELINE**: 2-3 дня до production ready system
