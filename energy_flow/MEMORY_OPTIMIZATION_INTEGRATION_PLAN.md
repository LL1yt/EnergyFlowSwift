# 🚀 План интеграции оптимизаций памяти и архитектурных улучшений

**Дата создания**: 2025-08-09  
**Цель**: Решение проблем с памятью во время обучения через оптимизацию архитектуры  
**Принцип**: Максимум потоков на текущем железе без уменьшения батчей

---

## 📋 Оглавление

1. [Анализ текущего состояния](#1-анализ-текущего-состояния)
2. [Архитектурные решения](#2-архитектурные-решения)
3. [План реализации модулей](#3-план-реализации-модулей)
4. [Интеграция без продакшена](#4-интеграция-без-продакшена)
5. [Тестирование и валидация](#5-тестирование-и-валидация)
6. [Метрики успеха](#6-метрики-успеха)

---

## 1. Анализ текущего состояния

### 🔴 Критические проблемы из ENERGY_FLOW_IMPROVEMENT_REPORT.md

| Проблема                                                            | Влияние на память             | Приоритет |
| ------------------------------------------------------------------- | ----------------------------- | --------- |
| Синтаксические ошибки в flow_processor.py                           | Код не исполняем              | КРИТИЧНО  |
| O(batch * W*H\*(D+1)) сложность в round_to_nearest_lattice_position | OOM при больших решетках      | КРИТИЧНО  |
| Отсутствие cleanup для spawn потоков                                | Лавинообразный рост памяти    | ВЫСОКИЙ   |
| Множественные clone() операции                                      | Избыточное потребление памяти | СРЕДНИЙ   |
| Отсутствие батчевой очистки                                         | Фрагментация памяти           | СРЕДНИЙ   |

### 🟡 Существующие попытки оптимизации

Из `performance_and_memory_fixes.py`:

- ✅ Ограничение max_active_flows до 50,000
- ✅ Отключение movement_based_spawn
- ⚠️ Создан FlowMemoryManager но не интегрирован
- ❌ position_history не используется эффективно

### 🟢 Хорошие практики в проекте

- Централизованная конфигурация через EnergyConfig
- Custom debug levels в логировании
- Device manager для GPU управления
- Предвычисление normalized lattice grid

---

## 2. Архитектурные решения

### 2.1 Централизованное логирование (energy_flow/utils/unified_logging.py)

```python
# Новый unified логгер с интеграцией метрик
class UnifiedLogger:
    """
    Централизованная система логирования с:
    - Автоматическим сбором метрик памяти
    - Частотным гейтингом для длинных циклов
    - Экспортом производительности в JSON
    - Интеграцией с существующими DEBUG_* уровнями
    """

    def __init__(self, config: LoggingConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.memory_tracker = MemoryTracker()
        self.performance_exporter = PerformanceExporter()

    @contextmanager
    def log_operation(self, name: str, log_memory: bool = True):
        """Context manager для автоматического трекинга операций"""
        start_time = time.perf_counter()
        start_memory = self.memory_tracker.get_current() if log_memory else None

        yield

        # Автоматический сбор метрик
        elapsed = time.perf_counter() - start_time
        if log_memory:
            memory_delta = self.memory_tracker.get_delta(start_memory)
            self.metrics_collector.add(name, elapsed, memory_delta)
```

### 2.2 Интеллектуальная очистка памяти (energy_flow/utils/smart_memory_cleaner.py)

```python
class SmartMemoryCleaner:
    """
    Умная система очистки памяти с:
    - Адаптивными порогами на основе текущей нагрузки
    - Приоритетной очисткой неактивных потоков
    - Метриками для принятия решений
    - Интеграцией с flow lifecycle
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cleanup_history = deque(maxlen=100)
        self.adaptive_threshold = config.initial_threshold

    def should_cleanup(self, active_flows: int, memory_usage: float) -> bool:
        """Умное решение о необходимости очистки"""
        # Адаптивный порог на основе истории
        if self.cleanup_history:
            avg_flows_before_cleanup = np.mean([h['flows'] for h in self.cleanup_history])
            self.adaptive_threshold = avg_flows_before_cleanup * 0.8

        return (active_flows > self.adaptive_threshold or
                memory_usage > self.config.memory_threshold_gb)

    def cleanup_flows(self, lattice: EnergyLattice) -> Dict[str, int]:
        """Приоритетная очистка потоков"""
        stats = {'removed': 0, 'kept': 0}

        # Сортируем потоки по приоритету удаления
        flows_with_priority = []
        for flow_id, flow in lattice.active_flows.items():
            priority = self._calculate_removal_priority(flow)
            flows_with_priority.append((priority, flow_id, flow))

        # Удаляем потоки с низким приоритетом
        flows_with_priority.sort(reverse=True)  # Высокий приоритет = удаляем первым

        target_removal = max(1, len(flows_with_priority) // 4)  # Удаляем 25%

        for priority, flow_id, flow in flows_with_priority[:target_removal]:
            if flow.age > 100 or flow.energy.abs().max() < 0.01:
                del lattice.active_flows[flow_id]
                stats['removed'] += 1
            else:
                stats['kept'] += 1

        return stats
```

### 2.3 Оптимизированная конфигурация (energy_flow/config/optimized_config.py)

```python
@dataclass
class OptimizedEnergyConfig(EnergyConfig):
    """Расширенная конфигурация с оптимизациями памяти"""

    # Memory management
    enable_smart_cleanup: bool = True
    cleanup_strategy: str = "adaptive"  # "adaptive", "aggressive", "conservative"
    memory_threshold_gb: float = 20.0  # Для RTX 5090

    # Flow management
    flow_priority_enabled: bool = True
    max_flow_age: int = 200
    min_energy_threshold: float = 0.001

    # Performance monitoring
    enable_metrics: bool = True
    metrics_export_interval: int = 100
    profile_memory: bool = False  # Тяжелая операция

    # Batching optimizations
    enable_flow_batching: bool = True
    flow_batch_size: int = 1000  # Обрабатывать потоки батчами

    # Logging optimizations
    log_frequency_gate: int = 10  # Логировать каждый N-й шаг
    enable_json_export: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Автоматическая настройка на основе размера решетки
        total_cells = self.lattice_width * self.lattice_height * self.lattice_depth

        if total_cells > 1000000:  # Большая решетка
            self.cleanup_strategy = "aggressive"
            self.flow_batch_size = 500
            self.log_frequency_gate = 50
        elif total_cells > 100000:  # Средняя решетка
            self.cleanup_strategy = "adaptive"
            self.flow_batch_size = 1000
            self.log_frequency_gate = 20
```

---

## 3. План реализации модулей

### Phase 1: Критические исправления (День 1)

#### 1.1 Исправление flow_processor.py

```python
# Задачи:
- [ ] Добавить недостающие импорты (torch, nn, time, numpy)
- [ ] Исправить пустые блоки if/for (добавить pass или логику)
- [ ] Исправить return statements в _collect_final_output
- [ ] Удалить дублирующиеся методы
```

#### 1.2 Оптимизация round_to_nearest_lattice_position

```python
def round_to_nearest_lattice_position_optimized(self, normalized_positions: torch.Tensor) -> torch.Tensor:
    """O(1) арифметическое квантование вместо O(N*M) поиска"""
    # Денормализация
    denorm = (normalized_positions + 1) * 0.5

    # Квантование по осям
    x_idx = (denorm[:, 0] * (self.width - 1)).round().clamp(0, self.width - 1)
    y_idx = (denorm[:, 1] * (self.height - 1)).round().clamp(0, self.height - 1)
    z_idx = (denorm[:, 2] * self.depth).round().clamp(0, self.depth)

    # Обратная нормализация
    norm_x = (x_idx / (self.width - 1)) * 2 - 1
    norm_y = (y_idx / (self.height - 1)) * 2 - 1
    norm_z = (z_idx / self.depth) * 2 - 1

    return torch.stack([norm_x, norm_y, norm_z], dim=1)
```

### Phase 2: Системы управления памятью (День 2)

#### 2.1 Создание SmartMemoryCleaner

```bash
# Файл: energy_flow/utils/smart_memory_cleaner.py
```

#### 2.2 Интеграция в FlowProcessor

```python
class FlowProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...

        # Новый memory cleaner
        from ..utils.smart_memory_cleaner import SmartMemoryCleaner
        self.memory_cleaner = SmartMemoryCleaner(config)

    def step(self, active_flows, global_training_step=None):
        # ... existing code ...

        # Умная очистка вместо периодической
        if self.memory_cleaner.should_cleanup(
            len(active_flows),
            torch.cuda.memory_allocated() / 1e9
        ):
            cleanup_stats = self.memory_cleaner.cleanup_flows(self.lattice)
            logger.log(DEBUG_MEMORY, f"🧹 Smart cleanup: {cleanup_stats}")
```

### Phase 3: Централизованное логирование (День 3)

#### 3.1 Создание UnifiedLogger

```bash
# Файл: energy_flow/utils/unified_logging.py
```

#### 3.2 Замена существующего логирования

```python
# В каждом модуле заменяем:
logger = get_logger(__name__)

# На:
from ..utils.unified_logging import UnifiedLogger
logger = UnifiedLogger.get_logger(__name__)
```

### Phase 4: Метрики и профилирование (День 4)

#### 4.1 Создание MetricsCollector

```python
class MetricsCollector:
    """Легковесный сборщик метрик"""

    def __init__(self, config: MetricsConfig):
        self.enabled = config.enable_metrics
        self.export_interval = config.metrics_export_interval

        # Кольцевые буферы для эффективности
        self.timing_buffer = deque(maxlen=1000)
        self.memory_buffer = deque(maxlen=1000)
        self.flow_stats_buffer = deque(maxlen=1000)

    @contextmanager
    def measure(self, operation: str):
        """Измерение времени и памяти операции"""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        yield

        elapsed = time.perf_counter() - start_time
        memory_delta = (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) - start_memory

        self.timing_buffer.append((operation, elapsed))
        self.memory_buffer.append((operation, memory_delta))
```

---

### 4.2 Feature flags в конфигурации

```python
class EnergyConfig:
    # Existing fields...

    # Feature flags для безопасного тестирования
    use_optimized_rounding: bool = False  # Новое квантование
    use_smart_memory_cleaner: bool = False  # Умная очистка
    use_unified_logger: bool = False  # Новый логгер
    use_metrics_collector: bool = False  # Сбор метрик
```
