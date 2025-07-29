# 📊 Metrics & Profiling System - Детальный План Реализации

## 🎯 **Основные принципы**

- **Минимальный overhead** - метрики не должны замедлять обучение 
- **Легкое отключение** - простые boolean флаги в конфиге
- **Центральная конфигурация** - все настройки в одном месте
- **Использование существующего логирования** - интеграция с DEBUG_PERFORMANCE/DEBUG_PROFILING
- **Модульная архитектура** - отдельный модуль, не влияющий на core компоненты

## 🏗️ **Архитектура модуля**

```
energy_flow/
├── utils/
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py      # MetricsCollector - центральный сбор метрик
│   │   ├── profiler.py       # ProfilerManager - torch.profiler integration  
│   │   ├── gpu_monitor.py    # GPUMonitor - lightweight GPU tracking
│   │   └── config.py         # MetricsConfig - все настройки метрик
│   └── logging.py           # Существующая система (расширим при необходимости)
```

## 📋 **1. MetricsConfig - Центральная конфигурация**

**Файл**: `energy_flow/utils/metrics/config.py`

```python
@dataclass
class MetricsConfig:
    # === MASTER SWITCHES ===
    enable_metrics: bool = False          # Главный выключатель всех метрик
    enable_profiler: bool = False         # PyTorch Profiler (тяжелый)
    enable_gpu_monitoring: bool = False   # GPU utilization tracking
    
    # === LIGHTWEIGHT METRICS ===
    collect_timing: bool = True           # Базовые timing метрики (почти бесплатно)
    collect_throughput: bool = True       # samples/sec, flows/sec (минимальный overhead)
    collect_memory_basic: bool = True     # torch.cuda.memory_allocated() только
    
    # === ADVANCED METRICS (больший overhead) ===
    collect_gpu_utilization: bool = False     # Требует nvidia-ml-py3 или polling
    collect_memory_detailed: bool = False     # Fragmentation, peak usage
    collect_component_profiling: bool = False # Детальное профилирование компонентов
    
    # === PROFILER SETTINGS ===
    profiler_record_shapes: bool = False      # Увеличивает размер traces
    profiler_profile_memory: bool = False     # Memory profiling (очень тяжелый)
    profiler_with_stack: bool = False         # Stack traces (тяжелый)
    profiler_export_interval: int = 1000      # Экспорт каждые N шагов
    profiler_trace_dir: str = "traces"        # Папка для Chrome traces
    
    # === COLLECTION INTERVALS ===
    metrics_log_interval: int = 10           # Логирование каждые N шагов  
    gpu_monitor_interval: float = 1.0        # GPU polling интервал (секунды)
    memory_cleanup_check_interval: int = 50  # Проверка memory leaks
    
    # === AUTOMATIC CONTROLS ===
    auto_disable_on_slow: bool = True        # Автоотключение при замедлении >5%
    performance_threshold: float = 0.05      # 5% overhead threshold
```

## 🔧 **2. MetricsCollector - Центральный сбор**

**Файл**: `energy_flow/utils/metrics/collector.py`

### Ключевые особенности:
- **Thread-safe** collections с минимальным locking'ом
- **Conditional collection** - метрики собираются только если включены
- **Lightweight storage** - используем deque с ограниченным размером
- **Integration с existing logging** - используем DEBUG_PERFORMANCE уровни

```python
class MetricsCollector:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.enabled = config.enable_metrics
        
        # Lightweight storage (fixed size для избежания memory leaks)
        self.timing_metrics = deque(maxlen=1000) if config.collect_timing else None
        self.throughput_metrics = deque(maxlen=1000) if config.collect_throughput else None
        self.memory_metrics = deque(maxlen=500) if config.collect_memory_basic else None
        
        # Thread safety только если нужно
        self._lock = threading.Lock() if config.enable_gpu_monitoring else None
        
    @contextmanager
    def time_component(self, component_name: str):
        \"\"\"Lightweight timing context manager\"\"\"
        if not self.config.collect_timing:
            yield  # No-op если timing отключен
            return
            
        start_time = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start_time
        
        # Сохраняем только если overhead приемлемый
        if elapsed > 0.001:  # Игнорируем < 1ms измерения
            self.timing_metrics.append((component_name, elapsed, time.time()))
```

## ⚡ **3. GPUMonitor - Легковесный GPU tracking**

**Файл**: `energy_flow/utils/metrics/gpu_monitor.py`

### Стратегия минимального overhead:
- **Polling в отдельном thread** только если включен
- **Graceful fallback** если nvidia-ml-py3 недоступен
- **Smart caching** - обновления раз в секунду максимум

```python
class GPUMonitor:
    def __init__(self, config: MetricsConfig):
        self.enabled = config.enable_gpu_monitoring
        self.interval = config.gpu_monitor_interval
        
        # Инициализация только если включен мониторинг
        if self.enabled:
            self._init_monitoring()
        
    def _init_monitoring(self):
        \"\"\"Lazy initialization только при необходимости\"\"\"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
        except ImportError:
            self.nvml_available = False
            logger.log(DEBUG_PERFORMANCE, \"GPU monitoring: pynvml не найден, используем torch fallback\")
    
    @property  
    def gpu_utilization(self) -> float:
        \"\"\"Cached GPU utilization с smart polling\"\"\"
        if not self.enabled:
            return 0.0
            
        # Используем cached значение если прошло < interval секунд
        now = time.time()
        if hasattr(self, '_last_poll') and now - self._last_poll < self.interval:
            return getattr(self, '_cached_utilization', 0.0)
            
        # Обновляем значение
        self._last_poll = now
        self._cached_utilization = self._poll_gpu_utilization()
        return self._cached_utilization
```

## 🔬 **4. ProfilerManager - torch.profiler integration**

**Файл**: `energy_flow/utils/metrics/profiler.py`

### Умное профилирование:
- **Context manager** для easy integration
- **Conditional activation** - профилер включается только когда нужно
- **Automatic export** с настраиваемым интервалом
- **Chrome Trace format** для визуализации

```python
class ProfilerManager:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.enabled = config.enable_profiler
        self.step_count = 0
        
    @contextmanager
    def profile_step(self, step_name: str = \"train_step\"):
        \"\"\"Context manager для профилирования одного шага\"\"\"
        if not self.enabled:
            yield  # No-op если профилер отключен
            return
            
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=self.config.profiler_record_shapes,
            profile_memory=self.config.profiler_profile_memory,
            with_stack=self.config.profiler_with_stack,
        ) as prof:
            yield prof
            
        # Автоматический экспорт каждые N шагов
        self.step_count += 1
        if self.step_count % self.config.profiler_export_interval == 0:
            self._export_trace(prof, step_name)
            
    def _export_trace(self, prof, step_name: str):
        \"\"\"Экспорт trace в Chrome format\"\"\"
        trace_path = f\"{self.config.profiler_trace_dir}/{step_name}_step_{self.step_count}.json\"
        prof.export_chrome_trace(trace_path)
        logger.log(DEBUG_PROFILING, f\"Exported profiler trace: {trace_path}\")
```

## 🔗 **5. Integration с EnergyTrainer**

### Минимальные изменения в existing коде:

```python
# В EnergyTrainer.__init__()
from ..utils.metrics import MetricsCollector, ProfilerManager, GPUMonitor, MetricsConfig

self.metrics_config = MetricsConfig()  # Или из основного config'а
self.metrics = MetricsCollector(self.metrics_config)
self.profiler = ProfilerManager(self.metrics_config) 
self.gpu_monitor = GPUMonitor(self.metrics_config)

# В train_step()
def train_step(self, ...):
    with self.profiler.profile_step(\"train_step\"):
        with self.metrics.time_component(\"total_step\"):
            
            # Existing код без изменений
            with self.metrics.time_component(\"flow_processor\"):
                cube_output_surface = self.flow_processor.forward(teacher_input_embeddings)
            
            with self.metrics.time_component(\"loss_computation\"):
                # Loss computation код
                
    # Логирование метрик (только если включено)
    if self.metrics.enabled and self.global_step % self.metrics_config.metrics_log_interval == 0:
        self._log_performance_metrics()
```

## 📊 **6. Configuration Integration**

### Добавление в EnergyConfig:

```python
@dataclass
class EnergyConfig:
    # ... существующие параметры ...
    
    # === METRICS & PROFILING ===
    # Простое включение/отключение одним флагом
    enable_performance_monitoring: bool = False    # Master switch
    
    # Детальная настройка (для advanced users)
    metrics_config: Optional[MetricsConfig] = None
    
    def __post_init__(self):
        # Автоматическая настройка метрик
        if self.metrics_config is None:
            self.metrics_config = MetricsConfig()
            self.metrics_config.enable_metrics = self.enable_performance_monitoring
            
            # В DEBUG режиме включаем больше метрик
            if self.debug_mode:
                self.metrics_config.enable_gpu_monitoring = True
                self.metrics_config.collect_component_profiling = True
```

## 🚀 **7. План поэтапной реализации**

### **Phase 1: Core Infrastructure (Priority: HIGH)**
1. Создать `MetricsConfig` с всеми настройками
2. Реализовать `MetricsCollector` с basic timing/throughput
3. Добавить integration points в `EnergyTrainer`
4. **Тестирование**: убедиться что overhead < 1%

### **Phase 2: GPU Monitoring (Priority: MEDIUM)**  
1. Реализовать `GPUMonitor` с fallback стратегией
2. Добавить real-time GPU utilization tracking
3. Integration с conditional logging
4. **Тестирование**: проверить что polling не влияет на performance

### **Phase 3: Advanced Profiling (Priority: LOW)**
1. Реализовать `ProfilerManager` с torch.profiler
2. Chrome Trace export functionality  
3. Automatic bottleneck detection
4. **Тестирование**: профилирование на больших batch'ах

### **Phase 4: Polish & Optimization**
1. Automatic performance regression detection
2. Optimization recommendations на основе метрик
3. TensorBoard integration (опционально)
4. Documentation и examples

## ⚠️ **Важные принципы реализации**

1. **Zero overhead по умолчанию** - все метрики отключены unless explicitly enabled
2. **Fail-safe** - ошибки в метриках не должны прерывать обучение
3. **Minimal dependencies** - используем только torch и standard library
4. **Graceful fallback** - если что-то недоступно, продолжаем работать
5. **Thread safety** только где необходимо - избегаем лишнего locking'а

## 🎯 **Expected Results**

- **< 1% overhead** при включенных базовых метриках
- **Easy on/off** через простые boolean флаги
- **Rich insights** когда метрики включены
- **Production ready** мониторинг для deployment
- **Chrome Profiler integration** для deep performance analysis