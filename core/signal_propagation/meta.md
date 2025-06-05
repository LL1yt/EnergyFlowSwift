# Signal Propagation Module - Metadata

## Module Information

**Name:** `signal_propagation`  
**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**Type:** Core Component  
**Phase:** 1 (Foundation)

## Description

Модуль временной динамики и распространения сигналов в 3D клеточной нейронной сети. Реализует сложное поведение сигналов во времени с множественными режимами распространения и автоматическим анализом паттернов.

## Dependencies

### Module Dependencies (Internal)

- **core.cell_prototype** - Для клеточной динамики
  - `CellPrototype` - Прототип клетки для обновления состояний
  - `CellConfig` - Конфигурация клеток
- **core.lattice_3d** - Для пространственной структуры
  - `Lattice3D` - 3D решетка клеток
  - `LatticeConfig` - Конфигурация решетки
  - `Position3D` - 3D координаты

### External Dependencies

- **torch** (>=1.9.0) - PyTorch для вычислений
  - `torch.Tensor` - Основной тип данных
  - `torch.nn.Module` - Базовый класс для neural networks
  - `torch.sin`, `torch.cos` - Математические функции
- **numpy** (>=1.20.0) - Численные операции
  - `numpy.ndarray` - Массивы для корреляционного анализа
  - `numpy.corrcoef` - Корреляционный анализ
- **logging** (standard library) - Логирование
- **dataclasses** (standard library) - Структуры данных
- **enum** (standard library) - Перечисления
- **typing** (standard library) - Типизация

### UI/DOM Dependencies

None - это модуль backend без UI компонентов.

## Exported API

### Classes

#### Core Classes

- **`TimeManager`** - Управление временной динамикой
- **`SignalPropagator`** - Основной класс распространения сигналов
- **`PatternAnalyzer`** - Анализ пространственно-временных паттернов
- **`ConvergenceDetector`** - Детекция сходимости системы
- **`AdaptiveConvergenceDetector`** - Адаптивная детекция сходимости

#### Configuration Classes

- **`TimeConfig`** - Конфигурация временного управления
- **`PropagationConfig`** - Конфигурация распространения сигналов
- **`PatternConfig`** - Конфигурация анализа паттернов
- **`ConvergenceConfig`** - Конфигурация детекции сходимости

#### Enums

- **`TimeMode`** - Режимы временного управления (FIXED, ADAPTIVE)
- **`PropagationMode`** - Режимы распространения (WAVE, DIFFUSION, DIRECTIONAL, CUSTOM)
- **`PatternType`** - Типы паттернов (WAVE, SPIRAL, UNIFORM, CLUSTERED, CHAOTIC, STATIC)
- **`ConvergenceMode`** - Критерии сходимости (ABSOLUTE, RELATIVE, ENERGY, GRADIENT, STATISTICAL, COMBINED)

### Functions

#### Factory Functions

- **`create_signal_propagator()`** - Создание SignalPropagator из конфигураций
- **`create_time_manager()`** - Создание TimeManager из конфигурации
- **`create_pattern_analyzer()`** - Создание PatternAnalyzer из конфигурации
- **`create_convergence_detector()`** - Создание ConvergenceDetector из конфигурации

#### Utility Functions

- **`load_signal_propagation_config()`** - Загрузка конфигурации из YAML
- **`validate_propagation_config()`** - Валидация конфигурации
- **`initialize_signals_on_face()`** - Инициализация сигналов на грани решетки

## File Structure

```
core/signal_propagation/
├── __init__.py                 # Экспорты модуля
├── time_manager.py            # Управление временем
├── signal_propagator.py       # Основная логика распространения
├── propagation_patterns.py    # Анализ паттернов
├── convergence_detector.py    # Детекция сходимости
├── README.md                  # Документация
├── plan.md                    # План реализации
├── meta.md                    # Метаданные (этот файл)
├── errors.md                  # Документированные ошибки
├── diagram.mmd                # Архитектурная диаграмма
└── examples.md                # Примеры использования
```

## Integration Points

### Input Interfaces

- **Lattice3D** - Получает решетку клеток для работы
- **Input Signals** - Принимает входные сигналы на гранях решетки
- **Configuration** - Загружает конфигурацию из YAML файлов

### Output Interfaces

- **Output Signals** - Предоставляет сигналы с выходных граней
- **Pattern Reports** - Экспортирует обнаруженные паттерны
- **Statistics** - Предоставляет статистику распространения
- **Convergence Status** - Сообщает о статусе сходимости

### Event Interfaces

- **Time Events** - События временных шагов
- **Pattern Events** - События обнаружения паттернов
- **Convergence Events** - События достижения сходимости

## Performance Characteristics

- **Time Complexity:** O(N³ × S) где N - размер решетки, S - количество шагов
- **Space Complexity:** O(N³ × H) где H - глубина истории
- **Memory Usage:** ~100MB для решетки 10×10×10 с историей 100 шагов
- **CPU Utilization:** Высокая (векторизованные операции PyTorch)
- **GPU Compatibility:** Готов к GPU ускорению (все операции PyTorch)

## Configuration Schema

```yaml
signal_propagation:
  propagation:
    mode: "wave" # wave|diffusion|directional|custom
    signal_strength: 1.0
    decay_rate: 0.1
    noise_level: 0.0
    boundary_condition: "reflective"
    wave_speed: 1.0
    diffusion_coefficient: 0.5
    direction_vector: [1.0, 0.0, 0.0]
    max_signal_amplitude: 10.0
    min_signal_threshold: 1e-6

  time:
    dt: 0.01
    max_time_steps: 1000
    mode: "fixed" # fixed|adaptive
    history_length: 100
    enable_checkpoints: true

  patterns:
    enable_analysis: true
    analysis_frequency: 10
    confidence_threshold: 0.5
    spatial_window_size: 3
    temporal_window_size: 5

  convergence:
    mode: "combined" # absolute|relative|energy|gradient|statistical|combined
    tolerance: 1e-6
    patience: 10
    check_frequency: 5
    adaptive_threshold: true
    min_improvement: 1e-8
```

## Testing Coverage

- ✅ **Unit Tests:** 100% coverage всех классов и методов
- ✅ **Integration Tests:** Взаимодействие всех компонентов
- ✅ **Performance Tests:** Тесты производительности на больших решетках
- ✅ **Error Handling Tests:** Обработка всех типов ошибок
- ✅ **Configuration Tests:** Валидация всех конфигураций

## Known Limitations

1. **CPU Only:** Текущая реализация работает только на CPU
2. **Memory Scale:** Память растет кубически с размером решетки
3. **Pattern Detection:** Некоторые сложные паттерны могут не распознаваться
4. **Real-time:** Не оптимизировано для real-time приложений

## Future Enhancements

- 🚀 GPU ускорение с CUDA
- 🚀 Динамическое управление памятью
- 🚀 Дополнительные типы паттернов
- 🚀 Real-time визуализация
- 🚀 Параллельная обработка больших решеток

## Maintenance

**Last Updated:** December 5, 2025  
**Maintainer:** AA Project Team  
**Review Cycle:** Each Phase  
**Update Frequency:** After major features or bug fixes

## License

Same as project license - See main project README.
