# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a research project implementing a **3D Cellular Neural Network** inspired by biological brain structures. The system uses cellular automata-like structures arranged in a 3D lattice where each cell runs the same neural network prototype but processes signals from neighboring cells.

### Контекст проекта

- Исследовательский проект на одного разработчика
- Структура: AA/ (легаси), AA/new_rebuild/ (активная разработка), AA/archive/ (старые версии)
- Тесты запускаются из корня AA/
- new_rebuild/Рабочие рещения из Legacy проекта, которые можно использовать.md - можно использовать как примеры

### Структура `new_rebuild`

Директория `new_rebuild` содержит основную, переработанную архитектуру проекта.

- **`new_rebuild/config/`**: Содержит централизованную конфигурацию проекта (`project_config.py`).
- **`new_rebuild/core/`**: Ядро архитектуры, включающее:
  - **`cells/`**: Определения различных типов клеток.
  - **`cnf/`**: Компоненты для Continuous Normalizing Flows.
  - **`lattice/`**: Реализация 3D-решетки, пространственного хеширования и оптимизаций.
  - **`moe/`**: Компоненты Mixture of Experts.
- **`new_rebuild/utils/`**: Вспомогательные утилиты, такие как управление устройствами (`device_manager.py`) и логирование (`logging.py`).

### Принципы работы

**приоритет на скорость**

- Минимальные церемонии, максимальная эффективность, в том смысле, что можно пожертвовать перфекционизмом в угоду простому и эффективному решению
- Используй современные языковые возможности
- Предпочитай прямолинейные решения сложным абстракциям
- при этом нас НЕ интересует продакшн и все что с этим связано, когда речь идет о больших коммерческих проектах с большим числом сотрудников
- **Конфигурации** - удачные параметры из экспериментов
- **Конфигурации** - стараемся сделать так, что бы конфигурации были централизованы и можно было бы лего настравивать из одного файла
- Использовать централизованное логирование, что бы можно было отследить какой модуль был вызван и кем
- **ИСКЛЮЧАЕМ CLI автоматизация**(за редким исключением, когда без этого нельзя и пользователь укажет отдельно) - оставляем только Python API
- **ИСКЛЮЧАЕМ Множественные конфигурации** - оставляем только один new_rebuild\config\project_config.py
- **ИСКЛЮЧАЕМ Legacy совместимость** - оставляем чистый код
- **ИСКЛЮЧАЕМ Динамические конфигурации**(за редким исключением, когда без этого нельзя и пользователь укажет отдельно) - оставляем статичные конфигурации

### Ключевые оптимизации производительности

**Connection Cache System (НОВИНКА 2024)**

Система pre-computed кэширования классификации связей для массивного ускорения больших решеток:

- **Pre-computed расстояния**: Статические distance matrices для всех пар клеток
- **Базовая классификация**: LOCAL/DISTANT связи вычисляются один раз при старте
- **Functional candidates**: Средние расстояния для динамической similarity проверки
- **Disk persistence**: Кэш сохраняется на диск с hash-ключами конфигурации
- **Automatic management**: Автоматическая перестройка при изменении параметров

**Настройки кэширования в project_config.py:**

```python
# Основные настройки
config.expert.cache.enabled = True  # Включить/выключить кэш
config.expert.cache.enable_performance_monitoring = True  # Мониторинг
config.expert.cache.enable_detailed_stats = False  # Детальная статистика

# Автоматические пороги (обновлены)
config.expert.cache.auto_enable_threshold = 3000   # >3k клеток (снижен)
config.expert.cache.auto_disable_threshold = 1000   # <1k клеток
config.expert.cache.small_lattice_fallback = True   # Fallback для малых решеток

# GPU ускорение (НОВИНКА)
config.expert.cache.use_gpu_acceleration = True  # RTX 5090 поддержка!
config.expert.cache.gpu_batch_size = 10000  # Размер GPU батча
config.expert.cache.prefer_gpu_for_large_lattices = True  # Автоматический GPU для >5k клеток
config.expert.cache.gpu_memory_fraction = 0.8  # Использовать 80% GPU памяти
```

**Производительность:**

- **5×5×5 (125 клеток)**: Кэш отключен, используется fallback
- **8×8×8 (512 клеток)**: CPU кэширование, ~2-3x ускорение
- **15×15×15 (3,375 клеток)**: GPU кэширование, ~5-10x ускорение
- **20×20×20 (8,000 клеток)**: GPU с RTX 5090, ~10-20x ускорение
- **27×27×27+ (большие решетки)**: GPU массивное ускорение, один раз потратить время на создание кэша

**GPU Benefits с RTX 5090:**

- Параллельные вычисления расстояний через `torch.cdist`
- Batch processing с умным управлением памятью
- Автоматический fallback на CPU при недостатке памяти
- Сохранение GPU-созданного кэша на диск для переиспользования

**Использование:**

```python
from new_rebuild.core.moe import create_connection_classifier
from new_rebuild.config import get_project_config

# Автоматические настройки на основе размера решетки
classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))

# Принудительное включение/выключение кэша
classifier = create_connection_classifier(
    lattice_dimensions=(15, 15, 15),
    enable_cache=True  # или False
)

# Получение статистики производительности
stats = classifier.get_classification_stats()
print(f"Cache hit rate: {stats['cache_performance']['cache_hit_rate']}")
print(f"Speedup: {stats['cache_performance'].get('speedup_ratio', 'N/A')}")
```

**Файловая интеграция:**

- `new_rebuild/core/moe/connection_cache.py` - основной кэш менеджер (GPU support)
- `new_rebuild/core/moe/connection_classifier.py` - интеграция с кэшем
- `new_rebuild/core/moe/__init__.py` - экспорт factory функций
- `new_rebuild/config/project_config.py` - централизованные настройки (GPU config)
- `test_connection_cache_settings.py` - тесты конфигурации
- `test_gpu_cache_demo.py` - демонстрация GPU ускорения для RTX 5090

**Тестирование GPU ускорения:**

```bash
# Демо GPU ускорения
python test_gpu_cache_demo.py

# Полные тесты настроек
python test_connection_cache_settings.py
```
