# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Research project implementing a **3D Cellular Neural Network** inspired by biological brain structures. Uses cellular automata-like structures in a 3D lattice where each cell runs the same neural network prototype but processes signals from neighboring cells.

### Контекст проекта

- Исследовательский проект на одного разработчика
- Структура: AA/ (легаси), AA/new_rebuild/ (активная разработка), AA/archive/ (старые версии)
- Тесты запускаются из корня AA/
- new_rebuild\Working_solutions_from_the_Legacy_project_that_can_be_used.md - примеры рабочих решений

## Архитектура new_rebuild

### config/ - Централизованная конфигурация

- **`__init__.py`** - Экспорт ProjectConfig и фабричных функций
- **`config_components.py`** - Модульные компоненты конфигурации
- **`simple_config.py`** - `SimpleProjectConfig` - единая точка доступа ко всем настройкам

### core/ - Ядро архитектуры

#### cells/ - Клетки нейронной сети

#### moe/ - Mixture of Experts

- **`moe_processor.py`** - `MoEConnectionProcessor` с тремя экспертами:
  - Local Expert (SimpleLinear)
  - Functional Expert (HybridGNN_CNF)
  - Distant Expert (GPUEnhancedCNF)

#### cnf/ - Continuous Normalizing Flows

- **`gpu_enhanced_cnf.py`** - `GPUEnhancedCNF` с векторизованным Neural ODE
- **`gpu_optimized_euler_solver.py`** - Адаптивная интеграция с Lipschitz control

#### lattice/ - 3D решетка

- **`lattice.py`** - `Lattice3D` - управление клетками и MoE

#### lattice/spatial_optimization/

#### training/ - Обучение

- **`embedding_trainer.py`** - `EmbeddingTrainer` - цикл: эмбединги → куб → эмбединги → текст
- **`embedding_lattice_mapper.py`** - Маппинг эмбеддингов в решетку

#### inference/ - Инференс

- **`text_decoder.py`** - `SimpleTextDecoder` и `JointTextDecoder` для декодирования в текст

#### common/ - Общие компоненты

- **`embedding_transformer.py`** - Преобразование размерностей эмбеддингов (768D ↔ lattice)
- **`interfaces.py`** - Абстрактные интерфейсы системы

### utils/ - Утилиты

- **`logging.py`** - Централизованное логирование с custom debug levels
- **`device_manager.py`** - `DeviceManager` для GPU/CPU управления
- **`model_cache.py`** - Кэширование моделей
- **`hardcoded_checker.py`** - Защита от hardcoded значений

### docs/ - Документация

- **`todo.md`** - Задачи на будущее

## Принципы работы

**Приоритет: вдумчивость, постепенность, эффективность**

- Сначала разобраться в проблеме, потом действовать
- Простой вариант → тесты → оптимизация при необходимости

### Основные принципы

- Централизованные конфигурации и логирование
- Нам не нужны сложные тесты. мы проверяем работоспособность и потом тестируем на реальных реализациях обучения, например `test_forward_simple_v2.py`
- Минимальные церемонии, максимальная эффективность
- Современные языковые возможности
- Прямолинейные решения вместо сложных абстракций
- Проект исследовательский, не продакшн
- Без fallback - лучше ошибка чем костыли
- RTX 5090 32GB - используем по максимуму

### Что исключаем

- **НЕТ** CLI автоматизации (только Python API)
- **НЕТ** Множественных конфигураций
- **НЕТ** Legacy совместимости
- **НЕТ** Динамических конфигураций

## Система конфигурации

### 3 режима работы

- **DEBUG** - быстрые тесты (15x15x15 решетка, максимум логов)
- **EXPERIMENT** - исследования (30x30x30, сбалансировано)
- **OPTIMIZED** - финальные прогоны (100x100x100, минимум логов)

```python
from new_rebuild.config import create_debug_config, set_project_config
config = create_debug_config()
set_project_config(config)
```

### Защита от hardcoded

- `@no_hardcoded` - декоратор для функций
- `strict_no_hardcoded()` - автоматическая замена
- `HardcodedValueError` - исключение с инструкцией
- `allow_hardcoded` - временное отключение

### Custom Debug Levels

- `DEBUG_VERBOSE` (11) - подробный вывод
- `DEBUG_CACHE` (12) - кэширование
- `DEBUG_SPATIAL` (13) - пространственная оптимизация
- `DEBUG_FORWARD` (14) - forward pass
- `DEBUG_MEMORY` (15) - управление памятью
- `DEBUG_TRAINING` (16) - прогресс обучения
- `DEBUG_INIT` (17) - инициализация
