# ❌ УСТАРЕВШИЕ КОМПОНЕНТЫ

## Компоненты old проекта, которые НЕ переносим в new_rebuild

> **ПРИНЦИП**: Сохраняем только эффективные, биологически корректные компоненты  
> **ЦЕЛЬ**: Избежать legacy сложности и technical debt

---

## 🚫 CLI И АВТОМАТИЗАЦИЯ (НЕ ИСПОЛЬЗУЕМ)

### ❌ Automated Training Scripts

- `automated_training_refactored.py`
- `automated_training.py`
- `training/automated_training/` (весь модуль)
- `smart_resume_training/` (весь модуль)
- `run_*.py` скрипты (весь набор)

**Причина**: Сложность интеграции, CLI зависимости  
**Замена**: `new_rebuild/training/trainer.py` (простой Python API)

### ❌ Dynamic Configuration System

- `utils/config_manager/dynamic_config.py`
- `production_training/config/`
- `training/automated_training/stage_runner.py`

**Причина**: Излишняя сложность, динамические конфигурации  
**Замена**: `new_rebuild/config/project_config.py` (статичный config)

---

## 🚫 СЛОЖНЫЕ АРХИТЕКТУРНЫЕ МОДУЛИ (НЕ ИСПОЛЬЗУЕМ)

### ❌ Legacy Training Infrastructure

- `emergent_training/` (частично - берем только EmergentGMLPCell)
- `training/embedding_trainer/` (сложная система адаптеров)
- `production_training/` (излишне сложная автоматизация)

**Причина**: Сложная интеграция, multiple configs  
**Замена**: Упрощенные версии в `new_rebuild/`

### ❌ Complex Embedding Processing

- `archive/embedding_processor/` (сложная система)
- `archive/embedding_reshaper/` (избыточная функциональность)
- `data/embedding_adapter/` (сложные адаптеры)

**Причина**: Overengineering для базовых задач  
**Замена**: `new_rebuild/data/embeddings.py` (простая обработка)

### ❌ Advanced Data Pipelines

- `training/embedding_trainer/advanced_dataset_expansion.py`
- `training/embedding_trainer/dialogue_dataset/`
- `data/data_visualization/` (сложная визуализация)

**Причина**: Не нужны для MVP  
**Замена**: Простая обработка phrase pairs

---

## 🚫 УСТАРЕВШИЕ CELL АРХИТЕКТУРЫ (НЕ ИСПОЛЬЗУЕМ)

### ❌ Complex gMLP Variants

- `core/cell_prototype/architectures/gmlp_opt_connections.py` (слишком сложный)
- `archive/cleanup_2024_06_19/gmlp_cell.py` (legacy версия)
- Multiple gMLP configurations в configs/

**Причина**: Излишняя сложность, параметрическое многообразие  
**Замена**: `EmergentGMLPCell` (простая, эффективная версия)

### ❌ Legacy Cell Prototypes

- `core/cell_prototype/main.py` (сложная factory система)
- Multiple prototype configurations
- Cell factory patterns

**Причина**: Overengineering, сложные зависимости  
**Замена**: Прямая инициализация клеток в `new_rebuild/`

---

## 🚫 СЛОЖНАЯ ТОПОЛОГИЯ И ПЛАСТИЧНОСТЬ (НЕ ИСПОЛЬЗУЕМ)

### ❌ Advanced Topology Systems

- `core/lattice_3d/spatial_hashing.py` (сложный spatial hashing)
- `core/lattice_3d/clustering/` (функциональная кластеризация)
- `core/lattice_3d/plasticity/` (сложные правила пластичности)

**Причина**: Преждевременная оптимизация для MVP  
**Замена**: Простая 26-neighbor topology

### ❌ Advanced Signal Propagation

- `core/signal_propagation/` (сложная система)
- `emergent_training/model/propagation.py` (3D convolutions)

**Причина**: Сложность без доказанной эффективности  
**Замена**: Простой forward pass через клетки

---

## 🚫 TESTING И DEBUGGING INFRASTRUCTURE (НЕ ИСПОЛЬЗУЕМ)

### ❌ Complex Test Suites

- `test_phase4_*.py` (множественные сложные тесты)
- `archive/debugging_sessions/` (debugging артефакты)
- `demos/test_versions/` (версионные тесты)

**Причина**: Сложность maintenance, legacy зависимости  
**Замена**: Простые unit tests в `new_rebuild/tests/`

### ❌ Advanced Debugging Tools

- `diagnose_*.py` (сложная диагностика)
- `debug_*.py` (debug скрипты)
- Complex logging infrastructure

**Причина**: Overengineering для debugging  
**Замена**: Простое logging в `new_rebuild/utils/logging.py`

---

## 🚫 LEGACY CONFIGURATIONS (НЕ ИСПОЛЬЗУЕМ)

### ❌ Multiple Config Systems

- `config/` (старая система с множественными конфигами)
- `utils/centralized_config.py` (попытка централизации, но сложная)
- YAML-based динамические конфигурации

**Причина**: Конфликт конфигураций, сложность интеграции  
**Замена**: Один `ProjectConfig` dataclass

### ❌ Legacy Model Managers

- `model_weights_manager.py`
- `utils/config_loader.py`
- Complex configuration validation

**Причина**: Избыточная сложность для простых задач  
**Замена**: Simple save/load в trainer

---

## ✅ ЧТО ПЕРЕНОСИМ (WHITELIST)

### ✅ Core Cell Architectures

- `core/cell_prototype/architectures/minimal_nca_cell.py` → `new_rebuild/core/cells/nca_cell.py`
- `emergent_training/model/cell.py` (EmergentGMLPCell) → `new_rebuild/core/cells/gmlp_cell.py`

### ✅ Basic 3D Lattice Concepts

- Базовая структура из `core/lattice_3d/lattice.py`
- Neighbor topology principles (упрощенные)
- 3D координатная система

### ✅ Training Loss Functions

- Базовые loss functions из `emergent_training/model/loss.py`
- MSE reconstruction loss

### ✅ Embedding Processing Concepts

- Phrase-based training идея
- Простая обработка embeddings

---

## 📋 MIGRATION CHECKLIST

### ✅ НЕ копировать:

- [ ] Любые CLI скрипты (`run_*.py`, `*_training.py`)
- [ ] Dynamic configuration системы
- [ ] Complex factory patterns
- [ ] Advanced debugging tools
- [ ] Multiple test suites
- [ ] YAML-based конфигурации
- [ ] Legacy compatibility layers

### ✅ Упростить и перенести:

- [ ] `MinimalNCACell` (убрать сложные зависимости)
- [ ] `EmergentGMLPCell` (убрать spatial complexity)
- [ ] Базовую 3D topology (только neighbor finding)
- [ ] Простые loss functions
- [ ] Базовые embedding utils

### ✅ Создать с нуля:

- [ ] `ProjectConfig` (dataclass)
- [ ] `SimpleTrainer` (без CLI)
- [ ] `HybridCell` (NCA + gMLP композиция)
- [ ] `SimpleLattice3D` (без complex topology)
- [ ] Basic test suite
- [ ] Simple logging

---

**РЕЗУЛЬТАТ**: Clean архитектура без legacy багажа, готовая к production scaling
