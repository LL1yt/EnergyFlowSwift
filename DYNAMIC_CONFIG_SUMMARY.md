# Динамическая система конфигурации на основе vlPFC

## 📋 Обзор

Реализована полнофункциональная **Dynamic Configuration System** для 3D Cellular Neural Network, основанная на биологических данных вентролатеральной префронтальной коры (vlPFC).

## 🧠 Биологические основы

### vlPFC характеристики:

- **Нейронов в одном полушарии**: 93,750,000
- **Нейронов в обоих полушариях**: 180,000,000
- **Целевое количество (среднее)**: 136,875,000
- **Синапсов на нейрон**: 5,000-15,000 (среднее: 10,000)
- **Биологические пропорции**: глубина = 0.5 × ширина

python experiment_with_scale.py
python run_dynamic_training.py --scale 0.02 --dataset-limit 1000 --epochs 20

# Стандартные режимы

python run_dynamic_training.py --mode development # scale=0.01
python run_dynamic_training.py --mode research # scale=0.1

## 🏗️ Архитектура системы

### 1. Основные компоненты

```python
# Биологические константы
BiologicalConstants()
- vlPFC нейроанатомические данные
- Синаптические характеристики
- Структурные пропорции

# Настройки масштабирования
ScaleSettings()
- Development: 1% (быстрая разработка)
- Research: 10% (исследования)
- Validation: 30% (валидация)
- Production: 100% (продакшен)

# Вычислитель выражений
ExpressionEvaluator()
- Поддержка выражений типа "{x*y}"
- Умное округление (smart_round)
- Математические функции

# Генератор конфигурации
DynamicConfigGenerator()
- Автоматические вычисления
- Биологически-корректное масштабирование
- Валидация конфигурации
```

### 2. Ключевые особенности

#### ✅ **Исправлено: Правильные округления**

- Все размеры решетки теперь целочисленные
- `smart_round()` функция для корректного округления
- Устранены дробные значения типа `199.79999999999998`

#### 🧮 **Динамические вычисления**

```yaml
lattice:
  x: 666 # Базовая ширина
  y: 666 # Базовая высота
  z: "{x*0.5}" # Биологически точная глубина

  # Масштабирование
  scale_factor: 0.1 # Текущий масштаб
  xs: "{smart_round(x*scale_factor)}" # Масштабированная ширина = 67
  ys: "{smart_round(y*scale_factor)}" # Масштабированная высота = 67
  zs: "{smart_round(z*scale_factor)}" # Масштабированная глубина = 33

  # Автоматические вычисления
  total_neurons: "{xs * ys * zs}" # Общее количество нейронов
  surface_size: "{xs * ys}" # Размер поверхности
  volume: "{xs * ys * zs}" # Объем

embeddings:
  embedding_dim: "{smart_round(xs*ys)}" # Размерность эмбедингов = surface_size
  teacher_embedding_dim: 768 # Размерность teacher модели
```

#### 🔧 **Автоопределение конфигурации**

- Автоматическое определение режима на основе GPU памяти
- RTX 5090 (≥20GB) → validation режим
- RTX 4070 Ti (≥12GB) → research режим
- Остальные → development режим

## 📁 Файловая структура

```
utils/config_manager/
├── dynamic_config.py          # 🆕 Основная реализация
├── config_manager.py          # ✏️ Обновлен с интеграцией
└── ...

test_dynamic_config.py         # 🆕 Базовое тестирование
test_dynamic_config_improved.py # 🆕 Улучшенное тестирование с проверкой округлений
run_overnight_training_fixed_d.py # ✏️ Обновлен для использования динамической конфигурации
DYNAMIC_CONFIG_SUMMARY.md      # 🆕 Этот файл
```

## 🚀 Использование

### 1. Standalone использование

```python
from utils.config_manager.dynamic_config import DynamicConfigManager

# Создание менеджера
manager = DynamicConfigManager()

# Генерация конфигурации для разных режимов
dev_config = manager.create_config_for_mode("development")
research_config = manager.create_config_for_mode("research")
prod_config = manager.create_config_for_mode("production")
```

### 2. Интеграция с основным ConfigManager

```python
from utils.config_manager.config_manager import ConfigManager, ConfigManagerSettings

# Настройки с динамической конфигурацией
settings = ConfigManagerSettings(
    enable_dynamic_config=True,
    dynamic_config_mode="auto",  # Автоопределение
    auto_hardware_detection=True
)

# Создание менеджера
config_manager = ConfigManager(settings)

# Получение конфигурации (с автоматически сгенерированными параметрами)
lattice_config = config_manager.get_config('lattice')
embedding_config = config_manager.get_config('embeddings')
```

### 3. Автоматическая конфигурация под железо

```python
from utils.config_manager.dynamic_config import generate_config_for_current_hardware

# Автоматическое определение оптимальной конфигурации
config = generate_config_for_current_hardware()
```

## 📊 Примеры результатов

### Development режим (scale=0.01)

```
Lattice: 7x7x3
Total neurons: 147
Embedding dimension: 49
Batch size: 2048
```

### Research режим (scale=0.1)

```
Lattice: 67x67x33
Total neurons: 148,017
Embedding dimension: 4,489
Batch size: 1024
```

### Validation режим (scale=0.3)

```
Lattice: 200x200x100
Total neurons: 4,000,000
Embedding dimension: 40,000
Batch size: 512
```

## ✅ Исправления округлений

### До исправлений:

- `199.79999999999998x199.79999999999998x99.89999999999999`
- `66.60000000000001x66.60000000000001x33.300000000000004`

### После исправлений:

- `200x200x100` ✅
- `67x67x33` ✅

Все размеры решетки теперь корректно округляются до целых чисел.

## 🎯 Основные преимущества

1. **Биологическая достоверность** - основано на реальных данных vlPFC
2. **Автоматическое масштабирование** - от разработки до продакшена
3. **Умные округления** - все размеры целочисленные
4. **Динамические вычисления** - embedding_dim автоматически = surface_size
5. **Адаптивность под железо** - автоопределение оптимального режима
6. **Полная интеграция** - работает с существующей системой конфигурации

## 🧪 Тестирование

Запустите тесты для проверки:

```bash
# Базовое тестирование
python test_dynamic_config.py

# Проверка округлений (рекомендуется)
python test_dynamic_config_improved.py
```

## 🔮 Статус: ГОТОВО К ПРОДУКТИВНОМУ ИСПОЛЬЗОВАНИЮ

Система полностью функциональна и интегрирована. Все проблемы с округлениями исправлены.

## ✅ Исправленные проблемы (2025-06-10)

### 1. Интеграция с ConfigManager ✅

- **Проблема**: `ConfigManagerSettings.__init__() got an unexpected keyword argument 'enable_dynamic_config'`
- **Решение**: Добавлены новые параметры в `ConfigManagerSettings`:
  - `enable_dynamic_config: bool = False`
  - `dynamic_config_mode: str = "auto"`
  - `auto_hardware_detection: bool = True`
  - `custom_scale_factor: Optional[float] = None`
- **Статус**: ✅ Интеграция работает корректно

### 2. Критические предупреждения PyTorch ✅

- **Проблема 1**: `std(): degrees of freedom is <= 0` в `neural_cellular_automata.py:453`
- **Решение**: Добавлена защита от проблем со `std()` при недостаточном количестве данных
- **Проблема 2**: `torch.utils.checkpoint: use_reentrant parameter should be passed explicitly`
- **Решение**: Добавлен параметр `use_reentrant=False` в файлы:
  - `emergent_training_stage_3_1_4_1.py`
  - `emergent_training_stage_3_1_4_1_no_st.py`
- **Статус**: ✅ Предупреждения устранены

### 3. Автоопределение железа ✅

- **Функция**: Автоматическое определение оптимального режима на основе GPU памяти
- **Логика**:
  - ≥20GB GPU → `validation` режим
  - ≥12GB GPU → `research` режим
  - <12GB GPU → `development` режим
- **Статус**: ✅ Работает корректно

## 🧪 Результаты последнего тестирования

```
🔳 Анализ загруженной lattice конфигурации:
   x              : 666             (int) ✅
   y              : 666             (int) ✅
   z              : 333             (int) ✅
   xs             : 67              (int) ✅
   ys             : 67              (int) ✅
   zs             : 33              (int) ✅
   total_neurons  : 148137          (int) ✅
   surface_size   : 4489            (int) ✅
   volume         : 148137          (int) ✅

🔗 Анализ embeddings конфигурации:
   embedding_dim: 4489 (int) ✅
   teacher_embedding_dim: 768 (int) ✅
```

**✅ ВСЕ ТЕСТЫ ОКРУГЛЕНИЙ ВЫПОЛНЕНЫ УСПЕШНО!**
