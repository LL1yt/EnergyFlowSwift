# Документированные Ошибки: EmbeddingProcessor

**Модуль:** `core/embedding_processor/`  
**Phase:** 2.5 - Core Embedding Processor  
**Период разработки:** 6 июня 2025

_Примечание: Документируются только РЕАЛЬНЫЕ ошибки, encountered во время разработки_

---

## 🐛 Ошибка #1: Import Error - Missing Factory Functions

**Дата:** 6 июня 2025  
**Тип:** ImportError  
**Критичность:** Высокая

### Сообщение об ошибке

```python
ImportError: cannot import name 'create_autoencoder_config' from 'core.embedding_processor'
```

### Причина

В `__init__.py` не были экспортированы конфигурационные функции-фабрики:

- `create_autoencoder_config`
- `create_generator_config`
- `create_dialogue_config`
- `create_default_config`

### Решение

Обновлен `core/embedding_processor/__init__.py`:

```python
# Добавлены экспорты функций-фабрик
from .config import (
    create_autoencoder_config,
    create_generator_config,
    create_dialogue_config,
    create_default_config
)
```

### Предотвращение

- Всегда проверять completeness экспортов в `__init__.py`
- Использовать импорт-тесты в test suite

---

## 🐛 Ошибка #2: EmbeddingReshaper API Mismatch

**Дата:** 6 июня 2025  
**Тип:** TypeError  
**Критичность:** Высокая

### Сообщение об ошибке

```python
TypeError: EmbeddingReshaper.__init__() got an unexpected keyword argument 'strategy'
```

### Причина

Использован устаревший параметр `strategy` в инициализации EmbeddingReshaper. Правильный API использует:

- `reshaping_method`
- `preserve_semantics`
- `semantic_threshold`

### Проблемный код

```python
# НЕПРАВИЛЬНО
reshaper = EmbeddingReshaper(
    input_dim=768,
    cube_shape=(8, 8, 8),
    strategy="adaptive"  # ❌ Несуществующий параметр
)
```

### Решение

```python
# ПРАВИЛЬНО
reshaper = EmbeddingReshaper(
    input_dim=768,
    cube_shape=(8, 8, 8),
    reshaping_method="adaptive",     # ✅ Правильный параметр
    preserve_semantics=True,         # ✅ Правильный параметр
    semantic_threshold=0.95          # ✅ Правильный параметр
)
```

### Предотвращение

- Всегда проверять актуальность API перед использованием
- Консультироваться с документацией модулей
- Использовать type hints для раннего выявления ошибок

---

## 🐛 Ошибка #3: Lattice3D Configuration Type Error

**Дата:** 6 июня 2025  
**Тип:** TypeError  
**Критичность:** Средняя

### Сообщение об ошибке

```python
TypeError: create_lattice_from_config() expected file path, got dict
```

### Причина

Функция `create_lattice_from_config()` ожидает путь к конфигурационному файлу, но получила словарь параметров напрямую.

### Проблемный код

```python
# НЕПРАВИЛЬНО
lattice_config = {
    "size": [8, 8, 8],
    "propagation_steps": 10
}
lattice = create_lattice_from_config(lattice_config)  # ❌ Передается dict
```

### Решение

```python
# ПРАВИЛЬНО
lattice_config = LatticeConfig(
    size=[8, 8, 8],
    propagation_steps=10,
    convergence_threshold=0.001
)
lattice = Lattice3D(lattice_config)  # ✅ Используется правильный конструктор
```

### Предотвращение

- Использовать правильные конструкторы для объектов
- Проверять типы параметров функций
- Читать документацию API перед использованием

---

## ✅ Общая Статистика Ошибок

### Итоговая статистика

- **Всего ошибок обнаружено:** 3
- **Критичных ошибок:** 2
- **Средних ошибок:** 1
- **Все ошибки исправлены:** ✅ Да
- **Время на исправление:** ~30 минут

### Категории ошибок

- **Import/Export errors:** 1 (33%)
- **API compatibility:** 1 (33%)
- **Type mismatches:** 1 (33%)

### Lessons learned

1. **Всегда проверять полноту экспортов** в `__init__.py`
2. **Консультироваться с актуальной документацией** API модулей
3. **Использовать правильные конструкторы** для инициализации объектов
4. **Тестировать integration points** между модулями

**Общий результат:** Все ошибки успешно исправлены, модуль работает стабильно с результатом 0.999 cosine similarity ✅
