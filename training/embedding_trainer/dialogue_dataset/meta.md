# DialogueDataset Module Metadata

**Модуль:** DialogueDataset - Stage 1.3  
**Версия:** v1.3.0  
**Статус:** ✅ Production Ready  
**Дата:** 7 июня 2025

---

## 📦 EXPORTS

### Основные классы

```python
from training.embedding_trainer.dialogue_dataset import (
    DialogueDataset,     # Основной dataset класс
    DialogueConfig,      # Конфигурация модуля
)
```

### Helper функции

```python
from training.embedding_trainer.dialogue_dataset import (
    create_dialogue_dataset,         # Создание из Q&A пар
    create_conversation_dataset,     # Создание из multi-turn диалогов
    load_dialogue_dataset_from_files # Загрузка из файлов
)
```

### Константы и флаги

```python
from training.embedding_trainer import (
    DIALOGUE_DATASET_AVAILABLE,     # Флаг доступности модуля
)
```

---

## 🔗 DEPENDENCIES

### Модульные зависимости

```python
# Внутренние модули проекта
data.embedding_loader.embedding_loader import EmbeddingLoader  # Teacher LLM интеграция
core.embedding_processor import EmbeddingProcessor             # 3D Cubic Core
training.embedding_trainer.cube_trainer import CubeTrainer    # Dialogue trainer
```

### Внешние зависимости

```yaml
torch: ">=1.9.0" # PyTorch для Dataset/DataLoader
transformers: ">=4.21.0" # Teacher LLM модели
numpy: ">=1.20.0" # Numerical operations
logging: "standard library" # Логирование
pathlib: "standard library" # Path management
hashlib: "standard library" # Cache key generation
json: "standard library" # JSON data parsing
csv: "standard library" # CSV data parsing
```

### Опциональные зависимости

```yaml
yaml: "*" # YAML config files (опционально)
pandas: "*" # DataFrame support (опционально)
```

---

## 🏗️ АРХИТЕКТУРНЫЕ ЗАВИСИМОСТИ

### Upstream модули (требуют DialogueDataset)

```python
training.embedding_trainer.cube_trainer.CubeTrainer
    ↳ Использует DialogueDataset для dialogue режима обучения

training.dialogue_trainer.*  # Планируемые модули Stage 2+
    ↳ Будут использовать DialogueDataset как источник данных
```

### Downstream модули (DialogueDataset использует)

```python
data.embedding_loader.EmbeddingLoader
    ↳ Teacher LLM интеграция для генерации Q&A эмбедингов

core.embedding_processor.EmbeddingProcessor
    ↳ Validation совместимости размеров [8,8,12] = 768D
```

---

## ⚙️ КОНФИГУРАЦИОННЫЕ ТРЕБОВАНИЯ

### Обязательные конфигурации

```yaml
# DialogueConfig основные параметры
teacher_model: str = "distilbert" # Teacher LLM модель
embedding_dim: int = 768 # Размерность эмбедингов
validation_split: float = 0.2 # Train/val split

# Quality control
enable_quality_filter: bool = True # Фильтрация качества
min_question_length: int = 5 # Минимальная длина Q
min_answer_length: int = 10 # Минимальная длина A
```

### Опциональные конфигурации

```yaml
# Advanced features
support_multiturn: bool = True # Multi-turn диалоги
use_cache: bool = True # Smart caching
normalize_embeddings: bool = True # Нормализация эмбедингов
cache_dir: str = "cache/dialogue_dataset" # Директория кэша

# Performance tuning
cache_batch_size: int = 500 # Batch size для кэширования
max_conversations: int = 5000 # Лимит диалогов
```

---

## 🔌 UI/DOM ВЗАИМОДЕЙСТВИЯ

**Статус:** Нет UI/DOM взаимодействий  
**Тип:** Backend data processing модуль

DialogueDataset - pure backend модуль для подготовки данных без UI компонентов.

---

## 📊 API ИНТЕРФЕЙС

### Публичный API

```python
class DialogueDataset(Dataset):
    def __init__(self, config, dialogue_pairs=None, conversations=None)
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]
    def get_dataloader(self, batch_size, shuffle, validation) -> DataLoader
    def get_statistics(self) -> Dict[str, Any]
    def get_sample_dialogues(self, n_samples) -> Dict[str, Any]
    def set_validation_mode(self, is_validation: bool)

# Helper functions API
def create_dialogue_dataset(dialogue_pairs, **kwargs) -> DialogueDataset
def create_conversation_dataset(conversations, **kwargs) -> DialogueDataset
def load_dialogue_dataset_from_files(file_paths, **kwargs) -> DialogueDataset
```

### Внутренний API

```python
class DialogueDataset:
    def _load_from_dialogue_pairs(self, pairs: List[Dict])
    def _load_from_conversations(self, conversations: List[List[Dict]])
    def _validate_teacher_model(self)
    def _filter_dialogue_quality(self, pairs: List[Dict]) -> List[Dict]
    def _create_train_val_split(self)
    def _create_cache_key_for_dialogues(self, pairs) -> str
    def _load_from_cache(self, cache_key) -> Optional[Dict]
    def _save_to_cache(self, cache_key, cache_data)
```

---

## 🧪 ТЕСТОВЫЕ ЗАВИСИМОСТИ

### Тестовые модули

```python
test_dialogue_dataset_basic.py     # Базовое тестирование
test_dialogue_dataset_fixed.py     # Исправленные тесты с batch обработкой
```

### Мок зависимости

Все тесты используют реальные модули - мока не требуются.

---

## 📈 ВЕРСИОНИРОВАНИЕ

### Текущая версия: v1.3.0

```yaml
Major: 1 # Stage 1 (основная реализация)
Minor: 3 # Stage 1.3 (DialogueDataset)
Patch: 0 # Первая production версия
```

### История версий

```yaml
v1.3.0: (7 июня 2025) - Production ready DialogueDataset
  ✅ Teacher LLM архитектура Q→A
  ✅ CubeTrainer совместимость [8,8,12] = 768D
  ✅ Smart caching & quality filtering
  ✅ Multi-turn dialogue поддержка
  ✅ Comprehensive testing (ALL passed)

v1.2.0: (6 июня 2025) - AutoencoderDataset (предыдущий Stage)
v1.1.0: (5 июня 2025) - CubeTrainer foundation
```

### Планируемые версии

```yaml
v2.1.0: Stage 2.1 - Dialogue Training integration
v2.2.0: Stage 2.2 - Advanced training features
v3.0.0: Stage 3.0 - Full production system
```

---

## 🔒 СОВМЕСТИМОСТЬ

### Backward compatibility

✅ **Полная обратная совместимость** с CubeTrainer v1.1.0+  
✅ **API compatibility** с EmbeddingLoader v2.0.0+  
✅ **Configuration compatibility** с существующими YAML конфигами

### Forward compatibility

✅ **Готов к Stage 2.1** dialogue training  
✅ **Expandable API** для future enhancements  
✅ **Configuration extensible** для new features

---

## 🚀 ГОТОВНОСТЬ К РАЗВЕРТЫВАНИЮ

**Production Status:** ✅ **READY**

- ✅ **All tests passed** (100% success rate)
- ✅ **Documentation complete** (README, plan, meta, errors, examples, diagram)
- ✅ **API stable** и backward compatible
- ✅ **Integration verified** с CubeTrainer
- ✅ **Performance validated** с smart caching

**Готов к Stage 2.1 - Dialogue Training!**
