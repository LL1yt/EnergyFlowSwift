# 🔧 Lightweight Decoder - Эмбединг → Текст

**Версия:** 0.1.0  
**Статус:** 🔄 **Phase 2.7 - Stage 1.1 ЗАВЕРШЕН!**  
**Модуль:** 3 (Lightweight Decoder)

## 🎉 **ПОСЛЕДНИЕ ДОСТИЖЕНИЯ**

- ✅ **PhraseBankDecoder ГОТОВ** - полностью функционален
- ✅ **Checkpoint 1.1 ЗАВЕРШЕН** - 5/5 тестов пройдено
- ✅ **RTX 5090 совместимость** - CPU-only режим работает
- ✅ **Module 1 ↔ Module 3 интеграция** - успешная интеграция с Teacher LLM Encoder

## 🎯 Назначение

Lightweight Decoder является **Модулем 3** в модульной архитектуре 3D Cellular Neural Network. Основная задача - преобразование обработанных эмбедингов (768D) обратно в связный текст с минимальными вычислительными затратами.

## 🏗️ Архитектура

### Три Варианта Декодера

```
┌─────────────────────┐
│  Processed Embedding │ (768D от Module 2)
│     (from 3D Core)   │
└──────────┬──────────┘
           │
      ┌────▼────┐
      │ DECODER │
      │ CHOICE  │
      └────┬────┘
           │
    ┌──────▼──────┬──────────────┬──────────────┐
    │             │              │              │
┌───▼───┐    ┌───▼───┐    ┌─────▼─────┐
│Phrase │    │Genera-│    │  Hybrid   │
│ Bank  │    │ tive  │    │ Approach  │
│Decoder│    │Decoder│    │  Decoder  │
└───┬───┘    └───┬───┘    └─────┬─────┘
    │            │              │
    └────────────┼──────────────┘
                 │
        ┌────────▼────────┐
        │  Generated Text │
        └─────────────────┘
```

### 1. PhraseBankDecoder

- **Метод:** Поиск ближайших семантических фраз
- **Размер:** ~100MB (phrase bank)
- **Скорость:** Очень быстрый
- **Качество:** Высокое для common phrases

### 2. GenerativeDecoder

- **Метод:** Compact transformer architecture
- **Размер:** ~1-2M parameters
- **Скорость:** Средняя
- **Качество:** Гибкая генерация

### 3. HybridDecoder

- **Метод:** Phrase bank + генерация для gaps
- **Размер:** ~2M total
- **Скорость:** Оптимизированная
- **Качество:** Лучшее из обоих подходов

## 🎯 Технические Характеристики

- **Input:** Processed embeddings 768D
- **Output:** Coherent text sequences
- **Target BLEU:** >0.4
- **Model Size:** <2M parameters
- **Integration:** Seamless с Modules 1 & 2

## 🚀 Использование

### PhraseBankDecoder (✅ ГОТОВ К ИСПОЛЬЗОВАНИЮ)

```python
from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
from data.embedding_loader import EmbeddingLoader

# Инициализация декодера
decoder = PhraseBankDecoder(
    embedding_dim=768,
    similarity_threshold=0.8
)

# Загрузка phrase bank
embedding_loader = EmbeddingLoader(cache_dir="./cache")
decoder.load_phrase_bank(embedding_loader=embedding_loader)

# Декодирование
processed_embedding = module_2.process(input_embedding)  # От 3D Core
output_text = decoder.decode(processed_embedding)

print(f"Generated: {output_text}")
```

### Пример полной интеграции с Modules 1 & 2

```python
from data.embedding_loader import EmbeddingLoader
from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder

# Module 1: Teacher LLM Encoder
encoder = EmbeddingLoader(cache_dir="./cache")

# Module 3: Lightweight Decoder
decoder = PhraseBankDecoder(embedding_dim=768)
decoder.load_phrase_bank(embedding_loader=encoder)

# Полный pipeline
input_text = "Hello, how are you today?"

# Текст → Эмбединг (Module 1)
embedding = encoder.load_from_llm(
    texts=[input_text],
    model_key="distilbert"
)[0]

# Эмбединг → Текст (Module 3)
output_text = decoder.decode(embedding)
print(f"Decoded: {output_text}")
```

## 📊 Интеграция с Общей Системой

```python
# Полная система Modules 1 + 2 + 3
class CompleteCognitiveSystem:
    def __init__(self):
        self.encoder = TeacherLLMEncoder()      # Module 1
        self.processor = EmbeddingProcessor()   # Module 2
        self.decoder = HybridDecoder()          # Module 3 (этот модуль)

    def forward(self, input_text):
        # Текст → Эмбединг
        embedding = self.encoder.encode(input_text)

        # Эмбединг → Обработанный эмбединг
        processed = self.processor.process(embedding)

        # Обработанный эмбединг → Текст
        output_text = self.decoder.decode(processed)

        return output_text
```

## 📋 План Разработки

- [x] **Phase 2.7.1:** PhraseBankDecoder implementation ✅ **ЗАВЕРШЕНО**
- [ ] **Phase 2.7.2:** GenerativeDecoder implementation 🔄 **СЛЕДУЮЩИЙ**
- [ ] **Phase 2.7.3:** HybridDecoder implementation
- [ ] **Phase 2.7.4:** Integration testing
- [ ] **Phase 2.7.5:** Performance optimization

## 🧪 Тестирование

Модуль будет тестироваться на:

- Качество генерации (BLEU score)
- Семантическое сохранение
- Computational efficiency
- Integration compatibility

## 🔗 Зависимости

- **Internal:** `core.embedding_processor`, `data.tokenizer`
- **External:** `torch`, `transformers`, `nltk`
- **Integration:** Modules 1 & 2 готовы
