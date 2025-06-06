# 🔧 Lightweight Decoder - Эмбединг → Текст

**Версия:** 0.1.0  
**Статус:** 🆕 Phase 2.7 - В разработке  
**Модуль:** 3 (Lightweight Decoder)

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

```python
from inference.lightweight_decoder import HybridDecoder

# Инициализация
decoder = HybridDecoder(
    embedding_dim=768,
    vocab_size=32000,
    max_length=512
)

# Декодирование
processed_embedding = module_2.process(input_embedding)  # От 3D Core
output_text = decoder.decode(processed_embedding)

print(f"Generated: {output_text}")
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

- [ ] **Phase 2.7.1:** PhraseBankDecoder implementation
- [ ] **Phase 2.7.2:** GenerativeDecoder implementation
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
