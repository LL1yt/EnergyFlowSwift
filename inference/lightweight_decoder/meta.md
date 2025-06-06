# 📊 METADATA: Lightweight Decoder

**Модуль:** inference/lightweight_decoder/  
**Версия:** 0.1.0  
**Дата создания:** 6 июня 2025  
**Последнее обновление:** 6 июня 2025  
**Статус:** 🆕 Инициализация Phase 2.7

---

## 🔗 ЗАВИСИМОСТИ

### Module Dependencies (Internal)

```python
# Прямые зависимости
from core.embedding_processor import EmbeddingProcessor  # Module 2 output
from data.tokenizer import TokenizerManager             # Text processing
from data.embedding_loader import EmbeddingLoader       # Reference embeddings

# Косвенные зависимости
from core.lattice_3d import Lattice3D                  # Через EmbeddingProcessor
from data.embedding_reshaper import EmbeddingReshaper   # Через EmbeddingProcessor
```

### External Dependencies

```python
# Core ML Framework
torch>=1.9.0                    # PyTorch для neural networks
transformers>=4.21.0            # Hugging Face transformers
torch-audio                     # Audio processing capabilities

# NLP Processing
nltk>=3.7                       # Natural language toolkit
sentence-transformers           # Sentence embeddings
spacy>=3.4.0                   # Advanced NLP

# Similarity Search & Indexing
faiss-cpu                       # Fast similarity search
annoy                          # Approximate nearest neighbors

# Evaluation & Metrics
sacrebleu                       # BLEU score calculation
rouge-score                     # ROUGE metrics
bert-score                      # Semantic similarity metrics

# Utilities
numpy>=1.20.0                  # Numerical operations
pyyaml                         # Configuration files
tqdm                           # Progress bars
```

### UI/DOM Dependencies

```python
# Не применимо - это backend inference модуль
ui_dependencies: None
dom_interactions: None
```

---

## 📤 EXPORTED API

### Main Classes

```python
# Phase 2.7.1 - Phrase Bank Approach
class PhraseBankDecoder:
    def __init__(self, embedding_dim, phrase_bank_size, similarity_threshold)
    def decode(self, embedding: torch.Tensor) -> str
    def load_phrase_bank(self, path: str) -> None
    def build_index(self) -> None

# Phase 2.7.2 - Generative Approach
class GenerativeDecoder:
    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers)
    def decode(self, embedding: torch.Tensor) -> str
    def generate(self, embedding: torch.Tensor, max_length: int) -> str
    def set_temperature(self, temperature: float) -> None

# Phase 2.7.3 - Hybrid Approach
class HybridDecoder:
    def __init__(self, phrase_decoder, generative_decoder, confidence_threshold)
    def decode(self, embedding: torch.Tensor) -> str
    def set_strategy(self, strategy: str) -> None  # 'phrase', 'generative', 'hybrid'
    def get_confidence(self, embedding: torch.Tensor) -> float

# Unified Interface
class DecoderFactory:
    @staticmethod
    def create_decoder(decoder_type: str, config: dict) -> BaseDecoder
    @staticmethod
    def load_pretrained(model_path: str) -> BaseDecoder
```

### Configuration Classes

```python
class DecoderConfig:
    def __init__(self, config_path: str)
    def get_decoder_config(self, decoder_type: str) -> dict
    def validate_config(self) -> bool

class PhraseConfig:
    phrase_bank_path: str
    similarity_threshold: float
    max_phrases: int

class GenerativeConfig:
    model_size: str          # 'small', 'medium'
    vocab_size: int
    max_length: int
    temperature: float

class HybridConfig:
    phrase_threshold: float
    generation_threshold: float
    confidence_weighting: bool
```

### Utility Functions

```python
# Evaluation utilities
def calculate_bleu(references: List[str], hypothesis: str) -> float
def calculate_rouge(references: List[str], hypothesis: str) -> dict
def semantic_similarity(text1: str, text2: str) -> float

# Integration utilities
def validate_embedding_input(embedding: torch.Tensor) -> bool
def preprocess_output_text(text: str) -> str
def postprocess_generated_text(text: str) -> str
```

---

## 🎛️ CONFIGURATION SCHEMA

### config/lightweight_decoder.yaml

```yaml
# Lightweight Decoder Configuration
lightweight_decoder:
  # General settings
  default_decoder: "hybrid" # phrase_bank, generative, hybrid
  embedding_dim: 768 # Input embedding dimension
  max_output_length: 512 # Maximum generated text length

  # PhraseBankDecoder settings
  phrase_bank:
    enabled: true
    bank_size: 50000 # Number of phrases in bank
    similarity_threshold: 0.8 # Minimum similarity for phrase selection
    index_type: "faiss" # faiss, annoy
    cache_size: 1000 # Cache frequently used phrases

  # GenerativeDecoder settings
  generative:
    enabled: true
    model_size: "medium" # small (~1M), medium (~2M)
    vocab_size: 32000 # Vocabulary size
    hidden_size: 1024 # Hidden layer dimension
    num_layers: 4 # Number of transformer layers
    num_heads: 8 # Attention heads
    temperature: 0.8 # Sampling temperature

  # HybridDecoder settings
  hybrid:
    enabled: true
    phrase_threshold: 0.8 # When to prefer phrase bank
    generation_threshold: 0.6 # When to prefer generation
    confidence_weighting: true # Combine confidence scores
    fallback_strategy: "phrase" # phrase, generative

  # Evaluation settings
  evaluation:
    calculate_bleu: true
    calculate_rouge: true
    semantic_similarity: true
    reference_corpus: "data/test/references.txt"
```

---

## 🏗️ АРХИТЕКТУРНЫЕ КОМПОНЕНТЫ

### Core Components

1. **BaseDecoder** - Abstract base class
2. **PhraseBankDecoder** - Phrase-based decoding
3. **GenerativeDecoder** - Neural generative model
4. **HybridDecoder** - Combined approach
5. **DecoderFactory** - Creation and management

### Supporting Components

1. **PhraseBank** - Phrase storage and indexing
2. **EmbeddingIndex** - Fast similarity search
3. **TextProcessor** - Pre/post-processing
4. **QualityAssessor** - Output quality evaluation
5. **ConfigurationManager** - Settings management

### Integration Components

1. **ModuleConnector** - Integration с Modules 1 & 2
2. **Pipeline** - End-to-end processing
3. **CacheManager** - Performance optimization
4. **MetricsCollector** - Quality monitoring

---

## 📊 МЕТРИКИ И МОНИТОРИНГ

### Quality Metrics

- **BLEU Score:** Text generation quality
- **ROUGE Score:** Summarization quality
- **Semantic Similarity:** Meaning preservation
- **Coherence Score:** Text fluency
- **Diversity Score:** Output variation

### Performance Metrics

- **Inference Time:** Decoding speed
- **Memory Usage:** Resource consumption
- **Throughput:** Items processed per second
- **Cache Hit Rate:** Efficiency optimization

### Integration Metrics

- **Module Compatibility:** Integration success rate
- **API Response Time:** Interface performance
- **Error Rate:** System reliability
- **Configuration Validation:** Setup success

---

## 🎯 ГОТОВНОСТЬ К ИНТЕГРАЦИИ

### Module 1 Integration ✅ READY

- Teacher LLM Encoder fully operational
- Standard embedding format (768D)
- Configuration compatibility confirmed

### Module 2 Integration ✅ READY

- EmbeddingProcessor производит processed embeddings
- Output format standardized (768D)
- Quality preserved (0.999 cosine similarity)

### Module 3 Implementation 🚀 STARTING

- Architecture designed
- Dependencies identified
- Integration points defined
- Ready to begin implementation

---

**СТАТУС:** 🎯 Готов к началу Phase 2.7 реализации!
