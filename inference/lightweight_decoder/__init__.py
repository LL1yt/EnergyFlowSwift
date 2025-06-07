"""
🔧 LIGHTWEIGHT DECODER - Компактный декодер эмбединг→текст

Модуль реализует три основных подхода к декодированию:
1. PhraseBankDecoder - поиск ближайших семантических фраз
2. GenerativeDecoder - генеративная модель ~1-2M параметров  
3. HybridDecoder - комбинированный подход для максимального качества

🎯 Технические характеристики:
- Input: эмбединги 768D (от EmbeddingProcessor)
- Output: coherent text sequences
- Target BLEU: >0.4
- Model size: <2M parameters
- Integration: seamless с Modules 1 & 2

📈 Метрики качества:
- BLEU score для text generation
- Semantic similarity preservation  
- Coherence и fluency оценки
- Computational efficiency
"""

__version__ = "0.1.0"

# Phase 2.7.1 exports
from .phrase_bank_decoder import PhraseBankDecoder, DecodingConfig
from .phrase_bank import PhraseBank, PhraseEntry, PhraseLoader

# Import GenerativeDecoder (Stage 2.1 Integration)
from .generative_decoder import GenerativeDecoder, GenerativeConfig, create_generative_decoder

# Экспорты с новым GenerativeDecoder
__all__ = [
    "PhraseBankDecoder",     # Phase 2.7.1 ✅
    "DecodingConfig",        # Configuration ✅
    "PhraseBank",            # Infrastructure ✅
    "PhraseEntry",           # Data structure ✅
    "PhraseLoader",          # Utilities ✅
    "GenerativeDecoder",     # Phase 2.7.2 ✅ STAGE 2.1 READY!
    "GenerativeConfig",      # Configuration ✅
    "create_generative_decoder",  # Factory ✅
    # "HybridDecoder",         # Phase 2.7.3 🔜 PLANNED
] 