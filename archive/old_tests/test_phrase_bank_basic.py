#!/usr/bin/env python3
"""
PHASE 2.7.1 - PhraseBankDecoder Infrastructure Test
=================================================

Checkpoint 1.1 Verification:
- Phrase bank loading and indexing 
- Similarity search functionality
- Performance testing (<10ms target)
- PhraseBankDecoder functionality
- Integration with Modules 1 & 2

Author: AI Assistant
Date: 6 июня 2025
"""

import time
import torch
import sys
import os

# [CONFIG] CUDA COMPATIBILITY FIX для RTX 5090
# Принудительно используем CPU для совместимости
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False

# Устанавливаем CPU как default device
if torch.cuda.is_available():
    print("[WARNING]  CUDA detected but forcing CPU mode for RTX 5090 compatibility")
torch.set_default_device('cpu')

# Добавляем корневую директорию в PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_phrase_bank_loading():
    """Тест загрузки и индексирования phrase bank"""
    print("\n🏦 Testing phrase bank loading and indexing...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseLoader
        from data.embedding_loader import EmbeddingLoader
        
        # Создание embedding loader для тестирования (правильный API)
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        
        # Создание phrase bank
        phrase_bank = PhraseBank(embedding_dim=768, index_type="linear")
        
        # Создание тестовых фраз через LLM
        print("   [WRITE] Creating sample phrases using LLM...")
        test_texts = [
            "Hello, how are you?",
            "Thank you very much", 
            "Good morning everyone",
            "Have a nice day",
            "See you later"
        ]
        
        # Генерируем эмбединги для фраз
        test_embeddings = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",  # Используем правильный ключ модели
            use_cache=True
        )
        
        # Создаем phrase entries
        from inference.lightweight_decoder.phrase_bank import PhraseEntry
        sample_phrases = []
        for i, text in enumerate(test_texts):
            # Определение категории на основе содержания
            category = "greeting" if "hello" in text.lower() or "good morning" in text.lower() else "general"
            
            phrase_entry = PhraseEntry(
                text=text,
                embedding=test_embeddings[i],
                frequency=1,
                category=category
            )
            sample_phrases.append(phrase_entry)
        
        # Добавляем фразы в банк
        phrase_bank.add_phrases(sample_phrases)
        
        # Проверяем загрузку
        stats = phrase_bank.get_statistics()
        print(f"   [OK] Loaded {stats['total_phrases']} phrases")
        print(f"   [DATA] Index type: {stats['index_type']}")
        print(f"   [FAST] Average search time: {stats['avg_search_time_ms']} ms")
        print(f"   [CONFIG] FAISS available: {stats['faiss_available']}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Phrase bank loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_search():
    """Тест функциональности поиска по сходству"""
    print("\n[MAGNIFY] Testing similarity search...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseEntry
        from data.embedding_loader import EmbeddingLoader
        
        # Подготовка (правильный API)
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        
        phrase_bank = PhraseBank(embedding_dim=768, index_type="linear")
        
        # Создание тестовых фраз
        test_texts = [
            "Hello, how are you?",
            "Hi there, how's it going?",
            "Good morning everyone",
            "Thank you very much",
            "See you later"
        ]
        
        # Генерируем эмбединги
        test_embeddings = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Создаем и добавляем фразы
        sample_phrases = []
        for i, text in enumerate(test_texts):
            phrase_entry = PhraseEntry(
                text=text,
                embedding=test_embeddings[i],
                frequency=1,
                category="test"
            )
            sample_phrases.append(phrase_entry)
        
        phrase_bank.add_phrases(sample_phrases)
        
        # Тест поиска с известной фразой
        print("   [TARGET] Testing search with known phrase...")
        test_query = "Hello, how are you?"
        query_embedding = embedding_loader.load_from_llm(
            texts=[test_query],
            model_key="distilbert",
            use_cache=True
        )[0]  # Берем первый результат
        
        # Поиск
        results = phrase_bank.search_phrases(query_embedding, k=3)
        
        if len(results) == 0:
            print("   [ERROR] No search results returned")
            return False
        
        print(f"   [OK] Found {len(results)} similar phrases")
        
        # Проверка качества результатов
        top_phrase, top_similarity = results[0]
        print(f"   [DATA] Top result: '{top_phrase.text}' (similarity: {top_similarity:.3f})")
        
        # Тест с произвольным эмбедингом
        print("   [DICE] Testing search with random embedding...")
        random_embedding = torch.randn(768)
        random_results = phrase_bank.search_phrases(random_embedding, k=3)
        
        print(f"   [OK] Random search returned {len(random_results)} results")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Similarity search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Тест производительности: <10ms на поиск фразы"""
    print("\n[FAST] Testing search performance...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseEntry
        from data.embedding_loader import EmbeddingLoader
        
        # Подготовка (правильный API)
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        
        phrase_bank = PhraseBank(embedding_dim=768, index_type="linear")
        
        # Создание большего набора фраз для тестирования производительности
        test_texts = [f"Test phrase number {i}" for i in range(20)]
        
        # Генерируем эмбединги
        test_embeddings = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Создаем и добавляем фразы
        sample_phrases = []
        for i, text in enumerate(test_texts):
            phrase_entry = PhraseEntry(
                text=text,
                embedding=test_embeddings[i],
                frequency=1,
                category="test"
            )
            sample_phrases.append(phrase_entry)
        
        phrase_bank.add_phrases(sample_phrases)
        
        # Подготовка тестовых запросов
        query_texts = ["Test phrase", "Random query", "Hello world"]
        query_embeddings = embedding_loader.load_from_llm(
            texts=query_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Измерение производительности
        print("   ⏱️  Measuring search performance...")
        
        total_time = 0
        num_searches = len(query_embeddings)
        
        for embedding in query_embeddings:
            start_time = time.time()
            results = phrase_bank.search_phrases(embedding, k=5)
            end_time = time.time()
            
            search_time = (end_time - start_time) * 1000  # в миллисекундах
            total_time += search_time
        
        avg_time = total_time / num_searches
        print(f"   [DATA] Average search time: {avg_time:.2f}ms")
        
        # Проверка критерия <10ms
        if avg_time < 10.0:
            print("   [OK] Performance target met (<10ms)")
            performance_ok = True
        else:
            print(f"   [WARNING]  Performance target missed (target: <10ms, actual: {avg_time:.2f}ms)")
            performance_ok = False
        
        # Статистика phrase bank
        stats = phrase_bank.get_statistics()
        print(f"   [CHART] Cache hit rate: {stats.get('cache_hit_rate', 'N/A')}")
        print(f"   [MAGNIFY] Total searches: {stats.get('total_searches', 0)}")
        
        return performance_ok
        
    except Exception as e:
        print(f"   [ERROR] Performance test failed: {e}")
        return False

def test_phrase_bank_decoder():
    """Тест основного PhraseBankDecoder"""
    print("\n🔤 Testing PhraseBankDecoder...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
        from data.embedding_loader import EmbeddingLoader
        
        # Создание декодера
        decoder = PhraseBankDecoder(
            embedding_dim=768,
            similarity_threshold=0.5,  # Немного ниже для тестирования
        )
        
        # Загрузка phrase bank (правильный API)
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        
        print("   [BOOKS] Loading phrase bank...")
        # Создаем простую phrase bank для тестирования
        test_texts = [
            "Thank you very much.",
            "Hello, how are you?",
            "Good morning everyone.",
            "Have a great day!",
            "See you later."
        ]
        
        test_embeddings = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Загружаем phrase bank в декодер (правильный способ)
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тест декодирования
        print("   🔤 Testing basic decoding...")
        test_query = "Thank you very much."
        test_embedding = embedding_loader.load_from_llm(
            texts=[test_query],
            model_key="distilbert",
            use_cache=True
        )[0]
        
        decoded_text = decoder.decode(test_embedding)
        print(f"   [WRITE] Decoded: '{decoded_text}'")
        
        # Тест batch декодирования
        print("   [PACKAGE] Testing batch decoding...")
        batch_queries = ["Hello there", "Good day"]
        batch_embeddings = embedding_loader.load_from_llm(
            texts=batch_queries,
            model_key="distilbert",
            use_cache=True
        )
        
        batch_results = decoder.batch_decode(batch_embeddings)
        print(f"   [WRITE] Batch decoded {len(batch_results)} texts")
        
        # Статистика декодера
        decoder_stats = decoder.get_statistics()
        print(f"   [DATA] Decoder stats:")
        print(f"      - Decode attempts: {decoder_stats.get('decode_attempts', 0)}")
        print(f"      - Success count: {decoder_stats.get('success_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] PhraseBankDecoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_modules():
    """Тест интеграции с Modules 1 & 2"""
    print("\n[LINK] Testing integration with existing modules...")
    
    try:
        from data.embedding_loader import EmbeddingLoader
        from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
        
        # Module 1: Teacher LLM Encoder (правильный API)
        print("   🔴 Setting up Module 1 (Teacher LLM Encoder)...")
        encoder = EmbeddingLoader(cache_dir="./cache")
        
        # Module 3: Lightweight Decoder
        print("   🟡 Setting up Module 3 (PhraseBankDecoder)...")
        decoder = PhraseBankDecoder(embedding_dim=768)
        
        # Создание тестового phrase bank
        test_texts = [
            "Hello, how are you today?",
            "Thank you for your help.",
            "Good morning everyone.",
            "Have a wonderful day!"
        ]
        
        test_embeddings = encoder.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Загружаем phrase bank в декодер (правильный способ)
        decoder.load_phrase_bank(embedding_loader=encoder)
        
        # Тест интеграции
        print("   🌊 Testing Module 1 → Module 3 pipeline...")
        
        test_text = "Hello, how are you today?"
        
        # Текст → Эмбединг (Module 1)
        embedding = encoder.load_from_llm(
            texts=[test_text],
            model_key="distilbert",
            use_cache=True
        )[0]
        
        print(f"   📏 Embedding shape: {embedding.shape}")
        
        # Эмбединг → Текст (Module 3)
        decoded_text = decoder.decode(embedding)
        print(f"   [WRITE] Decoded text: '{decoded_text}'")
        
        # Проверка результата
        if decoded_text and len(decoded_text) > 0:
            print("   [OK] Integration successful")
            return True
        else:
            print("   [ERROR] Integration failed - empty result")
            return False
        
    except Exception as e:
        print(f"   [ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("[START] PHASE 2.7.1 - PhraseBankDecoder Infrastructure Test")
    print("=" * 70)
    print("Checkpoint 1.1 Verification\n")
    
    # Список тестов
    tests = [
        ("Phrase Bank Loading", test_phrase_bank_loading),
        ("Similarity Search", test_similarity_search),
        ("Performance (<10ms)", test_performance),
        ("PhraseBankDecoder", test_phrase_bank_decoder),
        ("Module Integration", test_integration_with_modules),
    ]
    
    # Запуск тестов
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 70}")
        result = test_func()
        results.append((test_name, result))
    
    # Итоговые результаты
    print("\n" + "=" * 70)
    print("[DATA] CHECKPOINT 1.1 RESULTS")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n[TARGET] Checkpoint 1.1: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\n[SUCCESS] ALL TESTS PASSED! Ready for ETAP 1.2")
        print("[INFO] Next: PhraseBankDecoder refinement and optimization")
    elif success_rate >= 80:
        print("\n[WARNING]  MOSTLY SUCCESSFUL - Minor issues to fix")
    else:
        print("\n[ERROR] MULTIPLE FAILURES - Need debugging before proceeding")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 