#!/usr/bin/env python3
"""
🧪 BASIC TEST: Phase 2.7.1 - PhraseBankDecoder Infrastructure

Тест для Checkpoint 1.1:
- [ ] Phrase bank загружается и индексируется
- [ ] Similarity search работает корректно  
- [ ] Performance: <10ms на поиск фразы

Phase 2.7.1 - ЭТАП 1.1 Test
"""

import sys
import torch
import time
import logging
from pathlib import Path

def test_phrase_bank_loading():
    """Тест загрузки и индексирования phrase bank"""
    print("🏦 Testing phrase bank loading and indexing...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseLoader, PhraseEntry
        from data.embedding_loader import EmbeddingLoader
        
        # Создание embedding loader для тестирования
        embedding_loader = EmbeddingLoader(
            model_name="distilbert-base-uncased",  # Компактная модель для теста
            device="cpu"  # CPU для совместимости
        )
        
        # Создание phrase bank
        phrase_bank = PhraseBank(
            embedding_dim=768,
            similarity_threshold=0.8,
            index_type="linear"  # Linear search для стабильности
        )
        
        # Загрузка sample phrases
        print("   📝 Creating sample phrases...")
        sample_phrases = PhraseLoader.create_sample_phrases(embedding_loader)
        
        if len(sample_phrases) == 0:
            print("   ❌ No sample phrases created")
            return False
        
        print(f"   ✅ Created {len(sample_phrases)} sample phrases")
        
        # Добавление в phrase bank
        print("   🔨 Building phrase index...")
        phrase_bank.add_phrases(sample_phrases)
        
        print(f"   ✅ Phrase bank indexed with {len(phrase_bank.index.phrases)} phrases")
        return True
        
    except Exception as e:
        print(f"   ❌ Phrase bank loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_search():
    """Тест similarity search функциональности"""
    print("\n🔍 Testing similarity search...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseLoader
        from data.embedding_loader import EmbeddingLoader
        
        # Подготовка
        embedding_loader = EmbeddingLoader(
            model_name="distilbert-base-uncased",
            device="cpu"
        )
        
        phrase_bank = PhraseBank(embedding_dim=768, index_type="linear")
        sample_phrases = PhraseLoader.create_sample_phrases(embedding_loader)
        phrase_bank.add_phrases(sample_phrases)
        
        # Тест поиска с известной фразой
        print("   🎯 Testing search with known phrase...")
        test_text = "Hello, how are you?"
        test_embedding = embedding_loader.encode_text(test_text)
        
        # Поиск
        results = phrase_bank.search_phrases(test_embedding, k=5)
        
        if len(results) == 0:
            print("   ❌ No search results returned")
            return False
        
        print(f"   ✅ Found {len(results)} similar phrases")
        
        # Проверка качества результатов
        top_phrase, top_similarity = results[0]
        print(f"   📊 Top result: '{top_phrase.text}' (similarity: {top_similarity:.3f})")
        
        if top_similarity < 0.5:
            print("   ⚠️  Low similarity for exact match")
        
        # Тест с произвольным эмбедингом
        print("   🎲 Testing search with random embedding...")
        random_embedding = torch.randn(768)
        random_results = phrase_bank.search_phrases(random_embedding, k=3)
        
        print(f"   ✅ Random search returned {len(random_results)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Similarity search failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Тест производительности: <10ms на поиск фразы"""
    print("\n⚡ Testing search performance...")
    
    try:
        from inference.lightweight_decoder.phrase_bank import PhraseBank, PhraseLoader
        from data.embedding_loader import EmbeddingLoader
        
        # Подготовка
        embedding_loader = EmbeddingLoader(
            model_name="distilbert-base-uncased",
            device="cpu"
        )
        
        phrase_bank = PhraseBank(embedding_dim=768, index_type="linear")
        sample_phrases = PhraseLoader.create_sample_phrases(embedding_loader)
        phrase_bank.add_phrases(sample_phrases)
        
        # Подготовка тестовых эмбедингов
        test_embeddings = []
        for i in range(10):
            test_text = f"Test phrase number {i}"
            embedding = embedding_loader.encode_text(test_text)
            test_embeddings.append(embedding)
        
        # Измерение производительности
        print("   ⏱️  Measuring search performance...")
        
        total_time = 0
        num_searches = len(test_embeddings)
        
        for embedding in test_embeddings:
            start_time = time.time()
            results = phrase_bank.search_phrases(embedding, k=5)
            end_time = time.time()
            
            search_time = (end_time - start_time) * 1000  # в миллисекундах
            total_time += search_time
        
        avg_time = total_time / num_searches
        print(f"   📊 Average search time: {avg_time:.2f}ms")
        
        # Проверка критерия <10ms
        if avg_time < 10.0:
            print("   ✅ Performance target met (<10ms)")
            performance_ok = True
        else:
            print(f"   ⚠️  Performance target missed (target: <10ms, actual: {avg_time:.2f}ms)")
            performance_ok = False
        
        # Статистика phrase bank
        stats = phrase_bank.get_statistics()
        print(f"   📈 Cache hit rate: {stats['cache_hit_rate']}")
        print(f"   🔍 Total searches: {stats['total_searches']}")
        
        return performance_ok
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
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
            similarity_threshold=0.7,  # Немного ниже для тестирования
        )
        
        # Загрузка phrase bank
        embedding_loader = EmbeddingLoader(
            model_name="distilbert-base-uncased",
            device="cpu"
        )
        
        print("   📚 Loading phrase bank...")
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тест декодирования
        print("   🔤 Testing basic decoding...")
        test_text = "Thank you very much."
        test_embedding = embedding_loader.encode_text(test_text)
        
        decoded_text = decoder.decode(test_embedding)
        print(f"   📝 Decoded: '{decoded_text}'")
        
        # Тест с метриками
        print("   📊 Testing decoding with metrics...")
        decoded_text, metrics = decoder.decode_with_metrics(test_embedding)
        print(f"   📈 Quality metrics:")
        print(f"      - Confidence: {metrics['confidence']:.3f}")
        print(f"      - Quality Score: {metrics['quality_score']:.3f}")
        print(f"      - Candidates: {metrics['num_candidates']}")
        
        # Статистика декодера
        decoder_stats = decoder.get_statistics()
        print(f"   📊 Decoder stats:")
        print(f"      - Success rate: {decoder_stats['success_rate']}")
        print(f"      - Avg confidence: {decoder_stats['avg_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PhraseBankDecoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_modules():
    """Тест интеграции с Modules 1 & 2"""
    print("\n🔗 Testing integration with existing modules...")
    
    try:
        from data.embedding_loader import EmbeddingLoader
        from core.embedding_processor import EmbeddingProcessor
        from inference.lightweight_decoder.phrase_bank_decoder import PhraseBankDecoder
        
        # Module 1: Teacher LLM Encoder
        print("   🔴 Setting up Module 1 (Teacher LLM Encoder)...")
        encoder = EmbeddingLoader(
            model_name="distilbert-base-uncased",
            device="cpu"
        )
        
        # Module 2: 3D Cubic Core 
        print("   🔵 Setting up Module 2 (EmbeddingProcessor)...")
        processor = EmbeddingProcessor(
            lattice_size=(4, 4, 4),  # Маленький размер для теста
            propagation_steps=5
        )
        
        # Module 3: Lightweight Decoder
        print("   🟡 Setting up Module 3 (PhraseBankDecoder)...")
        decoder = PhraseBankDecoder(embedding_dim=768)
        decoder.load_phrase_bank(embedding_loader=encoder)
        
        # Тест полного pipeline
        print("   🌊 Testing end-to-end pipeline...")
        
        test_text = "Hello, how are you today?"
        
        # Текст → Эмбединг (Module 1)
        embedding = encoder.encode_text(test_text)
        print(f"   ✅ Module 1 output: {embedding.shape}")
        
        # Эмбединг → Обработанный эмбединг (Module 2)
        processed_embedding = processor.process(embedding)
        print(f"   ✅ Module 2 output: {processed_embedding.shape}")
        
        # Обработанный эмбединг → Текст (Module 3)
        output_text = decoder.decode(processed_embedding)
        print(f"   ✅ Module 3 output: '{output_text}'")
        
        print(f"\n   🎯 Full pipeline result:")
        print(f"      Input:  '{test_text}'")
        print(f"      Output: '{output_text}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция теста"""
    print("🚀 PHASE 2.7.1 - PhraseBankDecoder Infrastructure Test")
    print("=" * 70)
    print("Checkpoint 1.1 Verification")
    print()
    
    # Тесты для Checkpoint 1.1
    tests = [
        ("Phrase Bank Loading", test_phrase_bank_loading),
        ("Similarity Search", test_similarity_search),
        ("Performance (<10ms)", test_performance),
        ("PhraseBankDecoder", test_phrase_bank_decoder),
        ("Module Integration", test_integration_with_modules)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Результаты
    print("\n" + "=" * 70)
    print("📊 CHECKPOINT 1.1 RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Checkpoint 1.1: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Checkpoint 1.1 критерии
    checkpoint_criteria = [
        "Phrase bank загружается и индексируется",
        "Similarity search работает корректно", 
        "Performance: <10ms на поиск фразы"
    ]
    
    if passed == total:
        print("\n🎉 CHECKPOINT 1.1 COMPLETE!")
        print("\n✅ Критерии выполнены:")
        for criterion in checkpoint_criteria:
            print(f"   ✅ {criterion}")
        
        print("\n📝 Ready for Phase 2.7.1 - Этап 1.2:")
        print("   - PhraseBankDecoder implementation")
        print("   - Context-aware phrase selection")
        print("   - Post-processing для coherent text assembly")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Address issues before proceeding to Этап 1.2")
        return False

if __name__ == "__main__":
    # Убедимся что мы в правильной директории
    if not Path("config").exists() or not Path("core").exists():
        print("❌ Please run this test from the project root directory")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1) 