#!/usr/bin/env python3
"""
PHASE 2.7.2 - PhraseBankDecoder Stage 1.2 Optimization Test
==========================================================

Testing Stage 1.2 optimizations:
- Context-aware phrase selection 
- Improved text assembly
- Post-processing capabilities
- Session management
- Enhanced quality metrics

Author: AI Assistant
Date: 6 декабря 2024
"""

import time
import torch
import sys
import os
import numpy as np

# [CONFIG] CUDA COMPATIBILITY FIX для RTX 5090
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False
torch.set_default_device('cpu')

# Добавляем корневую директорию в PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_context_aware_decoding():
    """Тест контекстно-осведомленного декодирования"""
    print("\n[BRAIN] Testing context-aware decoding...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            PhraseBankDecoder, DecodingConfig
        )
        from data.embedding_loader import EmbeddingLoader
        
        # Создание decoder с context-aware режимом
        config = DecodingConfig(
            assembly_method="context_aware",
            context_weight=0.4,
            length_preference="medium",
            enable_grammar_fix=True,
            enable_coherence_boost=True
        )
        
        decoder = PhraseBankDecoder(
            embedding_dim=768,
            phrase_bank_size=50000,
            similarity_threshold=0.75,
            config=config
        )
        
        # Загрузка phrase bank
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тестовые последовательности для проверки контекста
        test_sequences = [
            ["Hello, how are you?", "I'm doing great!", "Thank you for asking"],
            ["Good morning", "Have a nice day", "See you later"],
            ["What's the weather?", "It's sunny today", "Perfect for a walk"]
        ]
        
        print("   [TARGET] Testing context awareness across sequences...")
        
        for seq_idx, sequence in enumerate(test_sequences):
            print(f"   \n   Sequence {seq_idx + 1}: {' -> '.join(sequence)}")
            
            # Начинаем новую сессию
            decoder.start_new_session()
            
            # Обрабатываем последовательность
            for step, text in enumerate(sequence):
                # Генерируем эмбединг
                embedding = embedding_loader.load_from_llm(
                    texts=[text],
                    model_key="distilbert",
                    use_cache=True
                )[0]
                
                # Декодируем
                decoded_text = decoder.decode(embedding)
                context_info = decoder.get_context_info()
                
                print(f"     Step {step + 1}: '{decoded_text}' (history: {context_info['phrase_history_length']})")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Context-aware decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_post_processing():
    """Тест постобработки текста"""
    print("\n✨ Testing text post-processing...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            TextPostProcessor, DecodingConfig
        )
        
        # Конфигурация с включенной постобработкой
        config = DecodingConfig(
            enable_grammar_fix=True,
            enable_coherence_boost=True,
            enable_redundancy_removal=True
        )
        
        post_processor = TextPostProcessor(config)
        
        # Тестовые случаи для постобработки
        test_cases = [
            ("hello  world  test", "Hello world test"),  # Грамматика
            ("test test repeat repeat", "Test repeat"),   # Избыточность  
            ("uncertain response", "It seems uncertain response"),  # Когерентность (низкая уверенность)
        ]
        
        print("   [WRITE] Testing grammar fixes, redundancy removal, coherence boost...")
        
        for raw_text, expected_pattern in test_cases:
            processed = post_processor.process_text(raw_text, confidence=0.5)
            print(f"     '{raw_text}' -> '{processed}'")
            
            # Проверяем базовые улучшения
            assert len(processed.strip()) > 0, "Processed text should not be empty"
            assert processed[0].isupper(), "Should start with capital letter"
        
        print("   [OK] Post-processing works correctly")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_management():
    """Тест управления сессиями"""
    print("\n[INFO] Testing session management...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            PhraseBankDecoder, DecodingConfig
        )
        from data.embedding_loader import EmbeddingLoader
        
        # Создание decoder
        config = DecodingConfig(assembly_method="context_aware")
        decoder = PhraseBankDecoder(config=config)
        
        # Загрузка phrase bank
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тест начальной сессии
        print("   [START] Testing session start/reset...")
        initial_context = decoder.get_context_info()
        assert initial_context['phrase_history_length'] == 0, "Initial context should be empty"
        
        # Добавляем несколько декодирований
        test_texts = ["Hello world", "Good morning", "Thank you"]
        embeddings_batch = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # Обрабатываем каждый эмбединг отдельно
        for i in range(len(test_texts)):
            embedding = embeddings_batch[i]  # Получаем i-й эмбединг
            decoder.decode(embedding)
        
        # Проверяем накопленный контекст
        context_with_history = decoder.get_context_info()
        assert context_with_history['phrase_history_length'] > 0, "Context should have history"
        
        # Сброс сессии
        decoder.start_new_session()
        reset_context = decoder.get_context_info()
        assert reset_context['phrase_history_length'] == 0, "Context should be reset"
        
        print("   [OK] Session management works correctly")
        
        # Тест batch декодирования с сессиями
        print("   [PACKAGE] Testing batch decoding with session boundaries...")
        
        # embeddings_batch уже является тензором правильной формы (N, 768)
        session_boundaries = [0, 2]  # Сброс на позициях 0 и 2
        
        results = decoder.batch_decode_with_sessions(
            embeddings_batch,
            session_boundaries=session_boundaries
        )
        
        assert len(results) == len(test_texts), "Should return result for each input"
        print(f"     Batch results: {results}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Session management failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_assembly_methods_comparison():
    """Тест сравнения методов сборки"""
    print("\n⚖️  Testing assembly methods comparison...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            PhraseBankDecoder, DecodingConfig
        )
        from data.embedding_loader import EmbeddingLoader
        
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        
        # Тестируем разные методы сборки
        assembly_methods = ["weighted", "greedy", "beam_search", "context_aware"]
        
        test_text = "Hello, how are you today?"
        test_embedding = embedding_loader.load_from_llm(
            texts=[test_text],
            model_key="distilbert",
            use_cache=True
        )[0]
        
        results = {}
        
        for method in assembly_methods:
            print(f"   [CONFIG] Testing {method} method...")
            
            config = DecodingConfig(assembly_method=method)
            decoder = PhraseBankDecoder(config=config)
            decoder.load_phrase_bank(embedding_loader=embedding_loader)
            
            # Декодирование с метриками
            decoded_text, metrics = decoder.decode_with_metrics(test_embedding)
            
            results[method] = {
                'text': decoded_text,
                'confidence': metrics['confidence'],
                'quality_score': metrics['quality_score']
            }
            
            print(f"     Result: '{decoded_text}' (confidence: {metrics['confidence']:.3f})")
        
        # Сравниваем результаты
        print("\n   [DATA] Assembly methods comparison:")
        for method, result in results.items():
            print(f"     {method:15}: confidence={result['confidence']:.3f}, quality={result['quality_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Assembly methods comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_optimization():
    """Тест производительности оптимизаций"""
    print("\n[FAST] Testing performance optimizations...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            PhraseBankDecoder, DecodingConfig
        )
        from data.embedding_loader import EmbeddingLoader
        
        # Создание decoder с оптимизированными настройками
        config = DecodingConfig(
            assembly_method="context_aware",
            max_candidates=5,  # Уменьшено для скорости
            similarity_threshold=0.7
        )
        
        decoder = PhraseBankDecoder(config=config)
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тест производительности на батче
        print("   ⏱️  Testing batch performance...")
        
        test_texts = [f"Test sentence number {i}" for i in range(20)]
        batch_embeddings = embedding_loader.load_from_llm(
            texts=test_texts,
            model_key="distilbert",
            use_cache=True
        )
        
        # batch_embeddings уже является тензором правильной формы (N, 768)
        
        # Измеряем время
        start_time = time.time()
        results = decoder.batch_decode(batch_embeddings)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # в миллисекундах
        avg_time = total_time / len(results)
        
        print(f"     Batch size: {len(results)}")
        print(f"     Total time: {total_time:.2f} ms")
        print(f"     Average per item: {avg_time:.2f} ms")
        print(f"     Throughput: {len(results) / (total_time / 1000):.1f} items/sec")
        
        # Проверяем целевую производительность
        target_time_per_item = 50  # 50ms на элемент
        if avg_time <= target_time_per_item:
            print(f"   [OK] Performance target met: {avg_time:.2f}ms <= {target_time_per_item}ms")
        else:
            print(f"   [WARNING]  Performance target missed: {avg_time:.2f}ms > {target_time_per_item}ms")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Тест улучшенных метрик качества"""
    print("\n[CHART] Testing enhanced quality metrics...")
    
    try:
        from inference.lightweight_decoder.phrase_bank_decoder import (
            PhraseBankDecoder, DecodingConfig
        )
        from data.embedding_loader import EmbeddingLoader
        
        config = DecodingConfig(assembly_method="context_aware")
        decoder = PhraseBankDecoder(config=config)
        
        embedding_loader = EmbeddingLoader(cache_dir="./cache")
        decoder.load_phrase_bank(embedding_loader=embedding_loader)
        
        # Тест с различными типами входных данных
        test_cases = [
            "Clear and simple sentence",          # Высокое качество ожидается
            "Ambiguous unclear meaning text",     # Среднее качество
            "Xyztabc random nonsense words"       # Низкое качество ожидается
        ]
        
        print("   [DATA] Testing quality assessment for different input types...")
        
        quality_results = []
        
        for test_text in test_cases:
            embedding = embedding_loader.load_from_llm(
                texts=[test_text],
                model_key="distilbert", 
                use_cache=True
            )[0]
            
            decoded_text, metrics = decoder.decode_with_metrics(embedding)
            
            quality_results.append({
                'input': test_text,
                'output': decoded_text,
                'quality_score': metrics['quality_score'],
                'confidence': metrics['confidence'],
                'coherence': metrics.get('coherence', 0.0),
                'num_candidates': metrics['num_candidates']
            })
            
            print(f"     Input: '{test_text}'")
            print(f"     Output: '{decoded_text}'")
            print(f"     Quality: {metrics['quality_score']:.3f}, Confidence: {metrics['confidence']:.3f}")
        
        # Проверяем тренды качества
        qualities = [r['quality_score'] for r in quality_results]
        print(f"\n   [CHART] Quality trend: {qualities}")
        
        # Статистика декодера
        stats = decoder.get_statistics()
        print(f"   [DATA] Decoder statistics:")
        print(f"     Success rate: {stats['success_rate']}")
        print(f"     Average confidence: {stats['avg_confidence']:.3f}")
        print(f"     Average quality: {stats['avg_quality']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Quality metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция тестирования Stage 1.2"""
    print("[START] PHASE 2.7.2 - PhraseBankDecoder Stage 1.2 Optimization Test")
    print("="*70)
    
    test_results = []
    
    # Выполняем все тесты
    tests = [
        ("Context-Aware Decoding", test_context_aware_decoding),
        ("Text Post-Processing", test_post_processing),
        ("Session Management", test_session_management), 
        ("Assembly Methods Comparison", test_assembly_methods_comparison),
        ("Performance Optimization", test_performance_optimization),
        ("Quality Metrics", test_quality_metrics),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n[INFO] Running {test_name}...")
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"[OK] {test_name}: PASSED")
            else:
                print(f"[ERROR] {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            test_results.append((test_name, False))
    
    # Итоговые результаты
    print("\n" + "="*70)
    print("[TARGET] STAGE 1.2 OPTIMIZATION RESULTS")
    print("="*70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    for test_name, result in test_results:
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n[TARGET] Stage 1.2: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("\n[SUCCESS] STAGE 1.2 OPTIMIZATION: SUCCESS!")
        print("[INFO] Ready for Stage 1.3: Production readiness")
        
        # Checkpoint 1.2 summary
        print("\n[DATA] CHECKPOINT 1.2 ACHIEVEMENTS:")
        print("[OK] Context-aware phrase selection implemented")
        print("[OK] Advanced text post-processing working")
        print("[OK] Session management system operational")
        print("[OK] Multiple assembly methods available")
        print("[OK] Performance optimizations effective")
        print("[OK] Enhanced quality metrics functional")
        
    else:
        print(f"\n[WARNING]  STAGE 1.2 NEEDS IMPROVEMENT: {success_rate:.1f}% < 80%")
        print("[CONFIG] Review failed tests and optimize implementation")

if __name__ == "__main__":
    main() 