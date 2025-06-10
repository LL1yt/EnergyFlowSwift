#!/usr/bin/env python3
"""
Basic Test for Tokenizer Module - 3D Cellular Neural Network

Простой тест для проверки основной функциональности модуля tokenizer.
Тестирует базовый токенайзер без зависимостей от transformers.

Автор: 3D CNN Team
Дата: Декабрь 2025
"""

import sys
import os
import logging
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_tokenizer():
    """Тест базового токенайзера."""
    print("🧪 Testing Basic Tokenizer...")
    
    try:
        from data.tokenizer import TokenizerManager
        
        # Создание токенайзера с базовым типом
        tokenizer = TokenizerManager(tokenizer_type='basic')
        
        # Тестовый текст
        test_text = "Hello world! This is a test of the basic tokenizer."
        
        print(f"[WRITE] Input text: {test_text}")
        
        # Тест токенизации
        tokens = tokenizer.tokenize(test_text)
        print(f"🔤 Tokens: {tokens}")
        
        # Тест кодирования
        token_ids = tokenizer.encode(test_text)
        print(f"🔢 Token IDs: {token_ids}")
        
        # Тест декодирования
        decoded_text = tokenizer.decode(token_ids)
        print(f"[FILE] Decoded: {decoded_text}")
        
        # Проверка метрик
        metrics = tokenizer.get_metrics()
        print(f"[DATA] Metrics: {metrics}")
        
        # Проверка специальных токенов
        special_tokens = tokenizer.get_special_tokens()
        print(f"[TARGET] Special tokens: {special_tokens}")
        
        print("[OK] Basic tokenizer test PASSED!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic tokenizer test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_text_processor():
    """Тест предобработчика текста."""
    print("\n🧪 Testing Text Processor...")
    
    try:
        from data.tokenizer.text_processor import TextProcessor
        
        # Создание процессора
        processor = TextProcessor()
        
        # Тестовый текст с различными проблемами
        test_text = "  Hello   WORLD!!! This is a TEST with   extra spaces.  "
        
        print(f"[WRITE] Input text: '{test_text}'")
        
        # Предобработка
        processed_text = processor.preprocess(test_text)
        print(f"[CONFIG] Processed: '{processed_text}'")
        
        # Статистика обработки
        stats = processor.get_processing_stats(test_text, processed_text)
        print(f"[DATA] Processing stats: {stats}")
        
        # Валидация
        is_valid = processor.validate_text(processed_text)
        print(f"[OK] Text valid: {is_valid}")
        
        print("[OK] Text processor test PASSED!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Text processor test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Тест batch обработки."""
    print("\n🧪 Testing Batch Processing...")
    
    try:
        from data.tokenizer import TokenizerManager
        
        # Создание токенайзера
        tokenizer = TokenizerManager(tokenizer_type='basic')
        
        # Тестовые тексты
        test_texts = [
            "First test sentence.",
            "Second test sentence with more words.",
            "Third sentence is here."
        ]
        
        print(f"[WRITE] Input texts: {test_texts}")
        
        # Batch кодирование
        batch_encoded = tokenizer.batch_encode(test_texts)
        print(f"🔢 Batch encoded: {batch_encoded}")
        
        # Batch декодирование
        batch_decoded = tokenizer.batch_decode(batch_encoded)
        print(f"[FILE] Batch decoded: {batch_decoded}")
        
        print("[OK] Batch processing test PASSED!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Batch processing test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_lattice_integration():
    """Тест интеграции с решеткой."""
    print("\n🧪 Testing Lattice Integration...")
    
    try:
        from data.tokenizer import TokenizerManager
        
        # Создание токенайзера
        tokenizer = TokenizerManager(tokenizer_type='basic')
        
        # Тестовый текст
        test_text = "Input for neural network processing"
        
        # Размер решетки
        lattice_size = (5, 5, 5)
        
        print(f"[WRITE] Input text: {test_text}")
        print(f"🔲 Lattice size: {lattice_size}")
        
        # Подготовка для решетки
        lattice_input = tokenizer.prepare_for_lattice(test_text, lattice_size)
        print(f"[TARGET] Lattice input shape: {lattice_input.shape}")
        print(f"🔢 Lattice input: {lattice_input}")
        
        # Проверка размера
        expected_shape = (lattice_size[0], lattice_size[1])
        if lattice_input.shape == expected_shape:
            print("[OK] Lattice shape is correct!")
        else:
            print(f"[ERROR] Expected shape {expected_shape}, got {lattice_input.shape}")
            return False
        
        print("[OK] Lattice integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Lattice integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Тест загрузки конфигурации."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from data.tokenizer import TokenizerManager
        
        # Тестовая конфигурация
        test_config = {
            'tokenizer': {
                'type': 'basic',
                'max_length': 100,
                'padding': True
            },
            'text_processing': {
                'lowercase': False,
                'remove_punctuation': True
            },
            'caching': {
                'enabled': False
            }
        }
        
        # Создание токенайзера с конфигурацией
        tokenizer = TokenizerManager(tokenizer_type='basic', config=test_config)
        
        print(f"[GEAR] Config loaded: {tokenizer.config['tokenizer']['max_length']}")
        print(f"[SAVE] Cache enabled: {tokenizer._cache_enabled}")
        
        # Тест с конфигурацией
        test_text = "Test with CUSTOM config!"
        tokens = tokenizer.encode(test_text, max_length=10)
        print(f"🔢 Tokens (max_length=10): {tokens}")
        
        print("[OK] Configuration test PASSED!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования."""
    print("[START] Starting Tokenizer Module Tests")
    print("=" * 50)
    
    tests = [
        test_basic_tokenizer,
        test_text_processor,
        test_batch_processing,
        test_lattice_integration,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"[TARGET] Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED! Tokenizer module is working correctly.")
        return True
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 