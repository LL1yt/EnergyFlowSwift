#!/usr/bin/env python3
"""
Скрипт для предварительной загрузки DistilBERT в локальную папку проекта.

Этот скрипт один раз загружает DistilBERT с HuggingFace и сохраняет 
его в папке models/local_cache для дальнейшего использования без интернета.
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def download_distilbert():
    """Загружает DistilBERT и сохраняет локально."""
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "distilbert-base-uncased"
        local_path = os.path.join("models", "local_cache", "distilbert-base-uncased")
        
        # Создаем папку если не существует
        os.makedirs(local_path, exist_ok=True)
        
        print(f"📥 Downloading DistilBERT: {model_name}")
        print(f"[SAVE] Saving to: {local_path}")
        
        # Проверяем не загружена ли модель уже
        if os.path.exists(os.path.join(local_path, "config.json")):
            print("[OK] DistilBERT already downloaded!")
            return local_path
        
        print("[REFRESH] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path)
        print("[OK] Tokenizer saved!")
        
        print("[REFRESH] Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(local_path)
        print("[OK] Model saved!")
        
        # Проверяем что все файлы на месте
        required_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(os.path.join(local_path, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"[WARNING]  Warning: Missing files: {missing_files}")
        else:
            print("[SUCCESS] DistilBERT successfully downloaded and cached!")
            print(f"[FOLDER] Location: {os.path.abspath(local_path)}")
        
        return local_path
        
    except ImportError:
        print("[ERROR] Error: transformers library not installed")
        print("Install with: pip install transformers")
        return None
        
    except Exception as e:
        print(f"[ERROR] Error downloading DistilBERT: {e}")
        return None

def test_cached_model():
    """Тестирует загруженную модель."""
    
    try:
        # Используем наш LLMHandler для тестирования
        from data.embedding_loader.format_handlers import create_llm_handler
        
        print("\n[TEST] Testing cached DistilBERT...")
        
        handler = create_llm_handler("distilbert")
        test_text = "This is a test sentence for DistilBERT."
        
        embedding = handler.generate_embeddings([test_text])
        
        print(f"[OK] Test successful!")
        print(f"[DATA] Embedding shape: {embedding.shape}")
        print(f"[BRAIN] Model info: {handler.get_model_info()}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def get_model_size():
    """Показывает размер загруженной модели."""
    
    local_path = os.path.join("models", "local_cache", "distilbert-base-uncased")
    
    if not os.path.exists(local_path):
        print("[FOLDER] Model not found locally")
        return
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            total_size += size
            file_count += 1
            
            # Показываем крупные файлы
            if size > 1024 * 1024:  # > 1MB
                print(f"  [FILE] {file}: {size / (1024*1024):.1f} MB")
    
    print(f"[DATA] Total size: {total_size / (1024*1024):.1f} MB ({file_count} files)")

def main():
    """Главная функция скрипта."""
    
    print("[TARGET] DistilBERT Local Cache Setup")
    print("=" * 40)
    
    # 1. Загружаем модель
    local_path = download_distilbert()
    
    if local_path:
        # 2. Показываем размер
        print("\n[DATA] Model Information:")
        get_model_size()
        
        # 3. Тестируем
        if test_cached_model():
            print("\n[SUCCESS] Setup completed successfully!")
            print("[IDEA] Now you can use DistilBERT without internet connection")
            print("[START] Run: python real_llama_training_production.py --distilbert")
        else:
            print("\n[WARNING]  Setup completed but test failed")
            print("[CONFIG] You may need to check your installation")
    else:
        print("\n[ERROR] Setup failed")
        print("[CONFIG] Please check your internet connection and try again")

if __name__ == "__main__":
    main() 