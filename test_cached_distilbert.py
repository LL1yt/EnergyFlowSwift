#!/usr/bin/env python3
"""
Тест для проверки локально кэшированной DistilBERT модели.
"""

import sys
import os
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cached_distilbert():
    """Тестирует кэшированную DistilBERT модель."""
    
    try:
        print("🧪 Testing cached DistilBERT...")
        
        # Проверяем есть ли локальная модель
        local_path = os.path.join("models", "local_cache", "distilbert-base-uncased")
        
        if not os.path.exists(local_path):
            print(f"❌ Local model not found at: {local_path}")
            return False
        
        print(f"✅ Local model found at: {local_path}")
        
        # Тестируем наш LLMHandler
        from data.embedding_loader.format_handlers import create_llm_handler
        
        print("🔄 Creating LLM handler...")
        handler = create_llm_handler("distilbert")
        
        print(f"📋 Handler model_name: {handler.model_name}")
        print(f"📋 Handler model_key: {getattr(handler, 'model_key', 'NOT SET')}")
        
        # Проверяем _get_model_path
        model_path = handler._get_model_path()
        print(f"📁 Resolved model path: {model_path}")
        
        # Тестируем генерацию эмбеддингов
        print("🔄 Testing embedding generation...")
        test_text = "This is a test sentence for DistilBERT."
        embedding = handler.generate_embeddings([test_text])
        
        print(f"✅ Embedding generated successfully!")
        print(f"📊 Embedding shape: {embedding.shape}")
        
        # Информация о модели
        model_info = handler.get_model_info()
        print(f"🧠 Model info: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция теста."""
    
    print("🎯 Testing Cached DistilBERT")
    print("=" * 40)
    
    success = test_cached_distilbert()
    
    if success:
        print("\n🎉 Test passed! Cached DistilBERT is working correctly.")
    else:
        print("\n❌ Test failed! There's an issue with the cached model.")

if __name__ == "__main__":
    main() 