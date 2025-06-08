#!/usr/bin/env python3
"""
Быстрый тест для проверки доступности локальной Meta-LLaMA-3-8B модели
"""

import os
import sys
from pathlib import Path

# Добавляем текущую директорию в Python path
sys.path.append(str(Path(__file__).parent))

from data.embedding_loader.format_handlers import SUPPORTED_LLM_MODELS, create_llm_handler

def test_llama_model():
    """Тест доступности локальной LLaMA модели"""
    
    print("🦙 Testing Local LLaMA-3-8B Model Availability")
    print("=" * 50)
    
    # 1. Проверяем поддерживаемые модели
    print("📋 Supported LLM Models:")
    for key, path in SUPPORTED_LLM_MODELS.items():
        print(f"   {key}: {path}")
    
    # 2. Проверяем путь к локальной модели
    local_path = r"C:\Users\n0n4a\Meta-Llama-3-8B"
    print(f"\n📁 Checking local path: {local_path}")
    
    if os.path.exists(local_path):
        print("✅ Local model directory exists")
        
        # Проверяем содержимое
        contents = list(os.listdir(local_path))
        print(f"📂 Directory contents ({len(contents)} items):")
        for item in contents[:10]:  # Показываем первые 10 элементов
            print(f"   - {item}")
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more items")
        
        # Проверяем ключевые файлы модели
        key_files = ['config.json', 'tokenizer.json', 'pytorch_model.bin', 'model.safetensors']
        for key_file in key_files:
            if key_file in contents:
                print(f"✅ Found {key_file}")
            else:
                print(f"❌ Missing {key_file}")
    else:
        print("❌ Local model directory does not exist")
        return False
    
    # 3. Пробуем создать LLM handler
    print(f"\n🧠 Testing LLM Handler creation:")
    
    try:
        print("   Creating handler for 'llama3-8b-local'...")
        handler = create_llm_handler("llama3-8b-local")
        print("✅ Handler created successfully")
        
        # Проверяем инициализацию модели (без загрузки)
        print(f"   Model name: {handler.model_name}")
        print(f"   Device: {handler._device}")
        
        # Пробуем сгенерировать тестовый эмбединг
        print("   Generating test embedding...")
        test_embedding = handler.generate_embeddings(["Hello world test"])
        print(f"✅ Test embedding generated: {test_embedding.shape}")
        print(f"   Dtype: {test_embedding.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating LLM handler: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llama_model()
    
    if success:
        print("\n🎉 Local LLaMA-3-8B model is accessible!")
    else:
        print("\n🔧 Local LLaMA-3-8B model needs configuration fixes") 