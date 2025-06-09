#!/usr/bin/env python3
"""
Тест исправления Data Pipeline - проверяем что embeddings теперь НЕ нулевые
"""

import torch
from pathlib import Path
import sys

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset, map_model_name_to_key

def test_model_name_mapping():
    """Тестирование mapping функции"""
    print("🔧 Тестирование model name mapping:")
    
    test_cases = [
        "distilbert-base-uncased",
        "distilbert", 
        "roberta-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "unknown-model"
    ]
    
    for model_name in test_cases:
        mapped_key = map_model_name_to_key(model_name)
        print(f"   '{model_name}' → '{mapped_key}'")

def test_fixed_data_pipeline():
    """Тестирование исправленного data pipeline"""
    print("\n🔍 Тестирование исправленного data pipeline:")
    
    # Тестируем с разными моделями
    test_models = [
        "distilbert-base-uncased",  # Полное имя - должно работать
        "distilbert",               # Ключ - должно работать  
        "sentence-transformers/all-MiniLM-L6-v2"  # Fallback - должно работать
    ]
    
    test_data = [
        {"question": "What is AI?", "answer": "AI is artificial intelligence."}
    ]
    
    for model_name in test_models:
        print(f"\n   📚 Testing with {model_name}:")
        
        try:
            dataset = create_dialogue_dataset(
                test_data,
                teacher_model=model_name,
                cache_embeddings=False,
                validation_split=0.0,
                normalize_embeddings=True
            )
            
            sample = dataset[0]
            if isinstance(sample, tuple):
                question_emb, answer_emb = sample
            else:
                question_emb = sample['question_embedding']
                answer_emb = sample['answer_embedding']
            
            q_norm = question_emb.norm().item()
            a_norm = answer_emb.norm().item()
            
            print(f"      Question embedding norm: {q_norm:.6f}")
            print(f"      Answer embedding norm: {a_norm:.6f}")
            
            if q_norm > 0.1 and a_norm > 0.1:
                print(f"      ✅ SUCCESS: Embeddings are non-zero!")
            elif q_norm == 0.0 or a_norm == 0.0:
                print(f"      ❌ FAILED: Still getting zero embeddings")
            else:
                print(f"      ⚠️ WARNING: Very small embeddings")
                
        except Exception as e:
            print(f"      ❌ ERROR: {e}")

def main():
    """Запуск тестов исправления"""
    print("🎯 ТЕСТ ИСПРАВЛЕНИЯ DATA PIPELINE")
    print("="*50)
    
    test_model_name_mapping()
    test_fixed_data_pipeline()
    
    print("\n" + "="*50)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    main() 