#!/usr/bin/env python3
"""
Диагностика Data Pipeline - почему embeddings приходят как нулевые векторы
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

class DataPipelineDiagnostics:
    """Диагностика проблем с data pipeline"""
    
    def __init__(self):
        print(f"🔍 Data Pipeline Diagnostics")
    
    def run_diagnostics(self):
        """Запуск полной диагностики data pipeline"""
        print("\n" + "="*60)
        print("🔍 ДИАГНОСТИКА DATA PIPELINE")
        print("="*60)
        
        # 1. Тестируем разные teacher models
        self._test_teacher_models()
        
        # 2. Тестируем разные настройки
        self._test_dataset_settings()
        
        # 3. Тестируем manual embedding creation
        self._test_manual_embeddings()
        
        print("\n" + "="*60)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("="*60)
    
    def _test_teacher_models(self):
        """Тестирование разных teacher models"""
        print("\n🤖 ТЕСТИРОВАНИЕ TEACHER MODELS:")
        
        test_data = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        ]
        
        models_to_test = [
            "distilbert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        for model_name in models_to_test:
            print(f"\n   📚 Testing {model_name}:")
            
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
                
                print(f"      ✅ Успешно создан dataset")
                print(f"      Question embedding: shape={question_emb.shape}, norm={question_emb.norm().item():.6f}")
                print(f"      Answer embedding: shape={answer_emb.shape}, norm={answer_emb.norm().item():.6f}")
                
                # Проверяем содержимое
                if question_emb.norm().item() == 0.0:
                    print(f"      🚨 ПРОБЛЕМА: Question embedding = нулевой вектор!")
                    print(f"         Первые 10 элементов: {question_emb[:10]}")
                    print(f"         Все ли элементы нули? {torch.all(question_emb == 0).item()}")
                else:
                    print(f"      ✅ Question embedding выглядит нормально")
                    print(f"         Первые 5 элементов: {question_emb[:5]}")
                
                if answer_emb.norm().item() == 0.0:
                    print(f"      🚨 ПРОБЛЕМА: Answer embedding = нулевой вектор!")
                    print(f"         Первые 10 элементов: {answer_emb[:10]}")
                    print(f"         Все ли элементы нули? {torch.all(answer_emb == 0).item()}")
                else:
                    print(f"      ✅ Answer embedding выглядит нормально")
                    print(f"         Первые 5 элементов: {answer_emb[:5]}")
                
            except Exception as e:
                print(f"      ❌ Ошибка с {model_name}: {e}")
    
    def _test_dataset_settings(self):
        """Тестирование разных настроек dataset"""
        print("\n⚙️ ТЕСТИРОВАНИЕ НАСТРОЕК DATASET:")
        
        test_data = [
            {"question": "What is machine learning?", "answer": "ML is a subset of AI."},
            {"question": "How do neural networks work?", "answer": "They process data through layers."}
        ]
        
        settings_to_test = [
            {"normalize_embeddings": True, "cache_embeddings": False},
            {"normalize_embeddings": False, "cache_embeddings": False},
            {"normalize_embeddings": True, "cache_embeddings": True},
            {"normalize_embeddings": False, "cache_embeddings": True}
        ]
        
        for i, settings in enumerate(settings_to_test):
            print(f"\n   ⚙️ Settings {i+1}: {settings}")
            
            try:
                dataset = create_dialogue_dataset(
                    test_data,
                    teacher_model="distilbert-base-uncased",
                    validation_split=0.0,
                    **settings
                )
                
                print(f"      Dataset size: {len(dataset)}")
                
                # Проверяем несколько samples
                for j in range(min(2, len(dataset))):
                    sample = dataset[j]
                    if isinstance(sample, tuple):
                        question_emb, answer_emb = sample
                    else:
                        question_emb = sample['question_embedding']
                        answer_emb = sample['answer_embedding']
                    
                    q_norm = question_emb.norm().item()
                    a_norm = answer_emb.norm().item()
                    
                    print(f"      Sample {j}: Q_norm={q_norm:.6f}, A_norm={a_norm:.6f}")
                    
                    if q_norm == 0.0 or a_norm == 0.0:
                        print(f"         🚨 Нулевые embeddings в sample {j}!")
                        if q_norm == 0.0:
                            print(f"            Question text: '{test_data[j]['question']}'")
                        if a_norm == 0.0:
                            print(f"            Answer text: '{test_data[j]['answer']}'")
                
            except Exception as e:
                print(f"      ❌ Ошибка с settings {settings}: {e}")
    
    def _test_manual_embeddings(self):
        """Тестирование manual создания embeddings"""
        print("\n🔧 ТЕСТИРОВАНИЕ MANUAL EMBEDDINGS:")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch.nn.functional as F
            
            # Загружаем модель напрямую
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            print(f"   📚 Manual loading {model_name}:")
            print(f"      Model loaded: {type(model)}")
            print(f"      Tokenizer loaded: {type(tokenizer)}")
            
            # Тестируем на простом тексте
            test_texts = [
                "What is AI?",
                "AI is artificial intelligence.",
                "This is a test sentence.",
                ""  # Пустая строка
            ]
            
            for text in test_texts:
                print(f"\n      Testing text: '{text}'")
                
                if not text.strip():
                    print(f"         ⚠️ Пустой текст - может вызвать проблемы")
                    continue
                
                # Токенизация
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                print(f"         Tokens: {inputs['input_ids'].shape}")
                
                # Получаем embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state
                    
                    # Pooling (mean)
                    pooled_embedding = embeddings.mean(dim=1).squeeze()
                    
                print(f"         Raw embedding shape: {embeddings.shape}")
                print(f"         Pooled embedding shape: {pooled_embedding.shape}")
                print(f"         Pooled embedding norm: {pooled_embedding.norm().item():.6f}")
                
                if pooled_embedding.norm().item() == 0.0:
                    print(f"         🚨 ПРОБЛЕМА: Manual embedding тоже нулевой!")
                    print(f"            Первые 10 значений: {pooled_embedding[:10]}")
                    print(f"            Все нули? {torch.all(pooled_embedding == 0).item()}")
                else:
                    print(f"         ✅ Manual embedding выглядит нормально")
                    print(f"            Первые 5 значений: {pooled_embedding[:5]}")
                    print(f"            Min: {pooled_embedding.min().item():.6f}")
                    print(f"            Max: {pooled_embedding.max().item():.6f}")
                    print(f"            Mean: {pooled_embedding.mean().item():.6f}")
                    print(f"            Std: {pooled_embedding.std().item():.6f}")
                
        except Exception as e:
            print(f"      ❌ Ошибка manual embedding: {e}")
    
    def _test_dialogue_dataset_internals(self):
        """Глубокое тестирование внутренностей dialogue_dataset"""
        print("\n🔬 ГЛУБОКОЕ ТЕСТИРОВАНИЕ DIALOGUE_DATASET:")
        
        # Импортируем внутренние компоненты
        try:
            from training.embedding_trainer.dialogue_dataset import DialogueDataset
            from training.embedding_trainer.embedding_loader import EmbeddingLoader
            
            # Создаем embedding loader напрямую
            loader = EmbeddingLoader("distilbert-base-uncased")
            
            print(f"   📚 EmbeddingLoader создан:")
            print(f"      Model: {type(loader.model)}")
            print(f"      Tokenizer: {type(loader.tokenizer)}")
            print(f"      Device: {loader.device}")
            
            # Тестируем loader напрямую
            test_texts = ["What is AI?", "AI is artificial intelligence."]
            
            for text in test_texts:
                embedding = loader.encode_text(text)
                print(f"      Text: '{text}'")
                print(f"         Embedding shape: {embedding.shape}")
                print(f"         Embedding norm: {embedding.norm().item():.6f}")
                
                if embedding.norm().item() == 0.0:
                    print(f"         🚨 Loader возвращает нулевой embedding!")
                else:
                    print(f"         ✅ Loader работает правильно")
            
        except Exception as e:
            print(f"   ❌ Ошибка тестирования internals: {e}")

def main():
    """Запуск диагностики data pipeline"""
    diagnostics = DataPipelineDiagnostics()
    diagnostics.run_diagnostics()
    
    print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Если все teacher models дают нулевые embeddings - проблема в окружении")
    print("2. Если только некоторые - проблема в конкретной модели")
    print("3. Если manual embeddings работают - проблема в dialogue_dataset коде")

if __name__ == "__main__":
    main() 