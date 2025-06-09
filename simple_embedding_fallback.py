#!/usr/bin/env python3
"""
Простой Fallback Embedding Loader - обходит сложную систему конфигурации
"""

import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SimpleFallbackEmbeddingLoader:
    """Простой загрузчик embeddings в обход сложной системы"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 SimpleFallbackEmbeddingLoader: {model_name} на {self.device}")
    
    def load_model(self):
        """Загрузка модели и tokenizer"""
        if self.model is None:
            print(f"📚 Loading {self.model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                raise
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Кодирование одного текста в embedding"""
        return self.encode_texts([text])[0]
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Кодирование списка текстов в embeddings"""
        self.load_model()
        
        if not texts:
            raise ValueError("Empty text list")
        
        # Добавляем padding token если нужно
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Токенизация
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Извлечение embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
            
            # Mean pooling с учетом attention mask
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        embeddings = embeddings.cpu()
        return embeddings


def create_dialogue_dataset_simple_fallback(dialogue_pairs: List[dict], 
                                           teacher_model: str = "distilbert-base-uncased",
                                           normalize_embeddings: bool = True,
                                           **kwargs):
    """
    Создание dataset с простым fallback loader
    """
    print(f"🔧 Creating dataset with SimpleFallbackEmbeddingLoader")
    
    # Создаем простой loader
    loader = SimpleFallbackEmbeddingLoader(teacher_model)
    
    # Извлекаем тексты
    questions = [pair['question'] for pair in dialogue_pairs]
    answers = [pair['answer'] for pair in dialogue_pairs]
    
    print(f"📝 Processing {len(questions)} question-answer pairs...")
    
    # Генерируем embeddings
    question_embeddings = loader.encode_texts(questions)
    answer_embeddings = loader.encode_texts(answers)
    
    print(f"📊 Generated embeddings:")
    print(f"   Questions: {question_embeddings.shape}, norm={question_embeddings.norm(dim=1).mean().item():.6f}")
    print(f"   Answers: {answer_embeddings.shape}, norm={answer_embeddings.norm(dim=1).mean().item():.6f}")
    
    # Нормализация если требуется
    if normalize_embeddings:
        question_embeddings = F.normalize(question_embeddings, p=2, dim=1)
        answer_embeddings = F.normalize(answer_embeddings, p=2, dim=1)
        print(f"✅ Embeddings normalized")
    
    return SimpleFallbackDataset(question_embeddings, answer_embeddings)


class SimpleFallbackDataset:
    """Простой dataset класс для fallback loader"""
    
    def __init__(self, question_embeddings, answer_embeddings):
        self.question_embeddings = question_embeddings
        self.answer_embeddings = answer_embeddings
    
    def __len__(self):
        return len(self.question_embeddings)
    
    def __getitem__(self, idx):
        return self.question_embeddings[idx], self.answer_embeddings[idx]


def test_simple_fallback():
    """Тест простого fallback loader"""
    print("🧪 Testing SimpleFallbackEmbeddingLoader")
    
    test_data = [
        {"question": "What is AI?", "answer": "AI is artificial intelligence."},
        {"question": "How does ML work?", "answer": "ML uses algorithms to learn patterns."}
    ]
    
    # Тестируем с разными моделями
    models_to_test = [
        "distilbert-base-uncased",
        # "roberta-base",  # раскомментировать если хотите протестировать
        # "gpt2"           # раскомментировать если хотите протестировать
    ]
    
    for model_name in models_to_test:
        print(f"\n📚 Testing {model_name}:")
        
        try:
            dataset = create_dialogue_dataset_simple_fallback(
                test_data,
                teacher_model=model_name,
                normalize_embeddings=True
            )
            
            # Проверяем первый sample
            sample = dataset[0]
            q_emb, a_emb = sample
            
            print(f"   Sample 0:")
            print(f"      Question embedding: norm={q_emb.norm().item():.6f}")
            print(f"      Answer embedding: norm={a_emb.norm().item():.6f}")
            
            if q_emb.norm().item() > 0.1 and a_emb.norm().item() > 0.1:
                print(f"      ✅ SUCCESS: Working embeddings!")
            else:
                print(f"      ❌ FAILED: Still zero embeddings")
            
        except Exception as e:
            print(f"      ❌ ERROR: {e}")


if __name__ == "__main__":
    test_simple_fallback() 