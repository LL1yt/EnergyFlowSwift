#!/usr/bin/env python3
"""
Упрощенный тест Dialogue Training для отладки

Простой тест для проверки основных компонентов dialogue training
без сложной архитектуры.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict

# Импорты компонентов
from training.embedding_trainer.cube_trainer import CubeTrainer, TrainingConfig
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_dialogue_data() -> List[Dict]:
    """Создает простые диалоговые данные"""
    return [
        {"question": "What is AI?", "answer": "AI is artificial intelligence."},
        {"question": "How are you?", "answer": "I am doing well, thank you."},
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "Hello there", "answer": "Hello! How can I help you today?"},
        {"question": "Good morning", "answer": "Good morning! Hope you have a great day."},
    ]


def test_components():
    """Тест отдельных компонентов"""
    logger.info("=== TESTING COMPONENTS ===")
    
    # 1. Тест создания dataset
    logger.info("1. Testing DialogueDataset creation...")
    dialogue_pairs = create_simple_dialogue_data()
    
    try:
        dataset = create_dialogue_dataset(
            dialogue_pairs=dialogue_pairs,
            teacher_model="distilbert",
            validation_split=0.4,  # Больше для validation при малом датасете
            embedding_dim=768,
            enable_quality_filter=False,  # Отключаем фильтрацию для простоты
            use_cache=True
        )
        logger.info(f"[OK] Dataset created: {len(dataset)} pairs")
        
        # Тест DataLoader
        dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)
        for i, (q_emb, a_emb) in enumerate(dataloader):
            logger.info(f"   Batch {i}: Q={q_emb.shape}, A={a_emb.shape}")
            if i >= 1:  # Только первые 2 batch'а
                break
                
    except Exception as e:
        logger.error(f"[ERROR] Dataset creation failed: {e}")
        return False
    
    # 2. Тест создания CubeTrainer
    logger.info("2. Testing CubeTrainer creation...")
    
    try:
        config = TrainingConfig(
            mode="dialogue",
            device="cpu",
            lattice_size=[8, 8, 12],  # 768D совместимость
            embedding_dim=768,
            batch_size=2,
            learning_rate=0.01,  # Выше для быстрого обучения
            epochs=3,
            target_similarity=0.70  # Пониженная цель для теста
        )
        
        trainer = CubeTrainer(config=config)
        trainer.initialize_components()
        logger.info("[OK] CubeTrainer created and initialized")
        
    except Exception as e:
        logger.error(f"[ERROR] CubeTrainer creation failed: {e}")
        return False
    
    # 3. Тест forward pass
    logger.info("3. Testing forward pass...")
    
    try:
        # Получаем один эмбединг из dataset
        sample_q, sample_a = next(iter(dataloader))
        
        # Тестируем forward для одного эмбединга
        single_q = sample_q[0]  # [768]
        logger.info(f"   Input shape: {single_q.shape}")
        
        with torch.no_grad():  # Без градиентов для теста
            output = trainer.forward(single_q)
            logger.info(f"   Output shape: {output.shape}")
            
            # Проверка качества
            similarity = torch.nn.functional.cosine_similarity(
                single_q.unsqueeze(0), output.unsqueeze(0), dim=1
            ).item()
            logger.info(f"   Identity similarity: {similarity:.4f}")
            
        logger.info("[OK] Forward pass successful")
        
    except Exception as e:
        logger.error(f"[ERROR] Forward pass failed: {e}")
        return False
    
    return True


def test_simple_training():
    """Упрощенный тест обучения"""
    logger.info("=== TESTING SIMPLE TRAINING ===")
    
    # Создание данных
    dialogue_pairs = create_simple_dialogue_data()
    dataset = create_dialogue_dataset(
        dialogue_pairs=dialogue_pairs,
        teacher_model="distilbert",
        validation_split=0.4,
        embedding_dim=768,
        enable_quality_filter=False,
        use_cache=True
    )
    
    # Создание trainer
    config = TrainingConfig(
        mode="dialogue",
        device="cpu",
        lattice_size=[8, 8, 12],
        embedding_dim=768,
        batch_size=1,  # Очень маленький batch
        learning_rate=0.01,
        epochs=2,
        target_similarity=0.60
    )
    
    trainer = CubeTrainer(config=config)
    trainer.initialize_components()
    
    # Получение данных
    dataloader = dataset.get_dataloader(batch_size=1, shuffle=False, validation=False)
    
    logger.info("Starting simplified training...")
    
    try:
        for epoch in range(2):
            logger.info(f"Epoch {epoch + 1}/2")
            
            for batch_idx, (question_emb, answer_emb) in enumerate(dataloader):
                if batch_idx >= 2:  # Только 2 batch'а для теста
                    break
                
                # Получаем одиночные эмбединги
                q_emb = question_emb[0]  # [768]
                a_emb = answer_emb[0]    # [768]
                
                logger.info(f"   Batch {batch_idx}: Q={q_emb.shape}, A={a_emb.shape}")
                
                # Проверяем что эмбединги не NaN
                if torch.isnan(q_emb).any() or torch.isnan(a_emb).any():
                    logger.error("[ERROR] NaN detected in embeddings")
                    return False
                
                # Простой forward без градиентов
                with torch.no_grad():
                    processed = trainer.forward(q_emb)
                    similarity = torch.nn.functional.cosine_similarity(
                        processed.unsqueeze(0), a_emb.unsqueeze(0), dim=1
                    ).item()
                    logger.info(f"      Similarity: {similarity:.4f}")
        
        logger.info("[OK] Simplified training completed")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Simplified training failed: {e}")
        return False


def main():
    """Main function"""
    logger.info("SIMPLE DIALOGUE TRAINING TEST")
    logger.info("=" * 50)
    
    # Тест компонентов
    if not test_components():
        logger.error("[ERROR] Component tests failed")
        return 1
    
    # Тест простого обучения
    if not test_simple_training():
        logger.error("[ERROR] Simple training test failed")
        return 1
    
    logger.info("[OK] ALL TESTS PASSED!")
    logger.info("[START] Components are ready for full dialogue training")
    return 0


if __name__ == "__main__":
    exit(main()) 