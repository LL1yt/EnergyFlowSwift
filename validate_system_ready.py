#!/usr/bin/env python3
"""
🎯 System Readiness Validation
Быстрая проверка готовности к реальному обучению

ПРОВЕРЯЕТ:
1. LLaMA-3-8B доступность и стабильность
2. 3D Cube processing работоспособность  
3. End-to-end pipeline функциональность
4. Mini training для подтверждения learning capability
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llama_availability():
    """Test LLaMA-3-8B basic functionality"""
    logger.info("🔍 Testing LLaMA-3-8B availability...")
    
    try:
        from utils.llm_handler import LLMHandler
        
        llm = LLMHandler('llama3-8b-local')
        embedding = llm.generate_embedding("Test text for validation")
        
        if embedding.shape[-1] == 4096:
            logger.info(f"✅ LLaMA-3-8B working: {embedding.shape}")
            return True
        else:
            logger.error(f"❌ Wrong embedding dimension: {embedding.shape}")
            return False
            
    except Exception as e:
        logger.error(f"❌ LLaMA-3-8B test failed: {e}")
        return False

def test_cube_processing():
    """Test 3D Cube basic processing"""
    logger.info("🎲 Testing 3D Cube processing...")
    
    try:
        from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
            EmergentCubeTrainer, EmergentTrainingConfig
        )
        
        config = EmergentTrainingConfig()
        config.cube_dimensions = (8, 8, 3)  # Smaller для validation
        config.enable_nca = True
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = EmergentCubeTrainer(config, device=device)
        
        # Test processing
        test_input = torch.randn(1, 8, 8, device=trainer.device)
        test_target = torch.randn(1, 8, 8, device=trainer.device)
        
        metrics = trainer.train_step(test_input, test_target)
        
        if metrics and 'loss' in metrics:
            logger.info(f"✅ Cube processing working: loss = {metrics['loss']:.4f}")
            return True
        else:
            logger.error("❌ Cube processing failed - no valid metrics")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cube processing test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete pipeline"""
    logger.info("🔄 Testing end-to-end pipeline...")
    
    try:
        from utils.llm_handler import LLMHandler
        from core.universal_adapter.universal_embedding_adapter import UniversalEmbeddingAdapter
        
        # Text → LLaMA
        llm = LLMHandler('llama3-8b-local')
        embedding = llm.generate_embedding("What is AI?")
        
        # LLaMA → Adapter → Surface
        adapter = UniversalEmbeddingAdapter(
            input_dim=4096,
            output_shape=(15, 15),
            strategy='hierarchical'
        )
        surface = adapter.forward(embedding.unsqueeze(0))
        
        logger.info(f"✅ Pipeline working: {embedding.shape} → {surface.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ End-to-end pipeline test failed: {e}")
        return False

def test_mini_training():
    """Test mini training session"""
    logger.info("🚀 Testing mini training session...")
    
    try:
        from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
        from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
            EmergentCubeTrainer, EmergentTrainingConfig
        )
        from core.universal_adapter.universal_embedding_adapter import UniversalEmbeddingAdapter
        import numpy as np
        
        # Mini dataset
        pairs = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."},
            {"question": "What is ML?", "answer": "ML is machine learning."}
        ]
        
        dataset = create_dialogue_dataset(
            pairs, teacher_model="llama3-8b-local", 
            cache_embeddings=False, validation_split=0.0
        )
        
        # Mini training
        config = EmergentTrainingConfig()
        config.cube_dimensions = (8, 8, 3)
        config.epochs = 3
        config.batch_size = 1
        config.learning_rate = 0.001
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = EmergentCubeTrainer(config, device=device)
        
        adapter = UniversalEmbeddingAdapter(4096, (8, 8), 'hierarchical')
        
        losses = []
        for epoch in range(3):
            sample = dataset[0]
            if isinstance(sample, tuple):
                q_emb, a_emb = sample
                q_surface = adapter.forward(q_emb.unsqueeze(0))
                a_surface = adapter.forward(a_emb.unsqueeze(0))
                
                metrics = trainer.train_step(q_surface, a_surface)
                losses.append(metrics.get('loss', 0.0))
        
        if len(losses) >= 2:
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            logger.info(f"✅ Mini training working: {improvement:.1%} improvement")
            return improvement > 0.05  # 5% minimum improvement
        else:
            logger.error("❌ Mini training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Mini training test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    logger.info("=" * 60)
    logger.info("🎯 SYSTEM READINESS VALIDATION")
    logger.info("=" * 60)
    
    tests = [
        ("LLaMA-3-8B Availability", test_llama_availability),
        ("3D Cube Processing", test_cube_processing), 
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Mini Training", test_mini_training)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        success = test_func()
        duration = time.time() - start_time
        
        results[test_name] = {'success': success, 'duration': duration}
        
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} ({duration:.1f}s)")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        logger.info(f"{status} {test_name}: {result['duration']:.1f}s")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 SYSTEM READY FOR REAL TRAINING!")
        logger.info("🚀 You can now run: python real_llama_training_production.py")
        return True
    else:
        logger.error("❌ System not ready - fix failing tests first")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 