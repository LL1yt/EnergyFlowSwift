#!/usr/bin/env python3
"""
🔬 Test Research Integration Phase 1: Critical Fixes
========================================================

Тест для проверки интеграции исследования:
- Task 1.1: Computational Graph Fix (strategic tensor lifecycle management)
- Task 1.2: Mixed Precision Training 
- Task 1.3: Stability validation

Цель: Решить блокирующую ошибку computational graph reuse
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath('.'))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, 
    EmergentTrainingConfig,
    create_emergent_trainer
)

# Настройка логирования (без Unicode для Windows совместимости)
logging.basicConfig(
    level=logging.INFO,  # Уменьшаем объем логов
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/research_integration_phase1.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def test_computational_graph_stability():
    """Test computational graph stability после исправлений"""
    
    logger.info("TESTING: Computational Graph Stability")
    logger.info("=" * 60)
    
    # Create trainer с новыми настройками
    config = EmergentTrainingConfig(
        mixed_precision=True,  # RESEARCH INTEGRATION: Enable mixed precision
        gradient_checkpointing=True,  # RESEARCH INTEGRATION: Enable checkpointing
        learning_rate=0.001,
        batch_size=4  # Smaller batch для testing
    )
    
    trainer = EmergentCubeTrainer(config=config, device="cpu")
    logger.info(f"Trainer created with mixed_precision={config.mixed_precision}")
    
    # Create test data
    batch_size = 4
    question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    
    logger.info(f"Test data: Q={question_embeddings.shape}, A={answer_embeddings.shape}")
    
    # Test multiple consecutive training steps (раньше падало на step 2)
    success_count = 0
    target_steps = 5  # Попробуем 5 consecutive steps
    
    try:
        for step in range(target_steps):
            logger.info(f"Training Step {step + 1}/{target_steps}")
            
            # Проверяем training_step counter
            logger.info(f"   Trainer internal step counter: {trainer.training_step}")
            
            # Run training step
            metrics = trainer.train_step(question_embeddings, answer_embeddings)
            
            logger.info(f"   Step {step + 1} completed successfully")
            logger.info(f"   Metrics: total_loss={metrics['total_loss']:.6f}, "
                       f"cos_sim={metrics['cosine_similarity']:.6f}")
            
            # Check for NaN/Inf values
            if torch.isnan(torch.tensor(metrics['total_loss'])) or torch.isinf(torch.tensor(metrics['total_loss'])):
                logger.error(f"Step {step + 1}: NaN/Inf loss detected!")
                break
            
            success_count += 1
            
        logger.info(f"RESULT: {success_count}/{target_steps} steps completed successfully")
        
        if success_count == target_steps:
            logger.info("COMPUTATIONAL GRAPH STABILITY: FIXED!")
            return True
        else:
            logger.warning(f"PARTIAL SUCCESS: {success_count}/{target_steps} steps")
            return False
            
    except RuntimeError as e:
        if "backward through the graph a second time" in str(e):
            logger.error("COMPUTATIONAL GRAPH REUSE ERROR STILL EXISTS!")
            logger.error(f"Error details: {e}")
            return False
        else:
            logger.error(f"Other error: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def test_mixed_precision_functionality():
    """Test mixed precision functionality"""
    
    logger.info("🔬 TESTING: Mixed Precision Functionality")
    logger.info("=" * 60)
    
    # Test с mixed precision
    config_mp = EmergentTrainingConfig(
        mixed_precision=True,
        gradient_checkpointing=True,
        batch_size=2
    )
    
    trainer_mp = EmergentCubeTrainer(config=config_mp, device="cpu")
    logger.info("[OK] Mixed precision trainer created")
    
    # Test без mixed precision
    config_no_mp = EmergentTrainingConfig(
        mixed_precision=False,
        gradient_checkpointing=True,
        batch_size=2
    )
    
    trainer_no_mp = EmergentCubeTrainer(config=config_no_mp, device="cpu")
    logger.info("[OK] Regular precision trainer created")
    
    # Test data
    batch_size = 2
    question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    
    try:
        # Test mixed precision
        logger.info("[REFRESH] Testing mixed precision mode...")
        metrics_mp = trainer_mp.train_step(question_embeddings, answer_embeddings)
        logger.info(f"   [OK] Mixed precision step completed: loss={metrics_mp['total_loss']:.6f}")
        
        # Test regular precision
        logger.info("[REFRESH] Testing regular precision mode...")
        metrics_no_mp = trainer_no_mp.train_step(question_embeddings, answer_embeddings)
        logger.info(f"   [OK] Regular precision step completed: loss={metrics_no_mp['total_loss']:.6f}")
        
        # Validate both modes work
        logger.info("[TARGET] RESULT: Mixed precision functionality working!")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Mixed precision test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage с новыми оптимизациями"""
    
    logger.info("🔬 TESTING: Memory Usage")
    logger.info("=" * 60)
    
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"[DATA] Baseline memory: {baseline_memory:.1f} MB")
    
    # Create trainer
    config = EmergentTrainingConfig(
        mixed_precision=True,
        gradient_checkpointing=True,
        batch_size=8
    )
    
    trainer = EmergentCubeTrainer(config=config, device="cpu")
    
    # Memory after trainer creation
    gc.collect()
    trainer_memory = process.memory_info().rss / 1024 / 1024  # MB
    trainer_usage = trainer_memory - baseline_memory
    logger.info(f"[DATA] Trainer memory: {trainer_memory:.1f} MB (+{trainer_usage:.1f} MB)")
    
    # Test data
    batch_size = 8
    question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    
    # Memory during training
    try:
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        gc.collect()
        training_memory = process.memory_info().rss / 1024 / 1024  # MB
        training_usage = training_memory - baseline_memory
        
        logger.info(f"[DATA] Training memory: {training_memory:.1f} MB (+{training_usage:.1f} MB)")
        logger.info(f"[DATA] Total overhead: {training_usage:.1f} MB")
        
        # Check if within expected range (target: <300MB based on research)
        if training_usage < 300:
            logger.info("[OK] Memory usage within target range!")
            return True
        else:
            logger.warning(f"[WARNING] Memory usage higher than expected: {training_usage:.1f} MB")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Memory test failed: {e}")
        return False

def test_gradient_checkpointing():
    """Test gradient checkpointing functionality"""
    
    logger.info("🔬 TESTING: Gradient Checkpointing")
    logger.info("=" * 60)
    
    config = EmergentTrainingConfig(
        mixed_precision=False,  # Focus on checkpointing
        gradient_checkpointing=True,
        batch_size=4
    )
    
    trainer = EmergentCubeTrainer(config=config, device="cpu")
    
    # Test data
    batch_size = 4
    question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
    
    try:
        # Run forward pass
        trainer.train()
        outputs = trainer.forward(question_embeddings)
        
        logger.info("[OK] Forward pass с gradient checkpointing completed")
        logger.info(f"[DATA] Output shapes: {[(k, v.shape) for k, v in outputs.items() if torch.is_tensor(v)]}")
        
        # Test training step
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        logger.info(f"[OK] Training step с checkpointing completed: loss={metrics['total_loss']:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Gradient checkpointing test failed: {e}")
        return False

def run_all_tests():
    """Run all Phase 1 tests"""
    
    logger.info("[START] RESEARCH INTEGRATION PHASE 1 TESTING")
    logger.info("=" * 80)
    logger.info("Testing critical fixes from research integration plan:")
    logger.info("- Task 1.1: Computational Graph Fix")
    logger.info("- Task 1.2: Mixed Precision Training") 
    logger.info("- Task 1.3: Stability Validation")
    logger.info("=" * 80)
    
    results = {}
    
    # Test 1: Computational Graph Stability (CRITICAL)
    results['computational_graph'] = test_computational_graph_stability()
    
    # Test 2: Mixed Precision Functionality 
    results['mixed_precision'] = test_mixed_precision_functionality()
    
    # Test 3: Memory Usage
    results['memory_usage'] = test_memory_usage()
    
    # Test 4: Gradient Checkpointing
    results['gradient_checkpointing'] = test_gradient_checkpointing()
    
    # Summary
    logger.info("[TARGET] PHASE 1 TESTING RESULTS")
    logger.info("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "[OK] PASS" if success else "[ERROR] FAIL"
        logger.info(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    logger.info("-" * 80)
    logger.info(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if results['computational_graph']:
        logger.info("[SUCCESS] CRITICAL: Computational graph stability FIXED!")
        logger.info("[START] Ready to proceed to Phase 2: GPU Optimization")
    else:
        logger.error("[ALERT] CRITICAL: Computational graph issue NOT resolved")
        logger.error("[ERROR] Cannot proceed to Phase 2 until this is fixed")
    
    return results

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        results = run_all_tests()
        
        # Exit code based on critical test
        if results['computational_graph']:
            logger.info("[OK] Phase 1 testing completed successfully")
            sys.exit(0)
        else:
            logger.error("[ERROR] Phase 1 critical test failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"[ERROR] Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 