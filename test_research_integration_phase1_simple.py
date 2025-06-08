#!/usr/bin/env python3
"""
Test Research Integration Phase 1: Critical Fixes (Simple Version)
==================================================================

Упрощенный тест для проверки интеграции исследования:
- Task 1.1: Computational Graph Fix
- Task 1.2: Mixed Precision Training 
- Task 1.3: Stability validation

Без эмодзи для Windows совместимости
"""

import sys
import os
import torch
import logging

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath('.'))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, 
    EmergentTrainingConfig
)

# Простое логирование
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_computational_graph_stability():
    """Test computational graph stability после исправлений"""
    
    logger.info("=== TESTING: Computational Graph Stability ===")
    
    # Enable anomaly detection для точного поиска inplace операций
    # torch.autograd.set_detect_anomaly(True)
    
    try:
        # Create trainer с новыми настройками
        config = EmergentTrainingConfig(
            mixed_precision=True,
            gradient_checkpointing=True,
            learning_rate=0.001,
            batch_size=2  # Very small batch для testing
        )
        
        trainer = EmergentCubeTrainer(config=config, device="cpu")
        logger.info(f"Trainer created successfully")
        
        # Create test data
        batch_size = 2
        question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
        answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
        
        # Test multiple consecutive training steps (раньше падало на step 2)
        success_count = 0
        target_steps = 3  # Minimal test
        
        for step in range(target_steps):
            logger.info(f"Training Step {step + 1}/{target_steps}")
            
            # Run training step
            metrics = trainer.train_step(question_embeddings, answer_embeddings)
            
            logger.info(f"Step {step + 1} completed: loss={metrics['total_loss']:.6f}")
            
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
            logger.error(f"Other runtime error: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Disable anomaly detection после теста
        torch.autograd.set_detect_anomaly(False)

def test_mixed_precision_functionality():
    """Test mixed precision functionality"""
    
    logger.info("=== TESTING: Mixed Precision Functionality ===")
    
    try:
        # Test с mixed precision
        config_mp = EmergentTrainingConfig(
            mixed_precision=True,
            gradient_checkpointing=True,
            batch_size=1
        )
        
        trainer_mp = EmergentCubeTrainer(config=config_mp, device="cpu")
        logger.info("Mixed precision trainer created")
        
        # Test data
        batch_size = 1
        question_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
        answer_embeddings = torch.randn(batch_size, 4096, requires_grad=True)
        
        # Test mixed precision
        logger.info("Testing mixed precision mode...")
        metrics_mp = trainer_mp.train_step(question_embeddings, answer_embeddings)
        logger.info(f"Mixed precision step completed: loss={metrics_mp['total_loss']:.6f}")
        
        logger.info("Mixed precision functionality working!")
        return True
        
    except Exception as e:
        logger.error(f"Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_basic_tests():
    """Run basic Phase 1 tests"""
    
    logger.info("RESEARCH INTEGRATION PHASE 1 TESTING")
    logger.info("=" * 50)
    logger.info("Testing critical fixes:")
    logger.info("- Task 1.1: Computational Graph Fix")
    logger.info("- Task 1.2: Mixed Precision Training")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: Computational Graph Stability (CRITICAL)
    results['computational_graph'] = test_computational_graph_stability()
    
    # Test 2: Mixed Precision Functionality 
    results['mixed_precision'] = test_mixed_precision_functionality()
    
    # Summary
    logger.info("PHASE 1 TESTING RESULTS")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    logger.info("-" * 50)
    logger.info(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if results['computational_graph']:
        logger.info("CRITICAL: Computational graph stability FIXED!")
        logger.info("Ready to proceed to Phase 2: GPU Optimization")
    else:
        logger.error("CRITICAL: Computational graph issue NOT resolved")
        logger.error("Cannot proceed to Phase 2 until this is fixed")
    
    return results

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        results = run_basic_tests()
        
        # Exit code based on critical test
        if results['computational_graph']:
            logger.info("Phase 1 testing completed successfully")
            sys.exit(0)
        else:
            logger.error("Phase 1 critical test failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 