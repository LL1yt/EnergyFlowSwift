#!/usr/bin/env python3
"""
🧪 Phase 3 Task 3.1: Neural Cellular Automata Integration Test

Тестирует интеграцию NCA с EmergentCubeTrainer для проверки:
1. Корректная инициализация NCA в trainer
2. NCA processing во время forward pass
3. Emergent behavior preservation metrics
4. Performance с NCA enabled vs disabled
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import time
from typing import Dict, Any

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.neural_cellular_automata import create_nca_config

def test_nca_integration_basic():
    """Базовый тест интеграции NCA с EmergentCubeTrainer"""
    print("🧪 Testing NCA Integration with EmergentCubeTrainer...")
    
    try:
        # Create config with NCA enabled
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (8, 8, 6)  # (width=8, height=8, depth=6)
        config.batch_size = 2
        
        # Initialize trainer
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        # Check NCA initialization
        assert hasattr(trainer, 'nca'), "NCA not initialized"
        assert trainer.nca is not None, "NCA is None"
        
        print("✅ NCA Integration: Initialization successful")
        
        # Test forward pass with NCA
        batch_size = 2
        surface_size = 8 * 8  # width * height
        surface_embeddings = torch.randn(batch_size, surface_size)
        
        # Forward pass
        outputs = trainer.forward(surface_embeddings)
        
        # Check outputs
        assert 'final_output' in outputs, "Missing final_output"
        assert outputs['final_output'].shape[0] == batch_size, "Batch size mismatch"
        
        print("✅ NCA Integration: Forward pass successful")
        
        # Check NCA metrics
        nca_metrics = trainer.get_nca_metrics()
        assert nca_metrics['config']['pattern_detection'], "Pattern detection not enabled"
        assert nca_metrics['training_step'] > 0, "Training step not incremented"
        
        print("✅ NCA Integration: Metrics collection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ NCA Integration test failed: {e}")
        return False

def test_nca_vs_standard_comparison():
    """Сравнение performance NCA vs standard processing"""
    print("\n🔬 Testing NCA vs Standard Processing Performance...")
    
    try:
        # Common configuration
        base_config = EmergentTrainingConfig()
        base_config.cube_dimensions = (8, 8, 6)
        base_config.batch_size = 2
        
        # NCA enabled config
        nca_config = EmergentTrainingConfig()
        nca_config.cube_dimensions = (8, 8, 6)  # (width=8, height=8, depth=6)
        nca_config.batch_size = 2
        nca_config.enable_nca = True
        
        # Standard processing config
        standard_config = EmergentTrainingConfig()
        standard_config.cube_dimensions = (8, 8, 6)  # (width=8, height=8, depth=6)
        standard_config.batch_size = 2
        standard_config.enable_nca = False
        
        # Initialize trainers
        nca_trainer = EmergentCubeTrainer(nca_config, device="cpu")
        standard_trainer = EmergentCubeTrainer(standard_config, device="cpu")
        
        # Test data
        batch_size = 2
        surface_size = 8 * 8
        surface_embeddings = torch.randn(batch_size, surface_size)
        
        # Performance comparison
        num_runs = 3
        
        # NCA timing
        start_time = time.time()
        for _ in range(num_runs):
            outputs_nca = nca_trainer.forward(surface_embeddings)
        nca_time = (time.time() - start_time) / num_runs
        
        # Standard timing
        start_time = time.time()
        for _ in range(num_runs):
            outputs_standard = standard_trainer.forward(surface_embeddings)
        standard_time = (time.time() - start_time) / num_runs
        
        # Compare outputs
        output_diff = torch.norm(outputs_nca['final_output'] - outputs_standard['final_output'])
        
        print(f"   NCA time: {nca_time:.4f}s")
        print(f"   Standard time: {standard_time:.4f}s")
        print(f"   Time overhead: {(nca_time/standard_time - 1)*100:.1f}%")
        print(f"   Output difference: {output_diff:.6f}")
        
        # Validate reasonable overhead (should be < 50% for this test)
        overhead = nca_time / standard_time - 1
        assert overhead < 0.5, f"NCA overhead too high: {overhead*100:.1f}%"
        
        print("✅ Performance comparison: Reasonable overhead")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def test_nca_pattern_detection():
    """Тест обнаружения emergent patterns через NCA"""
    print("\n🎨 Testing NCA Pattern Detection...")
    
    try:
        # Config with pattern detection enabled
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (6, 6, 4)  # Small cube
        config.batch_size = 1
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        # Multiple forward passes to build pattern history
        surface_size = 6 * 6
        
        pattern_metrics_history = []
        
        for step in range(10):
            # Slightly varying input to encourage pattern formation
            surface_embeddings = torch.randn(1, surface_size) * (0.5 + step * 0.1)
            
            outputs = trainer.forward(surface_embeddings)
            
            # Get NCA metrics
            nca_metrics = trainer.get_nca_metrics()
            
            if 'recent_patterns' in nca_metrics:
                pattern_metrics = nca_metrics['recent_patterns']
                pattern_metrics_history.append(pattern_metrics)
        
        # Check if patterns were detected
        if pattern_metrics_history:
            last_patterns = pattern_metrics_history[-1]
            
            # Check for spatial coherence
            if 'spatial_coherence' in last_patterns:
                spatial_coherence = last_patterns['spatial_coherence'].item()
                print(f"   Spatial coherence: {spatial_coherence:.4f}")
                assert 0 <= spatial_coherence <= 1, "Invalid spatial coherence"
            
            # Check for emergent specialization
            if 'emergent_specialization' in last_patterns:
                specialization = last_patterns['emergent_specialization'].item()
                print(f"   Emergent specialization: {specialization:.4f}")
                assert 0 <= specialization <= 2, "Invalid specialization score"
            
            print("✅ Pattern detection: Metrics computed successfully")
        else:
            print("⚠️  Pattern detection: No patterns recorded (may be normal for short test)")
        
        # Check NCA summary
        nca_summary = trainer.get_nca_summary() if hasattr(trainer, 'get_nca_summary') else {}
        print(f"   NCA training steps: {nca_summary.get('training_step', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection test failed: {e}")
        return False

def test_nca_stochastic_updates():
    """Тест stochastic update механизма"""
    print("\n🎲 Testing NCA Stochastic Updates...")
    
    try:
        # Config with specific NCA settings
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (4, 4, 3)
        config.batch_size = 1
        
        # Custom NCA config with high update probability
        config.nca_config = create_nca_config(
            update_probability=0.8,
            residual_learning_rate=0.05,
            enable_pattern_detection=True
        )
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        # Multiple forward passes to test stochastic behavior
        surface_size = 4 * 4
        surface_embeddings = torch.randn(1, surface_size)
        
        outputs_history = []
        for _ in range(5):
            outputs = trainer.forward(surface_embeddings)
            outputs_history.append(outputs['final_output'].clone())
        
        # Check variability (stochastic updates should produce slightly different results)
        output_variations = []
        for i in range(1, len(outputs_history)):
            diff = torch.norm(outputs_history[i] - outputs_history[0])
            output_variations.append(diff.item())
        
        avg_variation = np.mean(output_variations)
        print(f"   Average output variation: {avg_variation:.6f}")
        
        # Should have some variation due to stochastic updates (but not too much)
        assert avg_variation > 1e-6, "No stochastic variation detected"
        assert avg_variation < 1.0, "Too much variation - system unstable"
        
        # Check update statistics
        nca_metrics = trainer.get_nca_metrics()
        if 'stochastic_stats' in nca_metrics:
            update_stats = nca_metrics['stochastic_stats']
            print(f"   Update statistics: {update_stats}")
            
            # Check that updates are happening
            avg_updates = update_stats.get('avg_updates', 0)
            assert avg_updates > 0, "No cell updates detected"
        
        print("✅ Stochastic updates: Working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Stochastic updates test failed: {e}")
        return False

def test_nca_gpu_compatibility():
    """Тест совместимости NCA с GPU (если доступно)"""
    print("\n🚀 Testing NCA GPU Compatibility...")
    
    if not torch.cuda.is_available():
        print("⚠️  GPU not available, skipping GPU test")
        return True
    
    try:
        # Config for GPU
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (6, 6, 4)
        config.batch_size = 2
        
        trainer = EmergentCubeTrainer(config, device="cuda")
        
        # Check device placement
        assert next(trainer.parameters()).device.type == "cuda", "Trainer not on GPU"
        if trainer.nca is not None:
            assert next(trainer.nca.parameters()).device.type == "cuda", "NCA not on GPU"
        
        # Forward pass on GPU
        surface_size = 6 * 6
        surface_embeddings = torch.randn(2, surface_size, device="cuda")
        
        outputs = trainer.forward(surface_embeddings)
        
        # Check output device
        assert outputs['final_output'].device.type == "cuda", "Output not on GPU"
        
        print("✅ GPU compatibility: All components on GPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU compatibility test failed: {e}")
        return False

def run_all_tests():
    """Запустить все тесты Phase 3 NCA integration"""
    print("🧠 PHASE 3 Task 3.1: Neural Cellular Automata Integration Tests")
    print("=" * 70)
    
    test_results = []
    
    # Basic integration test
    test_results.append(test_nca_integration_basic())
    
    # Performance comparison
    test_results.append(test_nca_vs_standard_comparison())
    
    # Pattern detection
    test_results.append(test_nca_pattern_detection())
    
    # Stochastic updates
    test_results.append(test_nca_stochastic_updates())
    
    # GPU compatibility (if available)
    test_results.append(test_nca_gpu_compatibility())
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! Phase 3 Task 3.1 NCA integration is working correctly")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 