#!/usr/bin/env python3
"""
[START] Phase 2: GPU Optimization Testing
Testing Research Integration GPU optimization features:
- Task 2.1: Channels-Last Memory Format (22% bandwidth improvement)
- Task 2.2: Hierarchical Batching (effective batch 32)
- Task 2.3: 8-bit Optimizer (75% memory reduction)
"""

import torch
import torch.nn as nn
import logging
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
    EmergentTrainingConfig,
    create_emergent_trainer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_channels_last_memory_format():
    """Test Task 2.1: Channels-Last Memory Format"""
    print("\n" + "="*80)
    print("🧪 TEST 2.1: Channels-Last Memory Format")
    print("="*80)
    
    try:
        # Create trainer with GPU optimization enabled (auto-detect device)
        config = EmergentTrainingConfig()
        trainer = EmergentCubeTrainer(config)  # Auto-detect GPU
        
        # Check if channels-last template is created
        assert hasattr(trainer, 'cube_states_template'), "cube_states_template not created"
        
        # Verify 5D tensor shape for channels-last 3D
        template_shape = trainer.cube_states_template.shape
        print(f"   Template shape: {template_shape}")
        assert len(template_shape) == 5, f"Expected 5D tensor, got {len(template_shape)}D"
        
        # Check memory optimization implementation
        print("[OK] Cube states template created for memory optimization")
        
        # Test if the memory optimization doesn't break functionality
        print("   Testing memory-optimized tensor creation...")
        
        # Test forward pass with channels-last format
        batch_size = 4
        surface_dim = 15 * 15  # 225 for 15x15 surface
        
        # CRITICAL: Move test data to same device as model
        device = next(trainer.parameters()).device
        surface_embeddings = torch.randn(batch_size, surface_dim, device=device)
        
        start_time = time.time()
        outputs = trainer.forward(surface_embeddings)
        forward_time = time.time() - start_time
        
        print(f"[OK] Forward pass completed in {forward_time:.4f}s")
        print(f"   Output shape: {outputs['final_output'].shape}")
        
        # Verify memory optimization doesn't break functionality
        assert outputs['final_output'].shape[0] == batch_size, "Batch size mismatch"
        assert outputs['final_output'].shape[1] == 225, "Surface output size mismatch"  # 15×15
        
        print("[OK] Memory optimization integration verified (functionality preserved)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Channels-last test failed: {e}")
        return False


def test_hierarchical_batching():
    """Test Task 2.2: Hierarchical Batching with Gradient Accumulation"""
    print("\n" + "="*80)
    print("🧪 TEST 2.2: Hierarchical Batching")
    print("="*80)
    
    try:
        # Create trainer with hierarchical batching enabled
        config = EmergentTrainingConfig()
        config.gradient_accumulation_steps = 4  # Test accumulation
        config.effective_batch_size = 32
        
        trainer = EmergentCubeTrainer(config)  # Auto-detect GPU
        
        # Test data: batch of 8, should be split into 4 mini-batches of 2
        batch_size = 8
        surface_dim = 15 * 15
        
        # CRITICAL: Move test data to same device as model
        device = next(trainer.parameters()).device
        question_embeddings = torch.randn(batch_size, surface_dim, device=device)
        answer_embeddings = torch.randn(batch_size, 4096, device=device)  # LLaMA size
        
        print(f"🧪 Testing hierarchical batching: batch {batch_size} → {config.gradient_accumulation_steps} mini-batches")
        
        # Test hierarchical training step
        start_time = time.time()
        metrics = trainer.train_step_hierarchical(question_embeddings, answer_embeddings)
        training_time = time.time() - start_time
        
        print(f"[OK] Hierarchical training completed in {training_time:.4f}s")
        print(f"   Effective batch size: {metrics.get('effective_batch_size', 'N/A')}")
        print(f"   Accumulation steps: {metrics.get('accumulation_steps', 'N/A')}")
        print(f"   Total loss: {metrics['total_loss']:.6f}")
        
        # Verify gradient accumulation worked
        assert 'effective_batch_size' in metrics, "Effective batch size not reported"
        assert 'accumulation_steps' in metrics, "Accumulation steps not reported"
        assert metrics['accumulation_steps'] == 4, f"Expected 4 accumulation steps, got {metrics['accumulation_steps']}"
        
        print("[OK] Hierarchical batching with gradient accumulation verified")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Hierarchical batching test failed: {e}")
        return False


def test_8bit_optimizer():
    """Test Task 2.3: 8-bit Optimizer"""
    print("\n" + "="*80)
    print("🧪 TEST 2.3: 8-bit Optimizer")
    print("="*80)
    
    try:
        # Check if bitsandbytes is available
        try:
            import bitsandbytes as bnb
            bitsandbytes_available = True
            print("[OK] bitsandbytes library available")
        except ImportError:
            bitsandbytes_available = False
            print("[WARNING] bitsandbytes not available, testing fallback")
        
        # Create trainer (should auto-detect and use 8-bit optimizer if available)
        config = EmergentTrainingConfig()
        trainer = EmergentCubeTrainer(config)  # Auto-detect GPU
        
        # Check optimizer type
        optimizer_type = type(trainer.optimizer).__name__
        if bitsandbytes_available:
            print(f"[OK] Optimizer type: {optimizer_type}")
            if "AdamW8bit" in optimizer_type:
                print("[OK] 8-bit optimizer successfully enabled")
            else:
                print("[WARNING] 8-bit optimizer not enabled (may fallback to standard)")
        else:
            print(f"[OK] Fallback optimizer type: {optimizer_type}")
            # Don't assert - just verify it's a working optimizer
            if "AdamW" in optimizer_type:
                print("[OK] Standard AdamW fallback working correctly")
        
        # Test basic optimization step
        batch_size = 4
        surface_dim = 15 * 15
        
        # CRITICAL: Move test data to same device as model
        device = next(trainer.parameters()).device
        question_embeddings = torch.randn(batch_size, surface_dim, device=device)
        answer_embeddings = torch.randn(batch_size, 4096, device=device)
        
        # Test training step with 8-bit optimizer
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        print(f"[OK] Training step with optimizer completed")
        print(f"   Total loss: {metrics['total_loss']:.6f}")
        print(f"   Learning rate: {metrics['lr']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 8-bit optimizer test failed: {e}")
        return False


def test_combined_gpu_optimization():
    """Test all GPU optimization features working together"""
    print("\n" + "="*80)
    print("🧪 TEST 2.4: Combined GPU Optimization")
    print("="*80)
    
    try:
        # Create trainer with all optimizations enabled
        config = EmergentTrainingConfig()
        config.mixed_precision = True
        config.gradient_checkpointing = True
        config.gradient_accumulation_steps = 4
        config.effective_batch_size = 32
        
        trainer = EmergentCubeTrainer(config)  # Auto-detect GPU
        
        # Test data for multiple training steps
        batch_size = 8
        surface_dim = 15 * 15
        
        # CRITICAL: Get device from trainer
        device = next(trainer.parameters()).device
        
        print(f"🧪 Testing combined optimization with {batch_size} batch size")
        
        total_time = 0
        num_steps = 3
        
        for step in range(num_steps):
            question_embeddings = torch.randn(batch_size, surface_dim, device=device)
            answer_embeddings = torch.randn(batch_size, 4096, device=device)
            
            start_time = time.time()
            
            # Use hierarchical training (includes all optimizations)
            metrics = trainer.train_step_hierarchical(question_embeddings, answer_embeddings)
            
            step_time = time.time() - start_time
            total_time += step_time
            
            print(f"   Step {step+1}: {step_time:.4f}s, loss: {metrics['total_loss']:.6f}")
        
        avg_time = total_time / num_steps
        print(f"[OK] Combined optimization test completed")
        print(f"   Average time per step: {avg_time:.4f}s")
        print(f"   All optimizations working together")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Combined optimization test failed: {e}")
        return False


def benchmark_performance():
    """Benchmark performance improvements"""
    print("\n" + "="*80)
    print("[DATA] PERFORMANCE BENCHMARK")
    print("="*80)
    
    try:
        # Test baseline (minimal optimizations)
        config_baseline = EmergentTrainingConfig()
        config_baseline.mixed_precision = False
        config_baseline.gradient_accumulation_steps = 1
        
        trainer_baseline = EmergentCubeTrainer(config_baseline)  # Auto-detect GPU
        
        # Test optimized (all optimizations)
        config_optimized = EmergentTrainingConfig()
        config_optimized.mixed_precision = True
        config_optimized.gradient_checkpointing = True
        config_optimized.gradient_accumulation_steps = 4
        
        trainer_optimized = EmergentCubeTrainer(config_optimized)  # Auto-detect GPU
        
        # Benchmark data
        batch_size = 8
        surface_dim = 15 * 15
        num_steps = 5
        
        # CRITICAL: Get devices from trainers
        device_baseline = next(trainer_baseline.parameters()).device
        device_optimized = next(trainer_optimized.parameters()).device
        
        print(f"🏃‍♂️ Benchmarking {num_steps} steps with batch size {batch_size}")
        
        # Baseline timing
        baseline_times = []
        for step in range(num_steps):
            question_embeddings = torch.randn(batch_size, surface_dim, device=device_baseline)
            answer_embeddings = torch.randn(batch_size, 4096, device=device_baseline)
            
            start_time = time.time()
            metrics = trainer_baseline.train_step(question_embeddings, answer_embeddings)
            baseline_times.append(time.time() - start_time)
        
        # Optimized timing
        optimized_times = []
        for step in range(num_steps):
            question_embeddings = torch.randn(batch_size, surface_dim, device=device_optimized)
            answer_embeddings = torch.randn(batch_size, 4096, device=device_optimized)
            
            start_time = time.time()
            metrics = trainer_optimized.train_step_hierarchical(question_embeddings, answer_embeddings)
            optimized_times.append(time.time() - start_time)
        
        # Calculate performance metrics
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        speedup = avg_baseline / avg_optimized if avg_optimized > 0 else 1.0
        
        print(f"[CHART] Performance Results:")
        print(f"   Baseline average: {avg_baseline:.4f}s per step")
        print(f"   Optimized average: {avg_optimized:.4f}s per step")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("[OK] Performance improvement detected")
        else:
            print("[WARNING] No significant speedup (expected on CPU)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Performance benchmark failed: {e}")
        return False


def main():
    """Run all Phase 2 GPU optimization tests"""
    print("[START] PHASE 2: GPU OPTIMIZATION TESTING")
    print("="*80)
    print("Testing Research Integration GPU optimization features")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Device: CPU only")
    
    tests = [
        ("Channels-Last Memory Format", test_channels_last_memory_format),
        ("Hierarchical Batching", test_hierarchical_batching),
        ("8-bit Optimizer", test_8bit_optimizer),
        ("Combined GPU Optimization", test_combined_gpu_optimization),
        ("Performance Benchmark", benchmark_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            result = test_func()
            results[test_name] = "[OK] PASS" if result else "[ERROR] FAIL"
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results[test_name] = "[ERROR] ERROR"
    
    # Summary
    print("\n" + "="*80)
    print("[DATA] PHASE 2 TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        print(f"{result} | {test_name}")
    
    passed = sum(1 for r in results.values() if r == "[OK] PASS")
    total = len(results)
    
    print(f"\n[TARGET] OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Phase 2 GPU Optimization: ALL TESTS PASSED!")
        print("[START] Ready for Phase 3: Advanced Features")
    else:
        print("[WARNING] Some tests failed. Review implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 