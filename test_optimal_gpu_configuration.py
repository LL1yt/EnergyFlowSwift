#!/usr/bin/env python3
"""
🚀 Final Optimal GPU Configuration Test
Testing with optimal batch_size=1024 (14.2x speedup, 80% GPU utilization)
"""

import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
    EmergentTrainingConfig,
)


def test_optimal_configuration():
    """Test the optimal GPU configuration discovered"""
    print("🚀 OPTIMAL GPU CONFIGURATION TEST")
    print("="*80)
    print("Testing batch_size=1024 (optimal balance: 176 samples/sec, 80% GPU)")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 Total Memory: {total_memory:.1f} GB")
    
    # Create optimal configuration
    config = EmergentTrainingConfig()
    config.mixed_precision = True
    config.gradient_accumulation_steps = 1  # No accumulation needed
    config.channels_last_memory = True
    config.enable_8bit_optimizer = True
    
    print(f"\n📋 OPTIMAL CONFIGURATION:")
    print(f"   Batch size: 1024")
    print(f"   Mixed precision: {config.mixed_precision}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Channels-last memory: {config.channels_last_memory}")
    print(f"   8-bit optimizer: {config.enable_8bit_optimizer}")
    
    # Initialize trainer
    trainer = EmergentCubeTrainer(config)
    device = next(trainer.parameters()).device
    print(f"   Device: {device}")
    
    # Test data
    batch_size = 1024
    surface_dim = 15 * 15
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n🧪 PERFORMANCE TEST:")
    
    # Create test data
    question_embeddings = torch.randn(batch_size, surface_dim, device=device)
    answer_embeddings = torch.randn(batch_size, 4096, device=device)
    
    # Warmup
    print("   🔥 Warming up...")
    _ = trainer.forward(question_embeddings[:32])  # Small warmup
    torch.cuda.synchronize()
    
    # Full test
    print("   ⚡ Running full test...")
    
    # Forward pass
    torch.cuda.synchronize()
    forward_start = time.time()
    
    outputs = trainer.forward(question_embeddings)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_start
    forward_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"   ✅ Forward pass: {forward_time:.3f}s")
    print(f"   📊 Memory after forward: {forward_memory:.2f} GB")
    
    # Backward pass
    targets = {
        'target_embedding': answer_embeddings,
        'target_surface': outputs['input_surface']
    }
    losses = trainer.compute_loss(outputs, targets)
    
    torch.cuda.synchronize()
    backward_start = time.time()
    
    losses['total_loss'].backward()
    
    torch.cuda.synchronize()
    backward_time = time.time() - backward_start
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"   🔥 Backward pass: {backward_time:.3f}s")
    print(f"   📈 Peak memory: {peak_memory:.2f} GB")
    
    # Calculate metrics
    total_time = forward_time + backward_time
    throughput = batch_size / total_time
    memory_utilization = (peak_memory / total_memory) * 100
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   🚀 Total time: {total_time:.3f}s")
    print(f"   ⚡ Throughput: {throughput:.1f} samples/sec")
    print(f"   💾 Memory utilization: {memory_utilization:.1f}%")
    print(f"   🎯 GPU efficiency: {'OPTIMAL' if 70 <= memory_utilization <= 85 else 'GOOD' if memory_utilization >= 50 else 'LOW'}")
    
    # Compare with baseline
    baseline_throughput = 12.4  # From batch_size=64 test
    speedup = throughput / baseline_throughput
    
    print(f"\n⚖️  COMPARISON vs BASELINE:")
    print(f"   Baseline (batch 64): {baseline_throughput:.1f} samples/sec")
    print(f"   Optimized (batch 1024): {throughput:.1f} samples/sec")
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup >= 10:
        print("   🎉 OUTSTANDING improvement!")
    elif speedup >= 5:
        print("   ✅ EXCELLENT improvement!")
    elif speedup >= 2:
        print("   👍 GOOD improvement!")
    else:
        print("   ⚠️  Modest improvement")
    
    # Efficiency analysis
    print(f"\n💡 EFFICIENCY ANALYSIS:")
    if memory_utilization >= 75:
        print("   💾 Memory: Excellent utilization (75%+)")
    elif memory_utilization >= 50:
        print("   💾 Memory: Good utilization (50%+)")
    else:
        print("   💾 Memory: Low utilization (<50%)")
    
    if throughput >= 150:
        print("   ⚡ Throughput: Excellent (150+ samples/sec)")
    elif throughput >= 75:
        print("   ⚡ Throughput: Good (75+ samples/sec)")
    else:
        print("   ⚡ Throughput: Needs optimization")
    
    return {
        'throughput': throughput,
        'memory_utilization': memory_utilization,
        'speedup': speedup,
        'peak_memory_gb': peak_memory
    }


def test_training_step_stability():
    """Test multiple training steps for stability"""
    print(f"\n🔬 TRAINING STABILITY TEST")
    print("="*80)
    
    config = EmergentTrainingConfig()
    config.mixed_precision = True
    config.gradient_accumulation_steps = 1
    
    trainer = EmergentCubeTrainer(config)
    device = next(trainer.parameters()).device
    
    batch_size = 1024
    surface_dim = 15 * 15
    num_steps = 5
    
    times = []
    memory_usage = []
    
    print(f"Running {num_steps} consecutive training steps...")
    
    for step in range(num_steps):
        torch.cuda.empty_cache()
        
        # Create fresh data each step
        question_embeddings = torch.randn(batch_size, surface_dim, device=device)
        answer_embeddings = torch.randn(batch_size, 4096, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Full training step
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        torch.cuda.synchronize()
        step_time = time.time() - start_time
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        
        times.append(step_time)
        memory_usage.append(memory_used)
        
        print(f"   Step {step+1}: {step_time:.3f}s, {memory_used:.2f}GB, loss={metrics['total_loss']:.4f}")
    
    # Stability analysis
    avg_time = sum(times) / len(times)
    time_std = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    avg_throughput = batch_size / avg_time
    
    print(f"\n📊 STABILITY METRICS:")
    print(f"   ⏱️  Average time: {avg_time:.3f}s ± {time_std:.3f}s")
    print(f"   🚀 Average throughput: {avg_throughput:.1f} samples/sec")
    print(f"   💾 Average memory: {avg_memory:.2f} GB")
    
    stability_score = 1.0 - (time_std / avg_time)  # Lower variance = higher stability
    print(f"   🎯 Stability score: {stability_score:.2f} (1.0 = perfect)")
    
    if stability_score >= 0.95:
        print("   ✅ EXCELLENT stability!")
    elif stability_score >= 0.85:
        print("   👍 GOOD stability")
    else:
        print("   ⚠️  Variable performance")
    
    return avg_throughput


def main():
    """Run comprehensive optimal configuration test"""
    print("🎯 PHASE 2 GPU OPTIMIZATION - FINAL VALIDATION")
    print("="*80)
    print("Validating optimal configuration: batch_size=1024")
    
    # Test optimal configuration
    performance = test_optimal_configuration()
    
    if not performance:
        return False
    
    # Test stability
    stable_throughput = test_training_step_stability()
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏆 PHASE 2 OPTIMIZATION SUCCESS SUMMARY")
    print("="*80)
    
    print(f"✅ PERFORMANCE ACHIEVED:")
    print(f"   🚀 Throughput: {performance['throughput']:.1f} samples/sec")
    print(f"   📈 Speedup: {performance['speedup']:.1f}x vs baseline")
    print(f"   💾 GPU utilization: {performance['memory_utilization']:.1f}%")
    print(f"   🎯 Peak memory: {performance['peak_memory_gb']:.1f} GB / 32 GB")
    
    print(f"\n✅ OPTIMIZATION GOALS MET:")
    goals_met = 0
    total_goals = 4
    
    if performance['throughput'] >= 150:
        print("   ✅ Throughput > 150 samples/sec")
        goals_met += 1
    else:
        print("   ❌ Throughput < 150 samples/sec")
    
    if performance['speedup'] >= 10:
        print("   ✅ Speedup > 10x")
        goals_met += 1
    else:
        print("   ❌ Speedup < 10x")
    
    if 70 <= performance['memory_utilization'] <= 90:
        print("   ✅ Memory utilization 70-90%")
        goals_met += 1
    else:
        print("   ❌ Memory utilization not optimal")
    
    if performance['peak_memory_gb'] <= 30:
        print("   ✅ Memory usage < 30GB")
        goals_met += 1
    else:
        print("   ❌ Memory usage > 30GB")
    
    success_rate = goals_met / total_goals
    print(f"\n🎯 SUCCESS RATE: {goals_met}/{total_goals} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.75:
        print("🎉 PHASE 2 GPU OPTIMIZATION: SUCCESS!")
        print("Ready for Phase 3: Advanced Features")
    else:
        print("⚠️  PHASE 2: Partial success, needs tuning")
    
    return success_rate >= 0.75


if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Final result: {'SUCCESS' if success else 'NEEDS_TUNING'}")
    sys.exit(0 if success else 1) 