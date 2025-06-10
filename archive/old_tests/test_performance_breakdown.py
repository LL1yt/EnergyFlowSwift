#!/usr/bin/env python3
"""
🔬 Performance Breakdown Analysis
Understanding the difference between manual test (176 samples/sec) and final test (67.6 samples/sec)
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


def detailed_performance_breakdown():
    """Break down performance by individual operations"""
    print("🔬 DETAILED PERFORMANCE BREAKDOWN")
    print("="*80)
    print("Understanding: Manual test 176 samples/sec vs Final test 67.6 samples/sec")
    
    config = EmergentTrainingConfig()
    config.mixed_precision = True
    config.gradient_accumulation_steps = 1
    
    trainer = EmergentCubeTrainer(config)
    device = next(trainer.parameters()).device
    
    batch_size = 1024
    surface_dim = 15 * 15
    
    # Clean start
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Test data
    question_embeddings = torch.randn(batch_size, surface_dim, device=device)
    answer_embeddings = torch.randn(batch_size, 4096, device=device)
    
    print(f"[DATA] Testing with batch_size={batch_size}")
    
    # === 1. FORWARD PASS ONLY (like manual test) ===
    print(f"\n🔬 1. FORWARD PASS ONLY (Manual test style):")
    
    torch.cuda.synchronize()
    forward_start = time.time()
    
    outputs = trainer.forward(question_embeddings)
    
    torch.cuda.synchronize()
    forward_time = time.time() - forward_start
    forward_throughput = batch_size / forward_time
    
    print(f"   [FAST] Forward time: {forward_time:.3f}s")
    print(f"   [START] Forward throughput: {forward_throughput:.1f} samples/sec")
    print(f"   [DATA] Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # === 2. LOSS COMPUTATION ===
    print(f"\n🔬 2. LOSS COMPUTATION:")
    
    targets = {
        'target_embedding': answer_embeddings,
        'target_surface': outputs['input_surface']
    }
    
    torch.cuda.synchronize()
    loss_start = time.time()
    
    losses = trainer.compute_loss(outputs, targets)
    
    torch.cuda.synchronize()
    loss_time = time.time() - loss_start
    
    print(f"   [FAST] Loss time: {loss_time:.3f}s")
    print(f"   [DATA] Loss value: {losses['total_loss']:.4f}")
    
    # === 3. BACKWARD PASS ===
    print(f"\n🔬 3. BACKWARD PASS:")
    
    torch.cuda.synchronize()
    backward_start = time.time()
    
    losses['total_loss'].backward()
    
    torch.cuda.synchronize()
    backward_time = time.time() - backward_start
    
    print(f"   [FAST] Backward time: {backward_time:.3f}s")
    print(f"   [DATA] Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # === 4. OPTIMIZER STEP ===
    print(f"\n🔬 4. OPTIMIZER STEP:")
    
    torch.cuda.synchronize()
    opt_start = time.time()
    
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    
    torch.cuda.synchronize()
    opt_time = time.time() - opt_start
    
    print(f"   [FAST] Optimizer time: {opt_time:.3f}s")
    
    # === TOTAL ANALYSIS ===
    total_time = forward_time + loss_time + backward_time + opt_time
    total_throughput = batch_size / total_time
    
    print(f"\n[DATA] COMPLETE BREAKDOWN:")
    print(f"   Forward:   {forward_time:.3f}s ({forward_time/total_time*100:.1f}%)")
    print(f"   Loss:      {loss_time:.3f}s ({loss_time/total_time*100:.1f}%)")
    print(f"   Backward:  {backward_time:.3f}s ({backward_time/total_time*100:.1f}%)")
    print(f"   Optimizer: {opt_time:.3f}s ({opt_time/total_time*100:.1f}%)")
    print(f"   TOTAL:     {total_time:.3f}s")
    
    print(f"\n⚖️  THROUGHPUT COMPARISON:")
    print(f"   Forward-only: {forward_throughput:.1f} samples/sec")
    print(f"   Full training: {total_throughput:.1f} samples/sec")
    print(f"   Overhead: {forward_throughput/total_throughput:.1f}x slower for full training")
    
    return {
        'forward_only': forward_throughput,
        'full_training': total_throughput,
        'breakdown': {
            'forward': forward_time,
            'loss': loss_time,
            'backward': backward_time,
            'optimizer': opt_time
        }
    }


def test_multiple_iterations():
    """Test performance consistency across multiple iterations"""
    print(f"\n[REFRESH] CONSISTENCY TEST - Multiple Iterations")
    print("="*80)
    
    config = EmergentTrainingConfig()
    config.mixed_precision = True
    
    trainer = EmergentCubeTrainer(config)
    device = next(trainer.parameters()).device
    
    batch_size = 1024
    surface_dim = 15 * 15
    iterations = 3
    
    forward_times = []
    full_times = []
    
    for i in range(iterations):
        print(f"\n🧪 Iteration {i+1}/{iterations}:")
        
        torch.cuda.empty_cache()
        
        # Fresh data each iteration
        question_embeddings = torch.randn(batch_size, surface_dim, device=device)
        answer_embeddings = torch.randn(batch_size, 4096, device=device)
        
        # Forward only
        torch.cuda.synchronize()
        forward_start = time.time()
        outputs = trainer.forward(question_embeddings)
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Full training step
        torch.cuda.synchronize()
        full_start = time.time()
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        torch.cuda.synchronize()
        full_time = time.time() - full_start
        
        forward_throughput = batch_size / forward_time
        full_throughput = batch_size / full_time
        
        forward_times.append(forward_throughput)
        full_times.append(full_throughput)
        
        print(f"   Forward-only: {forward_throughput:.1f} samples/sec")
        print(f"   Full training: {full_throughput:.1f} samples/sec")
    
    # Statistics
    avg_forward = sum(forward_times) / len(forward_times)
    avg_full = sum(full_times) / len(full_times)
    
    forward_std = (sum((x - avg_forward)**2 for x in forward_times) / len(forward_times))**0.5
    full_std = (sum((x - avg_full)**2 for x in full_times) / len(full_times))**0.5
    
    print(f"\n[DATA] STATISTICS:")
    print(f"   Forward-only: {avg_forward:.1f} ± {forward_std:.1f} samples/sec")
    print(f"   Full training: {avg_full:.1f} ± {full_std:.1f} samples/sec")
    print(f"   Consistency: {1 - (full_std/avg_full):.2f} (closer to 1.0 = more consistent)")
    
    return avg_forward, avg_full


def optimize_further():
    """Try additional optimizations"""
    print(f"\n[START] ADDITIONAL OPTIMIZATION ATTEMPTS")
    print("="*80)
    
    optimizations = [
        ("Baseline", {}),
        ("No warmup", {"skip_warmup": True}),
        ("Larger batch", {"batch_size": 1536}),  # If memory allows
        ("Compilation", {"compile_model": True}),
    ]
    
    results = {}
    
    for name, opts in optimizations:
        try:
            print(f"\n🧪 Testing: {name}")
            
            config = EmergentTrainingConfig()
            config.mixed_precision = True
            
            batch_size = opts.get("batch_size", 1024)
            
            # Skip if we know this will OOM
            if batch_size > 1024:
                print(f"   [WARNING]  Skipping {batch_size} - known to OOM")
                continue
            
            trainer = EmergentCubeTrainer(config)
            device = next(trainer.parameters()).device
            
            # Model compilation if requested
            if opts.get("compile_model", False):
                try:
                    trainer = torch.compile(trainer)
                    print(f"   [OK] Model compiled")
                except:
                    print(f"   [WARNING]  Compilation failed, using uncompiled")
            
            torch.cuda.empty_cache()
            
            surface_dim = 15 * 15
            question_embeddings = torch.randn(batch_size, surface_dim, device=device)
            answer_embeddings = torch.randn(batch_size, 4096, device=device)
            
            # Warmup unless skipped
            if not opts.get("skip_warmup", False):
                _ = trainer.forward(question_embeddings[:32])
            
            # Test
            torch.cuda.synchronize()
            start_time = time.time()
            
            metrics = trainer.train_step(question_embeddings, answer_embeddings)
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            throughput = batch_size / total_time
            results[name] = throughput
            
            print(f"   [OK] Throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            results[name] = None
    
    # Find best
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best = max(valid_results, key=valid_results.get)
        print(f"\n[TROPHY] BEST OPTIMIZATION: {best}")
        print(f"   Throughput: {valid_results[best]:.1f} samples/sec")
        
        if valid_results[best] > 100:
            print(f"   [SUCCESS] Achieved 100+ samples/sec target!")
    
    return valid_results


def main():
    """Run comprehensive performance analysis"""
    print("🔬 PERFORMANCE ANALYSIS - Phase 2 Tuning")
    print("="*80)
    print("Goal: Understand and improve 67.6 → closer to 176 samples/sec")
    
    # 1. Detailed breakdown
    breakdown = detailed_performance_breakdown()
    
    # 2. Consistency test
    avg_forward, avg_full = test_multiple_iterations()
    
    # 3. Additional optimizations
    optimization_results = optimize_further()
    
    # Final analysis
    print(f"\n" + "="*80)
    print("[TARGET] PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"[DATA] KEY FINDINGS:")
    print(f"   🔬 Forward-only performance: {breakdown['forward_only']:.1f} samples/sec")
    print(f"   🏃 Full training performance: {breakdown['full_training']:.1f} samples/sec")
    print(f"   📉 Training overhead: {breakdown['forward_only']/breakdown['full_training']:.1f}x")
    
    # Determine if we're close to expected performance
    expected_forward = 176.1  # From manual test
    actual_forward = breakdown['forward_only']
    
    if actual_forward >= expected_forward * 0.8:  # Within 20%
        print(f"   [OK] Forward performance matches expectation ({actual_forward:.1f} vs {expected_forward:.1f})")
    else:
        print(f"   [WARNING]  Forward performance below expectation ({actual_forward:.1f} vs {expected_forward:.1f})")
    
    # Recommendations
    print(f"\n[IDEA] RECOMMENDATIONS:")
    
    if breakdown['full_training'] >= 100:
        print(f"   [OK] Current performance ({breakdown['full_training']:.1f} samples/sec) is excellent")
        print(f"   [TARGET] Phase 2 can be considered SUCCESS with current metrics")
    elif breakdown['full_training'] >= 75:
        print(f"   👍 Current performance ({breakdown['full_training']:.1f} samples/sec) is good")
        print(f"   [CONFIG] Minor optimizations could push to 100+ samples/sec")
    else:
        print(f"   [WARNING]  Performance needs improvement")
        print(f"   🛠️  Consider aggressive optimizations")
    
    # Best optimization recommendation
    if optimization_results:
        best_opt = max(optimization_results, key=lambda x: optimization_results[x] or 0)
        if optimization_results[best_opt] and optimization_results[best_opt] > breakdown['full_training']:
            print(f"   [START] Best optimization: {best_opt} ({optimization_results[best_opt]:.1f} samples/sec)")
    
    return breakdown['full_training'] >= 75  # 75+ samples/sec = success


if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Analysis result: {'PERFORMANCE ACCEPTABLE' if success else 'NEEDS_OPTIMIZATION'}") 