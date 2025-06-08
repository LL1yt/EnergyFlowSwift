#!/usr/bin/env python3
"""
🔍 GPU Performance Analysis - Phase 2 Optimization
Detailed analysis of GPU performance issues and optimization strategies
"""

import torch
import torch.nn as nn
import logging
import time
import sys
import os
import gc

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
    EmergentTrainingConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_gpu_utilization():
    """Analyze GPU utilization and identify bottlenecks"""
    print("\n" + "="*80)
    print("🔍 GPU UTILIZATION ANALYSIS")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping GPU analysis")
        return False
    
    device = torch.device("cuda")
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Memory baseline
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.memory_allocated()
    print(f"📈 Baseline GPU memory: {baseline_memory / 1024**2:.1f} MB")
    
    return True


def test_batch_size_impact():
    """Test impact of different batch sizes on GPU performance"""
    print("\n" + "="*80)
    print("🧪 BATCH SIZE IMPACT ANALYSIS")
    print("="*80)
    
    batch_sizes = [8, 16, 32, 64, 1024]  # Test various batch sizes
    surface_dim = 15 * 15
    results = {}
    
    for batch_size in batch_sizes:
        try:
            print(f"\n🔄 Testing batch size: {batch_size}")
            
            # Create optimized config for each batch size
            config = EmergentTrainingConfig()
            config.mixed_precision = True
            config.gradient_accumulation_steps = 1  # No accumulation for clean comparison
            
            trainer = EmergentCubeTrainer(config)
            device = next(trainer.parameters()).device
            
            # Test data
            question_embeddings = torch.randn(batch_size, surface_dim, device=device)
            answer_embeddings = torch.randn(batch_size, 4096, device=device)
            
            # Warmup
            _ = trainer.forward(question_embeddings)
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            num_runs = 3
            
            for run in range(num_runs):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                start_time = time.time()
                
                outputs = trainer.forward(question_embeddings)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            throughput = batch_size / avg_time  # samples per second
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            results[batch_size] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'memory_mb': memory_used
            }
            
            print(f"   Time: {avg_time:.4f}s")
            print(f"   Throughput: {throughput:.1f} samples/sec")
            print(f"   Memory: {memory_used:.1f} MB")
            
            # Cleanup
            del trainer, question_embeddings, answer_embeddings
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[batch_size] = None
    
    # Find optimal batch size
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_batch = max(valid_results, key=lambda x: valid_results[x]['throughput'])
        print(f"\n🎯 OPTIMAL BATCH SIZE: {best_batch}")
        print(f"   Best throughput: {valid_results[best_batch]['throughput']:.1f} samples/sec")
    
    return valid_results


def test_gpu_vs_cpu_optimized():
    """Fair comparison of CPU vs GPU with optimal settings for each"""
    print("\n" + "="*80)
    print("⚖️ OPTIMIZED CPU vs GPU COMPARISON")
    print("="*80)
    
    # Optimal batch size for GPU (from previous analysis or educated guess)
    gpu_batch_size = 1024
    cpu_batch_size = 8  # CPU often better with smaller batches
    surface_dim = 15 * 15
    num_steps = 5
    
    results = {}
    
    # === CPU OPTIMIZED ===
    print("\n🖥️ CPU Optimized Testing:")
    try:
        config_cpu = EmergentTrainingConfig()
        config_cpu.mixed_precision = False  # CPU doesn't benefit from mixed precision
        config_cpu.gradient_accumulation_steps = 1
        
        trainer_cpu = EmergentCubeTrainer(config_cpu, device="cpu")
        
        cpu_times = []
        for step in range(num_steps):
            question_embeddings = torch.randn(cpu_batch_size, surface_dim)
            answer_embeddings = torch.randn(cpu_batch_size, 4096)
            
            start_time = time.time()
            metrics = trainer_cpu.train_step(question_embeddings, answer_embeddings)
            cpu_times.append(time.time() - start_time)
        
        cpu_avg = sum(cpu_times) / len(cpu_times)
        cpu_throughput = cpu_batch_size / cpu_avg
        
        results['cpu'] = {
            'avg_time': cpu_avg,
            'throughput': cpu_throughput,
            'batch_size': cpu_batch_size
        }
        
        print(f"   Time per step: {cpu_avg:.4f}s")
        print(f"   Throughput: {cpu_throughput:.1f} samples/sec")
        
        del trainer_cpu
        
    except Exception as e:
        print(f"   ❌ CPU test failed: {e}")
        results['cpu'] = None
    
    # === GPU OPTIMIZED ===
    print("\n🚀 GPU Optimized Testing:")
    try:
        config_gpu = EmergentTrainingConfig()
        config_gpu.mixed_precision = True
        config_gpu.gradient_accumulation_steps = 1  # No accumulation for fair comparison
        
        trainer_gpu = EmergentCubeTrainer(config_gpu)  # Auto-detect GPU
        device = next(trainer_gpu.parameters()).device
        print(f"   Device: {device}")
        
        # Warmup
        warmup_q = torch.randn(gpu_batch_size, surface_dim, device=device)
        warmup_a = torch.randn(gpu_batch_size, 4096, device=device)
        _ = trainer_gpu.train_step(warmup_q, warmup_a)
        torch.cuda.synchronize()
        
        gpu_times = []
        for step in range(num_steps):
            question_embeddings = torch.randn(gpu_batch_size, surface_dim, device=device)
            answer_embeddings = torch.randn(gpu_batch_size, 4096, device=device)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            metrics = trainer_gpu.train_step(question_embeddings, answer_embeddings)
            
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start_time)
        
        gpu_avg = sum(gpu_times) / len(gpu_times)
        gpu_throughput = gpu_batch_size / gpu_avg
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        results['gpu'] = {
            'avg_time': gpu_avg,
            'throughput': gpu_throughput,
            'batch_size': gpu_batch_size,
            'memory_gb': memory_used
        }
        
        print(f"   Time per step: {gpu_avg:.4f}s")
        print(f"   Throughput: {gpu_throughput:.1f} samples/sec")
        print(f"   Memory used: {memory_used:.2f} GB")
        
        del trainer_gpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        results['gpu'] = None
    
    # === COMPARISON ===
    if results['cpu'] and results['gpu']:
        cpu_throughput = results['cpu']['throughput']
        gpu_throughput = results['gpu']['throughput']
        speedup = gpu_throughput / cpu_throughput
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   CPU: {cpu_throughput:.1f} samples/sec (batch {cpu_batch_size})")
        print(f"   GPU: {gpu_throughput:.1f} samples/sec (batch {gpu_batch_size})")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.2:
            print("✅ GPU shows meaningful speedup!")
        elif speedup > 0.8:
            print("⚖️ GPU performance comparable to CPU")
        else:
            print("❌ GPU underperforming - needs optimization")
            
        return speedup > 1.0
    
    return False


def analyze_memory_patterns():
    """Analyze GPU memory usage patterns"""
    print("\n" + "="*80)
    print("💾 MEMORY USAGE ANALYSIS")
    print("="*80)
    
    if not torch.cuda.is_available():
        return
    
    config = EmergentTrainingConfig()
    trainer = EmergentCubeTrainer(config)
    device = next(trainer.parameters()).device
    
    batch_size = 1024
    surface_dim = 15 * 15
    
    # Track memory at different stages
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"📊 Memory analysis with batch size {batch_size}:")
    
    # Initial state
    init_memory = torch.cuda.memory_allocated()
    print(f"   Initial: {init_memory / 1024**2:.1f} MB")
    
    # Create tensors
    question_embeddings = torch.randn(batch_size, surface_dim, device=device)
    answer_embeddings = torch.randn(batch_size, 4096, device=device)
    
    tensor_memory = torch.cuda.memory_allocated()
    print(f"   + Tensors: {(tensor_memory - init_memory) / 1024**2:.1f} MB")
    
    # Forward pass
    outputs = trainer.forward(question_embeddings)
    forward_memory = torch.cuda.memory_allocated()
    print(f"   + Forward: {(forward_memory - tensor_memory) / 1024**2:.1f} MB")
    
    # Loss computation
    targets = {
        'target_embedding': answer_embeddings,
        'target_surface': outputs['input_surface']
    }
    losses = trainer.compute_loss(outputs, targets)
    loss_memory = torch.cuda.memory_allocated()
    print(f"   + Loss: {(loss_memory - forward_memory) / 1024**2:.1f} MB")
    
    # Backward pass
    losses['total_loss'].backward()
    backward_memory = torch.cuda.memory_allocated()
    print(f"   + Backward: {(backward_memory - loss_memory) / 1024**2:.1f} MB")
    
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"   Peak: {peak_memory / 1024**2:.1f} MB")
    
    # Memory efficiency
    model_params = sum(p.numel() * p.element_size() for p in trainer.parameters())
    efficiency = model_params / peak_memory
    print(f"   Efficiency: {efficiency:.1%} (model params / peak memory)")


def recommend_optimizations():
    """Provide specific optimization recommendations"""
    print("\n" + "="*80)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    recommendations = [
        "🔧 Increase batch size to 32-64 for GPU (current: 8)",
        "⚡ Disable gradient accumulation for fair GPU comparison", 
        "📊 Use DataLoader with pin_memory=True for faster CPU→GPU transfer",
        "🔥 Enable Tensor Core optimization with proper tensor dimensions",
        "💾 Consider model parallelism for very large models",
        "⚖️ Profile actual operations to find specific bottlenecks",
        "🎯 Use torch.jit.script for frequently called functions",
        "🚀 Consider using torch.compile for PyTorch 2.x speedups"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n🎯 IMMEDIATE ACTIONS:")
    print(f"   1. Increase GPU batch size to 32+")
    print(f"   2. Disable gradient accumulation for baseline")
    print(f"   3. Profile with torch.profiler for detailed bottlenecks")


def main():
    """Run comprehensive GPU performance analysis"""
    print("🔍 GPU PERFORMANCE ANALYSIS")
    print("="*80)
    print("Analyzing Phase 2 GPU optimization performance issues")
    
    if not analyze_gpu_utilization():
        return False
    
    # Run analysis
    print("\n🧪 Running performance analysis...")
    
    # 1. Batch size impact
    batch_results = test_batch_size_impact()
    
    # 2. Optimized CPU vs GPU comparison  
    gpu_is_faster = test_gpu_vs_cpu_optimized()
    
    # 3. Memory analysis
    analyze_memory_patterns()
    
    # 4. Recommendations
    recommend_optimizations()
    
    # Summary
    print("\n" + "="*80)
    print("📋 ANALYSIS SUMMARY")
    print("="*80)
    
    if gpu_is_faster:
        print("✅ GPU optimization successful with proper settings")
    else:
        print("⚠️ GPU needs further optimization")
        print("   - Try larger batch sizes (32-64)")
        print("   - Disable gradient accumulation")
        print("   - Profile for specific bottlenecks")
    
    if batch_results:
        optimal_batch = max(batch_results, key=lambda x: batch_results[x]['throughput'] if batch_results[x] else 0)
        print(f"🎯 Recommended GPU batch size: {optimal_batch}")
    
    return gpu_is_faster


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 