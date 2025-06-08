#!/usr/bin/env python3
"""
🚀 Aggressive Batch Size Optimization - GPU Memory Utilization Test
Testing very large batch sizes to maximize 32GB GPU memory usage
"""

import torch
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


def test_memory_scaling():
    """Test batch sizes up to memory limit"""
    print("🚀 AGGRESSIVE BATCH SIZE SCALING TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total Memory: {total_memory / 1024**3:.1f} GB")
    
    # Test progressively larger batch sizes
    batch_sizes = [64, 128, 256, 512, 1024, 1536, 2048]  # Very aggressive scaling
    surface_dim = 15 * 15
    results = {}
    
    for batch_size in batch_sizes:
        try:
            print(f"\n🧪 Testing BATCH SIZE: {batch_size}")
            
            # Clean start
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create optimized config
            config = EmergentTrainingConfig()
            config.mixed_precision = True
            config.gradient_accumulation_steps = 1  # Pure batch processing
            
            trainer = EmergentCubeTrainer(config)
            device = next(trainer.parameters()).device
            
            # Memory before tensors
            baseline_memory = torch.cuda.memory_allocated()
            
            # Create large batch tensors
            question_embeddings = torch.randn(batch_size, surface_dim, device=device)
            answer_embeddings = torch.randn(batch_size, 4096, device=device)
            
            tensor_memory = torch.cuda.memory_allocated()
            print(f"   📊 Tensor memory: {(tensor_memory - baseline_memory) / 1024**2:.1f} MB")
            
            # Test forward pass (most memory intensive)
            start_time = time.time()
            
            try:
                outputs = trainer.forward(question_embeddings)
                
                forward_memory = torch.cuda.memory_allocated()
                forward_time = time.time() - start_time
                
                # Calculate throughput
                throughput = batch_size / forward_time
                memory_gb = forward_memory / 1024**3
                memory_utilization = (forward_memory / total_memory) * 100
                
                results[batch_size] = {
                    'success': True,
                    'forward_time': forward_time,
                    'throughput': throughput,
                    'memory_gb': memory_gb,
                    'memory_percent': memory_utilization
                }
                
                print(f"   ✅ SUCCESS!")
                print(f"   ⏱️  Forward time: {forward_time:.3f}s")
                print(f"   🚀 Throughput: {throughput:.1f} samples/sec")
                print(f"   💾 Memory used: {memory_gb:.2f} GB ({memory_utilization:.1f}%)")
                
                # Test backward pass if forward succeeded
                try:
                    targets = {
                        'target_embedding': answer_embeddings,
                        'target_surface': outputs['input_surface']
                    }
                    losses = trainer.compute_loss(outputs, targets)
                    
                    backward_start = time.time()
                    losses['total_loss'].backward()
                    backward_time = time.time() - backward_start
                    
                    peak_memory = torch.cuda.max_memory_allocated()
                    peak_gb = peak_memory / 1024**3
                    peak_percent = (peak_memory / total_memory) * 100
                    
                    results[batch_size].update({
                        'backward_time': backward_time,
                        'peak_memory_gb': peak_gb,
                        'peak_memory_percent': peak_percent,
                        'full_training_step': True
                    })
                    
                    print(f"   🔥 Backward time: {backward_time:.3f}s")
                    print(f"   📈 Peak memory: {peak_gb:.2f} GB ({peak_percent:.1f}%)")
                    
                except Exception as e:
                    print(f"   ⚠️  Backward failed: {e}")
                    results[batch_size]['full_training_step'] = False
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"   💥 OUT OF MEMORY: {e}")
                results[batch_size] = {
                    'success': False,
                    'error': 'OOM',
                    'attempted_memory_gb': tensor_memory / 1024**3
                }
                
                # Free memory and continue
                del question_embeddings, answer_embeddings, trainer
                torch.cuda.empty_cache()
                break
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Cleanup
            try:
                del trainer, question_embeddings, answer_embeddings
                if 'outputs' in locals():
                    del outputs
                torch.cuda.empty_cache()
            except:
                pass
                
        except Exception as e:
            print(f"   💥 CRITICAL ERROR: {e}")
            results[batch_size] = {'success': False, 'error': str(e)}
            break
    
    return results


def find_optimal_batch_size(results):
    """Find the optimal batch size based on throughput and memory efficiency"""
    print(f"\n📊 OPTIMIZATION ANALYSIS")
    print("="*80)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ No successful batch sizes found")
        return None
    
    print(f"📈 PERFORMANCE SUMMARY:")
    for batch_size, result in successful_results.items():
        throughput = result.get('throughput', 0)
        memory_percent = result.get('memory_percent', 0)
        peak_percent = result.get('peak_memory_percent', memory_percent)
        
        print(f"   Batch {batch_size:4d}: {throughput:6.1f} samples/sec, "
              f"{memory_percent:4.1f}% memory ({peak_percent:4.1f}% peak)")
    
    # Find optimal balance
    # Score = throughput * memory_utilization_factor
    best_batch = None
    best_score = 0
    
    for batch_size, result in successful_results.items():
        throughput = result.get('throughput', 0)
        memory_percent = result.get('peak_memory_percent', result.get('memory_percent', 0))
        
        # Reward both throughput and memory utilization
        memory_factor = min(memory_percent / 50.0, 2.0)  # Cap at 2x bonus for 50%+ usage
        score = throughput * memory_factor
        
        if score > best_score:
            best_score = score
            best_batch = batch_size
    
    if best_batch:
        result = successful_results[best_batch]
        print(f"\n🎯 OPTIMAL BATCH SIZE: {best_batch}")
        print(f"   🚀 Throughput: {result['throughput']:.1f} samples/sec")
        print(f"   💾 Memory usage: {result.get('peak_memory_percent', result.get('memory_percent', 0)):.1f}%")
        print(f"   ⚖️  Efficiency score: {best_score:.1f}")
    
    return best_batch


def test_memory_limits():
    """Test the absolute memory limits"""
    print(f"\n🔬 MEMORY LIMIT STRESS TEST")
    print("="*80)
    
    # Binary search for maximum batch size
    low, high = 64, 8192  # Start with reasonable range
    max_working_batch = 64
    
    while low <= high:
        mid = (low + high) // 2
        print(f"🧪 Testing batch size: {mid}")
        
        try:
            torch.cuda.empty_cache()
            
            config = EmergentTrainingConfig()
            config.mixed_precision = True
            trainer = EmergentCubeTrainer(config)
            device = next(trainer.parameters()).device
            
            # Just test tensor creation + forward pass
            question_embeddings = torch.randn(mid, 225, device=device)
            
            # Quick forward test
            _ = trainer.forward(question_embeddings)
            
            print(f"   ✅ SUCCESS at batch {mid}")
            max_working_batch = mid
            low = mid + 1
            
            del trainer, question_embeddings
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"   💥 OOM at batch {mid}")
            high = mid - 1
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error at batch {mid}: {e}")
            high = mid - 1
    
    print(f"\n🎯 MAXIMUM BATCH SIZE: {max_working_batch}")
    return max_working_batch


def main():
    """Run aggressive batch size optimization"""
    print("🚀 AGGRESSIVE GPU MEMORY UTILIZATION TEST")
    print("="*80)
    print("Finding optimal batch size for 32GB GPU memory")
    
    # 1. Progressive scaling test
    results = test_memory_scaling()
    
    # 2. Find optimal batch size
    optimal_batch = find_optimal_batch_size(results)
    
    # 3. Test absolute limits
    max_batch = test_memory_limits()
    
    # Final recommendations
    print(f"\n" + "="*80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*80)
    
    if optimal_batch:
        print(f"✅ RECOMMENDED BATCH SIZE: {optimal_batch}")
        print(f"   - Optimal balance of throughput and memory utilization")
        
        if optimal_batch in results:
            result = results[optimal_batch]
            memory_gb = result.get('peak_memory_gb', result.get('memory_gb', 0))
            print(f"   - Memory usage: {memory_gb:.1f} GB / 32 GB")
            print(f"   - Throughput: {result.get('throughput', 0):.1f} samples/sec")
    
    if max_batch and max_batch != optimal_batch:
        print(f"⚠️  MAXIMUM POSSIBLE: {max_batch}")
        print(f"   - May not be optimal for performance")
    
    # Update config recommendation
    if optimal_batch:
        print(f"\n📝 CONFIG UPDATE:")
        print(f"   # In your training script:")
        print(f"   batch_size = {optimal_batch}")
        print(f"   gradient_accumulation_steps = 1  # No accumulation needed")
        
        # Calculate expected memory usage
        if optimal_batch in results:
            result = results[optimal_batch]
            expected_memory = result.get('peak_memory_gb', result.get('memory_gb', 0))
            utilization = (expected_memory / 32) * 100
            print(f"   # Expected GPU memory: {expected_memory:.1f} GB ({utilization:.1f}%)")
    
    return optimal_batch


if __name__ == "__main__":
    optimal = main()
    print(f"\n🏁 Optimal batch size: {optimal}") 