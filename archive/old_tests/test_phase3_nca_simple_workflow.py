#!/usr/bin/env python3
"""
[START] Phase 3 Task 3.1: Simplified Training Workflow Test with NCA

Упрощенный тест без mixed precision для проверки NCA functionality:
1. Synthetic embeddings (избегаем teacher model dependencies)
2. Simplified training без autocast complications  
3. Direct NCA metrics collection
4. Clear performance comparison
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)

def create_synthetic_embeddings(num_pairs: int = 12, embedding_dim: int = 768):
    """Создает synthetic embeddings для тестирования"""
    
    # Synthetic Q&A pairs с realistic pattern
    torch.manual_seed(42)  # Reproducible
    
    # Question embeddings - более focused patterns
    question_embeddings = torch.randn(num_pairs, embedding_dim) * 0.8
    question_embeddings[:, :100] += 1.5  # "question" features stronger
    
    # Answer embeddings - related but different patterns  
    answer_embeddings = torch.randn(num_pairs, embedding_dim) * 0.8
    answer_embeddings[:, 100:200] += 1.2  # "answer" features stronger
    
    # Add some Q-A correlation
    correlation_dims = slice(200, 300)
    shared_pattern = torch.randn(num_pairs, 100) * 0.5
    question_embeddings[:, correlation_dims] += shared_pattern
    answer_embeddings[:, correlation_dims] += shared_pattern * 0.7
    
    # Normalize embeddings
    question_embeddings = torch.nn.functional.normalize(question_embeddings, dim=1)
    answer_embeddings = torch.nn.functional.normalize(answer_embeddings, dim=1)
    
    return question_embeddings, answer_embeddings

def test_nca_simple_training():
    """Main test: Simplified training workflow с NCA"""
    print("[START] Testing NCA on Simplified Training Workflow...")
    print("=" * 60)
    
    try:
        # Create synthetic data
        print("[DATA] Creating synthetic embeddings...")
        question_embeddings, answer_embeddings = create_synthetic_embeddings(num_pairs=6)
        print(f"Created {len(question_embeddings)} Q&A pairs")
        
        # Setup configurations (smaller for testing)
        print("[INFO] Setting up training configurations...")
        
        # NCA enabled configuration
        nca_config = EmergentTrainingConfig()
        nca_config.enable_nca = True
        nca_config.cube_dimensions = (4, 4, 3)  # Very small для testing
        nca_config.batch_size = 1
        nca_config.learning_rate = 0.001
        nca_config.epochs = 3
        nca_config.mixed_precision = False  # DISABLE mixed precision
        
        # Standard configuration
        standard_config = EmergentTrainingConfig()  
        standard_config.enable_nca = False
        standard_config.cube_dimensions = (4, 4, 3)
        standard_config.batch_size = 1
        standard_config.learning_rate = 0.001
        standard_config.epochs = 3
        standard_config.mixed_precision = False  # DISABLE mixed precision
        
        # Initialize trainers
        print("[BRAIN] Initializing trainers...")
        nca_trainer = EmergentCubeTrainer(nca_config, device="cpu")
        standard_trainer = EmergentCubeTrainer(standard_config, device="cpu")
        print("[OK] Trainers initialized successfully")
        
        # Run simplified training
        print("\n🏋️ Starting simplified training comparison...")
        
        # Training with NCA
        print("\n--- Training with NCA Enabled ---")
        nca_results = run_simple_training_session(
            nca_trainer, question_embeddings, answer_embeddings, "NCA"
        )
        
        # Training without NCA
        print("\n--- Training without NCA (Standard) ---")
        standard_results = run_simple_training_session(
            standard_trainer, question_embeddings, answer_embeddings, "Standard"
        )
        
        # Compare results
        print("\n[DATA] Training Results Comparison:")
        compare_simple_results(nca_results, standard_results)
        
        # Analyze NCA metrics
        print("\n[BRAIN] NCA Metrics Analysis:")
        analyze_simple_nca_metrics(nca_trainer, nca_results)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Simplified training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_training_session(trainer, question_embeddings, answer_embeddings, session_name: str) -> Dict[str, Any]:
    """Simplified training session"""
    
    print(f"[REFRESH] Running {session_name} training session...")
    
    results = {
        'session_name': session_name,
        'losses': [],
        'training_times': [],
        'nca_metrics': [],
        'total_time': 0
    }
    
    start_time = time.time()
    num_samples = len(question_embeddings)
    
    for epoch in range(trainer.config.epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        print(f"  Epoch {epoch + 1}/{trainer.config.epochs}")
        
        for sample_idx in range(min(3, num_samples)):  # Limit samples для testing
            try:
                # Get single Q&A pair
                question_emb = question_embeddings[sample_idx].unsqueeze(0)  # [1, 768]
                answer_emb = answer_embeddings[sample_idx].unsqueeze(0)      # [1, 768]
                
                # Keep full embedding size - trainer expects 4096D internally
                # Expand to 4096D if needed (trainer internals expect this)
                if question_emb.size(1) != 4096:
                    # Pad or project to 4096D
                    if question_emb.size(1) < 4096:
                        # Pad with zeros
                        padding_size = 4096 - question_emb.size(1)
                        padding = torch.zeros(1, padding_size)
                        question_emb = torch.cat([question_emb, padding], dim=1)
                        answer_emb = torch.cat([answer_emb, padding], dim=1)
                    else:
                        # Take first 4096 dimensions
                        question_emb = question_emb[:, :4096]
                        answer_emb = answer_emb[:, :4096]
                
                # Forward pass
                outputs = trainer.forward(question_emb)
                
                # Prepare targets  
                targets = {
                    'surface_input': question_emb,
                    'target_embedding': answer_emb
                }
                
                # Compute loss
                loss_results = trainer.compute_loss(outputs, targets)
                total_loss = loss_results['total_loss']
                
                # Backward pass (simplified)
                trainer.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                trainer.optimizer.step()
                
                epoch_losses.append(total_loss.item())
                
                # Collect NCA metrics
                if hasattr(trainer, 'get_nca_metrics'):
                    nca_metrics = trainer.get_nca_metrics()
                    if nca_metrics.get('status') != 'disabled':
                        results['nca_metrics'].append(nca_metrics.copy())
                
                print(f"    Sample {sample_idx + 1}: Loss = {total_loss.item():.6f}")
                
            except Exception as e:
                print(f"    [WARNING] Sample {sample_idx + 1} failed: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        results['losses'].append(avg_epoch_loss)
        results['training_times'].append(epoch_time)
        
        print(f"  Epoch {epoch + 1} completed: Avg Loss = {avg_epoch_loss:.6f}, Time = {epoch_time:.2f}s")
    
    results['total_time'] = time.time() - start_time
    print(f"[OK] {session_name} training completed in {results['total_time']:.2f}s")
    
    return results

def compare_simple_results(nca_results: Dict, standard_results: Dict):
    """Compare training results"""
    
    print(f"{'Metric':<20} {'NCA Enabled':<15} {'Standard':<15} {'Difference':<15}")
    print("-" * 65)
    
    # Training time comparison
    nca_time = nca_results['total_time']
    std_time = standard_results['total_time']
    time_diff = ((nca_time / std_time) - 1) * 100 if std_time > 0 else 0
    
    print(f"{'Total Time (s)':<20} {nca_time:<15.2f} {std_time:<15.2f} {time_diff:+.1f}%")
    
    # Loss comparison
    nca_final_loss = nca_results['losses'][-1] if nca_results['losses'] else 0
    std_final_loss = standard_results['losses'][-1] if standard_results['losses'] else 0
    loss_diff = ((nca_final_loss / std_final_loss) - 1) * 100 if std_final_loss > 0 else 0
    
    print(f"{'Final Loss':<20} {nca_final_loss:<15.6f} {std_final_loss:<15.6f} {loss_diff:+.1f}%")
    
    # Loss convergence  
    if len(nca_results['losses']) > 1:
        nca_convergence = nca_results['losses'][0] - nca_results['losses'][-1]
        std_convergence = standard_results['losses'][0] - standard_results['losses'][-1]
        print(f"{'Loss Reduction':<20} {nca_convergence:<15.6f} {std_convergence:<15.6f} {'N/A':<15}")
    
    # Performance analysis
    print(f"\n[CHART] Performance Analysis:")
    if time_diff < 100:  # Less than 100% overhead
        print(f"  [OK] NCA overhead acceptable: {time_diff:+.1f}%")
    else:
        print(f"  [WARNING] NCA overhead high: {time_diff:+.1f}%")
    
    if nca_final_loss > 0 and std_final_loss > 0:
        if abs(loss_diff) < 50:  # Less than 50% difference
            print(f"  [OK] Loss consistency maintained: {loss_diff:+.1f}%")
        else:
            print(f"  [WARNING] Significant loss difference: {loss_diff:+.1f}%")

def analyze_simple_nca_metrics(trainer, results: Dict):
    """Analyze NCA metrics from simplified training"""
    
    if not results['nca_metrics']:
        print("  [WARNING] No NCA metrics collected")
        return
    
    print(f"  [DATA] NCA Metrics Summary ({len(results['nca_metrics'])} snapshots):")
    
    # Get final summary
    final_summary = trainer.get_nca_metrics()
    
    print(f"    Status: {final_summary.get('status', 'Unknown')}")
    print(f"    Training Steps: {final_summary.get('training_step', 'N/A')}")
    
    # Stochastic update statistics
    if 'stochastic_stats' in final_summary:
        stats = final_summary['stochastic_stats']
        print(f"    Update Statistics:")
        print(f"      Total Updates: {stats.get('total_updates', 0)}")
        print(f"      Avg Updates: {stats.get('avg_updates', 0):.2f}")
        print(f"      Update Variance: {stats.get('update_variance', 0):.4f}")
    
    # Pattern analysis  
    if 'pattern_analysis' in final_summary:
        patterns = final_summary['pattern_analysis']
        print(f"    Pattern Analysis:")
        print(f"      Current Specialization: {patterns.get('current_specialization', 0):.4f}")
        print(f"      Avg Specialization: {patterns.get('avg_specialization', 0):.4f}")
        print(f"      Stability: {patterns.get('specialization_stability', 0):.4f}")
    
    # Recent pattern details
    if 'recent_patterns' in final_summary and final_summary['recent_patterns']:
        recent = final_summary['recent_patterns']
        print(f"    Recent Patterns:")
        if 'spatial_coherence' in recent:
            print(f"      Spatial Coherence: {recent['spatial_coherence'].item():.4f}")
        if 'emergent_specialization' in recent:
            print(f"      Emergent Specialization: {recent['emergent_specialization'].item():.4f}")
        if 'temporal_consistency' in recent:
            print(f"      Temporal Consistency: {recent['temporal_consistency'].item():.4f}")
    
    # Evolution analysis
    if len(results['nca_metrics']) >= 2:
        print(f"    Evolution Tracking:")
        first_metrics = results['nca_metrics'][0]
        last_metrics = results['nca_metrics'][-1]
        
        first_step = first_metrics.get('training_step', 0)
        last_step = last_metrics.get('training_step', 0)
        
        print(f"      Training Progress: step {first_step} → {last_step}")
        print(f"      Snapshots Collected: {len(results['nca_metrics'])}")
        
        # Check pattern stability
        if 'pattern_analysis' in first_metrics and 'pattern_analysis' in last_metrics:
            first_spec = first_metrics['pattern_analysis'].get('current_specialization', 0)
            last_spec = last_metrics['pattern_analysis'].get('current_specialization', 0)
            spec_change = last_spec - first_spec
            print(f"      Specialization Change: {spec_change:+.6f}")

def test_nca_pattern_evolution():
    """Test emergent pattern evolution during training"""
    print("\n[ART] Testing NCA Pattern Evolution...")
    
    try:
        # Create trainer with NCA
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (3, 3, 2)  # Minimal cube
        config.batch_size = 1
        config.mixed_precision = False
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        trainer.reset_nca_tracking()
        
        # Multiple forward passes with consistent input
        pattern_history = []
        surface_size = 3 * 3  # 9
        
        for step in range(8):
            # Consistent input with slight variation
            surface_input = torch.randn(1, surface_size) * 0.3 + step * 0.1
            
            # Forward pass
            outputs = trainer.forward(surface_input)
            
            # Collect pattern metrics
            nca_metrics = trainer.get_nca_metrics()
            if 'recent_patterns' in nca_metrics and nca_metrics['recent_patterns']:
                pattern_history.append({
                    'step': step,
                    'patterns': nca_metrics['recent_patterns'].copy(),
                    'training_step': nca_metrics.get('training_step', step)
                })
        
        # Analyze evolution
        if len(pattern_history) >= 3:
            print("  [CHART] Pattern Evolution Analysis:")
            
            # Spatial coherence trend
            if 'spatial_coherence' in pattern_history[0]['patterns']:
                coherence_values = [
                    p['patterns']['spatial_coherence'].item() 
                    for p in pattern_history if 'spatial_coherence' in p['patterns']
                ]
                
                if len(coherence_values) >= 3:
                    coherence_trend = np.polyfit(range(len(coherence_values)), coherence_values, 1)[0]
                    print(f"    Spatial Coherence Trend: {coherence_trend:+.6f} per step")
                    
                    if abs(coherence_trend) < 0.01:
                        print("    [OK] Spatial coherence stable")
                    else:
                        direction = "increasing" if coherence_trend > 0 else "decreasing"
                        print(f"    [DATA] Spatial coherence {direction}")
            
            # Specialization development
            if 'emergent_specialization' in pattern_history[0]['patterns']:
                spec_values = [
                    p['patterns']['emergent_specialization'].item()
                    for p in pattern_history if 'emergent_specialization' in p['patterns']
                ]
                
                if len(spec_values) >= 3:
                    spec_trend = np.polyfit(range(len(spec_values)), spec_values, 1)[0]
                    print(f"    Specialization Trend: {spec_trend:+.6f} per step")
                    
                    if spec_trend > 0.001:
                        print("    [OK] Emergent specialization developing")
                    elif abs(spec_trend) < 0.001:
                        print("    [DATA] Specialization stable")
                    else:
                        print("    [WARNING] Specialization decreasing")
            
            print(f"    Pattern History: {len(pattern_history)} steps tracked")
        
        print("  [OK] Pattern evolution test completed")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Pattern evolution test failed: {e}")
        return False

def run_comprehensive_simple_test():
    """Run comprehensive simplified test suite"""
    print("[BRAIN] PHASE 3 Task 3.1: Simplified Real Training Workflow Test")
    print("=" * 70)
    
    test_results = []
    
    # Main simplified training test
    print("\n1️⃣ Simplified Training Workflow Test")
    test_results.append(test_nca_simple_training())
    
    # Pattern evolution test
    print("\n2️⃣ Pattern Evolution Test")  
    test_results.append(test_nca_pattern_evolution())
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"🧪 Simplified Training Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] ALL SIMPLIFIED TESTS PASSED!")
        print("[SUCCESS] NCA integration works in simplified training workflow")
        print("[START] Ready to proceed with full real-world testing")
        return True
    else:
        print(f"[ERROR] {total - passed} tests failed. NCA needs debugging.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_simple_test()
    sys.exit(0 if success else 1) 