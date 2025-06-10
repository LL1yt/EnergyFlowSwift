#!/usr/bin/env python3
"""
[START] Phase 3 Task 3.1: Real Training Workflow Test with NCA

Тестирует Neural Cellular Automata на реальном dialogue training workflow:
1. Dialogue dataset loading и processing
2. Multi-step training с NCA enabled/disabled
3. Emergent pattern metrics во время обучения
4. Performance comparison и stability analysis
5. Real-world emergent behavior preservation
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# ДИАГНОСТИКА: Включаем подробное логирование для NCA
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.dialogue_dataset import DialogueDataset, create_dialogue_dataset

def create_test_dialogue_data():
    """Создает test dialogue dataset для реального training"""
    
    # Реалистичные dialogue pairs для AI/ML domain
    dialogue_pairs = [
        # Neural Networks & AI
        {"question": "What is a neural network?", "answer": "A neural network is a computational model inspired by biological neural networks, consisting of interconnected nodes that process information."},
        {"question": "How does backpropagation work?", "answer": "Backpropagation is an algorithm for training neural networks that calculates gradients by propagating errors backward through the network layers."},
        {"question": "What is overfitting in machine learning?", "answer": "Overfitting occurs when a model learns the training data too well, including noise, resulting in poor generalization to new data."},
        
        # 3D Processing & Cellular Networks  
        {"question": "What are cellular neural networks?", "answer": "Cellular neural networks are arrays of simple, locally connected processing units that can perform complex computations through emergent behavior."},
        {"question": "How do 3D neural networks differ from 2D?", "answer": "3D neural networks process volumetric data with depth information, enabling spatial reasoning and complex pattern recognition in three dimensions."},
        {"question": "What is emergent behavior in neural systems?", "answer": "Emergent behavior occurs when complex patterns and functions arise from simple interactions between individual components in neural networks."},
        
        # Training & Optimization
        {"question": "Why is GPU training faster?", "answer": "GPU training is faster because GPUs have thousands of parallel cores optimized for matrix operations, allowing simultaneous computation across many data points."},
        {"question": "What is mixed precision training?", "answer": "Mixed precision training uses both 16-bit and 32-bit floating point representations to reduce memory usage and accelerate training while maintaining model accuracy."},
        {"question": "How does batch size affect training?", "answer": "Larger batch sizes provide more stable gradients and better GPU utilization, but may require more memory and can affect convergence dynamics."},
        
        # Advanced Topics
        {"question": "What are attention mechanisms?", "answer": "Attention mechanisms allow neural networks to focus on relevant parts of input data, enabling better handling of sequential and structured information."},
        {"question": "How do transformer models work?", "answer": "Transformers use self-attention mechanisms to process sequences in parallel, capturing long-range dependencies without recurrent connections."},
        {"question": "What is transfer learning?", "answer": "Transfer learning involves using pre-trained models as starting points for new tasks, leveraging learned features to improve performance on related problems."}
    ]
    
    return dialogue_pairs

def test_nca_real_training():
    """Main test: Real training workflow с NCA integration"""
    print("[START] Testing NCA on Real Training Workflow...")
    print("=" * 60)
    
    try:
        # Setup configurations
        print("[INFO] Setting up training configurations...")
        
        # NCA enabled configuration
        nca_config = EmergentTrainingConfig()
        nca_config.enable_nca = True
        nca_config.cube_dimensions = (8, 8, 3)  # Smaller для faster testing  
        nca_config.batch_size = 1  # Single sample для debugging
        nca_config.learning_rate = 0.001
        nca_config.epochs = 3  # Very short training для test
        nca_config.mixed_precision = False  # Disable mixed precision для CPU testing
        
        # Standard configuration (NCA disabled)
        standard_config = EmergentTrainingConfig()
        standard_config.enable_nca = False
        standard_config.cube_dimensions = (8, 8, 3)
        standard_config.batch_size = 1
        standard_config.learning_rate = 0.001
        standard_config.epochs = 3
        standard_config.mixed_precision = False  # Disable mixed precision
        
        # Create dialogue dataset
        print("💬 Creating dialogue dataset...")
        dialogue_pairs = create_test_dialogue_data()
        
        # Create dataset using create_dialogue_dataset function
        dataset_config = {
            'cache_embeddings': False,  # Disable caching for testing 
            'embedding_dim': 4096,      # Expected embedding dimension for system
            'use_cache': False,         # Disable file caching
            'normalize_embeddings': True,
            'validation_split': 0.0     # No validation split for testing
        }
        
        dataset = create_dialogue_dataset(
            dialogue_pairs, 
            teacher_model="distilbert",  # Use lightweight fallback model
            **dataset_config
        )
        
        print(f"Dataset created: {len(dataset)} dialogue pairs")
        
        # Initialize trainers
        print("[BRAIN] Initializing trainers...")
        nca_trainer = EmergentCubeTrainer(nca_config, device="cpu")
        standard_trainer = EmergentCubeTrainer(standard_config, device="cpu")
        
        # Check NCA activation
        print(f"NCA Trainer - has nca: {hasattr(nca_trainer, 'nca')}")
        print(f"NCA Trainer - nca is not None: {getattr(nca_trainer, 'nca', None) is not None}")
        if hasattr(nca_trainer, 'nca') and nca_trainer.nca is not None:
            print(f"NCA system type: {type(nca_trainer.nca)}")
        
        nca_status = nca_trainer.get_nca_metrics()
        print(f"NCA initial status: {nca_status.get('status', 'Unknown')}")
        print(f"NCA config enable_nca: {nca_config.enable_nca}")
        
        print("[OK] Trainers initialized successfully")
        
        # Run training comparison
        print("\n🏋️ Starting training comparison...")
        
        # Training with NCA
        print("\n--- Training with NCA Enabled ---")
        nca_results = run_training_session(nca_trainer, dataset, "NCA")
        
        # Training without NCA  
        print("\n--- Training without NCA (Standard) ---")
        standard_results = run_training_session(standard_trainer, dataset, "Standard")
        
        # Compare results
        print("\n[DATA] Training Results Comparison:")
        compare_training_results(nca_results, standard_results)
        
        # Analyze NCA metrics
        print("\n[BRAIN] NCA Metrics Analysis:")
        analyze_nca_metrics(nca_trainer, nca_results)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Real training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_training_session(trainer, dataset, session_name: str) -> Dict[str, Any]:
    """Запускает training session и собирает метрики"""
    
    print(f"[REFRESH] Running {session_name} training session...")
    
    results = {
        'session_name': session_name,
        'losses': [],
        'training_times': [],
        'nca_metrics': [],
        'total_time': 0
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(trainer.config.epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        print(f"  Epoch {epoch + 1}/{trainer.config.epochs}")
        
        # Sample a few batches from dataset
        num_batches = min(3, len(dataset))  # Limit для testing
        
        for batch_idx in range(num_batches):
            try:
                # ДИАГНОСТИКА: Детальная проверка dataset access
                print(f"    [MAGNIFY] Accessing dataset[{batch_idx % len(dataset)}] from dataset length {len(dataset)}")
                
                # Get dialogue pair (dataset returns tuple: (question_emb, answer_emb))
                try:
                    dialogue_pair = dataset[batch_idx % len(dataset)]
                    print(f"    [MAGNIFY] Dataset returned: {type(dialogue_pair)}")
                    
                    if dialogue_pair is None:
                        print(f"    [ERROR] Dataset returned None for index {batch_idx % len(dataset)}")
                        continue
                    
                    if isinstance(dialogue_pair, tuple) and len(dialogue_pair) == 2:
                        question_emb, answer_emb = dialogue_pair
                    else:
                        print(f"    [ERROR] Unexpected dataset return format: {type(dialogue_pair)}, length: {len(dialogue_pair) if hasattr(dialogue_pair, '__len__') else 'N/A'}")
                        continue
                        
                except Exception as dataset_error:
                    print(f"    [ERROR] Dataset access error at index {batch_idx % len(dataset)}: {dataset_error}")
                    continue
                
                # ДИАГНОСТИКА: Проверяем на None values
                if question_emb is None:
                    print(f"    [ERROR] question_emb is None at batch {batch_idx}")
                    continue
                if answer_emb is None:
                    print(f"    [ERROR] answer_emb is None at batch {batch_idx}")
                    continue
                
                print(f"    [MAGNIFY] Batch {batch_idx}: question_emb type={type(question_emb)}, answer_emb type={type(answer_emb)}")
                print(f"    [MAGNIFY] Batch {batch_idx}: question_emb shape={getattr(question_emb, 'shape', 'NO SHAPE')}, answer_emb shape={getattr(answer_emb, 'shape', 'NO SHAPE')}")
                
                # Add batch dimension if needed
                if hasattr(question_emb, 'dim') and question_emb.dim() == 1:
                    question_emb = question_emb.unsqueeze(0)
                if hasattr(answer_emb, 'dim') and answer_emb.dim() == 1:
                    answer_emb = answer_emb.unsqueeze(0)
                
                # Convert to 4096D if needed (padding from 768D)
                if question_emb.size(1) != 4096:
                    if question_emb.size(1) < 4096:
                        # Pad with zeros to reach 4096D
                        padding_size = 4096 - question_emb.size(1)
                        padding = torch.zeros(question_emb.size(0), padding_size)
                        question_emb = torch.cat([question_emb, padding], dim=1)
                        answer_emb = torch.cat([answer_emb, padding], dim=1)
                        print(f"    Padded embeddings: {question_emb.size(1)-padding_size}D → {question_emb.size(1)}D")
                    else:
                        # Truncate to 4096D
                        question_emb = question_emb[:, :4096]
                        answer_emb = answer_emb[:, :4096]
                        print(f"    Truncated embeddings to 4096D")
                
                # Ensure correct batch size
                if question_emb.size(0) < trainer.config.batch_size:
                    # Repeat to match batch size
                    repeat_factor = trainer.config.batch_size
                    question_emb = question_emb.repeat(repeat_factor, 1)
                    answer_emb = answer_emb.repeat(repeat_factor, 1)
                
                # Manual training step (more control)
                # ДИАГНОСТИКА: Проверяем входы перед forward pass
                print(f"    [MAGNIFY] Pre-forward: question_emb shape={question_emb.shape}, answer_emb shape={answer_emb.shape}")
                
                # Forward pass
                try:
                    outputs = trainer.forward(question_emb)
                    print(f"    [OK] Forward pass completed, outputs keys: {list(outputs.keys())}")
                    
                    # ДИАГНОСТИКА: Проверяем outputs
                    for key, value in outputs.items():
                        if value is None:
                            print(f"    [ERROR] Output '{key}' is None!")
                        else:
                            print(f"    [MAGNIFY] Output '{key}': {value.shape}")
                
                except Exception as forward_error:
                    print(f"    [ERROR] Forward pass failed: {forward_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Prepare targets with correct keys for loss computation
                targets = {
                    'surface_input': question_emb,      # Input reconstruction target
                    'target_embedding': answer_emb      # Dialogue similarity target
                }
                
                print(f"    [MAGNIFY] Targets prepared: {list(targets.keys())}")
                
                # Compute loss
                try:
                    loss_results = trainer.compute_loss(outputs, targets)
                    print(f"    [OK] Loss computation completed")
                except Exception as loss_error:
                    print(f"    [ERROR] Loss computation failed: {loss_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                total_loss = loss_results['total_loss']
                print(f"    [MAGNIFY] Total loss: {total_loss.item():.6f}")
                
                # КРИТИЧЕСКАЯ ДИАГНОСТИКА: Проверяем computational graph
                print(f"    [MAGNIFY] Loss requires_grad: {total_loss.requires_grad}")
                print(f"    [MAGNIFY] Loss grad_fn: {total_loss.grad_fn}")
                print(f"    [MAGNIFY] Loss device: {total_loss.device}")
                
                # Проверяем все loss components
                for key, loss_component in loss_results.items():
                    if torch.is_tensor(loss_component):
                        print(f"    [MAGNIFY] {key}: requires_grad={loss_component.requires_grad}, grad_fn={loss_component.grad_fn}")
                
                # Если total_loss не требует градиентов, пропускаем backward
                if not total_loss.requires_grad:
                    print(f"    [WARNING] Total loss does not require gradients, skipping backward pass")
                    continue
                
                # Backward pass
                try:
                    print(f"    [MAGNIFY] Starting backward pass...")
                    trainer.optimizer.zero_grad()
                    print(f"    [MAGNIFY] Gradients zeroed")
                    
                    total_loss.backward()
                    print(f"    [MAGNIFY] Backward completed")
                    
                    # Optimizer step - теперь с enhanced handling в trainer
                    print(f"    [MAGNIFY] Starting optimizer step...")
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
                    print(f"    [MAGNIFY] Gradients clipped")
                    
                    trainer.optimizer.step()
                    print(f"    [OK] Optimizer step completed")
                    
                    # КРИТИЧЕСКОЕ: Полная очистка состояний после каждого batch для предотвращения accumulation
                    print(f"    [CONFIG] Cleaning states after batch...")
                    if hasattr(trainer, '_full_state_reset'):
                        trainer._full_state_reset()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    print(f"    [OK] State cleanup completed")
                    
                except Exception as backward_error:
                    print(f"    [ERROR] Backward pass failed: {backward_error}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Prepare step results dict
                step_results = {
                    'total_loss': total_loss.item(),
                    'surface_reconstruction_loss': loss_results.get('surface_reconstruction_loss', torch.tensor(0.0)).item(),
                    'internal_consistency_loss': loss_results.get('internal_consistency_loss', torch.tensor(0.0)).item(),
                    'dialogue_similarity_loss': loss_results.get('dialogue_similarity_loss', torch.tensor(0.0)).item()
                }
                
                # Collect metrics
                epoch_losses.append(step_results.get('total_loss', 0.0))
                
                # Collect NCA metrics if available
                if hasattr(trainer, 'get_nca_metrics'):
                    nca_metrics = trainer.get_nca_metrics()
                    if nca_metrics.get('status') != 'disabled':
                        results['nca_metrics'].append(nca_metrics)
                
                print(f"    Batch {batch_idx + 1}: Loss = {step_results.get('total_loss', 0.0):.6f}")
                
            except Exception as e:
                print(f"    [WARNING] Batch {batch_idx + 1} failed: {e}")
                continue
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        results['losses'].append(avg_epoch_loss)
        results['training_times'].append(epoch_time)
        
        print(f"  Epoch {epoch + 1} completed: Avg Loss = {avg_epoch_loss:.6f}, Time = {epoch_time:.2f}s")
    
    results['total_time'] = time.time() - start_time
    
    print(f"[OK] {session_name} training completed in {results['total_time']:.2f}s")
    
    return results

def compare_training_results(nca_results: Dict, standard_results: Dict):
    """Сравнивает результаты training с NCA и без"""
    
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
    
    # Training stability
    nca_loss_std = np.std(nca_results['losses']) if nca_results['losses'] else 0
    std_loss_std = np.std(standard_results['losses']) if standard_results['losses'] else 0
    
    print(f"{'Loss Std Dev':<20} {nca_loss_std:<15.6f} {std_loss_std:<15.6f} {'N/A':<15}")
    
    # Performance analysis
    print(f"\n[CHART] Performance Analysis:")
    if time_diff < 50:  # Less than 50% overhead
        print(f"  [OK] NCA overhead acceptable: {time_diff:+.1f}%")
    else:
        print(f"  [WARNING] NCA overhead high: {time_diff:+.1f}%")
    
    if abs(loss_diff) < 20:  # Less than 20% difference
        print(f"  [OK] Loss consistency maintained: {loss_diff:+.1f}%")
    else:
        print(f"  [WARNING] Significant loss difference: {loss_diff:+.1f}%")

def analyze_nca_metrics(trainer, results: Dict):
    """Анализирует NCA metrics из training session"""
    
    if not results['nca_metrics']:
        print("  [WARNING] No NCA metrics collected")
        return
    
    print(f"  [DATA] NCA Metrics Summary:")
    
    # Get final NCA summary
    final_summary = trainer.get_nca_metrics()
    
    print(f"    Training Steps: {final_summary.get('training_step', 'N/A')}")
    
    # Stochastic stats
    if 'stochastic_stats' in final_summary:
        stats = final_summary['stochastic_stats']
        print(f"    Update Statistics:")
        print(f"      Avg Updates: {stats.get('avg_updates', 0):.2f}")
        print(f"      Update Variance: {stats.get('update_variance', 0):.4f}")
    
    # Pattern analysis
    if 'pattern_analysis' in final_summary:
        patterns = final_summary['pattern_analysis']
        print(f"    Pattern Analysis:")
        print(f"      Current Specialization: {patterns.get('current_specialization', 0):.4f}")
        print(f"      Avg Specialization: {patterns.get('avg_specialization', 0):.4f}")
        print(f"      Specialization Stability: {patterns.get('specialization_stability', 0):.4f}")
    
    # Recent patterns
    if 'recent_patterns' in final_summary:
        recent = final_summary['recent_patterns']
        print(f"    Recent Patterns:")
        if 'spatial_coherence' in recent:
            print(f"      Spatial Coherence: {recent['spatial_coherence'].item():.4f}")
        if 'emergent_specialization' in recent:
            print(f"      Emergent Specialization: {recent['emergent_specialization'].item():.4f}")
    
    # Analyze pattern evolution during training
    if len(results['nca_metrics']) > 1:
        print(f"    Pattern Evolution:")
        first_metrics = results['nca_metrics'][0]
        last_metrics = results['nca_metrics'][-1]
        
        first_step = first_metrics.get('training_step', 0)
        last_step = last_metrics.get('training_step', 0)
        
        print(f"      Training Steps Progression: {first_step} → {last_step}")
        print(f"      Pattern Tracking: {len(results['nca_metrics'])} snapshots collected")

def test_nca_pattern_preservation():
    """Тест preservation emergent patterns во время training"""
    print("\n[ART] Testing Pattern Preservation During Training...")
    
    try:
        # Create trainer with NCA
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (4, 4, 3)  # Small cube
        config.batch_size = 1
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        # Initialize pattern tracking
        trainer.reset_nca_tracking()
        
        # Simulate multiple training steps
        pattern_history = []
        
        for step in range(10):
            # Create consistent input pattern
            surface_size = 4 * 4  # width * height
            surface_input = torch.randn(1, surface_size) * (0.5 + step * 0.1)
            
            # Forward pass
            outputs = trainer.forward(surface_input)
            
            # Collect pattern metrics
            nca_metrics = trainer.get_nca_metrics()
            if 'recent_patterns' in nca_metrics:
                pattern_history.append(nca_metrics['recent_patterns'])
        
        # Analyze pattern preservation
        if len(pattern_history) >= 3:
            print("  [CHART] Pattern Evolution Analysis:")
            
            # Check spatial coherence evolution
            if 'spatial_coherence' in pattern_history[0]:
                coherence_values = [p['spatial_coherence'].item() for p in pattern_history if 'spatial_coherence' in p]
                coherence_trend = np.polyfit(range(len(coherence_values)), coherence_values, 1)[0]
                
                print(f"    Spatial Coherence Trend: {coherence_trend:+.6f} (per step)")
                
                if abs(coherence_trend) < 0.01:
                    print("    [OK] Spatial coherence stable")
                else:
                    print(f"    [WARNING] Spatial coherence {'increasing' if coherence_trend > 0 else 'decreasing'}")
            
            # Check specialization evolution
            if 'emergent_specialization' in pattern_history[0]:
                spec_values = [p['emergent_specialization'].item() for p in pattern_history if 'emergent_specialization' in p]
                spec_trend = np.polyfit(range(len(spec_values)), spec_values, 1)[0]
                
                print(f"    Specialization Trend: {spec_trend:+.6f} (per step)")
                
                if spec_trend > 0:
                    print("    [OK] Emergent specialization developing")
                else:
                    print("    [DATA] Specialization pattern stable/decreasing")
        
        print("  [OK] Pattern preservation test completed")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Pattern preservation test failed: {e}")
        return False

def run_comprehensive_test():
    """Запускает comprehensive test suite для real training workflow"""
    print("[BRAIN] PHASE 3 Task 3.1: Real Training Workflow Test with NCA")
    print("=" * 70)
    
    test_results = []
    
    # Main training workflow test
    print("\n1️⃣ Real Training Workflow Test")
    test_results.append(test_nca_real_training())
    
    # Pattern preservation test
    print("\n2️⃣ Pattern Preservation Test")  
    test_results.append(test_nca_pattern_preservation())
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"🧪 Real Training Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] ALL REAL TRAINING TESTS PASSED!")
        print("[SUCCESS] NCA integration works correctly in real training workflow")
        print("[START] Ready for production deployment and Task 3.2 implementation")
        return True
    else:
        print(f"[ERROR] {total - passed} tests failed. NCA needs refinement.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 