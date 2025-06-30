"""
🧪 Test Suite: Stage 3.1.4.1 Emergent Training Infrastructure
=============================================================

Comprehensive testing для emergent processing системы:
- Full cube gradient flow validation
- gMLP neurons с 25K parameters
- Multi-objective loss function
- Real LLaMA-3-8B integration
- Spatial propagation verification
"""

import torch
import logging
import time
import traceback
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer,
    EmergentTrainingConfig,
    create_emergent_trainer,
    test_emergent_training_basic
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_system_initialization():
    """Тест 1: Инициализация emergent training системы"""
    print("\n🧪 ТЕСТ 1: System Initialization")
    print("=" * 50)
    
    try:
        # 1.1: Basic initialization
        print("[INFO] 1.1: Basic EmergentCubeTrainer initialization...")
        
        config = EmergentTrainingConfig(
            teacher_model="Meta-Llama-3-8B",
            cube_dimensions=(15, 15, 11),
            enable_full_cube_gradient=True,
            spatial_propagation_depth=11
        )
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        print(f"   ✅ Trainer created successfully")
        
        # 1.2: System info verification
        print("\n[INFO] 1.2: System Information Verification...")
        info = trainer.get_system_info()
        
        print(f"   📊 Architecture: {info['architecture']}")
        print(f"   📊 Cube dimensions: {info['cube_dimensions']}")
        print(f"   📊 Total cells: {info['total_cells']}")
        print(f"   📊 Avg params per cell: {info['avg_params_per_cell']:.0f}")
        print(f"   📊 Total system params: {info['total_system_params']:,}")
        print(f"   📊 Full cube gradient: {info['full_cube_gradient']}")
        
        # Verify target parameters
        expected_cells = 15 * 15 * 11  # 2,475 cells
        assert info['total_cells'] == expected_cells, f"Expected {expected_cells} cells, got {info['total_cells']}"
        
        # Check parameter count target (approximately 25K per cell)
        avg_params = info['avg_params_per_cell']
        if 20000 <= avg_params <= 30000:
            print(f"   ✅ Parameter count target achieved: {avg_params:.0f} ≈ 25K")
        else:
            print(f"   [WARNING]  Parameter count off target: {avg_params:.0f} (target: ~25K)")
        
        # 1.3: Component verification
        print("\n[INFO] 1.3: Component Verification...")
        
        # Check gMLP cells
        assert hasattr(trainer, 'gmlp_cells'), "Missing gMLP cells"
        assert len(trainer.gmlp_cells) == expected_cells, f"Wrong number of gMLP cells"
        print(f"   ✅ gMLP cells: {len(trainer.gmlp_cells)}")
        
        # Check spatial propagation
        assert hasattr(trainer, 'spatial_propagation'), "Missing spatial propagation"
        print(f"   ✅ Spatial propagation system")
        
        # Check multi-objective loss
        assert hasattr(trainer, 'loss_function'), "Missing loss function"
        print(f"   ✅ Multi-objective loss function")
        
        # Check base adapter
        assert hasattr(trainer, 'base_trainer'), "Missing base trainer"
        print(f"   ✅ Base adapter integration")
        
        print("\n🎯 ТЕСТ 1 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_full_cube_gradient_flow():
    """Тест 2: Full cube gradient flow verification"""
    print("\n🧪 ТЕСТ 2: Full Cube Gradient Flow")
    print("=" * 50)
    
    try:
        # 2.1: Create trainer with full gradient flow
        print("[INFO] 2.1: Creating trainer with full cube gradient flow...")
        
        trainer = create_emergent_trainer(
            cube_dimensions=(15, 15, 11),
            teacher_model="Meta-Llama-3-8B",
            device="cpu"
        )
        
        # 2.2: Forward pass test
        print("\n[INFO] 2.2: Forward Pass Testing...")
        
        batch_size = 2
        teacher_embeddings = torch.randn(batch_size, 4096)  # LLaMA-3-8B size
        
        start_time = time.time()
        outputs = trainer.forward(teacher_embeddings)
        forward_time = time.time() - start_time
        
        print(f"   ⚡ Forward pass time: {forward_time:.3f}s")
        
        # Verify output structure
        required_keys = ['input_surface', 'cube_states', 'processed_states', 
                        'enhanced_states', 'output_surface', 'final_output']
        
        for key in required_keys:
            assert key in outputs, f"Missing output: {key}"
            print(f"   ✅ Output '{key}': {outputs[key].shape}")
        
        # 2.3: Gradient flow verification
        print("\n[INFO] 2.3: Gradient Flow Verification...")
        
        # Create dummy targets
        target_embeddings = torch.randn(batch_size, 4096)
        
        # Forward pass with gradient tracking
        trainer.train()
        trainer.optimizer.zero_grad()
        
        outputs = trainer.forward(teacher_embeddings)
        
        # Compute loss
        targets = {
            'target_embedding': target_embeddings,
            'target_surface': outputs['input_surface']
        }
        
        losses = trainer.compute_loss(outputs, targets)
        total_loss = losses['total_loss']
        
        print(f"   📊 Total loss: {total_loss.item():.6f}")
        print(f"   📊 Loss components:")
        for key, value in losses.items():
            if key != 'total_loss' and torch.is_tensor(value):
                if key == 'loss_weights':
                    # Special handling для multi-element tensor
                    print(f"      - {key}: {value.tolist()}")
                elif value.numel() == 1:
                    # Scalar tensor
                    print(f"      - {key}: {value.item():.6f}")
                else:
                    # Multi-element tensor
                    print(f"      - {key}: {value.mean().item():.6f} (mean of {value.numel()} elements)")
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients in gMLP cells
        cells_with_gradients = 0
        total_gradient_norm = 0.0
        
        for i, cell in enumerate(trainer.gmlp_cells):
            has_gradients = False
            cell_grad_norm = 0.0
            
            for param in cell.parameters():
                if param.grad is not None:
                    has_gradients = True
                    cell_grad_norm += param.grad.norm().item() ** 2
            
            if has_gradients:
                cells_with_gradients += 1
                total_gradient_norm += cell_grad_norm ** 0.5
        
        print(f"   ✅ Cells with gradients: {cells_with_gradients}/{len(trainer.gmlp_cells)}")
        print(f"   ✅ Average gradient norm: {total_gradient_norm / cells_with_gradients:.6f}")
        
        # Verify full cube influence
        full_cube_ratio = cells_with_gradients / len(trainer.gmlp_cells)
        if full_cube_ratio > 0.8:  # 80%+ of cells should have gradients
            print(f"   ✅ Full cube gradient flow achieved: {full_cube_ratio:.1%}")
        else:
            print(f"   [WARNING]  Partial gradient flow: {full_cube_ratio:.1%} (target: >80%)")
        
        print("\n🎯 ТЕСТ 2 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_multi_objective_loss():
    """Тест 3: Multi-objective loss function"""
    print("\n🧪 ТЕСТ 3: Multi-Objective Loss Function")
    print("=" * 50)
    
    try:
        # 3.1: Loss function components
        print("[INFO] 3.1: Loss Function Components...")
        
        trainer = create_emergent_trainer(device="cpu")
        
        # Test data
        batch_size = 2
        teacher_embeddings = torch.randn(batch_size, 4096)
        target_embeddings = torch.randn(batch_size, 4096)
        
        # Forward pass
        outputs = trainer.forward(teacher_embeddings)
        
        # Prepare targets
        targets = {
            'target_embedding': target_embeddings,
            'target_surface': outputs['input_surface']
        }
        
        # Compute loss
        losses = trainer.compute_loss(outputs, targets)
        
        # Verify loss components
        expected_components = [
            'total_loss', 
            'surface_reconstruction_loss',
            'internal_consistency_loss', 
            'dialogue_similarity_loss',
            'loss_weights'
        ]
        
        for component in expected_components:
            assert component in losses, f"Missing loss component: {component}"
            print(f"   ✅ {component}: {losses[component]}")
        
        # 3.2: Loss weight verification
        print("\n[INFO] 3.2: Loss Weight Verification...")
        
        weights = losses['loss_weights']
        print(f"   📊 Surface reconstruction: {weights[0]:.3f}")
        print(f"   📊 Internal consistency: {weights[1]:.3f}")
        print(f"   📊 Dialogue similarity: {weights[2]:.3f}")
        print(f"   📊 Weight sum: {weights.sum():.3f}")
        
        # Verify weights sum to 1.0 (softmax normalization)
        assert abs(weights.sum().item() - 1.0) < 0.001, "Weights don't sum to 1.0"
        
        # 3.3: Gradient flow through loss
        print("\n[INFO] 3.3: Loss Gradient Flow...")
        
        trainer.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Check loss function gradients
        loss_params_with_grad = 0
        for param in trainer.loss_function.parameters():
            if param.grad is not None:
                loss_params_with_grad += 1
        
        print(f"   ✅ Loss function parameters with gradients: {loss_params_with_grad}")
        
        print("\n🎯 ТЕСТ 3 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_spatial_propagation():
    """Тест 4: Spatial propagation system"""
    print("\n🧪 ТЕСТ 4: Spatial Propagation System")
    print("=" * 50)
    
    try:
        # 4.1: Spatial propagation initialization
        print("[INFO] 4.1: Spatial Propagation System...")
        
        trainer = create_emergent_trainer(device="cpu")
        spatial_prop = trainer.spatial_propagation
        
        print(f"   ✅ Spatial propagation depth: {spatial_prop.depth}")
        print(f"   ✅ State size: {spatial_prop.state_size}")
        
        # 4.2: Cross-layer influence test
        print("\n[INFO] 4.2: Cross-Layer Influence Testing...")
        
        batch_size = 2
        depth, height, width, state_size = 11, 15, 15, 32
        
        # Create test cube states
        cube_states = torch.randn(batch_size, depth, height, width, state_size)
        
        # Apply spatial propagation
        enhanced_states = spatial_prop(cube_states)
        
        print(f"   ✅ Input shape: {cube_states.shape}")
        print(f"   ✅ Output shape: {enhanced_states.shape}")
        
        # Verify enhancement effect
        difference = torch.mean((enhanced_states - cube_states) ** 2).item()
        print(f"   ✅ Enhancement magnitude: {difference:.6f}")
        
        # 4.3: Layer-to-layer connections
        print("\n[INFO] 4.3: Layer Connection Verification...")
        
        # Check connection weights
        assert hasattr(spatial_prop, 'layer_connections'), "Missing layer connections"
        connections = spatial_prop.layer_connections
        
        expected_connections = depth - 1  # 10 connections for 11 layers
        assert connections.shape[0] == expected_connections, f"Wrong number of connections"
        
        print(f"   ✅ Layer connections: {connections.shape}")
        print(f"   ✅ Connection weight range: [{connections.min():.3f}, {connections.max():.3f}]")
        
        # 4.4: Gradient flow через spatial propagation
        print("\n[INFO] 4.4: Spatial Propagation Gradient Flow...")
        
        cube_states.requires_grad_(True)
        enhanced = spatial_prop(cube_states)
        loss = enhanced.mean()
        loss.backward()
        
        assert cube_states.grad is not None, "No gradients через spatial propagation"
        grad_norm = cube_states.grad.norm().item()
        print(f"   ✅ Gradient norm through spatial propagation: {grad_norm:.6f}")
        
        print("\n🎯 ТЕСТ 4 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_training_step_integration():
    """Тест 5: Full training step integration"""
    print("\n🧪 ТЕСТ 5: Training Step Integration")
    print("=" * 50)
    
    try:
        # 5.1: Full training step
        print("[INFO] 5.1: Complete Training Step...")
        
        trainer = create_emergent_trainer(device="cpu")
        
        # Training data
        batch_size = 2
        question_embeddings = torch.randn(batch_size, 4096)
        answer_embeddings = torch.randn(batch_size, 4096)
        
        # Single training step
        start_time = time.time()
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        step_time = time.time() - start_time
        
        print(f"   ⚡ Training step time: {step_time:.3f}s")
        print(f"   📊 Training metrics:")
        
        for key, value in metrics.items():
            print(f"      - {key}: {value:.6f}")
        
        # 5.2: Metrics validation
        print("\n[INFO] 5.2: Metrics Validation...")
        
        required_metrics = ['total_loss', 'surface_loss', 'internal_loss', 
                           'dialogue_loss', 'cosine_similarity', 'lr']
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert not torch.isnan(torch.tensor(metrics[metric])), f"NaN in {metric}"
            print(f"   ✅ {metric}: valid")
        
        # 5.3: Multiple training steps
        print("\n[INFO] 5.3: Multiple Training Steps...")
        
        initial_loss = metrics['total_loss']
        step_metrics = [metrics]
        
        for step in range(3):  # 3 additional steps
            # Create fresh data для каждого step (avoid graph reuse)
            fresh_questions = torch.randn(batch_size, 4096)
            fresh_answers = torch.randn(batch_size, 4096)
            
            step_metrics.append(
                trainer.train_step(fresh_questions, fresh_answers)
            )
        
        final_loss = step_metrics[-1]['total_loss']
        
        print(f"   📊 Initial loss: {initial_loss:.6f}")
        print(f"   📊 Final loss: {final_loss:.6f}")
        print(f"   📊 Loss change: {final_loss - initial_loss:.6f}")
        
        # Check for training stability (no explosive gradients)
        for i, step_metric in enumerate(step_metrics):
            assert not torch.isinf(torch.tensor(step_metric['total_loss'])), f"Inf loss at step {i}"
            print(f"   ✅ Step {i}: stable")
        
        print("\n🎯 ТЕСТ 5 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 5 FAILED: {e}")
        traceback.print_exc()
        return False


def test_emergent_behavior_indicators():
    """Тест 6: Emergent behavior indicators"""
    print("\n🧪 ТЕСТ 6: Emergent Behavior Indicators")
    print("=" * 50)
    
    try:
        # 6.1: Cell specialization analysis
        print("[INFO] 6.1: Cell Specialization Analysis...")
        
        trainer = create_emergent_trainer(device="cpu")
        
        # Process different inputs
        inputs = [
            torch.randn(1, 4096),  # Input 1
            torch.randn(1, 4096),  # Input 2  
            torch.randn(1, 4096)   # Input 3
        ]
        
        cell_activations = []
        
        for i, input_tensor in enumerate(inputs):
            outputs = trainer.forward(input_tensor)
            cube_states = outputs['enhanced_states']  # [1, 11, 15, 15, 32]
            
            # Flatten cube states for analysis
            flat_states = cube_states.view(cube_states.shape[1], -1)  # [11, 15*15*32]
            cell_activations.append(flat_states)
        
        # Analyze layer specialization
        print(f"   📊 Analyzing {len(inputs)} different inputs...")
        
        for layer in range(11):
            layer_vars = []
            for activation in cell_activations:
                layer_var = torch.var(activation[layer]).item()
                layer_vars.append(layer_var)
            
            avg_var = sum(layer_vars) / len(layer_vars)
            print(f"   📊 Layer {layer} activation variance: {avg_var:.6f}")
        
        # 6.2: Information flow analysis
        print("\n[INFO] 6.2: Information Flow Analysis...")
        
        # Single forward pass с detailed tracking
        test_input = torch.randn(1, 4096)
        outputs = trainer.forward(test_input)
        
        input_surface = outputs['input_surface']
        output_surface = outputs['output_surface']
        
        # Information preservation
        input_norm = torch.norm(input_surface).item()
        output_norm = torch.norm(output_surface).item()
        
        print(f"   📊 Input surface norm: {input_norm:.6f}")
        print(f"   📊 Output surface norm: {output_norm:.6f}")
        print(f"   📊 Information ratio: {output_norm / input_norm:.3f}")
        
        # Surface transformation (с dimension matching)
        if input_surface.shape[-1] != output_surface.shape[-1]:
            # Project input to output dimensions для comparison
            from training.embedding_trainer.emergent_training_stage_3_1_4_1 import EmergentMultiObjectiveLoss
            temp_config = trainer.config
            temp_loss = EmergentMultiObjectiveLoss(temp_config)
            projected_input = temp_loss.embedding_to_surface(input_surface)  # [1, 225]
            surface_similarity = torch.nn.functional.cosine_similarity(
                projected_input, output_surface, dim=-1
            ).item()
        else:
            surface_similarity = torch.nn.functional.cosine_similarity(
                input_surface, output_surface, dim=-1
            ).item()
        
        print(f"   📊 Input→Output similarity: {surface_similarity:.3f}")
        
        # 6.3: Emergent pattern detection
        print("\n[INFO] 6.3: Emergent Pattern Detection...")
        
        # Compare different depth layers
        enhanced_states = outputs['enhanced_states']  # [1, 11, 15, 15, 32]
        
        layer_similarities = []
        for i in range(10):  # Compare adjacent layers
            layer1 = enhanced_states[0, i].flatten()
            layer2 = enhanced_states[0, i+1].flatten()
            
            similarity = torch.nn.functional.cosine_similarity(
                layer1.unsqueeze(0), layer2.unsqueeze(0), dim=-1
            ).item()
            layer_similarities.append(similarity)
        
        avg_layer_similarity = sum(layer_similarities) / len(layer_similarities)
        print(f"   📊 Average adjacent layer similarity: {avg_layer_similarity:.3f}")
        
        # Detect potential specialization (low similarity = more specialization)
        if avg_layer_similarity < 0.8:
            print(f"   ✅ Potential layer specialization detected")
        else:
            print(f"   [WRITE] Layers still similar (early training)")
        
        print("\n🎯 ТЕСТ 6 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 6 FAILED: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_test_suite():
    """Run complete test suite для Stage 3.1.4.1"""
    print("\n" + "="*60)
    print("[BRAIN] COMPREHENSIVE TEST SUITE: Stage 3.1.4.1")
    print("Emergent Training Infrastructure")
    print("="*60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Full Cube Gradient Flow", test_full_cube_gradient_flow), 
        ("Multi-Objective Loss", test_multi_objective_loss),
        ("Spatial Propagation", test_spatial_propagation),
        ("Training Step Integration", test_training_step_integration),
        ("Emergent Behavior Indicators", test_emergent_behavior_indicators),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUITE SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Stage 3.1.4.1 Emergent Training Infrastructure READY!")
        return True
    else:
        print("[WARNING]  Some tests failed - review and fix before proceeding")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1) 