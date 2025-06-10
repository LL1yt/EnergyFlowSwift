#!/usr/bin/env python3
"""
[CONFIG] Debug Test для NCA Training Workflow

Простая диагностика проблемы с trainer.forward()
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)

def debug_trainer_forward():
    """Отладка trainer.forward() с разными входными данными"""
    print("[CONFIG] Debug: Trainer Forward Pass")
    print("=" * 50)
    
    try:
        # Setup simple configuration
        config = EmergentTrainingConfig()
        config.enable_nca = False  # Start without NCA
        config.cube_dimensions = (3, 3, 2)  # Minimal
        config.batch_size = 1
        config.mixed_precision = False
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        print("[OK] Trainer initialized")
        
        # Test 1: Surface-size input (what trainer expects)
        print("\n1️⃣ Testing surface-size input...")
        surface_size = 3 * 3  # 9
        surface_input = torch.randn(1, surface_size)
        print(f"Input shape: {surface_input.shape}")
        
        outputs = trainer.forward(surface_input)
        print(f"Outputs type: {type(outputs)}")
        if outputs is not None:
            print(f"Outputs keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not dict'}")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if value is not None:
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: None")
        else:
            print("[ERROR] Forward returned None!")
            
        # Test 2: 4096D input (что пытался передать before)
        print("\n2️⃣ Testing 4096D input...")
        large_input = torch.randn(1, 4096)
        print(f"Input shape: {large_input.shape}")
        
        try:
            outputs2 = trainer.forward(large_input)
            print(f"Outputs2 type: {type(outputs2)}")
            if outputs2 is not None and isinstance(outputs2, dict):
                for key, value in outputs2.items():
                    if value is not None:
                        print(f"  {key}: shape={value.shape}")
                    else:
                        print(f"  {key}: None")
        except Exception as e:
            print(f"[ERROR] 4096D input failed: {e}")
        
        # Test 3: Check trainer components
        print("\n3️⃣ Checking trainer components...")
        print(f"Has embedding_to_surface: {hasattr(trainer, 'embedding_to_surface')}")
        print(f"Has surface_to_embedding: {hasattr(trainer, 'surface_to_embedding')}")
        print(f"Has lattice_3d: {hasattr(trainer, 'lattice_3d')}")
        
        # Test 4: Direct lattice call
        print("\n4️⃣ Testing direct lattice call...")
        if hasattr(trainer, 'lattice_3d'):
            try:
                # Try injecting surface to cube
                cube_states = trainer._inject_surface_to_cube(surface_input)
                print(f"Cube states shape: {cube_states.shape}")
                
                # Direct lattice forward
                processed_cube = trainer.lattice_3d(cube_states)
                print(f"Processed cube shape: {processed_cube.shape}")
                
            except Exception as e:
                print(f"[ERROR] Direct lattice call failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_nca_trainer():
    """Отладка NCA trainer"""
    print("\n[BRAIN] Debug: NCA Trainer")
    print("=" * 50)
    
    try:
        # NCA configuration
        config = EmergentTrainingConfig()
        config.enable_nca = True  # Enable NCA
        config.cube_dimensions = (3, 3, 2)
        config.batch_size = 1
        config.mixed_precision = False
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        print("[OK] NCA Trainer initialized")
        
        # Check NCA components
        print(f"NCA enabled: {hasattr(trainer, 'nca_system')}")
        if hasattr(trainer, 'nca_system'):
            print(f"NCA system type: {type(trainer.nca_system)}")
        
        # Test forward pass
        surface_size = 3 * 3
        surface_input = torch.randn(1, surface_size)
        print(f"Input shape: {surface_input.shape}")
        
        outputs = trainer.forward(surface_input)
        print(f"NCA Outputs type: {type(outputs)}")
        
        if outputs is not None and isinstance(outputs, dict):
            for key, value in outputs.items():
                if value is not None:
                    print(f"  {key}: shape={value.shape}")
                else:
                    print(f"  {key}: None")
        
        # Check NCA metrics
        nca_metrics = trainer.get_nca_metrics()
        print(f"NCA metrics status: {nca_metrics.get('status', 'No status')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] NCA Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Тест loss computation с правильными данными"""
    print("\n[DATA] Debug: Loss Computation")
    print("=" * 50)
    
    try:
        config = EmergentTrainingConfig()
        config.enable_nca = False
        config.cube_dimensions = (3, 3, 2)
        config.batch_size = 1
        config.mixed_precision = False
        
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        # Create valid data
        surface_size = 3 * 3
        surface_input = torch.randn(1, surface_size)
        
        # Forward pass
        outputs = trainer.forward(surface_input)
        
        if outputs is not None:
            # Create proper targets
            targets = {
                'surface_input': surface_input,
                'target_embedding': torch.randn(1, surface_size)  # Same size as input
            }
            
            print("Outputs available:")
            for key, value in outputs.items():
                if value is not None:
                    print(f"  {key}: {value.shape}")
            
            print("Targets:")
            for key, value in targets.items():
                print(f"  {key}: {value.shape}")
            
            # Try loss computation
            loss_results = trainer.compute_loss(outputs, targets)
            print(f"Loss computation successful!")
            print(f"Total loss: {loss_results['total_loss'].item():.6f}")
            
            return True
        else:
            print("[ERROR] Forward pass returned None")
            return False
        
    except Exception as e:
        print(f"[ERROR] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_debug_suite():
    """Запуск полной отладочной suite"""
    print("[CONFIG] NCA Training Debug Suite")
    print("=" * 60)
    
    results = []
    
    # Standard trainer debug
    print("\n[MAGNIFY] Standard Trainer Debug")
    results.append(debug_trainer_forward())
    
    # NCA trainer debug  
    print("\n[MAGNIFY] NCA Trainer Debug")
    results.append(debug_nca_trainer())
    
    # Loss computation debug
    print("\n[MAGNIFY] Loss Computation Debug")
    results.append(test_loss_computation())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 60)
    print(f"🧪 Debug Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] All debug tests passed! Ready for training workflow")
    else:
        print(f"[ERROR] {total - passed} tests failed. Need fixes before training")
    
    return passed == total

if __name__ == "__main__":
    success = run_debug_suite()
    sys.exit(0 if success else 1) 