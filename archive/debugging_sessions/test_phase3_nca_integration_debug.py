#!/usr/bin/env python3
"""
🐛 Debug версия Phase 3 NCA Integration Test

Дополнительная диагностика для выявления проблемы с размерностями
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from typing import Dict, Any

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.neural_cellular_automata import create_nca_config

def debug_dimensions():
    """Диагностика размерностей для понимания проблемы"""
    print("🐛 Debug: Analyzing dimension mismatch...")
    
    try:
        # Create config
        config = EmergentTrainingConfig()
        config.enable_nca = True
        config.cube_dimensions = (8, 8, 6)  # (width=8, height=8, depth=6)
        config.batch_size = 2
        
        print(f"Config cube_dimensions: {config.cube_dimensions}")
        print(f"Config interpretation: width={config.cube_dimensions[0]}, height={config.cube_dimensions[1]}, depth={config.cube_dimensions[2]}")
        
        # Initialize trainer with debug
        trainer = EmergentCubeTrainer(config, device="cpu")
        
        print(f"NCA cube_dimensions: {trainer.nca.cube_dimensions}")
        print(f"NCA state_size: {trainer.nca.state_size}")
        
        # Test surface injection
        batch_size = 2
        surface_size = 8 * 8  # width * height = 64
        surface_embeddings = torch.randn(batch_size, surface_size)
        
        print(f"Surface embeddings shape: {surface_embeddings.shape}")
        
        # Manually call _inject_surface_to_cube to see what happens
        cube_states = trainer._inject_surface_to_cube(surface_embeddings)
        print(f"Cube states shape after injection: {cube_states.shape}")
        
        # Check if this matches NCA expectations
        print(f"Expected NCA input shape: [batch, depth, height, width, state_size]")
        print(f"Expected: [2, 6, 8, 8, 32]")
        print(f"Actual: {list(cube_states.shape)}")
        
        # Test NCA directly with cube_states
        if trainer.nca is not None:
            print("\nTesting NCA directly...")
            print(f"NCA expected cube dimensions: {trainer.nca.cube_dimensions}")
            
            # Create dummy raw_updates with same shape
            raw_updates = torch.randn_like(cube_states)
            
            print(f"Raw updates shape: {raw_updates.shape}")
            
            # Call NCA
            nca_results = trainer.nca(
                current_states=cube_states,
                raw_updates=raw_updates,
                enable_stochastic=True,
                enable_residual=True
            )
            
            print(f"NCA output shape: {nca_results['updated_states'].shape}")
            print("[OK] NCA call successful!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Debug failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Print traceback for detailed analysis
        import traceback
        print("Traceback:")
        traceback.print_exc()
        
        return False

def debug_nca_initialization():
    """Debug NCA initialization specifically"""
    print("\n[MAGNIFY] Debug: NCA Initialization...")
    
    try:
        # Test NCA creation directly
        config = EmergentTrainingConfig()
        config.cube_dimensions = (8, 8, 6)  # (width, height, depth)
        
        # Create NCA config
        nca_config = create_nca_config()
        
        print(f"Base config cube_dimensions: {config.cube_dimensions}")
        print(f"NCA config: {nca_config}")
        
        # Import and create NCA directly
        from training.embedding_trainer.neural_cellular_automata import NeuralCellularAutomata
        
        # The issue might be here - let's check parameter order
        nca = NeuralCellularAutomata(
            config=nca_config,
            cube_dimensions=config.cube_dimensions,  # This is (width, height, depth)
            state_size=32
        )
        
        print(f"NCA initialized with cube_dimensions: {nca.cube_dimensions}")
        
        # Test with correct tensor shape
        batch_size = 2
        # Based on error message, expected shape is [2, 6, 8, 8, 32]
        # This means [batch, depth, height, width, state_size]
        
        depth, height, width = 6, 8, 8  # From error message
        state_size = 32
        
        current_states = torch.randn(batch_size, depth, height, width, state_size)
        raw_updates = torch.randn(batch_size, depth, height, width, state_size)
        
        print(f"Test tensor shapes:")
        print(f"  current_states: {current_states.shape}")
        print(f"  raw_updates: {raw_updates.shape}")
        
        # Call NCA
        results = nca(current_states, raw_updates)
        
        print(f"NCA results shape: {results['updated_states'].shape}")
        print("[OK] Direct NCA test successful!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] NCA initialization debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🐛 Phase 3 NCA Integration Debug")
    print("=" * 50)
    
    # Run debug tests
    debug_dimensions()
    debug_nca_initialization() 