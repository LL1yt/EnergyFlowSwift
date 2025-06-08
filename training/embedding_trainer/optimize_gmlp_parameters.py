#!/usr/bin/env python3
"""
🔧 gMLP Parameter Optimization для 25K Target
==============================================

Находим оптимальную конфигурацию gMLP клетки для точного достижения ~25,000 параметров
"""

import sys
import os

# Добавляем корневую директорию в Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from core.cell_prototype.architectures.gmlp_cell import GatedMLPCell
from typing import Dict, Any


def count_parameters(model) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def optimize_gmlp_config() -> Dict[str, Any]:
    """Find optimal gMLP configuration для 25K parameters"""
    
    print("🔧 Optimizing gMLP Configuration для 25K Parameters")
    print("=" * 60)
    
    target_params = 25000
    best_config = None
    best_params = float('inf')
    best_diff = float('inf')
    
    # Test different configurations
    configurations = []
    
    # Try different hidden_dim values
    for hidden_dim in [32, 48, 64, 72, 80]:
        for memory_dim in [16, 24, 32]:
            for use_memory in [True, False]:
                config = {
                    'state_size': 32,
                    'neighbor_count': 6,
                    'hidden_dim': hidden_dim,
                    'external_input_size': 12,
                    'use_memory': use_memory,
                    'memory_dim': memory_dim if use_memory else 32,
                    'activation': 'gelu',
                    'dropout': 0.1
                }
                
                try:
                    # Create cell and count parameters
                    cell = GatedMLPCell(**config)
                    param_count = count_parameters(cell)
                    
                    diff = abs(param_count - target_params)
                    ratio = param_count / target_params
                    
                    configurations.append({
                        'config': config,
                        'params': param_count,
                        'diff': diff,
                        'ratio': ratio
                    })
                    
                    print(f"hidden_dim={hidden_dim:2d}, memory_dim={memory_dim:2d}, "
                          f"memory={use_memory}, params={param_count:6,d}, "
                          f"ratio={ratio:.2f}x, diff={diff:6,d}")
                    
                    # Track best configuration
                    if diff < best_diff:
                        best_diff = diff
                        best_config = config.copy()
                        best_params = param_count
                        
                except Exception as e:
                    print(f"❌ Failed config hidden_dim={hidden_dim}: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    if best_config:
        print(f"✅ Best Configuration:")
        for key, value in best_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n📊 Parameter Analysis:")
        print(f"   Target: {target_params:,} parameters")
        print(f"   Achieved: {best_params:,} parameters")
        print(f"   Difference: {best_diff:,} parameters")
        print(f"   Ratio: {best_params/target_params:.3f}x target")
        
        if best_diff <= 2000:  # Within 2K parameters
            print(f"🎉 EXCELLENT: Within 2K of target!")
        elif best_diff <= 5000:  # Within 5K parameters
            print(f"✅ GOOD: Within 5K of target")
        else:
            print(f"⚠️  ACCEPTABLE: {best_diff:,} parameters from target")
    
    # Show top 5 configurations
    print(f"\n📋 Top 5 Configurations:")
    configurations.sort(key=lambda x: x['diff'])
    
    for i, config_result in enumerate(configurations[:5]):
        config = config_result['config']
        params = config_result['params']
        ratio = config_result['ratio']
        diff = config_result['diff']
        
        print(f"   {i+1}. hidden_dim={config['hidden_dim']}, "
              f"memory_dim={config['memory_dim']}, "
              f"memory={config['use_memory']}, "
              f"params={params:,} ({ratio:.2f}x, diff={diff:,})")
    
    return best_config


def test_optimized_config(config: Dict[str, Any]):
    """Test optimized configuration"""
    print(f"\n🧪 Testing Optimized Configuration")
    print("=" * 40)
    
    try:
        # Create optimized cell
        cell = GatedMLPCell(**config)
        param_count = count_parameters(cell)
        
        print(f"✅ Cell created successfully")
        print(f"📊 Parameters: {param_count:,}")
        
        # Test forward pass
        batch_size = 2
        neighbor_states = torch.randn(batch_size, 6, 32)
        own_state = torch.randn(batch_size, 32)
        external_input = torch.randn(batch_size, 12)
        
        output = cell(neighbor_states, own_state, external_input)
        
        print(f"✅ Forward pass successful")
        print(f"📊 Input shape: {own_state.shape}")
        print(f"📊 Output shape: {output.shape}")
        
        # Test gradient flow
        loss = output.mean()
        loss.backward()
        
        grad_params = sum(1 for p in cell.parameters() if p.grad is not None)
        total_params = len(list(cell.parameters()))
        
        print(f"✅ Gradient flow successful")
        print(f"📊 Parameters with gradients: {grad_params}/{total_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def calculate_full_system_size(config: Dict[str, Any]):
    """Calculate full system parameter count"""
    print(f"\n🖥️  Full System Analysis")
    print("=" * 40)
    
    # Single cell parameters
    cell = GatedMLPCell(**config)
    cell_params = count_parameters(cell)
    
    # System configuration
    cube_dimensions = (15, 15, 11)
    total_cells = cube_dimensions[0] * cube_dimensions[1] * cube_dimensions[2]
    
    # Calculate system size
    lattice_params = cell_params * total_cells
    
    # Add other system components (rough estimates)
    adapter_params = 4096 * 225  # LLaMA → surface adapter
    spatial_propagation_params = 11 * 32 * 32  # Cross-layer connections
    loss_function_params = 100  # Minimal
    
    total_system_params = lattice_params + adapter_params + spatial_propagation_params + loss_function_params
    
    print(f"📊 Single Cell: {cell_params:,} parameters")
    print(f"📊 Total Cells: {total_cells:,} cells")
    print(f"📊 Lattice: {lattice_params:,} parameters")
    print(f"📊 Adapter: {adapter_params:,} parameters")
    print(f"📊 Spatial Prop: {spatial_propagation_params:,} parameters")
    print(f"📊 Total System: {total_system_params:,} parameters")
    
    # Memory estimation
    memory_gb = total_system_params * 4 / (1024**3)  # 4 bytes per float32
    print(f"💾 Estimated Memory: {memory_gb:.2f} GB")
    
    return {
        'cell_params': cell_params,
        'total_cells': total_cells,
        'lattice_params': lattice_params,
        'total_system_params': total_system_params,
        'memory_gb': memory_gb
    }


if __name__ == "__main__":
    # Find optimal configuration
    optimal_config = optimize_gmlp_config()
    
    if optimal_config:
        # Test configuration
        test_success = test_optimized_config(optimal_config)
        
        if test_success:
            # Analyze full system
            system_analysis = calculate_full_system_size(optimal_config)
            
            print(f"\n🎯 FINAL RECOMMENDATION:")
            print(f"✅ Use configuration: {optimal_config}")
            print(f"✅ System feasible: {system_analysis['memory_gb']:.1f}GB memory")
            print(f"✅ Ready для emergent training!")
        else:
            print(f"\n❌ Optimal configuration failed testing")
    else:
        print(f"\n❌ No suitable configuration found") 