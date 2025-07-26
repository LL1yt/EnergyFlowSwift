#!/usr/bin/env python3
"""
Test script to validate the training fixes
==========================================

This script tests the key fixes applied to resolve the training issues:
1. Gradient computation
2. Tokenizer encoding
3. Loss stability
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.training.energy_trainer import EnergyTrainer
from energy_flow.text_bridge import create_text_to_cube_encoder, create_cube_to_text_decoder

def test_gradient_computation():
    """Test gradient computation in training step"""
    print("🧪 Testing gradient computation...")
    
    config = create_debug_config()
    config.text_bridge_enabled = True
    config.text_loss_weight = 0.5
    set_energy_config(config)
    
    trainer = EnergyTrainer(config)
    
    # Test data
    input_texts = ["Hello world", "Test input"]
    target_texts = ["Hello response", "Test response"]
    teacher_input = torch.randn(2, 768, requires_grad=True)
    teacher_target = torch.randn(2, 768, requires_grad=True)
    
    # Test training step
    try:
        metrics = trainer.train_step(input_texts, target_texts, teacher_input, teacher_target)
        print(f"✅ Gradient computation test passed")
        print(f"   - Loss: {metrics['total_loss']:.4f}")
        print(f"   - Energy loss: {metrics['energy_loss']:.4f}")
        print(f"   - Text loss: {metrics['text_loss']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Gradient computation test failed: {e}")
        return False

def test_tokenizer_encoding():
    """Test tokenizer encoding in text bridge"""
    print("🧪 Testing tokenizer encoding...")
    
    config = create_debug_config()
    set_energy_config(config)
    
    # Test encoder
    encoder = create_text_to_cube_encoder(config)
    
    test_texts = [
        "Hello world",
        "This is a test sentence",
        "",  # Empty string
        "Short",
        "A much longer test sentence with multiple words to test tokenization"
    ]
    
    try:
        embeddings = encoder.encode_text(test_texts)
        print(f"✅ Tokenizer encoding test passed")
        print(f"   - Input texts: {len(test_texts)}")
        print(f"   - Output shape: {embeddings.shape}")
        print(f"   - Output range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        return True
    except Exception as e:
        print(f"❌ Tokenizer encoding test failed: {e}")
        return False

def test_decoder_functionality():
    """Test decoder functionality"""
    print("🧪 Testing decoder functionality...")
    
    config = create_debug_config()
    set_energy_config(config)
    
    decoder = create_cube_to_text_decoder(config)
    
    # Test surface embeddings
    surface_dim = config.lattice_width * config.lattice_height
    test_embeddings = torch.randn(2, surface_dim)
    
    try:
        decoded_texts = decoder.decode_surface(test_embeddings, max_length=32)
        print(f"✅ Decoder functionality test passed")
        print(f"   - Input shape: {test_embeddings.shape}")
        print(f"   - Output texts: {len(decoded_texts)}")
        for i, text in enumerate(decoded_texts):
            print(f"   - Text {i}: '{text[:50]}...'")
        return True
    except Exception as e:
        print(f"❌ Decoder functionality test failed: {e}")
        return False

def test_loss_stability():
    """Test loss computation stability"""
    print("🧪 Testing loss stability...")
    
    config = create_debug_config()
    config.text_bridge_enabled = True
    set_energy_config(config)
    
    trainer = EnergyTrainer(config)
    
    # Test with various edge cases
    test_cases = [
        # Normal case
        (["Hello"], ["World"], torch.randn(1, 768), torch.randn(1, 768)),
        # Empty texts
        (["", ""], ["", ""], torch.randn(2, 768), torch.randn(2, 768)),
        # Single character
        (["A"], ["B"], torch.randn(1, 768), torch.randn(1, 768)),
    ]
    
    all_passed = True
    for i, (input_texts, target_texts, teacher_input, teacher_target) in enumerate(test_cases):
        try:
            teacher_input.requires_grad_(True)
            teacher_target.requires_grad_(True)
            
            metrics = trainer.train_step(input_texts, target_texts, teacher_input, teacher_target)
            
            # Check for NaN or inf losses
            if any(torch.isnan(torch.tensor([metrics[k]])) or torch.isinf(torch.tensor([metrics[k]])) 
                   for k in ['total_loss', 'energy_loss', 'text_loss']):
                print(f"❌ Loss stability test case {i+1} failed: NaN/Inf detected")
                all_passed = False
            else:
                print(f"✅ Loss stability test case {i+1} passed")
                
        except Exception as e:
            print(f"❌ Loss stability test case {i+1} failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("🔬 Running comprehensive training fixes validation...")
    print("=" * 60)
    
    tests = [
        ("Gradient Computation", test_gradient_computation),
        ("Tokenizer Encoding", test_tokenizer_encoding),
        ("Decoder Functionality", test_decoder_functionality),
        ("Loss Stability", test_loss_stability),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All fixes validated successfully!")
        return True
    else:
        print("⚠️ Some tests failed - additional debugging may be needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)