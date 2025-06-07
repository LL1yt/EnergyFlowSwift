"""
🚀 RET v2.1 ULTRA-COMPACT - QUICK Test Suite

QUICK VALIDATION:
- ✅ Parameter target (CRITICAL)
- ⚡ Basic functionality  
- 🚀 RTX 5090 compatibility (minimal)
- 📊 Basic performance check

OPTIMIZED FOR SPEED: <30 seconds execution time
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Any

# Import our RET v2.1
from resource_efficient_decoder_v2_1 import (
    ResourceEfficientDecoderV21,
    RETConfigV21,
    create_ultra_compact_decoder
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_parameter_target():
    """🎯 QUICK TEST 1: Parameter Target (CRITICAL)"""
    print("🎯 Testing Parameter Target...")
    
    decoder = create_ultra_compact_decoder()
    param_count = decoder._count_parameters()
    target = decoder.config.target_parameters
    
    success = param_count <= target
    efficiency = (target - param_count) / target * 100
    
    print(f"   Parameters: {param_count:,} / {target:,}")
    print(f"   Efficiency: {efficiency:.1f}% under target")
    print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    return success


def test_basic_functionality():
    """⚡ QUICK TEST 2: Basic Functionality"""
    print("⚡ Testing Basic Functionality...")
    
    try:
        decoder = create_ultra_compact_decoder()
        test_embedding = torch.randn(768)
        
        # Single generation test
        result = decoder.decode(test_embedding, max_length=3, temperature=0.8)
        
        success = isinstance(result, str) and len(result) > 0
        print(f"   Generation: {result}")
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"   Error: {e}")
        print("   Result: ❌ FAILED")
        return False


def test_rtx_5090_quick():
    """🚀 QUICK TEST 3: RTX 5090 Quick Check"""
    print("🚀 Testing RTX 5090 Quick Check...")
    
    if not torch.cuda.is_available():
        print("   CUDA not available - SKIPPED")
        return True
    
    try:
        device = torch.device('cuda')
        decoder = create_ultra_compact_decoder()
        decoder.to(device)
        
        test_embedding = torch.randn(768, device=device)
        
        # Quick GPU test
        with torch.no_grad():
            result = decoder.decode(test_embedding, max_length=2)
        
        success = isinstance(result, str)
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Generation: {result}")
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"   GPU Error: {e}")
        print("   Result: ❌ FAILED")
        return False


def test_speed_quick():
    """📊 QUICK TEST 4: Speed Quick Check"""
    print("📊 Testing Speed Quick Check...")
    
    try:
        decoder = create_ultra_compact_decoder()
        test_embedding = torch.randn(768)
        if torch.cuda.is_available():
            decoder = decoder.cuda()
            test_embedding = test_embedding.cuda()
        
        # Quick speed test (only 3 runs)
        times = []
        for _ in range(3):
            start_time = time.time()
            _ = decoder.decode(test_embedding, max_length=5)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        target_time = 0.100  # 100ms target (more lenient)
        success = avg_time < target_time
        
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Target: <{target_time:.3f}s")
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"   Speed Error: {e}")
        print("   Result: ❌ FAILED")
        return False


def test_model_info():
    """📋 QUICK TEST 5: Model Info"""
    print("📋 Testing Model Info...")
    
    try:
        decoder = create_ultra_compact_decoder()
        model_info = decoder.get_model_info()
        
        required_keys = ['architecture', 'version', 'parameters', 'parameter_target_achieved']
        success = all(key in model_info for key in required_keys)
        
        print(f"   Architecture: {model_info.get('architecture', 'N/A')}")
        print(f"   Version: {model_info.get('version', 'N/A')}")
        print(f"   Target achieved: {model_info.get('parameter_target_achieved', False)}")
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"   Model Info Error: {e}")
        print("   Result: ❌ FAILED")
        return False


def run_quick_test_suite():
    """🚀 Run Quick Test Suite"""
    
    print("🧪" + "="*50)
    print("🚀 RET v2.1 ULTRA-COMPACT - QUICK TEST SUITE")
    print("⏱️ Expected time: <30 seconds")
    print("="*52)
    
    start_time = time.time()
    tests = []
    
    # Run quick tests
    tests.append(("Parameter Target", test_parameter_target()))
    tests.append(("Basic Functionality", test_basic_functionality()))
    tests.append(("RTX 5090 Quick", test_rtx_5090_quick()))
    tests.append(("Speed Quick", test_speed_quick()))
    tests.append(("Model Info", test_model_info()))
    
    total_time = time.time() - start_time
    
    # Results summary
    print("="*52)
    print("📊 QUICK TEST RESULTS:")
    print("="*52)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-"*52)
    print(f"📈 Summary: {passed}/{len(tests)} tests passed")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    if passed == len(tests):
        print("🎉 ALL QUICK TESTS PASSED!")
        print("✅ RET v2.1 ULTRA-COMPACT is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False


if __name__ == "__main__":
    success = run_quick_test_suite()
    
    if success:
        print("\n🚀 RET v2.1 is ready for integration!")
        print("💡 You can now proceed with GenerativeDecoder integration")
    else:
        print("\n⚠️ Fix issues before integration") 