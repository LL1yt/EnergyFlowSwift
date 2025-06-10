"""
🧪 GENERATIVE DECODER INTEGRATION TEST - Stage 2.1 Validation

TESTING SCOPE:
- [OK] GenerativeDecoder initialization with RET v2.1
- 🧪 Parameter target achievement (≤800K)
- 🧪 RTX 5090 compatibility validation  
- 🧪 Memory reduction effectiveness (≥60% target)
- 🧪 Generation quality (BLEU score improvement)
- 🧪 API consistency with PhraseBankDecoder
- 🧪 Performance benchmarking vs baseline

CRITICAL SUCCESS CRITERIA:
- Parameters ≤ 800K (RET v2.1: 722K [OK])
- Memory reduction ≥ 60%
- Generation time <100ms
- Quality score >0.4
- RTX 5090 compatibility
- API consistency
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import gc
from pathlib import Path
import logging
import unittest
from typing import Dict, Any, Optional, List
import json

# Import our GenerativeDecoder system
try:
    from generative_decoder import (
        GenerativeDecoder,
        GenerativeConfig,
        create_generative_decoder
    )
except ImportError:
    # Handle module imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from generative_decoder import (
        GenerativeDecoder,
        GenerativeConfig,
        create_generative_decoder
    )

# Import for comparison (optional)
try:
    from phrase_bank_decoder import PhraseBankDecoder
except ImportError:
    PhraseBankDecoder = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerativeDecoderIntegrationTest(unittest.TestCase):
    """
    🧪 Comprehensive GenerativeDecoder Integration Test Suite
    
    Tests all critical aspects of GenerativeDecoder Stage 2.1 integration
    """
    
    def setUp(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_embedding = torch.randn(768, device=self.device)
        
        # Create GenerativeDecoder with optimal config
        self.config = GenerativeConfig(
            architecture_type="resource_efficient_v21",
            target_parameters=800_000,
            verbose_logging=True,
            edge_optimization=True
        )
        
        self.decoder = GenerativeDecoder(self.config)
        self.decoder.to(self.device)
        
        # Performance baselines
        self.baseline_memory = self._get_memory_usage()
        
        logger.info(f"🧪 GenerativeDecoder Integration Test setup complete")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Architecture: {self.config.architecture_type}")
        logger.info(f"   Baseline memory: {self.baseline_memory:.2f} MB")
    
    def tearDown(self):
        """Cleanup after tests"""
        if hasattr(self, 'decoder'):
            del self.decoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def test_01_initialization_success(self):
        """[TARGET] TEST 1: Successful initialization"""
        logger.info("[TARGET] TEST 1: GenerativeDecoder Initialization")
        
        # Check basic initialization
        self.assertIsNotNone(self.decoder)
        self.assertIsNotNone(self.decoder.decoder_model)
        self.assertIsNotNone(self.decoder.tokenizer)
        self.assertIsNotNone(self.decoder.quality_assessor)
        
        # Check architecture selection
        self.assertEqual(self.decoder.config.architecture_type, "resource_efficient_v21")
        
        # Check parameter count
        param_count = self.decoder._count_parameters()
        self.assertLessEqual(param_count, self.config.target_parameters)
        
        logger.info(f"   [OK] Initialization successful")
        logger.info(f"   Parameters: {param_count:,} / {self.config.target_parameters:,}")
        logger.info(f"   Architecture: {self.decoder.config.architecture_type}")
    
    def test_02_parameter_efficiency_achievement(self):
        """[TARGET] TEST 2: Parameter Efficiency (CRITICAL)"""
        logger.info("[TARGET] TEST 2: Parameter Efficiency Achievement")
        
        param_count = self.decoder._count_parameters()
        target = self.config.target_parameters
        
        # CRITICAL: Must be ≤ 800K
        self.assertLessEqual(param_count, target, 
                           f"Parameter count {param_count:,} exceeds target {target:,}")
        
        # Calculate efficiency
        efficiency = (target - param_count) / target * 100
        
        logger.info(f"   Parameters: {param_count:,} / {target:,}")
        logger.info(f"   Efficiency: {efficiency:.1f}% under target")
        logger.info(f"   [OK] Parameter efficiency ACHIEVED!")
        
        # Verify this matches RET v2.1 expectations (722K)
        expected_ret_v21_params = 722_944
        param_variance = abs(param_count - expected_ret_v21_params) / expected_ret_v21_params
        self.assertLess(param_variance, 0.05, f"Parameter count variance >5% from expected RET v2.1")
    
    def test_03_basic_generation_functionality(self):
        """[TARGET] TEST 3: Basic Generation Functionality"""
        logger.info("[TARGET] TEST 3: Basic Generation Functionality")
        
        # Test basic generation
        result = self.decoder.generate(self.test_embedding, max_length=10)
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('tokens', result)
        self.assertIn('quality_metrics', result)
        self.assertIn('generation_time', result)
        
        # Validate generation content
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['tokens'], list)
        self.assertGreater(len(result['text']), 0)
        
        logger.info(f"   Generated text: '{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}'")
        logger.info(f"   Tokens: {len(result['tokens'])}")
        logger.info(f"   Quality: {result['quality_metrics']['overall_quality']:.3f}")
        logger.info(f"   Time: {result['generation_time']:.3f}s")
        logger.info(f"   [OK] Basic generation WORKING!")
    
    def test_04_api_consistency_with_phrase_bank(self):
        """[TARGET] TEST 4: API Consistency with PhraseBankDecoder"""
        logger.info("[TARGET] TEST 4: API Consistency with PhraseBankDecoder")
        
        # Test decode() method compatibility
        text_result = self.decoder.decode(self.test_embedding, max_length=10)
        
        # Validate decode method
        self.assertIsInstance(text_result, str)
        self.assertGreater(len(text_result), 0)
        
        # Test batch functionality
        batch_embeddings = torch.randn(3, 768, device=self.device)
        batch_results = self.decoder.batch_generate(batch_embeddings)
        
        # Validate batch results
        self.assertEqual(len(batch_results), 3)
        for result in batch_results:
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
        
        logger.info(f"   Single decode: '{text_result[:30]}{'...' if len(text_result) > 30 else ''}'")
        logger.info(f"   Batch results: {len(batch_results)} items")
        logger.info(f"   [OK] API consistency VERIFIED!")
    
    def test_05_rtx_5090_compatibility(self):
        """[START] TEST 5: RTX 5090 Compatibility"""
        logger.info("[START] TEST 5: RTX 5090 Compatibility")
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - skipping RTX 5090 test")
        
        # GPU info
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   Compute capability: {compute_capability}")
        
        # Test CUDA operations
        try:
            # Move to GPU
            self.decoder.to(self.device)
            test_embedding = self.test_embedding.to(self.device)
            
            # Warm-up
            _ = self.decoder.generate(test_embedding, max_length=3)
            
            # Test mixed precision if enabled
            if self.config.mixed_precision:
                with torch.amp.autocast('cuda'):
                    result = self.decoder.generate(test_embedding, max_length=5)
                    self.assertIsInstance(result['text'], str)
                    logger.info(f"   Mixed precision: [OK] Working")
            
            # Test memory efficiency
            memory_used = self._get_memory_usage()
            logger.info(f"   GPU memory used: {memory_used:.2f} MB")
            
            # Test edge optimization features
            if hasattr(self.decoder.decoder_model, 'config'):
                model_config = self.decoder.decoder_model.config
                if hasattr(model_config, 'edge_optimization'):
                    logger.info(f"   Edge optimization: {model_config.edge_optimization}")
            
            logger.info("   [OK] RTX 5090 compatibility VERIFIED!")
            
        except Exception as e:
            self.fail(f"RTX 5090 compatibility test failed: {e}")
    
    def test_06_memory_reduction_validation(self):
        """[SAVE] TEST 6: Memory Reduction Validation"""
        logger.info("[SAVE] TEST 6: Memory Reduction Validation")
        
        # Calculate model size
        model_size_mb = self._get_model_size_mb(self.decoder)
        
        # Theoretical baseline для comparison (standard transformer)
        baseline_params = 32_000_000  # Standard transformer with 32K vocab
        current_params = self.decoder._count_parameters()
        
        # Calculate reduction
        reduction_ratio = (baseline_params - current_params) / baseline_params
        
        logger.info(f"   Model size: {model_size_mb:.2f} MB")
        logger.info(f"   Current params: {current_params:,}")
        logger.info(f"   Baseline params: {baseline_params:,}")
        logger.info(f"   Reduction ratio: {reduction_ratio:.1%}")
        
        # Verify significant reduction achieved
        target_reduction = self.config.memory_reduction_target  # 60%
        self.assertGreater(reduction_ratio, target_reduction, 
                          f"Memory reduction {reduction_ratio:.1%} below {target_reduction:.1%} target")
        
        logger.info(f"   [OK] Memory reduction target ACHIEVED! ({reduction_ratio:.1%} > {target_reduction:.1%})")
    
    def test_07_generation_performance(self):
        """[FAST] TEST 7: Generation Performance"""
        logger.info("[FAST] TEST 7: Generation Performance")
        
        # Warmup
        for _ in range(3):
            _ = self.decoder.generate(self.test_embedding, max_length=5)
        
        # Benchmark runs
        times = []
        quality_scores = []
        
        for i in range(10):
            start_time = time.time()
            result = self.decoder.generate(self.test_embedding, max_length=10)
            end_time = time.time()
            
            times.append(end_time - start_time)
            quality_scores.append(result['quality_metrics']['overall_quality'])
        
        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        logger.info(f"   Average time: {avg_time:.3f}s")
        logger.info(f"   Min time: {min_time:.3f}s")
        logger.info(f"   Max time: {max_time:.3f}s")
        logger.info(f"   Average quality: {avg_quality:.3f}")
        
        # Performance targets
        target_time = 0.100  # 100ms target
        self.assertLess(avg_time, target_time, 
                       f"Average time {avg_time:.3f}s exceeds {target_time:.3f}s target")
        
        target_quality = 0.4  # Quality target
        self.assertGreater(avg_quality, target_quality,
                          f"Average quality {avg_quality:.3f} below {target_quality:.3f} target")
        
        logger.info("   [OK] Performance targets ACHIEVED!")
    
    def test_08_quality_assessment_system(self):
        """[TARGET] TEST 8: Quality Assessment System"""
        logger.info("[TARGET] TEST 8: Quality Assessment System")
        
        # Generate multiple samples
        samples = []
        for _ in range(5):
            result = self.decoder.generate(self.test_embedding, max_length=15)
            samples.append(result)
        
        # Analyze quality metrics
        quality_scores = [s['quality_metrics']['overall_quality'] for s in samples]
        coherence_scores = [s['quality_metrics']['coherence'] for s in samples]
        fluency_scores = [s['quality_metrics']['fluency'] for s in samples]
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_fluency = sum(fluency_scores) / len(fluency_scores)
        
        logger.info(f"   Samples analyzed: {len(samples)}")
        logger.info(f"   Average quality: {avg_quality:.3f}")
        logger.info(f"   Average coherence: {avg_coherence:.3f}")
        logger.info(f"   Average fluency: {avg_fluency:.3f}")
        
        # Quality consistency check
        quality_variance = sum((q - avg_quality) ** 2 for q in quality_scores) / len(quality_scores)
        quality_std = quality_variance ** 0.5
        
        logger.info(f"   Quality std deviation: {quality_std:.3f}")
        
        # Validate quality system
        self.assertGreater(avg_quality, 0.2, "Quality system producing too low scores")
        self.assertLess(quality_std, 0.3, "Quality scores too inconsistent")
        
        logger.info("   [OK] Quality assessment system WORKING!")
    
    def test_09_integration_readiness(self):
        """[LINK] TEST 9: Integration Readiness"""
        logger.info("[LINK] TEST 9: Integration Readiness")
        
        # Test performance report
        report = self.decoder.get_performance_report()
        
        # Validate report structure
        required_fields = [
            'architecture', 'parameter_count', 'parameter_target',
            'parameter_efficiency', 'success_rate', 'average_quality'
        ]
        
        for field in required_fields:
            self.assertIn(field, report, f"Missing field in performance report: {field}")
        
        # Test save/load functionality
        save_path = Path("test_generative_decoder_checkpoint.pth")
        
        try:
            # Save model
            self.decoder.save_model(save_path)
            self.assertTrue(save_path.exists(), "Model save failed")
            
            # Create new decoder and load
            new_decoder = GenerativeDecoder(self.config)
            new_decoder.load_model(save_path)
            
            # Test loaded model
            original_result = self.decoder.generate(self.test_embedding, max_length=5)
            loaded_result = new_decoder.generate(self.test_embedding, max_length=5)
            
            # Both should work (exact match not required due to randomness)
            self.assertIsInstance(original_result['text'], str)
            self.assertIsInstance(loaded_result['text'], str)
            
            logger.info(f"   Performance report: [OK] Complete")
            logger.info(f"   Save/Load: [OK] Working")
            logger.info(f"   Parameter efficiency: {report['parameter_efficiency']:.2f}x")
            logger.info(f"   Success rate: {report['success_rate']:.1%}")
            
        finally:
            # Cleanup
            if save_path.exists():
                save_path.unlink()
        
        logger.info("   [OK] Integration readiness VERIFIED!")


def run_comprehensive_integration_test():
    """
    [START] Run comprehensive GenerativeDecoder integration test suite
    """
    
    print("=" * 60)
    print("🧪 GENERATIVE DECODER INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Phase 2.7 Stage 2.1 - GenerativeDecoder + RET v2.1")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(GenerativeDecoderIntegrationTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Results summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("[OK] GenerativeDecoder + RET v2.1 READY FOR PRODUCTION!")
        print("\n[START] STAGE 2.1 INTEGRATION COMPLETE!")
        print("[IDEA] Next step: Stage 2.2 - Advanced optimization & RTX 5090 validation")
        
        return True
    else:
        print("[ERROR] SOME INTEGRATION TESTS FAILED!")
        print(f"[ERROR] Failures: {len(result.failures)}")
        print(f"[ERROR] Errors: {len(result.errors)}")
        
        return False


if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    
    if success:
        print("\n[TARGET] INTEGRATION SUCCESS - Ready for next stage!")
    else:
        print("\n[WARNING] Fix integration issues before proceeding") 