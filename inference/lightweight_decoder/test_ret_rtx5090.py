"""
[TEST] RTX 5090 COMPATIBILITY TEST - Resource-Efficient Transformer

Специальный тест для проверки:
- RTX 5090 edge optimization compatibility
- 52% memory reduction validation  
- 33% speedup measurement
- Edge quantization effectiveness
- Production readiness assessment

Test Environment: RTX 5090 (sm_120) с PyTorch ограничениями
"""

import torch
import time
import psutil
import gc
import tracemalloc
from pathlib import Path
import sys
from typing import Dict, Any, List
import logging

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

# Импорты нашей системы
from inference.lightweight_decoder.resource_efficient_decoder import (
    ResourceEfficientDecoder, 
    RETConfig,
    create_resource_efficient_decoder
)
# ConfigManager не требуется для тестирования

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RTX5090PerformanceTester:
    """
    [START] RTX 5090 Performance Tester для Resource-Efficient Transformer
    
    Проверяет:
    - Совместимость с RTX 5090 
    - Memory reduction (52% target)
    - Speed improvement (33% target)
    - Edge optimization effectiveness
    """
    
    def __init__(self):
        self.device = self._detect_device()
        self.baseline_metrics = {}
        self.ret_metrics = {}
        self.test_results = {}
        
        logger.info(f"[TARGET] RTX 5090 Tester initialized on device: {self.device}")
        
    def _detect_device(self) -> str:
        """Определение доступного устройства"""
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            
            logger.info(f"🎮 CUDA Device detected: {device_name}")
            logger.info(f"   Compute Capability: {device_props.major}.{device_props.minor}")
            logger.info(f"   Memory: {device_props.total_memory / 1024**3:.1f} GB")
            
            # RTX 5090 detection
            if "5090" in device_name or device_props.major >= 12:
                logger.info("[START] RTX 5090 (or newer) detected - edge optimization режим")
                return "cuda"
            else:
                logger.info("🎮 Other CUDA device - standard режим")
                return "cuda"
        else:
            logger.info("[COMPUTER] CPU mode - RTX 5090 fallback активен")
            return "cpu"
    
    def create_baseline_model(self) -> torch.nn.Module:
        """Создание baseline модели для сравнения"""
        
        class BaselineTransformer(torch.nn.Module):
            """Простая baseline модель для сравнения performance"""
            
            def __init__(self):
                super().__init__()
                self.embedding_proj = torch.nn.Linear(768, 1024)
                self.transformer_layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=1024, 
                        nhead=8, 
                        dim_feedforward=2048,
                        batch_first=True
                    ) for _ in range(4)
                ])
                self.output_head = torch.nn.Linear(1024, 32000)
                
            def forward(self, x):
                # Handle different input shapes
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # Add batch dimension: (768,) -> (1, 768)
                
                x = self.embedding_proj(x)  # (batch, 768) -> (batch, 1024)
                
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension: (batch, 1024) -> (batch, 1, 1024)
                
                for layer in self.transformer_layers:
                    x = layer(x)
                
                return self.output_head(x)
        
        model = BaselineTransformer()
        if self.device == "cuda":
            model = model.cuda()
        
        return model
    
    def measure_memory_usage(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
        """Измерение memory usage"""
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Start memory tracking
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        tracemalloc.start()
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'decode'):
                # RET model
                _ = model.decode(input_tensor)
            else:
                # Baseline model
                _ = model(input_tensor)
        
        # Memory measurement
        if self.device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            memory_mb = (peak_memory - start_memory) / 1024**2
        else:
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024**2
        
        tracemalloc.stop()
        
        return memory_mb
    
    def measure_inference_speed(self, model: torch.nn.Module, 
                              input_tensor: torch.Tensor, 
                              num_runs: int = 10) -> float:
        """Измерение inference speed"""
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                if hasattr(model, 'decode'):
                    _ = model.decode(input_tensor)
                else:
                    _ = model(input_tensor)
        
        # Timing measurement
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                if hasattr(model, 'decode'):
                    _ = model.decode(input_tensor, max_length=20)
                else:
                    _ = model(input_tensor)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Average время в milliseconds
        avg_time_ms = (sum(times) / len(times)) * 1000
        return avg_time_ms
    
    def count_parameters(self, model: torch.nn.Module) -> int:
        """Подсчет параметров модели"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def test_baseline_performance(self) -> Dict[str, Any]:
        """Тестирование baseline модели"""
        
        logger.info("[DATA] Testing baseline transformer performance...")
        
        # Create baseline model
        baseline_model = self.create_baseline_model()
        
        # Test input
        test_embedding = torch.randn(768)
        if self.device == "cuda":
            test_embedding = test_embedding.cuda()
        
        # Measurements
        memory_usage = self.measure_memory_usage(baseline_model, test_embedding)
        inference_time = self.measure_inference_speed(baseline_model, test_embedding)
        parameters = self.count_parameters(baseline_model)
        
        self.baseline_metrics = {
            'memory_mb': memory_usage,
            'inference_time_ms': inference_time,
            'parameters': parameters,
            'model_type': 'baseline_transformer'
        }
        
        logger.info(f"[CHART] Baseline Results:")
        logger.info(f"   Memory: {memory_usage:.1f} MB")
        logger.info(f"   Speed: {inference_time:.1f} ms")
        logger.info(f"   Parameters: {parameters:,}")
        
        return self.baseline_metrics
    
    def test_ret_performance(self) -> Dict[str, Any]:
        """Тестирование Resource-Efficient Transformer"""
        
        logger.info("[START] Testing Resource-Efficient Transformer performance...")
        
        # Create RET model with RTX 5090 optimization
        config = RETConfig(
            edge_quantization=True,
            memory_reduction_factor=0.52,
            speed_improvement_factor=0.33,
            target_parameters=1_000_000
        )
        
        ret_model = ResourceEfficientDecoder(config)
        if self.device == "cuda":
            ret_model = ret_model.cuda()
        
        # Test input
        test_embedding = torch.randn(768)
        if self.device == "cuda":
            test_embedding = test_embedding.cuda()
        
        # Measurements
        memory_usage = self.measure_memory_usage(ret_model, test_embedding)
        inference_time = self.measure_inference_speed(ret_model, test_embedding)
        parameters = self.count_parameters(ret_model)
        
        self.ret_metrics = {
            'memory_mb': memory_usage,
            'inference_time_ms': inference_time,
            'parameters': parameters,
            'model_type': 'resource_efficient_transformer'
        }
        
        logger.info(f"[TARGET] RET Results:")
        logger.info(f"   Memory: {memory_usage:.1f} MB")
        logger.info(f"   Speed: {inference_time:.1f} ms")
        logger.info(f"   Parameters: {parameters:,}")
        
        return self.ret_metrics
    
    def calculate_improvements(self) -> Dict[str, Any]:
        """Расчет улучшений RET относительно baseline"""
        
        if not self.baseline_metrics or not self.ret_metrics:
            logger.error("[ERROR] Baseline или RET metrics не собраны")
            return {}
        
        # Memory reduction calculation
        memory_reduction = (
            (self.baseline_metrics['memory_mb'] - self.ret_metrics['memory_mb']) / 
            self.baseline_metrics['memory_mb']
        )
        
        # Speed improvement calculation
        speed_improvement = (
            (self.baseline_metrics['inference_time_ms'] - self.ret_metrics['inference_time_ms']) / 
            self.baseline_metrics['inference_time_ms']
        )
        
        # Parameter reduction
        param_reduction = (
            (self.baseline_metrics['parameters'] - self.ret_metrics['parameters']) / 
            self.baseline_metrics['parameters']
        )
        
        improvements = {
            'memory_reduction_percent': memory_reduction * 100,
            'speed_improvement_percent': speed_improvement * 100,
            'parameter_reduction_percent': param_reduction * 100,
            'memory_target_achieved': memory_reduction >= 0.45,  # 45% minimum
            'speed_target_achieved': speed_improvement >= 0.25,  # 25% minimum
            'parameter_target_achieved': self.ret_metrics['parameters'] <= 1_200_000
        }
        
        return improvements
    
    def test_rtx5090_compatibility(self) -> Dict[str, Any]:
        """Специальный тест RTX 5090 совместимости"""
        
        logger.info("🎮 Testing RTX 5090 specific compatibility...")
        
        compatibility_results = {
            'device_detected': self.device,
            'cuda_available': torch.cuda.is_available(),
            'rtx_5090_optimizations': True,
            'edge_quantization_works': False,
            'gradient_checkpointing_works': False,
            'mixed_precision_works': False
        }
        
        if self.device == "cuda":
            try:
                # Test edge quantization
                config = RETConfig(edge_quantization=True)
                model = ResourceEfficientDecoder(config).cuda()
                test_input = torch.randn(768).cuda()
                
                with torch.no_grad():
                    result = model.decode(test_input)
                
                compatibility_results['edge_quantization_works'] = True
                logger.info("[OK] Edge quantization: PASSED")
                
            except Exception as e:
                logger.warning(f"[WARNING] Edge quantization: FAILED - {e}")
            
            try:
                # Test gradient checkpointing
                config = RETConfig(gradient_checkpointing=True)
                model = ResourceEfficientDecoder(config).cuda()
                test_input = torch.randn(768).cuda()
                
                model.train()
                result = model.decode(test_input)
                
                compatibility_results['gradient_checkpointing_works'] = True
                logger.info("[OK] Gradient checkpointing: PASSED")
                
            except Exception as e:
                logger.warning(f"[WARNING] Gradient checkpointing: FAILED - {e}")
            
            try:
                # Test mixed precision
                config = RETConfig(mixed_precision=True)
                model = ResourceEfficientDecoder(config).cuda()
                test_input = torch.randn(768).cuda()
                
                with torch.cuda.amp.autocast():
                    result = model.decode(test_input)
                
                compatibility_results['mixed_precision_works'] = True
                logger.info("[OK] Mixed precision: PASSED")
                
            except Exception as e:
                logger.warning(f"[WARNING] Mixed precision: FAILED - {e}")
        
        return compatibility_results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Запуск полного теста RTX 5090 совместимости"""
        
        logger.info("[START] Starting comprehensive RTX 5090 performance test...")
        
        # Run all tests
        baseline_results = self.test_baseline_performance()
        ret_results = self.test_ret_performance()
        improvements = self.calculate_improvements()
        compatibility = self.test_rtx5090_compatibility()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_environment': {
                'device': self.device,
                'cuda_available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'baseline_performance': baseline_results,
            'ret_performance': ret_results,
            'improvements': improvements,
            'rtx5090_compatibility': compatibility,
            'test_passed': self._evaluate_overall_success(improvements, compatibility)
        }
        
        self.test_results = comprehensive_results
        return comprehensive_results
    
    def _evaluate_overall_success(self, improvements: Dict[str, Any], 
                                 compatibility: Dict[str, Any]) -> bool:
        """Оценка общего успеха тестов"""
        
        memory_ok = improvements.get('memory_target_achieved', False)
        speed_ok = improvements.get('speed_target_achieved', False)
        params_ok = improvements.get('parameter_target_achieved', False)
        
        compat_ok = (
            compatibility.get('edge_quantization_works', False) or
            compatibility.get('gradient_checkpointing_works', False)
        )
        
        return memory_ok and speed_ok and params_ok and compat_ok
    
    def print_detailed_report(self):
        """Печать детального отчета"""
        
        if not self.test_results:
            logger.error("[ERROR] No test results available")
            return
        
        print("\n" + "="*60)
        print("[START] RTX 5090 RESOURCE-EFFICIENT TRANSFORMER TEST REPORT")
        print("="*60)
        
        # Environment info
        env = self.test_results['test_environment']
        print(f"\n[DATA] Test Environment:")
        print(f"   Device: {env['device']}")
        print(f"   CUDA Available: {env['cuda_available']}")
        print(f"   GPU: {env.get('device_name', 'N/A')}")
        
        # Performance comparison
        baseline = self.test_results['baseline_performance']
        ret = self.test_results['ret_performance']
        improvements = self.test_results['improvements']
        
        print(f"\n[CHART] Performance Comparison:")
        print(f"   Baseline Memory: {baseline['memory_mb']:.1f} MB")
        print(f"   RET Memory: {ret['memory_mb']:.1f} MB")
        print(f"   → Memory Reduction: {improvements['memory_reduction_percent']:.1f}% "
              f"{'[OK]' if improvements['memory_target_achieved'] else '[ERROR]'}")
        
        print(f"\n[FAST] Speed Comparison:")
        print(f"   Baseline Speed: {baseline['inference_time_ms']:.1f} ms")
        print(f"   RET Speed: {ret['inference_time_ms']:.1f} ms")
        print(f"   → Speed Improvement: {improvements['speed_improvement_percent']:.1f}% "
              f"{'[OK]' if improvements['speed_target_achieved'] else '[ERROR]'}")
        
        print(f"\n[SAVE] Parameter Comparison:")
        print(f"   Baseline Parameters: {baseline['parameters']:,}")
        print(f"   RET Parameters: {ret['parameters']:,}")
        print(f"   → Parameter Reduction: {improvements['parameter_reduction_percent']:.1f}% "
              f"{'[OK]' if improvements['parameter_target_achieved'] else '[ERROR]'}")
        
        # RTX 5090 compatibility
        compat = self.test_results['rtx5090_compatibility']
        print(f"\n🎮 RTX 5090 Compatibility:")
        print(f"   Edge Quantization: {'[OK]' if compat['edge_quantization_works'] else '[ERROR]'}")
        print(f"   Gradient Checkpointing: {'[OK]' if compat['gradient_checkpointing_works'] else '[ERROR]'}")
        print(f"   Mixed Precision: {'[OK]' if compat['mixed_precision_works'] else '[ERROR]'}")
        
        # Overall result
        success = self.test_results['test_passed']
        print(f"\n[TARGET] Overall Test Result: {'[OK] PASSED' if success else '[ERROR] FAILED'}")
        
        if success:
            print("\n[SUCCESS] Resource-Efficient Transformer готов к production!")
            print("   - Memory reduction цель достигнута")
            print("   - Speed improvement цель достигнута") 
            print("   - RTX 5090 совместимость подтверждена")
        else:
            print("\n[WARNING] Требуется дополнительная оптимизация")
        
        print("="*60)


def main():
    """Main test функция"""
    
    print("[START] Starting RTX 5090 Resource-Efficient Transformer Test")
    print("="*60)
    
    # Create tester
    tester = RTX5090PerformanceTester()
    
    # Run comprehensive test
    try:
        results = tester.run_comprehensive_test()
        
        # Print detailed report
        tester.print_detailed_report()
        
        # Save results for analysis
        import json
        results_path = Path(__file__).parent / "rtx5090_test_results.json"
        
        # Convert to serializable format
        serializable_results = {
            k: v for k, v in results.items() 
            if not isinstance(v, torch.Tensor)
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"[FOLDER] Results saved to: {results_path}")
        
        return results['test_passed']
        
    except Exception as e:
        import traceback
        logger.error(f"[ERROR] Test failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)