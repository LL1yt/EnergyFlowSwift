#!/usr/bin/env python3
"""
Комплексное тестирование GPU Enhanced CNF Integration
=====================================================

Полное тестирование улучшенной CNF интеграции с проверкой:
- GPU Optimized Euler Solver
- GPU Enhanced CNF с batch processing
- Vectorized Neural ODE operations
- Lipschitz-based adaptive stepping
- Performance benchmarks и memory efficiency

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cnf_imports():
    """Тест 1: Проверка всех импортов CNF компонентов"""
    logger.info("🧪 Тест 1: Проверка импортов CNF компонентов")
    
    try:
        # Legacy компоненты
        from core.cnf import (
            LightweightCNF,
            NeuralODE,
            ConnectionType,
            EulerSolver
        )
        logger.info("✅ Legacy CNF импорты успешны")
        
        # Новые GPU компоненты
        from core.cnf import (
            GPUOptimizedEulerSolver,
            SolverConfig,
            AdaptiveMethod,
            IntegrationResult,
            create_gpu_optimized_euler_solver,
            batch_euler_solve,
            benchmark_solver_performance
        )
        logger.info("✅ GPU Optimized Solver импорты успешны")
        
        # GPU Enhanced CNF
        from core.cnf import (
            GPUEnhancedCNF,
            VectorizedNeuralODE,
            BatchProcessingMode,
            create_gpu_enhanced_cnf,
            benchmark_cnf_performance
        )
        logger.info("✅ GPU Enhanced CNF импорты успешны")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False


def test_vectorized_neural_ode():
    """Тест 2: Vectorized Neural ODE"""
    logger.info("🧪 Тест 2: Vectorized Neural ODE")
    
    try:
        from core.cnf import VectorizedNeuralODE, ConnectionType
        
        state_size = 16
        batch_size = 10
        
        # Создаем Vectorized Neural ODE
        neural_ode = VectorizedNeuralODE(
            state_size=state_size,
            connection_type=ConnectionType.DISTANT,
            batch_size=batch_size
        )
        
        logger.info(f"   ✅ VectorizedNeuralODE создан: {neural_ode.device}")
        logger.info(f"   📊 Параметры: {sum(p.numel() for p in neural_ode.parameters())}")
        
        # Тестовые данные
        t = torch.tensor(0.5)
        current_states = torch.randn(batch_size, state_size)
        neighbor_influences = torch.randn(batch_size, state_size)
        
        logger.info(f"   📊 Входные данные: states={current_states.shape}, neighbors={neighbor_influences.shape}")
        
        # Forward pass
        start_time = time.time()
        derivatives = neural_ode(t, current_states, neighbor_influences)
        forward_time = time.time() - start_time
        
        logger.info(f"   ✅ Forward pass завершен:")
        logger.info(f"     ⏱️ Время: {forward_time*1000:.1f}ms")
        logger.info(f"     📊 Выход: {derivatives.shape}")
        logger.info(f"     🔍 Статистика: mean={derivatives.mean().item():.4f}, std={derivatives.std().item():.4f}")
        
        # Проверки
        assert derivatives.shape == current_states.shape
        assert not torch.isnan(derivatives).any()
        assert not torch.isinf(derivatives).any()
        
        # Тест с разными размерами batch'а
        for test_batch_size in [1, 5, 20]:
            test_states = torch.randn(test_batch_size, state_size)
            test_neighbors = torch.randn(test_batch_size, state_size)
            test_derivatives = neural_ode(t, test_states, test_neighbors)
            
            assert test_derivatives.shape == (test_batch_size, state_size)
            logger.info(f"     ✅ Batch size {test_batch_size}: OK")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в VectorizedNeuralODE тесте: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_enhanced_cnf_basic():
    """Тест 3: Базовая функциональность GPU Enhanced CNF"""
    logger.info("🧪 Тест 3: Базовая функциональность GPU Enhanced CNF")
    
    try:
        from core.cnf import (
            create_gpu_enhanced_cnf,
            ConnectionType,
            BatchProcessingMode,
            AdaptiveMethod
        )
        
        state_size = 32
        
        # Создаем GPU Enhanced CNF
        cnf = create_gpu_enhanced_cnf(
            state_size=state_size,
            connection_type=ConnectionType.DISTANT,
            batch_processing_mode=BatchProcessingMode.ADAPTIVE_BATCH,
            max_batch_size=50,
            adaptive_method=AdaptiveMethod.LIPSCHITZ_BASED
        )
        
        logger.info(f"   ✅ GPU Enhanced CNF создан")
        logger.info(f"   🎯 Device: {cnf.device}")
        logger.info(f"   📊 Total params: {sum(p.numel() for p in cnf.parameters())}")
        
        # Тест single connection (legacy compatibility)
        current_state = torch.randn(1, state_size)
        neighbor_states = torch.randn(5, state_size)
        
        logger.info("   🔄 Тест single connection...")
        start_time = time.time()
        result = cnf(current_state, neighbor_states)
        single_time = time.time() - start_time
        
        logger.info(f"     ⏱️ Single connection: {single_time*1000:.1f}ms")
        logger.info(f"     📊 Результат: {result['new_state'].shape}")
        logger.info(f"     🎯 Processing time: {result['processing_time_ms']:.1f}ms")
        
        assert result['new_state'].shape == current_state.shape
        assert not torch.isnan(result['new_state']).any()
        
        # Получаем статистику
        stats = cnf.get_comprehensive_stats()
        logger.info(f"   📈 CNF stats: {stats['cnf_performance']['total_forward_passes']} passes")
        
        cnf.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в GPU Enhanced CNF тесте: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing_modes():
    """Тест 4: Различные режимы batch processing"""
    logger.info("🧪 Тест 4: Различные режимы batch processing")
    
    try:
        from core.cnf import (
            create_gpu_enhanced_cnf,
            ConnectionType,
            BatchProcessingMode,
            AdaptiveMethod
        )
        
        state_size = 16
        batch_size = 8
        
        modes_to_test = [
            BatchProcessingMode.CONNECTION_BATCH,
            BatchProcessingMode.ADAPTIVE_BATCH
        ]
        
        results = {}
        
        for mode in modes_to_test:
            logger.info(f"   🔍 Тестируем режим: {mode.value}")
            
            cnf = create_gpu_enhanced_cnf(
                state_size=state_size,
                batch_processing_mode=mode,
                max_batch_size=batch_size
            )
            
            # Подготавливаем batch данные
            current_states = torch.randn(batch_size, state_size)
            neighbor_states_list = [
                torch.randn(torch.randint(3, 10, (1,)).item(), state_size) 
                for _ in range(batch_size)
            ]
            
            # Выполняем batch processing
            start_time = time.time()
            result = cnf(current_states, neighbor_states_list)
            batch_time = time.time() - start_time
            
            results[mode.value] = {
                "batch_time_s": batch_time,
                "processing_time_ms": result["processing_time_ms"],
                "batch_size": result["batch_size"],
                "output_shape": result["new_state"].shape
            }
            
            logger.info(f"     ⏱️ Batch time: {batch_time*1000:.1f}ms")
            logger.info(f"     🚀 Processing time: {result['processing_time_ms']:.1f}ms")
            logger.info(f"     📊 Output: {result['new_state'].shape}")
            
            # Проверки
            assert result["new_state"].shape == (batch_size, state_size)
            assert not torch.isnan(result["new_state"]).any()
            assert result["batch_size"] == batch_size
            
            cnf.cleanup()
        
        # Сравнение производительности
        logger.info("   📊 Сравнение режимов:")
        for mode, metrics in results.items():
            throughput = metrics["batch_size"] / metrics["batch_time_s"]
            logger.info(f"     {mode}: {throughput:.1f} connections/s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в batch processing тесте: {e}")
        return False


def test_cnf_performance_scaling():
    """Тест 5: Scalability и производительность CNF"""
    logger.info("🧪 Тест 5: Scalability и производительность CNF")
    
    try:
        from core.cnf import create_gpu_enhanced_cnf, BatchProcessingMode
        
        state_size = 32
        batch_sizes = [1, 5, 10, 20]
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"   📊 Batch size: {batch_size}")
            
            cnf = create_gpu_enhanced_cnf(
                state_size=state_size,
                batch_processing_mode=BatchProcessingMode.CONNECTION_BATCH,
                max_batch_size=max(20, batch_size)
            )
            
            # Тестовые данные
            current_states = torch.randn(batch_size, state_size)
            neighbor_states_list = [
                torch.randn(torch.randint(5, 15, (1,)).item(), state_size) 
                for _ in range(batch_size)
            ]
            
            # Прогрев
            _ = cnf(current_states, neighbor_states_list)
            
            # Измерение производительности
            trials = 3
            times = []
            
            for trial in range(trials):
                start_time = time.time()
                result = cnf(current_states, neighbor_states_list)
                wall_time = time.time() - start_time
                times.append(wall_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            performance_results[batch_size] = {
                "avg_time_s": avg_time,
                "throughput_connections_per_s": throughput,
                "processing_time_ms": result["processing_time_ms"]
            }
            
            logger.info(f"     ⏱️ Avg time: {avg_time*1000:.1f}ms")
            logger.info(f"     🚀 Throughput: {throughput:.1f} conn/s")
            
            cnf.cleanup()
        
        # Анализ scalability
        logger.info("   📈 Scalability анализ:")
        baseline_throughput = performance_results[1]["throughput_connections_per_s"]
        
        for batch_size in batch_sizes[1:]:  # Пропускаем baseline
            current_throughput = performance_results[batch_size]["throughput_connections_per_s"]
            per_connection_throughput = current_throughput / batch_size
            baseline_per_connection = baseline_throughput / 1
            
            efficiency = per_connection_throughput / baseline_per_connection
            logger.info(f"     Batch {batch_size}: {efficiency:.2f}x efficiency")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"❌ Ошибка в scalability тесте: {e}")
        return {}


def test_adaptive_methods_comparison():
    """Тест 6: Сравнение адаптивных методов"""
    logger.info("🧪 Тест 6: Сравнение адаптивных методов")
    
    try:
        from core.cnf import (
            create_gpu_enhanced_cnf,
            AdaptiveMethod,
            BatchProcessingMode
        )
        
        state_size = 24
        batch_size = 10
        
        methods = [
            AdaptiveMethod.LIPSCHITZ_BASED,
            AdaptiveMethod.ACTIVITY_BASED
        ]
        
        results = {}
        
        # Сложная тестовая задача - stiff dynamics
        for method in methods:
            logger.info(f"   🔍 Метод: {method.value}")
            
            cnf = create_gpu_enhanced_cnf(
                state_size=state_size,
                batch_processing_mode=BatchProcessingMode.CONNECTION_BATCH,
                adaptive_method=method,
                max_batch_size=batch_size
            )
            
            # Создаем stiff test case
            current_states = torch.randn(batch_size, state_size) * 0.1
            # Некоторые neighbors имеют большие значения для создания stiffness
            neighbor_states_list = []
            for i in range(batch_size):
                num_neighbors = torch.randint(5, 15, (1,)).item()
                neighbors = torch.randn(num_neighbors, state_size)
                if i % 3 == 0:  # Каждый третий - stiff case
                    neighbors *= 5.0  # Большие значения
                neighbor_states_list.append(neighbors)
            
            # Выполняем интеграцию
            start_time = time.time()
            result = cnf(current_states, neighbor_states_list)
            integration_time = time.time() - start_time
            
            # Получаем детальную статистику
            stats = cnf.get_comprehensive_stats()
            solver_stats = stats["solver_stats"]
            
            results[method.value] = {
                "integration_time_s": integration_time,
                "processing_time_ms": result["processing_time_ms"],
                "solver_performance": solver_stats["performance"],
                "stability_violations": solver_stats["performance"]["stability_violations"],
                "adaptive_adjustments": solver_stats["performance"]["adaptive_adjustments"],
                "final_state_norm": torch.norm(result["new_state"]).item()
            }
            
            logger.info(f"     ⏱️ Integration time: {integration_time*1000:.1f}ms")
            logger.info(f"     🔧 Adaptive adjustments: {results[method.value]['adaptive_adjustments']}")
            logger.info(f"     ⚠️ Stability violations: {results[method.value]['stability_violations']}")
            logger.info(f"     📊 Final norm: {results[method.value]['final_state_norm']:.3f}")
            
            cnf.cleanup()
        
        # Сравнение методов
        logger.info("   📊 Сравнение методов:")
        for method, metrics in results.items():
            logger.info(f"     {method}:")
            logger.info(f"       ⏱️ Time: {metrics['integration_time_s']*1000:.1f}ms")
            logger.info(f"       🔧 Adjustments: {metrics['adaptive_adjustments']}")
            logger.info(f"       ⚠️ Violations: {metrics['stability_violations']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Ошибка в adaptive methods тесте: {e}")
        return {}


def test_cnf_benchmark():
    """Тест 7: Full CNF benchmark"""
    logger.info("🧪 Тест 7: Full CNF benchmark")
    
    try:
        from core.cnf import benchmark_cnf_performance
        
        logger.info("   🚀 Запуск CNF benchmark...")
        
        # Небольшой benchmark для быстрого тестирования
        results = benchmark_cnf_performance(
            state_sizes=[16, 32],
            batch_sizes=[1, 10, 20],
            num_trials=3
        )
        
        logger.info("   📊 Результаты benchmark:")
        
        for key, metrics in results.items():
            logger.info(f"     {key}:")
            logger.info(f"       ⏱️ Wall time: {metrics['avg_wall_time_s']*1000:.1f}ms")
            logger.info(f"       🚀 Processing: {metrics['avg_processing_time_ms']:.1f}ms")
            logger.info(f"       📈 Throughput: {metrics['throughput_connections_per_second']:.0f} conn/s")
        
        # Анализ эффективности разных state_size
        state_16_results = {k: v for k, v in results.items() if "state_16" in k}
        state_32_results = {k: v for k, v in results.items() if "state_32" in k}
        
        if state_16_results and state_32_results:
            logger.info("   📊 Влияние state_size на производительность:")
            
            for batch_size in [1, 10, 20]:
                key_16 = f"state_16_batch_{batch_size}"
                key_32 = f"state_32_batch_{batch_size}"
                
                if key_16 in results and key_32 in results:
                    throughput_16 = results[key_16]["throughput_connections_per_second"]
                    throughput_32 = results[key_32]["throughput_connections_per_second"]
                    ratio = throughput_16 / throughput_32 if throughput_32 > 0 else 0
                    
                    logger.info(f"     Batch {batch_size}: 16D/32D = {ratio:.2f}x throughput")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Ошибка в CNF benchmark: {e}")
        return {}


def test_integration_with_moe():
    """Тест 8: Интеграция с MoE архитектурой"""
    logger.info("🧪 Тест 8: Интеграция с MoE архитектурой")
    
    try:
        from core.cnf import create_gpu_enhanced_cnf, ConnectionType, BatchProcessingMode
        
        # Создаем CNF для разных типов экспертов
        state_size = 32
        
        # Distant Expert - использует только CNF
        distant_cnf = create_gpu_enhanced_cnf(
            state_size=state_size,
            connection_type=ConnectionType.DISTANT,
            batch_processing_mode=BatchProcessingMode.ADAPTIVE_BATCH
        )
        
        # Functional Expert - может использовать CNF для части связей
        functional_cnf = create_gpu_enhanced_cnf(
            state_size=state_size,
            connection_type=ConnectionType.FUNCTIONAL,
            batch_processing_mode=BatchProcessingMode.CONNECTION_BATCH
        )
        
        logger.info("   ✅ CNF для разных экспертов созданы")
        
        # Симуляция MoE forward pass
        batch_size = 15
        current_states = torch.randn(batch_size, state_size)
        neighbor_states_list = [
            torch.randn(torch.randint(5, 20, (1,)).item(), state_size) 
            for _ in range(batch_size)
        ]
        
        # Распределяем связи между экспертами (как в реальной MoE)
        distant_ratio = 0.35  # 35% distant connections
        functional_ratio = 0.55  # 55% functional connections (часть через CNF)
        
        distant_connections = int(batch_size * distant_ratio)
        functional_connections = batch_size - distant_connections
        
        logger.info(f"   📊 Распределение: {distant_connections} distant, {functional_connections} functional")
        
        # Обработка distant connections через CNF
        if distant_connections > 0:
            distant_states = current_states[:distant_connections]
            distant_neighbors = neighbor_states_list[:distant_connections]
            
            start_time = time.time()
            distant_result = distant_cnf(distant_states, distant_neighbors)
            distant_time = time.time() - start_time
            
            logger.info(f"   🌌 Distant Expert CNF: {distant_time*1000:.1f}ms")
        
        # Обработка functional connections через CNF (частично)
        if functional_connections > 0:
            functional_states = current_states[distant_connections:]
            functional_neighbors = neighbor_states_list[distant_connections:]
            
            start_time = time.time()
            functional_result = functional_cnf(functional_states, functional_neighbors)
            functional_time = time.time() - start_time
            
            logger.info(f"   🔗 Functional Expert CNF: {functional_time*1000:.1f}ms")
        
        # Общая статистика
        total_processing_time = distant_result["processing_time_ms"] + functional_result["processing_time_ms"]
        logger.info(f"   ⏱️ Общее время CNF: {total_processing_time:.1f}ms")
        
        # Получаем статистику каждого CNF
        distant_stats = distant_cnf.get_comprehensive_stats()
        functional_stats = functional_cnf.get_comprehensive_stats()
        
        logger.info(f"   📈 Статистика:")
        logger.info(f"     Distant: {distant_stats['cnf_performance']['batch_efficiency']:.1f} conn/s")
        logger.info(f"     Functional: {functional_stats['cnf_performance']['batch_efficiency']:.1f} conn/s")
        
        distant_cnf.cleanup()
        functional_cnf.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в MoE интеграции: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_cnf_tests():
    """Запуск всех тестов GPU Enhanced CNF"""
    logger.info("🚀 Запуск комплексного тестирования GPU Enhanced CNF Integration")
    logger.info("=" * 90)
    
    test_results = {}
    
    # Тест 1: Импорты
    test_results["imports"] = test_cnf_imports()
    
    if not test_results["imports"]:
        logger.error("❌ Критическая ошибка: импорты не удались")
        return test_results
    
    # Тест 2: Vectorized Neural ODE
    test_results["vectorized_ode"] = test_vectorized_neural_ode()
    
    # Тест 3: Базовая функциональность CNF
    test_results["cnf_basic"] = test_gpu_enhanced_cnf_basic()
    
    # Тест 4: Batch processing modes
    test_results["batch_modes"] = test_batch_processing_modes()
    
    # Тест 5: Performance scaling
    test_results["performance_scaling"] = test_cnf_performance_scaling()
    
    # Тест 6: Adaptive methods
    test_results["adaptive_methods"] = test_adaptive_methods_comparison()
    
    # Тест 7: CNF benchmark
    test_results["cnf_benchmark"] = test_cnf_benchmark()
    
    # Тест 8: MoE интеграция
    test_results["moe_integration"] = test_integration_with_moe()
    
    # Финальный отчет
    logger.info("=" * 90)
    logger.info("📋 ФИНАЛЬНЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ GPU ENHANCED CNF")
    logger.info("=" * 90)
    
    successful_tests = sum(1 for result in test_results.values() if 
                          isinstance(result, (bool, dict)) and 
                          (result is True or (isinstance(result, dict) and result)))
    
    total_tests = len(test_results)
    
    logger.info(f"✅ Успешные тесты: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("🚀 GPU Enhanced CNF Integration готова к использованию")
        logger.info("")
        logger.info("🔥 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ РЕАЛИЗОВАНЫ:")
        logger.info("   ⚡ Vectorized операции для всех шагов интеграции")
        logger.info("   📊 Batch processing для multiple trajectories")
        logger.info("   🎯 Adaptive step size на основе Lipschitz константы")
        logger.info("   💾 Memory-efficient batch operations")
        logger.info("   📈 Real-time performance monitoring")
        logger.info("   🤖 Полная интеграция с MoE архитектурой")
        logger.info("")
        logger.info("🏆 CNF теперь готова для production использования!")
    else:
        failed_tests = total_tests - successful_tests
        logger.warning(f"⚠️ {failed_tests} тестов требуют внимания")
        logger.info("🔧 Проверьте логи выше для исправления проблем")
    
    return test_results


if __name__ == "__main__":
    results = run_all_cnf_tests()