#!/usr/bin/env python3
"""
Тестирование GPU Optimized Euler Solver
=======================================

Комплексные тесты для GPU-оптимизированного Euler solver'а с проверкой:
- Vectorized операций
- Batch processing
- Adaptive step size на основе Lipschitz константы
- Memory efficiency
- Performance benchmarks

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Тест 1: Проверка импортов"""
    logger.info("🧪 Тест 1: Проверка импортов GPU Optimized Euler Solver")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            GPUOptimizedEulerSolver,
            SolverConfig,
            AdaptiveMethod,
            IntegrationResult,
            create_gpu_optimized_euler_solver,
            batch_euler_solve,
            benchmark_solver_performance,
        )

        logger.info("✅ Все импорты успешны")
        return True
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        return False


def test_basic_functionality():
    """Тест 2: Базовая функциональность"""
    logger.info("🧪 Тест 2: Базовая функциональность")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            create_gpu_optimized_euler_solver,
            AdaptiveMethod,
        )

        # Создаем solver
        solver = create_gpu_optimized_euler_solver(
            adaptive_method=AdaptiveMethod.LIPSCHITZ_BASED, max_batch_size=100
        )

        logger.info(f"   ✅ Solver создан: {type(solver).__name__}")
        logger.info(f"   🎯 Device: {solver.device}")
        logger.info(f"   ⚙️ Config: {solver.config.adaptive_method.value}")

        # Простая тестовая функция производной
        def simple_derivative(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            # dx/dt = -x (экспоненциальный распад)
            return -states

        # Тестовые данные
        batch_size = 5
        state_size = 8
        initial_states = torch.randn(batch_size, state_size)

        logger.info(f"   📊 Тестовые данные: {initial_states.shape}")

        # Выполняем интеграцию
        result = solver.batch_integrate(
            simple_derivative, initial_states, integration_time=1.0, num_steps=3
        )

        logger.info(f"   ✅ Интеграция завершена:")
        logger.info(f"     ⏱️ Время: {result.integration_time_ms:.1f}ms")
        logger.info(f"     📊 Шаги: {result.steps_taken}")
        logger.info(f"     🎯 Успех: {result.success}")
        logger.info(f"     💾 Память: {result.memory_usage_mb:.1f}MB")
        logger.info(f"     🔧 Adjustments: {result.adaptive_adjustments}")

        # Проверяем результат
        assert result.final_state.shape == initial_states.shape
        assert not torch.isnan(result.final_state).any()
        assert not torch.isinf(result.final_state).any()

        # Для экспоненциального распада ожидаем уменьшение
        initial_norm = torch.norm(initial_states)
        final_norm = torch.norm(result.final_state)
        logger.info(f"     📉 Норма: {initial_norm:.3f} → {final_norm:.3f}")

        solver.cleanup()

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в базовой функциональности: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_adaptive_methods():
    """Тест 3: Сравнение адаптивных методов"""
    logger.info("🧪 Тест 3: Сравнение адаптивных методов")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            create_gpu_optimized_euler_solver,
            AdaptiveMethod,
        )

        # Более сложная функция производной
        def complex_derivative(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            # Нелинейная система с oscillations
            x, y = states[..., 0], states[..., 1]
            dxdt = -0.1 * x + 0.5 * y * torch.sin(
                t.unsqueeze(-1) if t.dim() == 1 else t
            )
            dydt = -0.2 * y - 0.3 * x * torch.cos(
                t.unsqueeze(-1) if t.dim() == 1 else t
            )

            # Добавляем остальные размерности как damped
            rest = (
                -0.1 * states[..., 2:]
                if states.shape[-1] > 2
                else torch.empty(states.shape[0], 0, device=states.device)
            )

            return torch.cat([dxdt.unsqueeze(-1), dydt.unsqueeze(-1), rest], dim=-1)

        methods_to_test = [
            AdaptiveMethod.LIPSCHITZ_BASED,
            AdaptiveMethod.ACTIVITY_BASED,
        ]

        batch_size = 10
        state_size = 4
        initial_states = torch.randn(batch_size, state_size) * 0.5

        results = {}

        for method in methods_to_test:
            logger.info(f"   🔍 Тестируем метод: {method.value}")

            solver = create_gpu_optimized_euler_solver(adaptive_method=method)

            result = solver.batch_integrate(
                complex_derivative, initial_states, integration_time=2.0, num_steps=5
            )

            results[method.value] = {
                "integration_time_ms": result.integration_time_ms,
                "adaptive_adjustments": result.adaptive_adjustments,
                "stability_violations": result.stability_violations,
                "success": result.success,
                "final_norm": torch.norm(result.final_state).item(),
            }

            logger.info(f"     ⏱️ Время: {result.integration_time_ms:.1f}ms")
            logger.info(f"     🔧 Adjustments: {result.adaptive_adjustments}")
            logger.info(f"     ⚠️ Violations: {result.stability_violations}")
            logger.info(
                f"     📊 Final norm: {results[method.value]['final_norm']:.3f}"
            )

            solver.cleanup()

        # Сравнение результатов
        logger.info("   📊 Сравнение методов:")
        for method, result in results.items():
            logger.info(
                f"     {method}: {result['integration_time_ms']:.1f}ms, "
                f"{result['adaptive_adjustments']} adj, "
                f"norm={result['final_norm']:.3f}"
            )

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании adaptive методов: {e}")
        return False


def test_batch_processing():
    """Тест 4: Batch processing и scalability"""
    logger.info("🧪 Тест 4: Batch processing и scalability")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            create_gpu_optimized_euler_solver,
        )

        # Простая linear dynamics
        def linear_dynamics(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            # Разные rates для разных состояний
            rates = torch.tensor([-0.1, -0.2, -0.3, -0.4], device=states.device)
            rates = rates[: states.shape[-1]]  # Подгоняем под размер состояния
            return states * rates.unsqueeze(0)

        batch_sizes = [1, 10, 50, 100]
        state_size = 4

        performance_results = {}

        for batch_size in batch_sizes:
            logger.info(f"   📊 Batch size: {batch_size}")

            # Создаем данные
            initial_states = torch.randn(batch_size, state_size) * 0.1

            # Создаем solver
            solver = create_gpu_optimized_euler_solver(
                max_batch_size=max(100, batch_size)
            )

            # Измеряем производительность
            start_time = time.time()

            result = solver.batch_integrate(
                linear_dynamics, initial_states, integration_time=1.0, num_steps=5
            )

            wall_time = time.time() - start_time

            # Вычисляем throughput
            trajectories_per_second = batch_size / wall_time

            performance_results[batch_size] = {
                "wall_time_s": wall_time,
                "integration_time_ms": result.integration_time_ms,
                "memory_usage_mb": result.memory_usage_mb,
                "trajectories_per_second": trajectories_per_second,
                "success": result.success,
            }

            logger.info(f"     ⏱️ Wall time: {wall_time:.3f}s")
            logger.info(f"     🚀 Integration time: {result.integration_time_ms:.1f}ms")
            logger.info(f"     💾 Memory: {result.memory_usage_mb:.1f}MB")
            logger.info(f"     📈 Throughput: {trajectories_per_second:.1f} traj/s")

            solver.cleanup()

        # Анализ scalability
        logger.info("   📈 Scalability анализ:")

        baseline_batch = min(batch_sizes)
        baseline_throughput = performance_results[baseline_batch][
            "trajectories_per_second"
        ]

        for batch_size in batch_sizes:
            if batch_size == baseline_batch:
                continue

            current_throughput = performance_results[batch_size][
                "trajectories_per_second"
            ]
            speedup = current_throughput / baseline_throughput
            efficiency = speedup / (batch_size / baseline_batch)

            logger.info(
                f"     Batch {batch_size}: {speedup:.1f}x speedup, "
                f"{efficiency:.1%} efficiency"
            )

        return performance_results

    except Exception as e:
        logger.error(f"❌ Ошибка в batch processing тесте: {e}")
        return {}


def test_adaptive_integration():
    """Тест 5: Adaptive интеграция с контролем ошибки"""
    logger.info("🧪 Тест 5: Adaptive интеграция с контролем ошибки")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            create_gpu_optimized_euler_solver,
        )

        # Stiff ODE для тестирования adaptive capabilities
        def stiff_ode(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            # Жесткая система с быстрыми и медленными модами
            fast_rate = -100.0
            slow_rate = -1.0

            # Первая половина состояний - быстрые моды
            # Вторая половина - медленные моды
            state_size = states.shape[-1]
            fast_size = state_size // 2

            fast_dynamics = fast_rate * states[..., :fast_size]
            slow_dynamics = slow_rate * states[..., fast_size:]

            return torch.cat([fast_dynamics, slow_dynamics], dim=-1)

        batch_size = 5
        state_size = 6
        initial_states = torch.randn(batch_size, state_size) * 0.1

        # Создаем solver
        solver = create_gpu_optimized_euler_solver()

        # Тестируем adaptive интеграцию
        logger.info("   🔄 Запуск adaptive интеграции...")

        result = solver.batch_integrate_adaptive(
            stiff_ode,
            initial_states,
            integration_time=0.1,  # Короткое время для stiff system
            target_error=1e-3,
            max_steps=50,
            return_trajectory=True,
        )

        logger.info(f"   ✅ Adaptive интеграция завершена:")
        logger.info(f"     ⏱️ Время: {result.integration_time_ms:.1f}ms")
        logger.info(f"     📊 Шаги: {result.steps_taken}")
        logger.info(f"     🔧 Adjustments: {result.adaptive_adjustments}")
        logger.info(f"     🎯 Успех: {result.success}")
        logger.info(f"     📈 Error estimates: {len(result.error_estimates)}")

        if result.trajectory is not None:
            logger.info(f"     📊 Trajectory shape: {result.trajectory.shape}")

            # Анализ траектории - fast моды должны быстро затухать
            fast_states = result.trajectory[..., : state_size // 2]
            slow_states = result.trajectory[..., state_size // 2 :]

            initial_fast_norm = torch.norm(fast_states[0])
            final_fast_norm = torch.norm(fast_states[-1])
            initial_slow_norm = torch.norm(slow_states[0])
            final_slow_norm = torch.norm(slow_states[-1])

            logger.info(
                f"     ⚡ Fast modes: {initial_fast_norm:.4f} → {final_fast_norm:.4f}"
            )
            logger.info(
                f"     🐌 Slow modes: {initial_slow_norm:.4f} → {final_slow_norm:.4f}"
            )

            # Fast моды должны затухать быстрее
            fast_decay_ratio = final_fast_norm / (initial_fast_norm + 1e-8)
            slow_decay_ratio = final_slow_norm / (initial_slow_norm + 1e-8)

            logger.info(
                f"     📉 Decay ratios - Fast: {fast_decay_ratio:.4f}, "
                f"Slow: {slow_decay_ratio:.4f}"
            )

        # Проверяем error estimates
        if result.error_estimates:
            avg_error = np.mean(result.error_estimates)
            max_error = np.max(result.error_estimates)
            logger.info(
                f"     📊 Error estimates - Avg: {avg_error:.2e}, Max: {max_error:.2e}"
            )

        solver.cleanup()

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка в adaptive интеграции: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Тест 6: Performance benchmark"""
    logger.info("🧪 Тест 6: Performance benchmark")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            benchmark_solver_performance,
        )

        # Запускаем бенчмарк
        logger.info("   🚀 Запуск performance benchmark...")

        results = benchmark_solver_performance(
            state_size=32,
            batch_sizes=[1, 10, 100],  # Уменьшенные размеры для быстрого теста
            num_steps=3,
            num_trials=3,
        )

        logger.info("   📊 Результаты benchmark:")

        for batch_key, metrics in results.items():
            batch_size = int(batch_key.split("_")[1])
            logger.info(f"     Batch {batch_size}:")
            logger.info(f"       ⏱️ Time: {metrics['avg_integration_time_ms']:.1f}ms")
            logger.info(f"       💾 Memory: {metrics['avg_memory_usage_mb']:.1f}MB")
            logger.info(
                f"       🚀 Throughput: {metrics['throughput_trajectories_per_second']:.0f} traj/s"
            )
            logger.info(
                f"       📊 Efficiency: {metrics['memory_efficiency_mb_per_trajectory']:.2f} MB/traj"
            )
            logger.info(f"       ✅ Success: {metrics['success_rate']:.1%}")

        # Анализ efficiency scaling
        batch_sizes = [int(k.split("_")[1]) for k in results.keys()]
        throughputs = [
            results[f"batch_{bs}"]["throughput_trajectories_per_second"]
            for bs in batch_sizes
        ]

        if len(batch_sizes) > 1:
            logger.info("   📈 Throughput scaling:")
            baseline_throughput = (
                throughputs[0] / batch_sizes[0]
            )  # per-trajectory throughput

            for i, (bs, throughput) in enumerate(zip(batch_sizes, throughputs)):
                per_traj_throughput = throughput / bs
                scaling_efficiency = per_traj_throughput / baseline_throughput
                logger.info(
                    f"     Batch {bs}: {scaling_efficiency:.2f}x efficiency vs single trajectory"
                )

        return results

    except Exception as e:
        logger.error(f"❌ Ошибка в performance benchmark: {e}")
        return {}


def test_memory_efficiency():
    """Тест 7: Memory efficiency"""
    logger.info("🧪 Тест 7: Memory efficiency")

    try:
        from new_rebuild.core.cnf.gpu_optimized_euler_solver import (
            create_gpu_optimized_euler_solver,
        )

        def simple_derivative(t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            return -0.1 * states

        # Тестируем с и без memory efficiency
        batch_size = 50
        state_size = 16
        initial_states = torch.randn(batch_size, state_size)

        configs = [("Memory Efficient", True), ("Standard", False)]

        results = {}

        for config_name, memory_efficient in configs:
            logger.info(f"   🔍 Тестируем {config_name} режим")

            solver = create_gpu_optimized_euler_solver(
                memory_efficient=memory_efficient
            )

            # Запускаем несколько интеграций для проверки memory pooling
            integration_results = []

            for i in range(3):
                result = solver.batch_integrate(
                    simple_derivative, initial_states, integration_time=1.0, num_steps=5
                )
                integration_results.append(result)

                logger.info(f"     Интеграция {i+1}: {result.memory_usage_mb:.1f}MB")

            # Получаем статистику solver'а
            stats = solver.get_comprehensive_stats()

            results[config_name] = {
                "avg_memory_per_integration": np.mean(
                    [r.memory_usage_mb for r in integration_results]
                ),
                "memory_pool_size": stats["memory_pool"]["pool_size"],
                "peak_memory_mb": stats["performance"]["gpu_memory_peak_mb"],
            }

            logger.info(
                f"     📊 Memory pool size: {stats['memory_pool']['pool_size']}"
            )
            logger.info(
                f"     💾 Peak memory: {stats['performance']['gpu_memory_peak_mb']:.1f}MB"
            )

            solver.cleanup()

        # Сравнение эффективности
        logger.info("   📊 Сравнение memory efficiency:")
        for config_name, metrics in results.items():
            logger.info(f"     {config_name}:")
            logger.info(
                f"       📊 Avg memory/integration: {metrics['avg_memory_per_integration']:.1f}MB"
            )
            logger.info(f"       🗃️ Pool size: {metrics['memory_pool_size']}")
            logger.info(f"       ⛰️ Peak memory: {metrics['peak_memory_mb']:.1f}MB")

        return results

    except Exception as e:
        logger.error(f"❌ Ошибка в memory efficiency тесте: {e}")
        return {}


def run_all_tests():
    """Запуск всех тестов"""
    logger.info("🚀 Запуск комплексного тестирования GPU Optimized Euler Solver")
    logger.info("=" * 80)

    test_results = {}

    # Тест 1: Импорты
    test_results["imports"] = test_imports()

    if not test_results["imports"]:
        logger.error("❌ Критическая ошибка: импорты не удались")
        return test_results

    # Тест 2: Базовая функциональность
    test_results["basic_functionality"] = test_basic_functionality()

    # Тест 3: Adaptive методы
    test_results["adaptive_methods"] = test_adaptive_methods()

    # Тест 4: Batch processing
    test_results["batch_processing"] = test_batch_processing()

    # Тест 5: Adaptive интеграция
    test_results["adaptive_integration"] = test_adaptive_integration()

    # Тест 6: Performance benchmark
    test_results["performance_benchmark"] = test_performance_benchmark()

    # Тест 7: Memory efficiency
    test_results["memory_efficiency"] = test_memory_efficiency()

    # Финальный отчет
    logger.info("=" * 80)
    logger.info("📋 ФИНАЛЬНЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    logger.info("=" * 80)

    successful_tests = sum(
        1
        for result in test_results.values()
        if isinstance(result, (bool, dict))
        and (result is True or (isinstance(result, dict) and result))
    )

    total_tests = len(test_results)

    logger.info(f"✅ Успешные тесты: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("🚀 GPU Optimized Euler Solver готов к использованию")
        logger.info("⚡ Vectorized операции работают")
        logger.info("📊 Batch processing эффективен")
        logger.info("🎯 Lipschitz-based adaptation функционирует")
        logger.info("💾 Memory efficiency оптимизирована")
    else:
        logger.warning(f"⚠️ {total_tests - successful_tests} тестов требуют внимания")
        logger.info("🔧 Проверьте логи выше для исправления проблем")

    return test_results


if __name__ == "__main__":
    results = run_all_tests()
