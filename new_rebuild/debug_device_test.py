#!/usr/bin/env python3
"""
Упрощенный тест для отладки device mismatch в GPU Optimized Euler Solver
"""

import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_adaptive_integration():
    """Простейший тест adaptive интеграции для отладки device issues"""
    print("🔍 Отладка device mismatch в adaptive интеграции")
    
    try:
        from core.cnf import create_gpu_optimized_euler_solver, AdaptiveMethod
        
        # Создаем solver
        solver = create_gpu_optimized_euler_solver(
            adaptive_method=AdaptiveMethod.LIPSCHITZ_BASED,
            memory_efficient=True
        )
        
        print(f"✅ Solver создан на устройстве: {solver.device}")
        
        # Простейшая тестовая функция производной
        def simple_derivative(t, states):
            print(f"  derivative_fn called with t device: {t.device if torch.is_tensor(t) else 'scalar'}")
            print(f"  derivative_fn called with states device: {states.device}")
            result = -0.1 * states  # Простое затухание
            print(f"  derivative_fn result device: {result.device}")
            return result
        
        # Минимальные тестовые данные
        batch_size = 2
        state_size = 4
        initial_states = torch.randn(batch_size, state_size)
        
        print(f"📊 Исходные данные:")
        print(f"  initial_states device: {initial_states.device}")
        print(f"  initial_states shape: {initial_states.shape}")
        
        # Запускаем adaptive интеграцию
        print(f"🚀 Запуск batch_integrate_adaptive...")
        result = solver.batch_integrate_adaptive(
            simple_derivative,
            initial_states,
            integration_time=0.1,  # Короткое время
            max_steps=2,           # Всего 2 шага
            return_trajectory=False
        )
        
        print(f"✅ Интеграция завершена успешно!")
        print(f"  final_state device: {result.final_state.device}")
        print(f"  final_state shape: {result.final_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Очистка
        if 'solver' in locals():
            solver.cleanup()

if __name__ == "__main__":
    debug_adaptive_integration()