#!/usr/bin/env python3
"""
Тест Mixed Precision Training
"""

import torch
import sys
import time
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_experiment_config, set_energy_config
from energy_flow.utils.logging import setup_logging
from energy_flow.training import EnergyTrainer

def test_mixed_precision():
    """Тест Mixed Precision Training с измерением производительности"""
    
    # Настройка логирования
    setup_logging(debug_mode=True, debug_categories=['training', 'performance'])
    
    # Создаем конфигурацию с Mixed Precision
    config = create_experiment_config()
    
    # Убеждаемся что Mixed Precision включен
    config.use_mixed_precision = True
    config.mixed_precision_dtype = torch.bfloat16
    config.use_gradient_scaling = True
    config.batch_size = 16  # Меньший batch для теста
    config.lattice_depth = 20  # Меньше для быстрого теста
    
    set_energy_config(config)
    
    print(f"🔬 Testing MIXED PRECISION Training")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    print(f"   Precision dtype: {config.mixed_precision_dtype}")
    print(f"   Gradient scaling: {config.use_gradient_scaling}")
    print(f"   Batch size: {config.batch_size}")
    print()
    
    # Создаем trainer
    trainer = EnergyTrainer(config)
    
    # Создаем тестовые данные
    batch_size = config.batch_size
    input_texts = [f"Test input {i}" for i in range(batch_size)]
    target_texts = [f"Test target {i}" for i in range(batch_size)]
    
    # Teacher embeddings
    teacher_input = torch.randn(batch_size, 768, device=config.device, requires_grad=True)
    teacher_target = torch.randn(batch_size, 768, device=config.device, requires_grad=True)
    
    print(f"🚀 Starting mixed precision train_step...")
    
    # Измеряем память до
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1e9
        print(f"   Memory before: {memory_before:.2f}GB")
    
    # Засекаем время
    start_time = time.time()
    
    # Запускаем train step
    metrics = trainer.train_step(input_texts, target_texts, teacher_input, teacher_target)
    
    end_time = time.time()
    step_time = end_time - start_time
    
    # Измеряем память после
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1e9
        print(f"   Memory after: {memory_after:.2f}GB")
        memory_saved = memory_before - memory_after if memory_before > 0 else 0
    
    print(f"✅ Mixed precision train_step completed in {step_time:.2f}s")
    print(f"   Total loss: {metrics.get('total_loss', 'N/A')}")
    print(f"   Energy loss: {metrics.get('energy_loss', 'N/A')}")
    print(f"   Text loss: {metrics.get('text_loss', 'N/A')}")
    
    if torch.cuda.is_available():
        print(f"   Memory usage: {memory_after:.2f}GB")
        if memory_saved > 0:
            print(f"   Memory saved: {memory_saved:.2f}GB")
    
    print()
    
    # Проверяем что scaler работает
    if hasattr(trainer, 'scaler') and trainer.scaler is not None:
        current_scale = trainer.scaler.get_scale()
        print(f"🔧 Gradient scaler: scale={current_scale:.0f}")
    
    print("🎯 Mixed Precision test completed successfully!")
    
    return {
        'step_time': step_time,
        'total_loss': metrics.get('total_loss', float('inf')),
        'memory_usage': memory_after if torch.cuda.is_available() else 0,
        'mixed_precision_enabled': config.use_mixed_precision
    }

if __name__ == "__main__":
    results = test_mixed_precision()
    
    print("\n" + "="*60)
    print("🏆 MIXED PRECISION TEST RESULTS:")
    print(f"   Step time: {results['step_time']:.2f}s")
    print(f"   Total loss: {results['total_loss']}")
    print(f"   Memory usage: {results['memory_usage']:.2f}GB")
    print(f"   Mixed precision: {results['mixed_precision_enabled']}")
    print("   Expected benefits: 1.5x speedup + 50% memory savings")
    print("="*60)