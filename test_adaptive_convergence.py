#!/usr/bin/env python3
"""
Тест для adaptive max_steps с convergence detection
"""

import torch
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.utils.logging import setup_logging
from energy_flow.core import create_flow_processor

def test_adaptive_convergence():
    """Тест adaptive max_steps оптимизации"""
    
    # Настройка логирования с convergence категорией
    setup_logging(debug_mode=True, debug_categories=['convergence'])
    
    # Создаем конфигурацию для тестирования
    config = create_debug_config()
    
    # Настраиваем параметры конвергенции для быстрого тестирования  
    config.convergence_enabled = True
    config.convergence_threshold = 0.8  # 80% потоков должны достичь выхода
    config.convergence_min_steps = 3    # Минимум 3 шага
    config.convergence_patience = 2     # Терпение 2 шага
    config.lattice_depth = 15           # Максимум 15 шагов
    
    set_energy_config(config)
    
    print(f"🧪 Testing adaptive max_steps with convergence detection")
    print(f"   Lattice: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    print(f"   Convergence: threshold={config.convergence_threshold}, patience={config.convergence_patience}")
    print(f"   Max steps: {config.lattice_depth}")
    print()
    
    # Создаем FlowProcessor
    processor = create_flow_processor(config)
    
    # Создаем тестовые входные данные
    batch_size = 4
    input_embeddings = torch.randn(batch_size, config.input_embedding_dim_from_teacher, device=config.device)
    
    print(f"🚀 Starting forward pass with {batch_size} inputs...")
    
    # Запускаем forward pass
    output_embeddings = processor.forward(input_embeddings, max_steps=config.lattice_depth)
    
    print(f"✅ Forward pass completed")
    print(f"   Input shape: {input_embeddings.shape}")
    print(f"   Output shape: {output_embeddings.shape}")
    print()
    
    # Получаем статистику производительности
    perf_stats = processor.get_performance_stats()
    
    print("📊 Performance Statistics:")
    print(f"   Average step time: {perf_stats.get('avg_step_time', 0)*1000:.2f}ms")
    print(f"   Max flows per step: {perf_stats.get('max_flows_per_step', 0)}")
    
    if 'convergence_stats' in perf_stats:
        conv_stats = perf_stats['convergence_stats']
        print(f"   Best completion count: {conv_stats['best_completion_count']}")
        print(f"   Final completion count: {conv_stats['final_completion_count']}")
        print(f"   Convergence trend steps: {conv_stats['completion_trend']}")
    
    lattice_stats = perf_stats.get('lattice_stats', {})
    if lattice_stats:
        print(f"   Total completed flows: {lattice_stats.get('total_completed', 0)}")
        print(f"   Total died flows: {lattice_stats.get('total_died', 0)}")
    
    print()
    print("🎯 Test completed successfully!")

if __name__ == "__main__":
    test_adaptive_convergence()