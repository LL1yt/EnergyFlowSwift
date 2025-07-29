#!/usr/bin/env python3
"""
Тест для векторизованного FlowProcessor
"""

import torch
import sys
import time
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.utils.logging import setup_logging
from energy_flow.core import create_flow_processor

def test_vectorized_flow_processor():
    """Тест векторизованного FlowProcessor с измерением производительности"""
    
    # Настройка логирования 
    setup_logging(debug_mode=True, debug_categories=['convergence', 'performance'])
    
    # Создаем конфигурацию для тестирования
    config = create_debug_config()
    
    # Настройки для тестирования векторизации
    config.convergence_enabled = True
    config.convergence_threshold = 0.8
    config.convergence_min_steps = 3
    config.convergence_patience = 2
    config.lattice_depth = 12   # Меньше шагов для быстрого тестирования
    config.max_active_flows = 20000  # Больше потоков для тестирования векторизации
    
    set_energy_config(config)
    
    print(f"🧪 Testing VECTORIZED FlowProcessor")
    print(f"   Lattice: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    print(f"   Max flows: {config.max_active_flows}")
    print(f"   Convergence: threshold={config.convergence_threshold}")
    print()
    
    # Создаем FlowProcessor
    processor = create_flow_processor(config)
    
    # Создаем тестовые входные данные (больший batch для тестирования)
    batch_size = 8
    input_embeddings = torch.randn(batch_size, config.input_embedding_dim_from_teacher, device=config.device)
    
    print(f"🚀 Starting vectorized forward pass with {batch_size} inputs...")
    
    # Засекаем время
    start_time = time.time()
    
    # Запускаем forward pass
    output_embeddings = processor.forward(input_embeddings, max_steps=config.lattice_depth)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Vectorized forward pass completed in {total_time:.2f}s")
    print(f"   Input shape: {input_embeddings.shape}")
    print(f"   Output shape: {output_embeddings.shape}")
    print()
    
    # Получаем статистику производительности
    perf_stats = processor.get_performance_stats()
    
    print("📊 Vectorized Performance Statistics:")
    print(f"   Average step time: {perf_stats.get('avg_step_time', 0)*1000:.2f}ms")
    print(f"   Max flows per step: {perf_stats.get('max_flows_per_step', 0):,}")
    print(f"   Total forward time: {total_time:.2f}s")
    print(f"   Throughput: {batch_size/total_time:.2f} samples/s")
    
    if 'convergence_stats' in perf_stats:
        conv_stats = perf_stats['convergence_stats']
        print(f"   Best completion count: {conv_stats['best_completion_count']:,}")
        print(f"   Final completion count: {conv_stats['final_completion_count']:,}")
        print(f"   Convergence trend steps: {conv_stats['completion_trend']}")
    
    lattice_stats = perf_stats.get('lattice_stats', {})
    if lattice_stats:
        print(f"   Total completed flows: {lattice_stats.get('total_completed', 0):,}")
        print(f"   Total died flows: {lattice_stats.get('total_died', 0):,}")
        completion_rate = lattice_stats.get('total_completed', 0) / (lattice_stats.get('total_completed', 0) + lattice_stats.get('total_died', 0)) * 100
        print(f"   Completion rate: {completion_rate:.1f}%")
    
    print()
    print("🎯 Vectorized FlowProcessor test completed successfully!")
    
    return {
        'total_time': total_time,
        'throughput': batch_size / total_time,
        'avg_step_time': perf_stats.get('avg_step_time', 0),
        'max_flows': perf_stats.get('max_flows_per_step', 0),
        'completion_rate': completion_rate if 'completion_rate' in locals() else 0
    }

if __name__ == "__main__":
    results = test_vectorized_flow_processor()
    
    print("\n" + "="*60)
    print("🏆 VECTORIZATION TEST RESULTS:")
    print(f"   Total time: {results['total_time']:.2f}s")
    print(f"   Throughput: {results['throughput']:.2f} samples/s")
    print(f"   Step time: {results['avg_step_time']*1000:.2f}ms")
    print(f"   Max flows: {results['max_flows']:,}")
    print(f"   Completion rate: {results['completion_rate']:.1f}%")
    print("="*60)