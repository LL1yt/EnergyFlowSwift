#!/usr/bin/env python3
"""
Примеры использования Unified Spatial Optimizer
==============================================

Практические примеры использования унифицированной системы пространственной
оптимизации для различных задач 3D решеток.

Автор: 3D Cellular Neural Network Project
Версия: 3.0.0 (2024-12-27)
"""

import torch
import numpy as np
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """Пример 1: Базовое использование"""
    logger.info("📝 Пример 1: Базовое использование UnifiedSpatialOptimizer")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer
    )
    
    # Создаем оптимизатор с автоматическими настройками
    dimensions = (8, 8, 8)
    optimizer = create_unified_spatial_optimizer(dimensions)
    
    # Создаем тестовые данные
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    states = torch.randn(total_cells, 32)  # 32-мерные состояния
    
    logger.info(f"   🔧 Создан оптимизатор для решетки {dimensions}")
    logger.info(f"   📊 Входные данные: {states.shape}")
    
    # Выполняем оптимизацию
    result = optimizer.optimize_lattice_forward(states)
    
    # Выводим результаты
    logger.info(f"   ✅ Обработка завершена:")
    logger.info(f"     ⏱️ Время: {result.processing_time_ms:.1f}ms")
    logger.info(f"     🎯 Режим: {result.mode_used.value}")
    logger.info(f"     💾 Память: {result.memory_usage_mb:.1f}MB")
    logger.info(f"     🔍 Соседей: {result.neighbors_found}")
    
    # Очистка
    optimizer.cleanup()
    logger.info("   🧹 Ресурсы освобождены")


def example_2_custom_configuration():
    """Пример 2: Кастомная конфигурация"""
    logger.info("📝 Пример 2: Кастомная конфигурация")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode
    )
    
    # Создаем кастомную конфигурацию
    config = OptimizationConfig(
        mode=OptimizationMode.HYBRID,  # Гибридный режим
        enable_moe=False,              # Без MoE пока
        enable_morton_encoding=True,   # С Morton encoding
        enable_adaptive_chunking=True, # С adaptive chunking
        max_memory_gb=4.0,            # Ограничение памяти
        target_performance_ms=20.0,   # Целевая производительность
        fallback_enabled=True          # С fallback
    )
    
    dimensions = (12, 12, 12)
    optimizer = create_unified_spatial_optimizer(dimensions, config)
    
    logger.info(f"   ⚙️ Конфигурация:")
    logger.info(f"     🎯 Режим: {config.mode.value}")
    logger.info(f"     🔢 Morton: {config.enable_morton_encoding}")
    logger.info(f"     📦 Chunking: {config.enable_adaptive_chunking}")
    logger.info(f"     💾 Макс память: {config.max_memory_gb}GB")
    
    # Тестируем поиск соседей
    test_coords = (6, 6, 6)
    neighbors = optimizer.find_neighbors_optimized(test_coords, radius=3.0)
    
    logger.info(f"   🔍 Поиск соседей для {test_coords}:")
    logger.info(f"     📊 Найдено соседей: {len(neighbors)}")
    logger.info(f"     🎯 Первые 10: {neighbors[:10]}")
    
    # Получаем статистику
    stats = optimizer.get_comprehensive_stats()
    unified_stats = stats['unified_optimizer']
    
    logger.info(f"   📈 Статистика:")
    logger.info(f"     🎯 Активный режим: {unified_stats['active_mode']}")
    logger.info(f"     🔢 Morton включен: {unified_stats['morton_enabled']}")
    logger.info(f"     📝 История производительности: {unified_stats['performance_history_length']}")
    
    optimizer.cleanup()


def example_3_moe_integration():
    """Пример 3: Интеграция с MoE архитектурой"""
    logger.info("📝 Пример 3: Интеграция с MoE архитектурой")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode
    )
    
    # Создаем простой MoE processor
    class SimpleMoEProcessor(torch.nn.Module):
        def __init__(self, state_size, num_experts=3):
            super().__init__()
            self.num_experts = num_experts
            
            # Создаем экспертов (простые линейные слои)
            self.experts = torch.nn.ModuleList([
                torch.nn.Linear(state_size, state_size) for _ in range(num_experts)
            ])
            
            # Gating network
            self.gate = torch.nn.Linear(state_size, num_experts)
            self.softmax = torch.nn.Softmax(dim=-1)
        
        def forward(self, current_state, neighbor_states, cell_idx, neighbor_indices, **kwargs):
            # Получаем веса экспертов
            gate_weights = self.softmax(self.gate(current_state))
            
            # Применяем экспертов
            expert_outputs = []
            for expert in self.experts:
                expert_output = expert(current_state)
                expert_outputs.append(expert_output)
            
            # Взвешенная комбинация экспертов
            expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, state_size, num_experts]
            gate_weights = gate_weights.unsqueeze(1)  # [batch, 1, num_experts]
            
            new_state = (expert_outputs * gate_weights).sum(dim=-1)
            
            return {"new_state": new_state}
    
    # Создаем MoE processor
    state_size = 64
    moe_processor = SimpleMoEProcessor(state_size)
    
    # Конфигурация с MoE
    config = OptimizationConfig(
        mode=OptimizationMode.AUTO,
        enable_moe=True,                    # Включаем MoE
        enable_morton_encoding=True,
        target_performance_ms=50.0
    )
    
    dimensions = (6, 6, 6)
    optimizer = create_unified_spatial_optimizer(dimensions, config, moe_processor)
    
    logger.info(f"   🤖 MoE Processor создан:")
    logger.info(f"     👥 Количество экспертов: {moe_processor.num_experts}")
    logger.info(f"     📊 Размер состояния: {state_size}")
    
    # Создаем данные и обрабатываем
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    states = torch.randn(total_cells, state_size)
    
    logger.info(f"   🔄 Запуск MoE оптимизации...")
    start_time = time.time()
    result = optimizer.optimize_lattice_forward(states)
    moe_time = time.time() - start_time
    
    logger.info(f"   ✅ MoE оптимизация завершена:")
    logger.info(f"     ⏱️ Общее время: {moe_time:.3f}s")
    logger.info(f"     🏃 Время обработки: {result.processing_time_ms:.1f}ms")
    logger.info(f"     🎯 Режим: {result.mode_used.value}")
    logger.info(f"     💾 Память: {result.memory_usage_mb:.1f}MB")
    logger.info(f"     🏎️ GPU утилизация: {result.gpu_utilization:.1%}")
    
    # Проверяем что MoE действительно изменил состояния
    state_diff = torch.norm(result.new_states - states)
    logger.info(f"     📊 Изменение состояний: {state_diff:.3f}")
    
    optimizer.cleanup()


def example_4_performance_comparison():
    """Пример 4: Сравнение производительности разных режимов"""
    logger.info("📝 Пример 4: Сравнение производительности разных режимов")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode
    )
    
    dimensions = (10, 10, 10)
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    state_size = 32
    states = torch.randn(total_cells, state_size)
    
    # Режимы для тестирования
    modes_to_test = [
        ("CPU_ONLY", OptimizationMode.CPU_ONLY),
        ("HYBRID", OptimizationMode.HYBRID),
        ("AUTO", OptimizationMode.AUTO),
    ]
    
    # Добавляем GPU режим если доступен
    if torch.cuda.is_available():
        modes_to_test.append(("GPU_ONLY", OptimizationMode.GPU_ONLY))
    
    results = {}
    
    for mode_name, mode in modes_to_test:
        logger.info(f"   🧪 Тестируем режим: {mode_name}")
        
        try:
            config = OptimizationConfig(mode=mode)
            optimizer = create_unified_spatial_optimizer(dimensions, config)
            
            # Прогрев
            _ = optimizer.optimize_lattice_forward(states)
            
            # Измеряем производительность
            times = []
            for i in range(5):
                start_time = time.time()
                result = optimizer.optimize_lattice_forward(states)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[mode_name] = {
                "avg_time_s": avg_time,
                "std_time_s": std_time,
                "mode_used": result.mode_used.value,
                "memory_mb": result.memory_usage_mb,
                "neighbors_found": result.neighbors_found
            }
            
            logger.info(f"     ⏱️ Среднее время: {avg_time:.3f}±{std_time:.3f}s")
            logger.info(f"     🎯 Режим использован: {result.mode_used.value}")
            logger.info(f"     💾 Память: {result.memory_usage_mb:.1f}MB")
            
            optimizer.cleanup()
            
        except Exception as e:
            logger.warning(f"     ⚠️ Режим {mode_name} не удался: {e}")
            results[mode_name] = {"error": str(e)}
    
    # Анализ результатов
    logger.info("   📊 Сравнение производительности:")
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    
    if len(successful_results) > 1:
        fastest = min(successful_results.items(), key=lambda x: x[1]["avg_time_s"])
        logger.info(f"     🏆 Самый быстрый: {fastest[0]} ({fastest[1]['avg_time_s']:.3f}s)")
        
        for mode_name, result in successful_results.items():
            if mode_name != fastest[0]:
                slowdown = result["avg_time_s"] / fastest[1]["avg_time_s"]
                logger.info(f"     📈 {mode_name}: {slowdown:.2f}x медленнее")
    
    return results


def example_5_memory_estimation():
    """Пример 5: Оценка требований к памяти"""
    logger.info("📝 Пример 5: Оценка требований к памяти")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        estimate_unified_memory_requirements,
        OptimizationConfig,
        OptimizationMode
    )
    
    # Тестируем разные размеры решеток
    test_cases = [
        ("Маленькая", (5, 5, 5)),
        ("Средняя", (20, 20, 20)),
        ("Большая", (50, 50, 50)),
        ("Очень большая", (100, 100, 100))
    ]
    
    for case_name, dimensions in test_cases:
        logger.info(f"   📏 {case_name} решетка {dimensions}:")
        
        # Базовая конфигурация
        config_basic = OptimizationConfig(mode=OptimizationMode.CPU_ONLY)
        estimates_basic = estimate_unified_memory_requirements(dimensions, config_basic)
        
        # GPU конфигурация
        config_gpu = OptimizationConfig(mode=OptimizationMode.GPU_ONLY)
        estimates_gpu = estimate_unified_memory_requirements(dimensions, config_gpu)
        
        # GPU + MoE конфигурация
        config_moe = OptimizationConfig(
            mode=OptimizationMode.GPU_ONLY,
            enable_moe=True,
            enable_morton_encoding=True
        )
        estimates_moe = estimate_unified_memory_requirements(dimensions, config_moe)
        
        logger.info(f"     💻 CPU только: {estimates_basic['total_memory_gb']:.2f}GB")
        logger.info(f"     🚀 GPU: {estimates_gpu['total_memory_gb']:.2f}GB")
        logger.info(f"     🤖 GPU+MoE: {estimates_moe['total_memory_gb']:.2f}GB")
        logger.info(f"     🎯 Рекомендуемая GPU память: {estimates_moe['recommended_gpu_memory_gb']:.2f}GB")
        
        # Проверка на практичность
        if estimates_moe['recommended_gpu_memory_gb'] > 16:
            logger.warning(f"     ⚠️ Требует >16GB GPU памяти - может быть непрактично")
        elif estimates_moe['recommended_gpu_memory_gb'] > 8:
            logger.info(f"     💡 Требует высококлассную GPU (>8GB)")
        else:
            logger.info(f"     ✅ Подходит для обычных GPU")


def example_6_adaptive_optimization():
    """Пример 6: Адаптивная оптимизация производительности"""
    logger.info("📝 Пример 6: Адаптивная оптимизация производительности")
    
    from core.lattice.spatial_optimization.unified_spatial_optimizer import (
        create_unified_spatial_optimizer,
        OptimizationConfig,
        OptimizationMode
    )
    
    # Создаем оптимизатор в AUTO режиме для адаптации
    config = OptimizationConfig(
        mode=OptimizationMode.AUTO,
        target_performance_ms=30.0  # Целевая производительность
    )
    
    dimensions = (8, 8, 8)
    optimizer = create_unified_spatial_optimizer(dimensions, config)
    
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    state_size = 32
    
    logger.info(f"   🎯 Целевая производительность: {config.target_performance_ms}ms")
    logger.info(f"   🔄 Запуск серии оптимизаций для адаптации...")
    
    # Запускаем серию оптимизаций для демонстрации адаптации
    for iteration in range(10):
        # Создаем новые данные каждый раз
        states = torch.randn(total_cells, state_size)
        
        result = optimizer.optimize_lattice_forward(states)
        
        logger.info(
            f"     🔄 Итерация {iteration + 1}: "
            f"{result.processing_time_ms:.1f}ms, "
            f"режим: {result.mode_used.value}"
        )
        
        # Показываем адаптацию каждые 3 итерации
        if (iteration + 1) % 3 == 0:
            stats = optimizer.get_comprehensive_stats()
            if 'performance_analysis' in stats:
                perf = stats['performance_analysis']
                logger.info(
                    f"       📊 Средняя производительность: {perf['avg_time_ms']:.1f}ms"
                )
                logger.info(
                    f"       📈 Распределение режимов: {perf['mode_distribution']}"
                )
    
    # Финальная статистика
    final_stats = optimizer.get_comprehensive_stats()
    logger.info("   📋 Финальная статистика адаптации:")
    
    if 'performance_analysis' in final_stats:
        perf = final_stats['performance_analysis']
        logger.info(f"     ⏱️ Итоговая средняя производительность: {perf['avg_time_ms']:.1f}ms")
        
        target_achieved = perf['avg_time_ms'] <= config.target_performance_ms
        logger.info(f"     🎯 Целевая производительность достигнута: {'✅' if target_achieved else '❌'}")
    
    # Принудительная оптимизация
    logger.info("   🔧 Запуск принудительной оптимизации...")
    optimizer.optimize_performance()
    
    optimizer.cleanup()


def run_all_examples():
    """Запуск всех примеров"""
    logger.info("🚀 Запуск всех примеров использования Unified Spatial Optimizer")
    logger.info("=" * 80)
    
    examples = [
        ("Базовое использование", example_1_basic_usage),
        ("Кастомная конфигурация", example_2_custom_configuration),
        ("MoE интеграция", example_3_moe_integration),
        ("Сравнение производительности", example_4_performance_comparison),
        ("Оценка памяти", example_5_memory_estimation),
        ("Адаптивная оптимизация", example_6_adaptive_optimization),
    ]
    
    for example_name, example_func in examples:
        try:
            logger.info(f"\n{'=' * 60}")
            example_func()
            logger.info(f"✅ {example_name} - выполнен успешно")
            
        except Exception as e:
            logger.error(f"❌ {example_name} - ошибка: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 Все примеры выполнены!")
    logger.info("📖 Подробнее в SPATIAL_OPTIMIZATION_MIGRATION_GUIDE.md")


if __name__ == "__main__":
    run_all_examples()