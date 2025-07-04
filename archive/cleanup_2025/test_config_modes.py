#!/usr/bin/env python3
"""
Тест 3 режимов конфигурации: DEBUG, EXPERIMENT, OPTIMIZED
=========================================================

Проверяет, что все режимы корректно применяются и параметры
правильно устанавливаются.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import (
    create_debug_config,
    create_experiment_config,
    create_optimized_config,
    ConfigMode,
    set_project_config,
    get_project_config,
)


def test_debug_mode():
    """Тест DEBUG режима"""
    print("\n=== Тестирование DEBUG режима ===")
    
    config = create_debug_config()
    set_project_config(config)
    
    # Проверяем основные параметры
    assert config.mode.mode == ConfigMode.DEBUG
    assert config.lattice.dimensions == (8, 8, 8)
    assert config.model.state_size == 32
    assert config.model.target_params == 8000
    
    # Проверяем логирование
    assert config.logging.debug_mode == True
    assert config.logging.level == "DEBUG"
    assert config.logging.performance_tracking == True
    
    # Проверяем параметры обучения
    assert config.training_embedding.max_total_samples == 50
    assert config.training_embedding.num_epochs == 1
    assert config.training_embedding.test_mode == True
    
    # Проверяем централизованные параметры
    assert config.architecture.moe_functional_params == 2000
    assert config.architecture.moe_distant_params == 1000
    assert config.architecture.spatial_max_neighbors == 100
    
    # Проверяем память
    assert config.memory_management.training_memory_reserve_gb == 2.0
    assert config.memory_management.dataloader_workers == 2
    
    print(f"✅ DEBUG режим: решетка {config.lattice.dimensions}, "
          f"{config.lattice.total_cells} клеток, "
          f"state_size={config.model.state_size}")
    print(f"   Логирование: {config.logging.level}, debug={config.logging.debug_mode}")
    print(f"   Обучение: {config.training_embedding.max_total_samples} samples, "
          f"{config.training_embedding.num_epochs} epochs")
    print(f"   MoE: functional={config.architecture.moe_functional_params}, "
          f"distant={config.architecture.moe_distant_params}")


def test_experiment_mode():
    """Тест EXPERIMENT режима"""
    print("\n=== Тестирование EXPERIMENT режима ===")
    
    config = create_experiment_config()
    set_project_config(config)
    
    # Проверяем основные параметры
    assert config.mode.mode == ConfigMode.EXPERIMENT
    assert config.lattice.dimensions == (15, 15, 15)
    assert config.model.state_size == 64
    assert config.model.target_params == 25000
    
    # Проверяем логирование
    assert config.logging.debug_mode == False
    assert config.logging.level == "INFO"
    
    # Проверяем параметры обучения
    assert config.training_embedding.max_total_samples == 1000
    assert config.training_embedding.num_epochs == 10
    assert config.training_embedding.test_mode == False
    
    # Проверяем централизованные параметры
    assert config.architecture.moe_functional_params == 8000
    assert config.architecture.moe_distant_params == 4000
    assert config.architecture.spatial_max_neighbors == 1000
    
    # Проверяем память
    assert config.memory_management.training_memory_reserve_gb == 10.0
    assert config.memory_management.dataloader_workers == 4
    
    print(f"✅ EXPERIMENT режим: решетка {config.lattice.dimensions}, "
          f"{config.lattice.total_cells} клеток, "
          f"state_size={config.model.state_size}")
    print(f"   Логирование: {config.logging.level}, debug={config.logging.debug_mode}")
    print(f"   Обучение: {config.training_embedding.max_total_samples} samples, "
          f"{config.training_embedding.num_epochs} epochs")
    print(f"   MoE: functional={config.architecture.moe_functional_params}, "
          f"distant={config.architecture.moe_distant_params}")


def test_optimized_mode():
    """Тест OPTIMIZED режима"""
    print("\n=== Тестирование OPTIMIZED режима ===")
    
    config = create_optimized_config()
    set_project_config(config)
    
    # Проверяем основные параметры
    assert config.mode.mode == ConfigMode.OPTIMIZED
    assert config.lattice.dimensions == (30, 30, 30)
    assert config.model.state_size == 128
    assert config.model.target_params == 100000
    
    # Проверяем логирование
    assert config.logging.debug_mode == False
    assert config.logging.level == "WARNING"
    assert config.logging.performance_tracking == False
    
    # Проверяем параметры обучения
    assert config.training_embedding.max_total_samples is None
    assert config.training_embedding.num_epochs == 100
    assert config.training_embedding.test_mode == False
    
    # Проверяем централизованные параметры
    assert config.architecture.moe_functional_params == 15000
    assert config.architecture.moe_distant_params == 8000
    assert config.architecture.spatial_max_neighbors == 2000
    
    # Проверяем память
    assert config.memory_management.training_memory_reserve_gb == 20.0
    assert config.memory_management.dataloader_workers == 8
    
    # Проверяем оптимизации
    assert config.performance is not None
    assert config.performance.enable_jit == True
    assert config.performance.benchmark_mode == True
    
    print(f"✅ OPTIMIZED режим: решетка {config.lattice.dimensions}, "
          f"{config.lattice.total_cells} клеток, "
          f"state_size={config.model.state_size}")
    print(f"   Логирование: {config.logging.level}, debug={config.logging.debug_mode}")
    print(f"   Обучение: samples=unlimited, "
          f"{config.training_embedding.num_epochs} epochs")
    print(f"   MoE: functional={config.architecture.moe_functional_params}, "
          f"distant={config.architecture.moe_distant_params}")
    print(f"   Оптимизации: JIT={config.performance.enable_jit}, "
          f"benchmark={config.performance.benchmark_mode}")


def test_mode_switching():
    """Тест переключения между режимами"""
    print("\n=== Тестирование переключения режимов ===")
    
    # Начинаем с DEBUG
    config = create_debug_config()
    assert config.lattice.dimensions == (8, 8, 8)
    print("✅ Создан DEBUG конфиг")
    
    # Переключаемся на EXPERIMENT
    config = create_experiment_config()
    assert config.lattice.dimensions == (15, 15, 15)
    print("✅ Переключились на EXPERIMENT")
    
    # Переключаемся на OPTIMIZED
    config = create_optimized_config()
    assert config.lattice.dimensions == (30, 30, 30)
    print("✅ Переключились на OPTIMIZED")
    
    # Проверяем глобальный конфиг
    set_project_config(config)
    global_config = get_project_config()
    assert global_config.mode.mode == ConfigMode.OPTIMIZED
    print("✅ Глобальный конфиг обновлен")


def test_custom_overrides():
    """Тест переопределения параметров"""
    print("\n=== Тестирование переопределения параметров ===")
    
    # Создаем DEBUG конфиг с кастомными параметрами
    config = create_debug_config(
        lattice={"dimensions": (10, 10, 10)},
        model={"state_size": 48}
    )
    
    # Режим применился, но наши переопределения тоже работают
    assert config.mode.mode == ConfigMode.DEBUG
    assert config.lattice.dimensions == (10, 10, 10)  # Наше значение
    assert config.model.state_size == 48  # Наше значение
    assert config.logging.debug_mode == True  # Из режима DEBUG
    
    print("✅ Переопределение параметров работает корректно")
    print(f"   Кастомная решетка: {config.lattice.dimensions}")
    print(f"   Кастомный state_size: {config.model.state_size}")
    print(f"   Режим логирования из DEBUG: {config.logging.level}")


def test_centralized_parameters():
    """Тест новых централизованных параметров"""
    print("\n=== Тестирование централизованных параметров ===")
    
    config = create_experiment_config()
    
    # Проверяем TrainingOptimizerSettings
    assert config.training_optimizer.learning_rate == 1e-4
    assert config.training_optimizer.weight_decay == 1e-5
    assert config.training_optimizer.gradient_clip_max_norm == 1.0
    print("✅ TrainingOptimizerSettings работает")
    
    # Проверяем EmbeddingMappingSettings
    assert config.embedding_mapping.surface_coverage == 0.8
    assert config.embedding_mapping.lattice_steps == 5
    assert config.embedding_mapping.attention_num_heads == 4
    print("✅ EmbeddingMappingSettings работает")
    
    # Проверяем MemoryManagementSettings
    assert config.memory_management.min_gpu_memory_gb == 8.0
    assert config.memory_management.gpu_memory_safety_factor == 0.85
    print("✅ MemoryManagementSettings работает")
    
    # Проверяем ArchitectureConstants
    assert config.architecture.teacher_embedding_dim == 768
    assert config.architecture.spatial_consistency_range == 27
    print("✅ ArchitectureConstants работает")
    
    # Проверяем AlgorithmicStrategies
    assert "faces" in config.strategies.placement_strategies
    assert config.strategies.default_cnf_mode == "adaptive"
    print("✅ AlgorithmicStrategies работает")


def main():
    """Запуск всех тестов"""
    print("🔧 Тестирование системы конфигурации с 3 режимами")
    print("=" * 60)
    
    try:
        test_debug_mode()
        test_experiment_mode()
        test_optimized_mode()
        test_mode_switching()
        test_custom_overrides()
        test_centralized_parameters()
        
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! 🎉")
        print("\nТеперь можно использовать:")
        print("  - create_debug_config() для быстрых тестов")
        print("  - create_experiment_config() для экспериментов")
        print("  - create_optimized_config() для финальных прогонов")
        
    except AssertionError as e:
        print(f"\n❌ Тест провален: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()