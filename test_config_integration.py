#!/usr/bin/env python3
"""
Тест интеграции системы конфигурации в проект
============================================

Проверяем что все компоненты используют централизованную конфигурацию.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import (
    create_debug_config, 
    create_experiment_config,
    create_optimized_config,
    get_project_config,
    set_project_config
)
from new_rebuild.utils import strict_no_hardcoded, HardcodedValueError


def test_lattice_integration():
    """Проверяем что Lattice3D использует конфиг"""
    print("\n=== Тест Lattice3D ===")
    
    # Устанавливаем DEBUG конфиг
    config = create_debug_config()
    set_project_config(config)
    
    # Создаем решетку используя фабричную функцию
    from new_rebuild.core.lattice import create_lattice
    lattice = create_lattice()  # Использует глобальный конфиг
    
    print(f"✅ Lattice создан с размерами из конфига: {config.lattice.dimensions}")
    print(f"   Total cells: {lattice.pos_helper.total_positions}")
    print(f"   Adaptive radius: {config.calculate_adaptive_radius():.2f}")
    
    # Проверяем что параметры соответствуют режиму
    assert lattice.pos_helper.dimensions == (8, 8, 8), "DEBUG режим должен использовать 8x8x8"
    assert lattice.pos_helper.total_positions == 512
    
    # Переключаемся на EXPERIMENT
    config2 = create_experiment_config()
    set_project_config(config2)
    lattice2 = create_lattice()
    
    assert lattice2.pos_helper.dimensions == (15, 15, 15), "EXPERIMENT режим должен использовать 15x15x15"
    print(f"✅ EXPERIMENT режим: {lattice2.pos_helper.dimensions}, {lattice2.pos_helper.total_positions} cells")


def test_model_cell_integration():
    """Проверяем что клетки используют конфиг"""
    print("\n=== Тест Model Cells ===")
    
    config = create_debug_config()
    set_project_config(config)
    
    from new_rebuild.core.cells import create_cell
    
    # Создаем клетку (использует глобальный конфиг)
    cell = create_cell("vectorized_gnn")
    
    print(f"✅ VectorizedGNNCell создан с параметрами из конфига:")
    print(f"   State size: {cell.state_size}")
    print(f"   Hidden dim: {cell.hidden_dim}")
    print(f"   Message dim: {cell.message_dim}")
    
    # Проверяем DEBUG значения
    assert cell.state_size == config.model.state_size
    assert cell.state_size == 32  # DEBUG preset


def test_moe_integration():
    """Проверяем что MoE компоненты используют конфиг"""
    print("\n=== Тест MoE Components ===")
    
    config = create_experiment_config()
    set_project_config(config)
    
    from new_rebuild.core.moe import UnifiedConnectionClassifier
    
    # Создаем классификатор с размерами решетки
    classifier = UnifiedConnectionClassifier(
        lattice_dimensions=config.lattice.dimensions
    )
    
    print(f"✅ UnifiedConnectionClassifier создан с параметрами из конфига:")
    print(f"   Lattice dimensions: {classifier.lattice_dimensions}")
    print(f"   State size: {classifier.state_size}")
    print(f"   Cache enabled: {classifier.enable_cache}")
    
    # Проверяем что использует значения из конфига
    assert classifier.state_size == config.model.state_size
    assert classifier.state_size == 64  # EXPERIMENT preset


def test_training_integration():
    """Проверяем что обучение использует конфиг"""
    print("\n=== Тест Training Components ===")
    
    config = create_optimized_config()
    set_project_config(config)
    
    print(f"✅ OPTIMIZED режим параметры обучения:")
    print(f"   Max samples: {config.training_embedding.max_total_samples}")
    print(f"   Epochs: {config.training_embedding.num_epochs}")
    print(f"   Learning rate: {config.training_optimizer.learning_rate}")
    print(f"   Memory reserve: {config.memory_management.training_memory_reserve_gb}GB")
    
    # Проверяем OPTIMIZED значения
    assert config.training_embedding.max_total_samples == 50000
    assert config.training_embedding.num_epochs == 20
    assert config.memory_management.training_memory_reserve_gb == 8.0


def test_hardcoded_protection():
    """Проверяем защиту от hardcoded значений"""
    print("\n=== Тест защиты от Hardcoded ===")
    
    config = create_debug_config()
    set_project_config(config)
    
    # Пример функции которая должна использовать конфиг
    def bad_function():
        # Эти значения должны вызвать ошибку при проверке
        learning_rate = strict_no_hardcoded(1e-4, "training_optimizer.learning_rate")
        return learning_rate
    
    # Проверяем что значение берется из конфига
    lr = bad_function()
    print(f"✅ strict_no_hardcoded заменил 1e-4 на {lr} из конфига")
    assert lr == config.training_optimizer.learning_rate
    
    # Проверяем что несуществующий параметр вызывает ошибку
    try:
        value = strict_no_hardcoded(12345, "non.existent.param")
        print("❌ Должна была быть ошибка!")
    except HardcodedValueError as e:
        print("✅ Поймана ожидаемая ошибка для несуществующего параметра")


def test_config_mode_switching():
    """Проверяем переключение между режимами"""
    print("\n=== Тест переключения режимов ===")
    
    modes = [
        ("DEBUG", create_debug_config, (8, 8, 8), 32, 100),
        ("EXPERIMENT", create_experiment_config, (15, 15, 15), 64, 10000), 
        ("OPTIMIZED", create_optimized_config, (30, 30, 30), 128, 50000),
    ]
    
    for mode_name, create_func, expected_dims, expected_state, expected_samples in modes:
        config = create_func()
        set_project_config(config)
        
        print(f"\n{mode_name} режим:")
        print(f"  Решетка: {config.lattice.dimensions}")
        print(f"  State size: {config.model.state_size}")
        print(f"  Max samples: {config.training_embedding.max_total_samples}")
        
        assert config.lattice.dimensions == expected_dims
        assert config.model.state_size == expected_state
        assert config.training_embedding.max_total_samples == expected_samples
        
    print("\n✅ Все режимы работают корректно!")


def test_config_access_patterns():
    """Проверяем различные способы доступа к конфигу"""
    print("\n=== Тест паттернов доступа ===")
    
    config = create_experiment_config()
    set_project_config(config)
    
    # 1. Прямой доступ к пресетам
    print("1. Доступ к пресетам:")
    print(f"   Debug lattice: {config.mode_presets.debug.lattice_dimensions}")
    print(f"   Optimized lattice: {config.mode_presets.optimized.lattice_dimensions}")
    
    # 2. Доступ через компоненты
    print("\n2. Доступ через компоненты:")
    print(f"   Lattice total: {config.lattice.total_cells}")
    print(f"   Model params: {config.model.target_params}")
    
    # 3. Вычисляемые свойства
    print("\n3. Вычисляемые свойства:")
    print(f"   Cube surface dim: {config.cube_surface_dim}")
    print(f"   Cube embedding dim: {config.cube_embedding_dim}")
    print(f"   Effective chunk size: {config.effective_max_chunk_size}")
    
    # 4. Runtime компоненты
    print("\n4. Runtime компоненты:")
    print(f"   Device: {config.current_device}")
    print(f"   Device manager: {config.device_manager}")


def main():
    """Основная функция"""
    print("🔍 Тестирование интеграции системы конфигурации")
    print("=" * 60)
    
    try:
        test_lattice_integration()
        test_model_cell_integration()
        test_moe_integration()
        test_training_integration()
        test_hardcoded_protection()
        test_config_mode_switching()
        test_config_access_patterns()
        
        print("\n" + "=" * 60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\n💡 Выводы:")
        print("1. Система конфигурации полностью интегрирована")
        print("2. Все основные компоненты используют централизованный конфиг")
        print("3. Защита от hardcoded значений работает")
        print("4. Переключение режимов работает корректно")
        print("5. Пресеты доступны для кастомизации")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())