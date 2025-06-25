#!/usr/bin/env python3
"""
Тест Phase 2: Базовое тестирование 3D Lattice
==============================================

Тестирует интеграцию клеток с 3D решеткой.
"""

import torch
import logging

# Настройка централизованного логирования
from new_rebuild.utils.logging import setup_logging
from new_rebuild.config import get_project_config

# Получаем конфигурацию для определения debug_mode
config = get_project_config()
setup_logging(debug_mode=config.debug_mode)


def test_phase2_lattice_integration():
    """Тестирует базовую интеграцию lattice с клетками."""

    print("🧪 PHASE 2 TEST: Lattice Integration")
    print("=" * 50)

    try:
        # Импорт компонентов
        from new_rebuild.config import get_project_config
        from new_rebuild.core.lattice import create_lattice

        print("✅ Imports successful")

        # Получаем конфигурацию
        config = get_project_config()
        print(f"✅ Config loaded: {config.architecture_type}")
        print(f"   Lattice dimensions: {config.lattice_dimensions}")
        print(f"   Device: {config.device}")

        # Создаем решетку
        print("\n🏗️ Creating lattice...")
        lattice = create_lattice()
        print(f"✅ Lattice created successfully")

        # Проверяем основные компоненты
        print(f"   Total cells: {lattice.pos_helper.total_positions}")
        print(f"   State shape: {lattice.states.shape}")
        print(f"   Cell type: {type(lattice.cells).__name__}")
        print(f"   Input points: {len(lattice.input_points)}")
        print(f"   Output points: {len(lattice.output_points)}")

        # Тестируем forward pass
        print("\n🔄 Testing forward pass...")
        initial_states = lattice.states.clone()

        # Выполняем несколько шагов
        for step in range(3):
            new_states = lattice.forward()
            print(
                f"   Step {step+1}: states changed = {not torch.equal(initial_states, new_states)}"
            )
            initial_states = new_states.clone()

        # Проверяем валидацию
        print("\n📊 Validation...")
        validation_stats = lattice.validate_lattice()
        print(f"   Architecture: {validation_stats['architecture_type']}")
        print(
            f"   Topology neighbors: {validation_stats['topology']['avg_neighbors']:.1f}"
        )

        # Проверяем производительность
        perf_stats = lattice.get_performance_stats()
        print(f"   Steps performed: {perf_stats['total_steps']}")
        print(f"   Avg time per step: {perf_stats['avg_time_per_step']*1000:.2f}ms")

        # Тестируем I/O операции
        print("\n📥📤 Testing I/O operations...")
        input_states = lattice.get_input_states()
        output_states = lattice.get_output_states()
        print(f"   Input states shape: {input_states.shape}")
        print(f"   Output states shape: {output_states.shape}")

        # Устанавливаем случайные входные состояния
        random_inputs = torch.randn_like(input_states) * 0.5
        lattice.set_input_states(random_inputs)
        print("   ✅ Input states set successfully")

        # Выполняем forward pass с новыми входами
        states_after_input = lattice.forward()
        print(f"   ✅ Forward pass with inputs successful")

        print("\n🎉 PHASE 2 TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Statistics:")
        print(
            f"   Total parameters: ~{sum(p.numel() for p in lattice.cells.parameters()):,}"
        )
        print(f"   Memory usage: ~{lattice.states.numel() * 4 / 1024:.1f} KB")
        print(
            f"   Topology efficiency: {validation_stats['topology']['avg_neighbors']:.1f}/{validation_stats['topology']['target_neighbors']} neighbors"
        )

        return True

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_lattice_configurations():
    """Тестирует различные конфигурации решетки."""

    print("\n🔧 Testing Different Configurations")
    print("-" * 40)

    try:
        from new_rebuild.config import get_project_config

        config = get_project_config()
        original_arch = config.architecture_type

        # Тестируем разные архитектуры
        architectures = ["nca", "gmlp", "hybrid"]

        for arch in architectures:
            print(f"\n🧪 Testing {arch.upper()} architecture...")

            # Временно меняем архитектуру
            config.architecture_type = arch

            from new_rebuild.core.lattice import create_lattice

            lattice = create_lattice()

            # Быстрый тест
            states = lattice.forward()
            print(
                f"   ✅ {arch}: forward pass successful, output shape: {states.shape}"
            )

            # Очистка для следующего теста
            del lattice
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Восстанавливаем исходную архитектуру
        config.architecture_type = original_arch

        print("\n✅ All architecture configurations tested successfully!")
        return True

    except Exception as e:
        print(f"❌ Configuration test error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 STARTING PHASE 2 LATTICE TESTS")
    print("=" * 60)

    success = True

    # Основной тест
    success &= test_phase2_lattice_integration()

    # Тест конфигураций
    success &= test_lattice_configurations()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Phase 2 lattice integration is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")

    print("=" * 60)
