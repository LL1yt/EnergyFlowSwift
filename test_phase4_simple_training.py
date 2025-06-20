#!/usr/bin/env python3
"""
Простой тест обучения Phase 4 с минимальными данными
"""

import torch
import tempfile
import yaml
from pathlib import Path
import logging


def create_minimal_training_data():
    """Создание минимальных тренировочных данных"""
    print("📊 СОЗДАНИЕ МИНИМАЛЬНЫХ ДАННЫХ")
    print("=" * 40)

    # Простые диалоги для тестирования
    dialogue_pairs = [
        {
            "question": "Hello, how are you?",
            "answer": "I'm doing well, thank you!",
            "quality_score": 0.9,
        },
        {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "quality_score": 0.8,
        },
        {
            "question": "Tell me about neural networks.",
            "answer": "Neural networks are computational models inspired by the brain.",
            "quality_score": 0.85,
        },
    ]

    print(f"✅ Создано {len(dialogue_pairs)} диалогов")
    return dialogue_pairs


def test_simple_lattice_creation():
    """Тест создания простой решетки"""
    print("\n🧱 СОЗДАНИЕ ПРОСТОЙ РЕШЕТКИ")
    print("=" * 40)

    try:
        from core.lattice_3d.config import LatticeConfig
        from core.lattice_3d.lattice import Lattice3D

        # Создаем минимальную конфигурацию
        config = LatticeConfig(
            dimensions=(6, 6, 6),  # 216 клеток - достаточно мало для теста
            gpu_enabled=True,
            parallel_processing=True,
            enable_logging=True,
            batch_size=1,
        )

        print(f"Конфигурация решетки:")
        print(f"  Размеры: {config.dimensions}")
        print(f"  Общее клеток: {config.total_cells}")
        print(f"  GPU включен: {config.gpu_enabled}")

        # Создаем решетку
        lattice = Lattice3D(config)

        print(f"✅ Решетка создана:")
        print(f"  Device: {lattice.device}")
        print(f"  States shape: {lattice.states.shape}")
        print(f"  Cell prototype: {type(lattice.cell_prototype).__name__}")

        # Проверяем GPU память
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**2
            print(f"  GPU память: {memory_used:.1f} MB")

        return lattice, config

    except Exception as e:
        print(f"❌ Ошибка создания решетки: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_simple_forward_pass():
    """Тест простого прохода вперед"""
    print("\n⚡ ТЕСТ ПРОСТОГО ПРОХОДА")
    print("=" * 40)

    lattice, config = test_simple_lattice_creation()
    if lattice is None:
        return False

    try:
        # Создаем простые входные данные
        batch_size = 1
        input_size = lattice.cell_prototype.external_input_size

        # Простые входные данные
        inputs = torch.randn(batch_size, len(lattice.input_indices), input_size)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        print(f"Входные данные: {inputs.shape}")
        print(f"Входные точки: {len(lattice.input_indices)}")
        print(f"Выходные точки: {len(lattice.output_indices)}")

        # Делаем проход вперед
        initial_memory = (
            torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        )

        outputs = lattice.forward(inputs)

        final_memory = (
            torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        )

        print(f"✅ Проход завершен:")
        print(f"  Выходные данные: {outputs.shape}")
        print(f"  Output device: {outputs.device}")
        if torch.cuda.is_available():
            print(f"  Память использована: {final_memory - initial_memory:.1f} MB")

        return True

    except Exception as e:
        print(f"❌ Ошибка прохода: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_training_step():
    """Тест простого шага обучения"""
    print("\n🎓 ТЕСТ ПРОСТОГО ОБУЧЕНИЯ")
    print("=" * 40)

    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        # Создаем конфигурацию обучения
        generator = DynamicConfigGenerator()
        config = generator.generate_config("development")

        # Уменьшаем размеры для быстрого теста
        config["lattice"]["xs"] = 8
        config["lattice"]["ys"] = 8
        config["lattice"]["zs"] = 8

        print(f"Конфигурация обучения:")
        print(
            f"  Lattice: {config['lattice']['xs']}×{config['lattice']['ys']}×{config['lattice']['zs']}"
        )
        print(f"  Architecture: {config['architecture']['neuron_architecture']}")
        print(f"  Hybrid mode: {config['architecture']['hybrid_mode']}")
        print(f"  Cell arch: {config['emergent_training']['cell_architecture']}")

        # Проверяем GPU настройки
        lattice_3d = config.get("lattice_3d", {})
        training = config.get("training", {})

        print(f"  GPU enabled: {lattice_3d.get('gpu_enabled')}")
        print(f"  Training device: {training.get('device')}")
        print(f"  Mixed precision: {training.get('mixed_precision')}")

        print("✅ Конфигурация готова для обучения")
        return True

    except Exception as e:
        print(f"❌ Ошибка конфигурации обучения: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_optimization():
    """Тест оптимизации памяти"""
    print("\n💾 ТЕСТ ОПТИМИЗАЦИИ ПАМЯТИ")
    print("=" * 40)

    if not torch.cuda.is_available():
        print("⚠️  CUDA недоступен, пропускаем тест памяти")
        return True

    try:
        # Очищаем GPU память
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1024**2

        print(f"Начальная память: {initial_memory:.1f} MB")

        # Создаем решетку среднего размера
        from core.lattice_3d.config import LatticeConfig
        from core.lattice_3d.lattice import Lattice3D

        config = LatticeConfig(
            dimensions=(16, 16, 16),  # 4096 клеток
            gpu_enabled=True,
            mixed_precision=True,  # Включаем mixed precision
            memory_efficient=True,
        )

        lattice = Lattice3D(config)

        after_lattice_memory = torch.cuda.memory_allocated(0) / 1024**2
        lattice_memory = after_lattice_memory - initial_memory

        print(
            f"Память после создания решетки: {after_lattice_memory:.1f} MB (+{lattice_memory:.1f} MB)"
        )

        # Тест с mixed precision
        with torch.cuda.amp.autocast():
            inputs = torch.randn(
                1,
                len(lattice.input_indices),
                lattice.cell_prototype.external_input_size,
            ).cuda()
            outputs = lattice.forward(inputs)

        final_memory = torch.cuda.memory_allocated(0) / 1024**2
        forward_memory = final_memory - after_lattice_memory

        print(
            f"Память после forward pass: {final_memory:.1f} MB (+{forward_memory:.1f} MB)"
        )

        # Очищаем память
        del lattice, inputs, outputs
        torch.cuda.empty_cache()

        cleaned_memory = torch.cuda.memory_allocated(0) / 1024**2
        print(f"Память после очистки: {cleaned_memory:.1f} MB")

        print("✅ Тест оптимизации памяти завершен")
        return True

    except Exception as e:
        print(f"❌ Ошибка теста памяти: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 ПРОСТОЙ ТЕСТ ОБУЧЕНИЯ - PHASE 4")
    print("=" * 60)

    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    results = []

    # Тест 1: Создание данных
    dialogue_data = create_minimal_training_data()
    results.append(("Data Creation", dialogue_data is not None))

    # Тест 2: Простой проход
    results.append(("Forward Pass", test_simple_forward_pass()))

    # Тест 3: Конфигурация обучения
    results.append(("Training Config", test_simple_training_step()))

    # Тест 4: Оптимизация памяти
    results.append(("Memory Optimization", test_memory_optimization()))

    # Результаты
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ПРОСТОГО ТЕСТА:")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} | {name}")
        if result:
            passed += 1

    print(f"\n🎯 ИТОГО: {passed}/{len(results)} тестов прошли")

    if passed == len(results):
        print("🎉 ВСЕ ПРОСТЫЕ ТЕСТЫ ПРОШЛИ!")
        print("🚀 ГОТОВО К ПОЛНОМУ ТЕСТУ ОБУЧЕНИЯ!")
        print("\nСледующий шаг:")
        print("  python test_phase4_full_training_cycle.py")
    else:
        print("⚠️  Некоторые тесты не прошли. Требуется диагностика.")


if __name__ == "__main__":
    main()
