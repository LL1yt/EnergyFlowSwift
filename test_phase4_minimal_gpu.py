#!/usr/bin/env python3
"""
Минимальный тест GPU и эмбедингов для Phase 4
"""

import torch
import tempfile
import yaml
from pathlib import Path


def test_gpu_detection():
    """Тест обнаружения GPU"""
    print("🔍 ПРОВЕРКА GPU")
    print("=" * 40)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступен: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # Тест создания тензора на GPU
        try:
            test_tensor = torch.randn(100, 100).cuda()
            print(f"✅ Тензор на GPU создан: {test_tensor.device}")

            # Проверяем использование памяти
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            print(f"GPU память использована: {memory_allocated:.1f} MB")

            return True
        except Exception as e:
            print(f"❌ Ошибка создания тензора на GPU: {e}")
            return False
    else:
        print("❌ CUDA недоступен")
        return False


def test_lattice_gpu_config():
    """Тест конфигурации GPU для lattice"""
    print("\n🧱 ПРОВЕРКА LATTICE GPU КОНФИГУРАЦИИ")
    print("=" * 40)

    try:
        from core.lattice_3d.config import LatticeConfig

        # Создаем конфигурацию
        config = LatticeConfig(
            dimensions=(4, 4, 4),  # Маленькая решетка для теста
            gpu_enabled=True,
            parallel_processing=True,
        )

        print(f"gpu_enabled: {config.gpu_enabled}")
        print(f"parallel_processing: {config.parallel_processing}")
        print(f"total_cells: {config.total_cells}")

        # Создаем решетку
        from core.lattice_3d.lattice import Lattice3D

        lattice = Lattice3D(config)
        print(f"Lattice device: {lattice.device}")
        print(f"States shape: {lattice.states.shape}")
        print(f"States device: {lattice.states.device}")

        if torch.cuda.is_available():
            expected_device = "cuda"
        else:
            expected_device = "cpu"

        if str(lattice.device) == expected_device:
            print("✅ Lattice использует правильное устройство")
            return True
        else:
            print(f"❌ Lattice на {lattice.device}, ожидалось {expected_device}")
            return False

    except Exception as e:
        print(f"❌ Ошибка создания lattice: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_minimal_training_config():
    """Тест минимальной конфигурации обучения"""
    print("\n⚙️ ПРОВЕРКА МИНИМАЛЬНОЙ КОНФИГУРАЦИИ ОБУЧЕНИЯ")
    print("=" * 40)

    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        # Создаем генератор
        generator = DynamicConfigGenerator()
        config = generator.generate_config("development")

        # Проверяем ключевые секции
        print("Проверка секций конфигурации:")

        # Architecture
        architecture = config.get("architecture", {})
        print(f"  architecture.hybrid_mode: {architecture.get('hybrid_mode')}")
        print(
            f"  architecture.neuron_architecture: {architecture.get('neuron_architecture')}"
        )

        # Lattice_3d
        lattice_3d = config.get("lattice_3d", {})
        print(f"  lattice_3d.gpu_enabled: {lattice_3d.get('gpu_enabled')}")
        print(
            f"  lattice_3d.parallel_processing: {lattice_3d.get('parallel_processing')}"
        )

        # Training
        training = config.get("training", {})
        print(f"  training.device: {training.get('device')}")
        print(f"  training.mixed_precision: {training.get('mixed_precision')}")

        # Emergent training
        emergent = config.get("emergent_training", {})
        print(
            f"  emergent_training.cell_architecture: {emergent.get('cell_architecture')}"
        )

        # Lattice размеры
        lattice = config.get("lattice", {})
        print(
            f"  lattice размеры: {lattice.get('xs')}×{lattice.get('ys')}×{lattice.get('zs')}"
        )

        return True

    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_embedding_creation():
    """Тест простого создания эмбедингов"""
    print("\n📊 ПРОВЕРКА ПРОСТЫХ ЭМБЕДИНГОВ")
    print("=" * 40)

    try:
        # Создаем простые тестовые данные
        test_texts = ["Hello world", "Test sentence", "Simple example"]

        # Пытаемся создать простые эмбединги
        from sentence_transformers import SentenceTransformer

        # Используем простую модель
        model = SentenceTransformer("all-MiniLM-L6-v2")

        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ Модель перемещена на GPU")

        embeddings = model.encode(test_texts)
        print(f"✅ Эмбединги созданы: {embeddings.shape}")

        # Конвертируем в torch tensor
        embeddings_tensor = torch.tensor(embeddings)
        if torch.cuda.is_available():
            embeddings_tensor = embeddings_tensor.cuda()

        print(
            f"✅ Tensor эмбедингов: {embeddings_tensor.shape}, device: {embeddings_tensor.device}"
        )

        return True

    except Exception as e:
        print(f"❌ Ошибка создания эмбедингов: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 МИНИМАЛЬНЫЙ ТЕСТ GPU И ЭМБЕДИНГОВ - PHASE 4")
    print("=" * 60)

    results = []

    # Тест 1: GPU обнаружение
    results.append(("GPU Detection", test_gpu_detection()))

    # Тест 2: Lattice GPU конфигурация
    results.append(("Lattice GPU Config", test_lattice_gpu_config()))

    # Тест 3: Минимальная конфигурация
    results.append(("Training Config", test_minimal_training_config()))

    # Тест 4: Простые эмбединги
    results.append(("Simple Embeddings", test_simple_embedding_creation()))

    # Результаты
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} | {name}")
        if result:
            passed += 1

    print(f"\n🎯 ИТОГО: {passed}/{len(results)} тестов прошли")

    if passed == len(results):
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ! GPU ГОТОВ К ИСПОЛЬЗОВАНИЮ!")
    else:
        print("⚠️  Некоторые тесты не прошли. Требуется диагностика.")


if __name__ == "__main__":
    main()
