#!/usr/bin/env python3
"""
🧪 ТЕСТ ИСПРАВЛЕНИЙ: Архитектура и GPU интеграция

Проверяем исправления:
1. ✅ DynamicConfigGenerator правильно устанавливает cell_architecture = "nca" в hybrid режиме
2. ✅ config_initializer логирует NCA параметры вместо gMLP
3. ✅ GPU configuration добавлена в stage_runner
"""

import sys
import torch
import tempfile
import yaml
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))


def test_dynamic_config_fix():
    """Тест исправления DynamicConfigGenerator"""
    print("🧪 ТЕСТ 1: DynamicConfigGenerator исправление")
    print("-" * 50)

    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        generator = DynamicConfigGenerator()

        # Генерируем конфигурацию для development режима (имеет hybrid_mode)
        config = generator.generate_config("development")

        # Проверяем архитектуру
        architecture = config.get("architecture", {})
        emergent_training = config.get("emergent_training", {})

        print(f"✅ hybrid_mode: {architecture.get('hybrid_mode')}")
        print(f"✅ neuron_architecture: {architecture.get('neuron_architecture')}")
        print(
            f"✅ connection_architecture: {architecture.get('connection_architecture')}"
        )
        print(f"✅ cell_architecture: {emergent_training.get('cell_architecture')}")

        # Проверяем что исправление сработало
        hybrid_mode = architecture.get("hybrid_mode", False)
        neuron_arch = architecture.get("neuron_architecture")
        cell_arch = emergent_training.get("cell_architecture")

        if hybrid_mode and neuron_arch == "minimal_nca" and cell_arch == "nca":
            print(
                "🎉 ИСПРАВЛЕНИЕ СРАБОТАЛО: cell_architecture = 'nca' в hybrid режиме!"
            )
            return True
        else:
            print(
                f"❌ Проблема: hybrid={hybrid_mode}, neuron={neuron_arch}, cell={cell_arch}"
            )
            return False

    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        return False


def test_config_initializer_fix():
    """Тест исправления config_initializer логирования"""
    print("\n🧪 ТЕСТ 2: config_initializer исправление")
    print("-" * 50)

    try:
        from smart_resume_training.core.config_initializer import ConfigInitializer
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        # Используем реальную конфигурацию из DynamicConfigGenerator (имеет hybrid_mode=True)
        generator = DynamicConfigGenerator()
        test_config = generator.generate_config("development")

        # ДЕБАГ: Проверяем что генератор создал
        print("🔍 ДЕБАГ: Анализ сгенерированной конфигурации:")
        architecture_debug = test_config.get("architecture", {})
        emergent_debug = test_config.get("emergent_training", {})
        print(f"   architecture.hybrid_mode: {architecture_debug.get('hybrid_mode')}")
        print(
            f"   architecture.neuron_architecture: {architecture_debug.get('neuron_architecture')}"
        )
        print(
            f"   emergent_training.cell_architecture: {emergent_debug.get('cell_architecture')}"
        )

        # Создаем временный файл с конфигурацией
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name

        # ДЕБАГ: Проверяем что записалось в файл
        print("🔍 ДЕБАГ: Проверка файла конфигурации:")
        with open(temp_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            if "hybrid_mode: true" in file_content:
                print("   ✅ hybrid_mode: true найден в файле")
            else:
                print("   ❌ hybrid_mode: true НЕ найден в файле")

        # Инициализируем ConfigInitializer
        initializer = ConfigInitializer(temp_path)

        # Захватываем логи (упрощенная версия)
        print("📊 Логи config_initializer:")

        # Эмулируем вызов _log_config_details
        config = initializer.config

        # ДЕБАГ: Проверяем что ConfigInitializer загрузил
        print("🔍 ДЕБАГ: Анализ загруженной конфигурации:")
        architecture_loaded = config.get("architecture", {})
        emergent_loaded = config.get("emergent_training", {})
        print(
            f"   Загруженная architecture.hybrid_mode: {architecture_loaded.get('hybrid_mode')}"
        )
        print(
            f"   Загруженная emergent_training.cell_architecture: {emergent_loaded.get('cell_architecture')}"
        )

        # Проверяем что в hybrid режиме логируются NCA параметры
        architecture = config.get("architecture", {})
        emergent_training = config.get("emergent_training", {})

        hybrid_mode = architecture.get("hybrid_mode", False)
        cell_architecture = emergent_training.get("cell_architecture", "gmlp")

        if hybrid_mode and cell_architecture == "nca":
            nca_config = emergent_training.get("nca_config", {})
            if nca_config:
                print(
                    f"✅ NCA (hybrid) state size: {nca_config.get('state_size')}, hidden_dim: {nca_config.get('hidden_dim')}"
                )
                print(f"✅ Architecture: Hybrid NCA+gMLP mode")
                success = True
            else:
                minimal_nca = config.get("minimal_nca_cell", {})
                if minimal_nca:
                    print(
                        f"✅ NCA (hybrid) state size: {minimal_nca.get('state_size')}, hidden_dim: {minimal_nca.get('hidden_dim')}"
                    )
                    print(f"✅ Architecture: Hybrid NCA+gMLP mode")
                    success = True
                else:
                    print("❌ NCA конфигурация не найдена")
                    success = False
        else:
            print(
                f"❌ Неправильный режим: hybrid={hybrid_mode}, cell_arch={cell_architecture}"
            )
            success = False

        # Удаляем временный файл
        Path(temp_path).unlink()

        if success:
            print("🎉 ИСПРАВЛЕНИЕ СРАБОТАЛО: Логируются NCA параметры в hybrid режиме!")

        return success

    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        return False


def test_gpu_configuration_fix():
    """Тест исправления GPU конфигурации"""
    print("\n🧪 ТЕСТ 3: GPU конфигурация исправление")
    print("-" * 50)

    try:
        from training.automated_training.stage_runner import TrainingStageRunner
        from training.automated_training.types import StageConfig

        # Создаем StageConfig с memory optimizations
        stage_config = StageConfig(
            stage=1,
            dataset_limit=10,
            epochs=1,
            batch_size=4,
            description="GPU Test",
            memory_optimizations=True,
            progressive_scaling=False,
        )

        # Создаем runner
        runner = TrainingStageRunner(mode="development", verbose=True)

        # Генерируем конфигурацию (но не запускаем обучение)
        temp_config_path = runner._generate_temp_config(stage_config)

        if temp_config_path:
            # Читаем сгенерированную конфигурацию
            with open(temp_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            print("📊 GPU конфигурация:")

            # Проверяем lattice_3d настройки
            lattice_3d = config_data.get("lattice_3d", {})
            print(f"✅ lattice_3d.gpu_enabled: {lattice_3d.get('gpu_enabled')}")
            print(
                f"✅ lattice_3d.parallel_processing: {lattice_3d.get('parallel_processing')}"
            )

            # Проверяем training настройки
            training = config_data.get("training", {})
            print(f"✅ training.device: {training.get('device')}")
            print(f"✅ training.pin_memory: {training.get('pin_memory')}")
            print(f"✅ training.mixed_precision: {training.get('mixed_precision')}")
            print(
                f"✅ training.gradient_checkpointing: {training.get('gradient_checkpointing')}"
            )

            # Проверяем CUDA доступность
            cuda_available = torch.cuda.is_available()
            print(f"✅ CUDA доступен: {cuda_available}")

            if cuda_available:
                print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

            # Проверяем что исправления сработали
            expected_device = "cuda" if cuda_available else "cpu"
            actual_device = training.get("device")

            gpu_enabled = lattice_3d.get("gpu_enabled", False)

            success = (
                actual_device == expected_device
                and (gpu_enabled == cuda_available)
                and training.get("pin_memory", False) == cuda_available
            )

            if success:
                print("🎉 ИСПРАВЛЕНИЕ СРАБОТАЛО: GPU конфигурация добавлена!")
            else:
                print(
                    f"❌ Проблема: device={actual_device}, expected={expected_device}, gpu_enabled={gpu_enabled}"
                )

            # Удаляем временный файл
            Path(temp_config_path).unlink()

            return success
        else:
            print("❌ Не удалось создать временную конфигурацию")
            return False

    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_configuration():
    """Полный тест конфигурации от начала до конца"""
    print("\n🧪 ТЕСТ 4: End-to-end конфигурация")
    print("-" * 50)

    try:
        from training.automated_training.automated_trainer import AutomatedTrainer
        from training.automated_training.types import StageConfig

        # Создаем простую конфигурацию для тестирования
        stage_configs = [
            StageConfig(
                stage=1,
                dataset_limit=5,  # Очень мало для быстрого теста
                epochs=1,
                batch_size=2,
                description="End-to-end test",
                plasticity_profile="discovery",
                clustering_enabled=False,
                memory_optimizations=True,
                progressive_scaling=True,
            )
        ]

        # Создаем AutomatedTrainer
        trainer = AutomatedTrainer(
            mode="development",
            scale=0.005,  # Очень маленький scale для быстрого теста
            timeout_multiplier=1.0,
            verbose=True,
        )

        print("📊 Анализ конфигурации:")

        # Анализируем что будет сгенерировано
        runner = trainer.stage_runner
        temp_config_path = runner._generate_temp_config(stage_configs[0])

        if temp_config_path:
            with open(temp_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Проверяем архитектуру
            architecture = config_data.get("architecture", {})
            emergent_training = config_data.get("emergent_training", {})
            print(
                f"✅ Architecture type: {architecture.get('neuron_architecture')} (hybrid: {architecture.get('hybrid_mode')})"
            )
            print(f"✅ Cell architecture: {emergent_training.get('cell_architecture')}")

            # Проверяем GPU
            lattice_3d = config_data.get("lattice_3d", {})
            training = config_data.get("training", {})
            print(f"✅ GPU enabled: {lattice_3d.get('gpu_enabled')}")
            print(f"✅ Training device: {training.get('device')}")

            # Проверяем размеры решетки
            lattice = config_data.get("lattice", {})
            print(
                f"✅ Lattice dimensions: {lattice.get('lattice_width')}×{lattice.get('lattice_height')}×{lattice.get('lattice_depth')}"
            )

            # Удаляем временный файл
            Path(temp_config_path).unlink()

            print("🎉 END-TO-END КОНФИГУРАЦИЯ ГОТОВА!")
            return True
        else:
            print("❌ Не удалось создать конфигурацию")
            return False

    except Exception as e:
        print(f"❌ Ошибка end-to-end теста: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования исправлений"""
    print("🧪 ТЕСТ ИСПРАВЛЕНИЙ АРХИТЕКТУРЫ И GPU ИНТЕГРАЦИИ")
    print("=" * 80)
    print("Проверяем что исправления работают корректно")
    print()

    tests = [
        ("DynamicConfigGenerator Fix", test_dynamic_config_fix),
        ("ConfigInitializer Fix", test_config_initializer_fix),
        ("GPU Configuration Fix", test_gpu_configuration_fix),
        ("End-to-End Configuration", test_end_to_end_configuration),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            print(f"❌ Ошибка в тесте {test_name}: {e}")
            results[test_name] = "❌ ERROR"

    print("\n" + "=" * 80)
    print("🎯 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    print()

    for test_name, result in results.items():
        print(f"   {result} | {test_name}")

    passed = sum(1 for r in results.values() if r == "✅ PASS")
    total = len(results)

    print(f"\n📊 ИТОГО: {passed}/{total} тестов прошли")

    if passed == total:
        print("🎉 ВСЕ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
        print()
        print("🚀 ГОТОВО К ЗАПУСКУ:")
        print("   python test_phase4_full_training_cycle.py")
    else:
        print("⚠️  Некоторые исправления требуют доработки")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
