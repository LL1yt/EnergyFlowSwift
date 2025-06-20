#!/usr/bin/env python3
"""
🔍 ДИАГНОСТИКА: Проблемы архитектуры и GPU интеграции

Обнаруженные проблемы:
1. gMLP архитектура используется вместо NCA (8/8 parameters)
2. GPU память не используется (0.0MB GPU)

Цель: Найти корень проблемы и исправить конфигурацию
"""

import sys
import torch
import yaml
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))


def diagnose_gpu_integration():
    """Диагностика GPU интеграции"""
    print("🔍 ДИАГНОСТИКА GPU ИНТЕГРАЦИИ")
    print("=" * 60)

    # 1. Проверка CUDA доступности
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA доступен: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"✅ Количество GPU: {device_count}")

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {name} ({memory:.1f}GB)")

    # 2. Проверка device selection в различных модулях
    print("\n📊 ПРОВЕРКА DEVICE SELECTION:")

    # Проверяем dynamic config
    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        generator = DynamicConfigGenerator()
        detected_mode = generator.detect_hardware_mode()
        print(f"   DynamicConfigGenerator mode: {detected_mode}")
    except Exception as e:
        print(f"   DynamicConfigGenerator error: {e}")

    # Проверяем config loader
    try:
        from utils.config_loader import ConfigManager

        cm = ConfigManager()
        device_config = cm.get_device_config()
        print(f"   ConfigManager device: {device_config}")
    except Exception as e:
        print(f"   ConfigManager error: {e}")

    # 3. Проверка lattice_3d GPU settings
    print("\n🧱 ПРОВЕРКА LATTICE_3D GPU SETTINGS:")

    try:
        from core.lattice_3d.config import LatticeConfig

        config = LatticeConfig()
        print(f"   gpu_enabled: {config.gpu_enabled}")
        print(f"   parallel_processing: {config.parallel_processing}")
    except Exception as e:
        print(f"   LatticeConfig error: {e}")

    return cuda_available


def diagnose_architecture_selection():
    """Диагностика выбора архитектуры"""
    print("\n🏗️ ДИАГНОСТИКА ВЫБОРА АРХИТЕКТУРЫ")
    print("=" * 60)

    # 1. Анализ debug_final_config.yaml
    print("📋 АНАЛИЗ debug_final_config.yaml:")

    config_path = "debug_final_config.yaml"
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Проверяем ключевые секции
        architecture = config.get("architecture", {})
        print(f"   hybrid_mode: {architecture.get('hybrid_mode')}")
        print(f"   neuron_architecture: {architecture.get('neuron_architecture')}")
        print(
            f"   connection_architecture: {architecture.get('connection_architecture')}"
        )

        emergent_training = config.get("emergent_training", {})
        print(f"   cell_architecture: {emergent_training.get('cell_architecture')}")

        gmlp_config = emergent_training.get("gmlp_config", {})
        print(f"   gmlp state_size: {gmlp_config.get('state_size')}")
        print(f"   gmlp hidden_dim: {gmlp_config.get('hidden_dim')}")

        minimal_nca = config.get("minimal_nca_cell", {})
        print(f"   nca state_size: {minimal_nca.get('state_size')}")
        print(f"   nca hidden_dim: {minimal_nca.get('hidden_dim')}")

        # ПРОБЛЕМА: config_initializer логирует gmlp секцию даже в hybrid режиме!

    else:
        print(f"   ❌ {config_path} не найден")

    # 2. Проверка кода config_initializer
    print("\n🔧 ПРОВЕРКА CONFIG_INITIALIZER:")

    config_init_path = "smart_resume_training/core/config_initializer.py"
    if Path(config_init_path).exists():
        with open(config_init_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Ищем логирование gmlp
        if "gMLP state size" in content:
            print("   ❌ НАЙДЕНА ПРОБЛЕМА: config_initializer всегда логирует gMLP")
            print("      Код логирует секцию 'gmlp' даже в hybrid режиме")
            print("      Нужно добавить проверку architecture_type!")
        else:
            print("   ✅ gMLP логирование не найдено")

    # 3. Проверка архитектуры в emergent_training
    print("\n⚙️ ПРОВЕРКА EMERGENT_TRAINING АРХИТЕКТУРЫ:")

    try:
        from emergent_training.config.config import EmergentTrainingConfig
        from emergent_training.core.trainer import EmergentCubeTrainer

        # Создаем конфигурацию
        config = EmergentTrainingConfig()
        print(f"   enable_nca: {config.enable_nca}")
        print(f"   gmlp_config: {config.gmlp_config}")
        print(f"   nca_config: {config.nca_config}")

        # Анализируем что происходит при создании trainer'а

    except Exception as e:
        print(f"   EmergentTraining error: {e}")


def diagnose_hybrid_architecture_confusion():
    """Диагностика путаницы в hybrid архитектуре"""
    print("\n🔀 ДИАГНОСТИКА HYBRID АРХИТЕКТУРЫ")
    print("=" * 60)

    # Проблема: в логах config_initializer.py показывает gMLP параметры
    # но конфигурация указывает на hybrid_mode с minimal_nca

    print("🎯 АНАЛИЗ ПРОБЛЕМЫ:")
    print("   1. debug_final_config.yaml показывает:")
    print("      - hybrid_mode: true")
    print("      - neuron_architecture: minimal_nca")
    print("      - cell_architecture: gmlp (в emergent_training)")
    print()
    print("   2. config_initializer.py логирует:")
    print("      - 'gMLP state size: 8, hidden_dim: 8'")
    print("      - Но должно быть NCA параметры!")
    print()
    print("   3. Возможные причины:")
    print("      a) config_initializer не учитывает hybrid_mode")
    print(
        "      b) emergent_training.cell_architecture переопределяет neuron_architecture"
    )
    print("      c) Неправильный mapping в DynamicConfigGenerator")


def create_architecture_fix():
    """Создание исправления для архитектуры"""
    print("\n🔧 СОЗДАНИЕ ИСПРАВЛЕНИЯ АРХИТЕКТУРЫ")
    print("=" * 60)

    print("📝 ПЛАН ИСПРАВЛЕНИЯ:")
    print()
    print("1. ИСПРАВИТЬ config_initializer.py:")
    print("   - Добавить проверку hybrid_mode")
    print("   - Логировать правильную архитектуру (NCA vs gMLP)")
    print()
    print("2. ИСПРАВИТЬ DynamicConfigGenerator:")
    print("   - Убедиться что hybrid_mode правильно обрабатывается")
    print("   - emergent_training.cell_architecture должен быть 'nca' в hybrid режиме")
    print()
    print("3. ИСПРАВИТЬ GPU интеграцию:")
    print("   - Проверить что gpu_enabled передается правильно")
    print("   - Убедиться что модели инициализируются на GPU")


def create_gpu_fix():
    """Создание исправления для GPU"""
    print("\n💾 СОЗДАНИЕ ИСПРАВЛЕНИЯ GPU")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("❌ CUDA не доступен - проверьте установку PyTorch с CUDA")
        return False

    print("📝 ПЛАН ИСПРАВЛЕНИЯ GPU:")
    print()
    print("1. ДОБАВИТЬ EXPLICIT GPU SETTINGS:")
    print("   - Убедиться что lattice_3d.gpu_enabled = True")
    print("   - Добавить device='cuda' в конфигурацию")
    print()
    print("2. ПРОВЕРИТЬ DEVICE PLACEMENT:")
    print("   - Все модели должны быть на GPU")
    print("   - Все tensors должны быть на GPU")
    print()
    print("3. ДОБАВИТЬ GPU MEMORY MONITORING:")
    print("   - torch.cuda.memory_allocated() должен показывать использование")

    return True


def main():
    """Основная функция диагностики"""
    print("🔍 ДИАГНОСТИКА АРХИТЕКТУРЫ И GPU ИНТЕГРАЦИИ")
    print("=" * 80)
    print("Анализ проблем:")
    print("- gMLP архитектура вместо NCA (8/8 params)")
    print("- GPU память не используется (0.0MB)")
    print()

    try:
        # 1. Диагностика GPU
        gpu_ok = diagnose_gpu_integration()

        # 2. Диагностика архитектуры
        diagnose_architecture_selection()

        # 3. Диагностика hybrid confusion
        diagnose_hybrid_architecture_confusion()

        # 4. План исправления
        create_architecture_fix()
        gpu_fix_ok = create_gpu_fix()

        print("\n" + "=" * 80)
        print("🎯 ВЫВОДЫ ДИАГНОСТИКИ:")
        print()

        print("❌ ПРОБЛЕМА 1: config_initializer.py неправильно логирует архитектуру")
        print(
            "   Решение: Добавить проверку hybrid_mode и логировать правильные параметры"
        )
        print()

        print("❌ ПРОБЛЕМА 2: Путаница между neuron_architecture и cell_architecture")
        print("   neuron_architecture = minimal_nca (должно быть primary)")
        print("   cell_architecture = gmlp (переопределяет neuron_architecture)")
        print()

        if gpu_ok and gpu_fix_ok:
            print("✅ ПРОБЛЕМА 3: GPU доступен, но не используется")
            print("   Решение: Добавить explicit GPU configuration")
        else:
            print("❌ ПРОБЛЕМА 3: GPU недоступен или некорректно настроен")

        print()
        print("🔧 КРИТИЧНЫЕ ИСПРАВЛЕНИЯ:")
        print("1. Исправить mapping архитектуры в DynamicConfigGenerator")
        print("2. Обновить config_initializer для hybrid режима")
        print("3. Добавить explicit GPU device configuration")
        print("4. Тест полного цикла после исправлений")

        return True

    except Exception as e:
        print(f"❌ ОШИБКА В ДИАГНОСТИКЕ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
