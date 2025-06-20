#!/usr/bin/env python3
"""
Тест новых чистых конфигураций Phase 4
======================================

Проверяет что новые конфигурации:
1. Правильно загружаются
2. Используют гибридную архитектуру NCA+gMLP
3. Включают пластичность и оптимизации
4. Совместимы с системой обучения
"""

import sys
import yaml
from pathlib import Path


def test_lattice_3d_config():
    """Тест конфигурации lattice_3d"""
    print("🧪 ТЕСТ 1: Конфигурация lattice_3d")
    print("-" * 40)

    try:
        from core.lattice_3d.config import load_lattice_config

        config_path = "core/lattice_3d/config/hybrid_nca_gmlp.yaml"

        if not Path(config_path).exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            return False

        # Загружаем конфигурацию
        lattice_config = load_lattice_config(config_path)

        print(f"✅ Конфигурация загружена")
        print(f"✅ Размеры решетки: {lattice_config.dimensions}")
        print(f"✅ GPU включен: {lattice_config.gpu_enabled}")
        print(f"✅ Пластичность: {lattice_config.enable_plasticity}")
        print(f"✅ Конкурентное обучение: {lattice_config.enable_competitive_learning}")
        print(f"✅ BCM метапластичность: {lattice_config.enable_metaplasticity}")
        print(f"✅ Кластеризация: {lattice_config.enable_clustering}")
        print(f"✅ Mixed precision: {lattice_config.mixed_precision}")
        print(f"✅ Checkpointing: {lattice_config.use_checkpointing}")

        # Проверяем что все критичные параметры правильные
        assert tuple(lattice_config.dimensions) == (
            16,
            16,
            16,
        ), f"Неправильные размеры: {lattice_config.dimensions}"
        assert lattice_config.gpu_enabled == True, "GPU должен быть включен"
        assert (
            lattice_config.neighbors == 26
        ), f"Неправильное количество соседей: {lattice_config.neighbors}"
        assert (
            lattice_config.enable_plasticity == True
        ), "Пластичность должна быть включена"

        print("🎉 LATTICE_3D КОНФИГУРАЦИЯ КОРРЕКТНА!")
        return True

    except Exception as e:
        print(f"❌ Ошибка в lattice_3d конфигурации: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cell_prototype_config():
    """Тест конфигурации cell_prototype"""
    print("\n🧪 ТЕСТ 2: Конфигурация cell_prototype")
    print("-" * 40)

    try:
        config_path = "core/cell_prototype/config/hybrid_nca_gmlp.yaml"

        if not Path(config_path).exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            return False

        # Загружаем YAML
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        print(f"✅ YAML загружен")

        # Проверяем структуру
        assert "architecture" in config_data, "Отсутствует секция architecture"
        assert "minimal_nca_cell" in config_data, "Отсутствует секция minimal_nca_cell"
        assert "gmlp_cell" in config_data, "Отсутствует секция gmlp_cell"
        assert "integration" in config_data, "Отсутствует секция integration"

        # Проверяем архитектуру
        arch = config_data["architecture"]
        assert arch["hybrid_mode"] == True, "Гибридный режим должен быть включен"
        assert arch["neuron_architecture"] == "minimal_nca", "Нейроны должны быть NCA"
        assert arch["connection_architecture"] == "gated_mlp", "Связи должны быть gMLP"

        # Проверяем NCA конфигурацию
        nca = config_data["minimal_nca_cell"]
        assert (
            nca["state_size"] == 4
        ), f"Неправильный state_size для NCA: {nca['state_size']}"
        assert (
            nca["neighbor_count"] == 26
        ), f"Неправильный neighbor_count для NCA: {nca['neighbor_count']}"
        assert (
            nca["target_params"] == 362
        ), f"Неправильный target_params для NCA: {nca['target_params']}"

        # Проверяем gMLP конфигурацию
        gmlp = config_data["gmlp_cell"]
        assert (
            gmlp["state_size"] == 8
        ), f"Неправильный state_size для gMLP: {gmlp['state_size']}"
        assert (
            gmlp["neighbor_count"] == 26
        ), f"Неправильный neighbor_count для gMLP: {gmlp['neighbor_count']}"
        assert gmlp["use_memory"] == False, "gMLP не должен использовать память"

        # Проверяем интеграцию
        integration = config_data["integration"]
        assert (
            integration["state_synchronization"] == True
        ), "Синхронизация состояний должна быть включена"

        print(f"✅ Гибридная архитектура: {arch['hybrid_mode']}")
        print(f"✅ NCA state_size: {nca['state_size']}")
        print(f"✅ NCA target_params: {nca['target_params']}")
        print(f"✅ gMLP state_size: {gmlp['state_size']}")
        print(f"✅ gMLP use_memory: {gmlp['use_memory']}")
        print(f"✅ Интеграция включена: {integration['state_synchronization']}")

        print("🎉 CELL_PROTOTYPE КОНФИГУРАЦИЯ КОРРЕКТНА!")
        return True

    except Exception as e:
        print(f"❌ Ошибка в cell_prototype конфигурации: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_integration():
    """Тест интеграции конфигураций"""
    print("\n🧪 ТЕСТ 3: Интеграция конфигураций")
    print("-" * 40)

    try:
        # Создаем объединенную конфигурацию
        lattice_path = "core/lattice_3d/config/hybrid_nca_gmlp.yaml"
        cell_path = "core/cell_prototype/config/hybrid_nca_gmlp.yaml"

        # Загружаем обе конфигурации
        with open(lattice_path, "r", encoding="utf-8") as f:
            lattice_data = yaml.safe_load(f)

        with open(cell_path, "r", encoding="utf-8") as f:
            cell_data = yaml.safe_load(f)

        # Объединяем
        combined_config = {}
        combined_config.update(lattice_data)
        combined_config.update(cell_data)

        print("✅ Конфигурации объединены")

        # Проверяем совместимость архитектур
        lattice_arch = lattice_data.get("architecture", {})
        cell_arch = cell_data.get("architecture", {})

        assert lattice_arch.get("hybrid_mode") == cell_arch.get(
            "hybrid_mode"
        ), "Несовместимые hybrid_mode"
        assert lattice_arch.get("neuron_architecture") == cell_arch.get(
            "neuron_architecture"
        ), "Несовместимые neuron_architecture"

        # Проверяем синхронизацию neighbor_count
        lattice_neighbors = lattice_data["lattice_3d"]["topology"]["neighbors"]
        nca_neighbors = cell_data["minimal_nca_cell"]["neighbor_count"]
        gmlp_neighbors = cell_data["gmlp_cell"]["neighbor_count"]

        assert (
            lattice_neighbors == nca_neighbors == gmlp_neighbors
        ), f"Несинхронизированные neighbors: {lattice_neighbors}, {nca_neighbors}, {gmlp_neighbors}"

        print(f"✅ Архитектуры совместимы")
        print(f"✅ Neighbors синхронизированы: {lattice_neighbors}")

        # Проверяем что можем создать LatticeConfig
        from core.lattice_3d.config import _build_lattice_config_from_data

        lattice_config = _build_lattice_config_from_data(combined_config)

        print(f"✅ LatticeConfig создан успешно")
        print(f"✅ Размеры: {lattice_config.dimensions}")
        print(f"✅ Neighbors: {lattice_config.neighbors}")
        print(f"✅ Пластичность: {lattice_config.enable_plasticity}")

        print("🎉 ИНТЕГРАЦИЯ КОНФИГУРАЦИЙ УСПЕШНА!")
        return True

    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dynamic_config_compatibility():
    """Тест совместимости с DynamicConfigGenerator"""
    print("\n🧪 ТЕСТ 4: Совместимость с DynamicConfigGenerator")
    print("-" * 40)

    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        # Создаем генератор
        generator = DynamicConfigGenerator()

        # Генерируем конфигурацию в режиме development
        config = generator.generate_config("development")

        print("✅ DynamicConfigGenerator работает")

        # Проверяем что hybrid режим включен
        architecture = config.get("architecture", {})
        assert (
            architecture.get("hybrid_mode") == True
        ), "Hybrid режим должен быть включен"

        # Проверяем что есть NCA конфигурация
        assert "minimal_nca_cell" in config, "Отсутствует minimal_nca_cell"

        # Проверяем emergent_training
        emergent = config.get("emergent_training", {})
        assert (
            emergent.get("cell_architecture") == "nca"
        ), f"Неправильная cell_architecture: {emergent.get('cell_architecture')}"

        print(f"✅ Hybrid mode: {architecture.get('hybrid_mode')}")
        print(f"✅ Cell architecture: {emergent.get('cell_architecture')}")
        print(f"✅ NCA state_size: {config['minimal_nca_cell']['state_size']}")

        print("🎉 СОВМЕСТИМОСТЬ С DYNAMIC CONFIG ПОДТВЕРЖДЕНА!")
        return True

    except Exception as e:
        print(f"❌ Ошибка совместимости: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования"""
    print("🧪 ТЕСТ НОВЫХ ЧИСТЫХ КОНФИГУРАЦИЙ PHASE 4")
    print("=" * 60)
    print("Проверяем что новые конфигурации работают корректно")
    print()

    tests = [
        ("Lattice 3D Config", test_lattice_3d_config),
        ("Cell Prototype Config", test_cell_prototype_config),
        ("Config Integration", test_config_integration),
        ("Dynamic Config Compatibility", test_dynamic_config_compatibility),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            print(f"❌ Ошибка в тесте {test_name}: {e}")
            results[test_name] = "❌ ERROR"

    print("\n" + "=" * 60)
    print("🎯 РЕЗУЛЬТАТЫ ТЕСТОВ:")
    print()

    for test_name, result in results.items():
        print(f"   {result} | {test_name}")

    passed = sum(1 for r in results.values() if r == "✅ PASS")
    total = len(results)

    print(f"\n📊 ИТОГО: {passed}/{total} тестов прошли")

    if passed == total:
        print("🎉 ВСЕ КОНФИГУРАЦИИ РАБОТАЮТ!")
        print()
        print("🚀 ГОТОВО К ИСПОЛЬЗОВАНИЮ:")
        print("   - core/lattice_3d/config/hybrid_nca_gmlp.yaml")
        print("   - core/cell_prototype/config/hybrid_nca_gmlp.yaml")
        print()
        print("🎯 СЛЕДУЮЩИЙ ШАГ:")
        print("   python test_phase4_full_training_cycle.py")
    else:
        print("⚠️  Некоторые конфигурации требуют исправления")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
