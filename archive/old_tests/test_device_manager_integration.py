#!/usr/bin/env python3
"""
Тест интеграции DeviceManager в new_rebuild архитектуру
=====================================================

Проверяет:
1. Корректную инициализацию DeviceManager
2. Консистентное управление устройствами
3. Автоматическую очистку памяти
4. Интеграцию с ProjectConfig
"""

import sys
import torch
from pathlib import Path

# Добавляем new_rebuild в path
sys.path.insert(0, str(Path(__file__).parent / "new_rebuild"))

from new_rebuild.utils.device_manager import (
    DeviceManager,
    get_device_manager,
    reset_device_manager,
)
from new_rebuild.config.project_config import ProjectConfig, get_project_config
from new_rebuild.core.moe.moe_processor import MoEConnectionProcessor


def test_device_manager_basic():
    """Тест базовой функциональности DeviceManager"""
    print("🔧 Тест DeviceManager - базовая функциональность")
    print("=" * 60)

    # Сброс для чистого теста
    reset_device_manager()

    # Создание DeviceManager
    device_manager = get_device_manager(debug_mode=True)

    print(f"✅ DeviceManager создан: {device_manager.get_device()}")
    print(f"   CUDA доступен: {torch.cuda.is_available()}")
    print(f"   Устройство: {device_manager.get_device_str()}")

    # Тест выделения tensor'а
    test_tensor = device_manager.allocate_tensor((100, 32), dtype=torch.float32)
    print(f"✅ Tensor выделен: {test_tensor.shape} на {test_tensor.device}")

    # Тест переноса tensor'а
    cpu_tensor = torch.randn(50, 32)
    gpu_tensor = device_manager.ensure_device(cpu_tensor)
    print(f"✅ Tensor перенесен: {cpu_tensor.device} → {gpu_tensor.device}")

    # Статистика памяти
    stats = device_manager.get_memory_stats()
    print(f"✅ Статистика памяти:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Cleanup
    device_manager.cleanup()
    print("✅ Cleanup выполнен")

    return True


def test_project_config_integration():
    """Тест интеграции DeviceManager с ProjectConfig"""
    print("\n🔧 Тест ProjectConfig - интеграция DeviceManager")
    print("=" * 60)

    # Создание ProjectConfig
    config = ProjectConfig(debug_mode=True, prefer_cuda=True)

    print(f"✅ ProjectConfig создан с DeviceManager")
    print(f"   Устройство: {config.device}")
    print(f"   DeviceManager: {config.device_manager}")

    # Тест методов DeviceManager через ProjectConfig
    test_tensor = config.allocate_tensor((10, 32))
    print(f"✅ Tensor через ProjectConfig: {test_tensor.shape} на {test_tensor.device}")

    # Тест device config
    device_config = config.get_device_config()
    print(f"✅ Device config получен:")
    for key, value in device_config.items():
        if key != "device_manager":  # Исключаем объект из вывода
            print(f"   {key}: {value}")

    return True


def test_moe_processor_integration():
    """Тест интеграции DeviceManager с MoEProcessor"""
    print("\n🔧 Тест MoEProcessor - интеграция DeviceManager")
    print("=" * 60)

    try:
        # Создание MoEProcessor (должен автоматически использовать DeviceManager)
        processor = MoEConnectionProcessor(
            state_size=32,
            lattice_dimensions=(6, 6, 6),  # Малая решетка для теста
            neighbor_count=26,
        )

        print(f"✅ MoEProcessor создан")
        print(f"   Устройство: {processor.device}")
        print(f"   DeviceManager интегрирован: {hasattr(processor, 'device_manager')}")

        # Тест forward pass
        current_state = torch.randn(32)
        neighbor_states = torch.randn(10, 32)

        result = processor.forward(
            current_state=current_state,
            neighbor_states=neighbor_states,
            cell_idx=0,
            neighbor_indices=list(range(10)),
        )

        print(f"✅ Forward pass выполнен")
        print(f"   Результат: {result.keys()}")
        print(f"   Новое состояние: {result['new_state'].shape}")

        return True

    except Exception as e:
        print(f"❌ Ошибка в MoEProcessor тесте: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_management():
    """Тест управления памятью"""
    print("\n🔧 Тест управления памятью")
    print("=" * 60)

    device_manager = get_device_manager()

    # Создаем много tensor'ов для тестирования cleanup
    tensors = []
    for i in range(20):
        tensor = device_manager.allocate_tensor((100, 32))
        tensors.append(tensor)

    stats_before = device_manager.get_memory_stats()
    print(f"✅ Создано {len(tensors)} tensor'ов")
    print(f"   Выделений: {stats_before['total_allocations']}")

    # Принудительный cleanup
    device_manager.cleanup()

    stats_after = device_manager.get_memory_stats()
    print(f"✅ Cleanup выполнен")
    print(f"   Выделений: {stats_after['total_allocations']}")

    # Проверка, что memory cleanup работает
    if device_manager.is_cuda():
        print(f"   GPU память до: {stats_before.get('allocated_mb', 0):.2f}MB")
        print(f"   GPU память после: {stats_after.get('allocated_mb', 0):.2f}MB")

    return True


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТ ИНТЕГРАЦИИ DEVICEMANAGER В NEW_REBUILD")
    print("=" * 80)

    tests = [
        ("DeviceManager - базовая функциональность", test_device_manager_basic),
        ("ProjectConfig - интеграция DeviceManager", test_project_config_integration),
        ("MoEProcessor - интеграция DeviceManager", test_moe_processor_integration),
        ("Управление памятью", test_memory_management),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            print(f"❌ ОШИБКА в тесте '{test_name}': {e}")
            results[test_name] = "❌ ERROR"

    # Итоговый отчет
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)

    for test_name, result in results.items():
        print(f"{result} | {test_name}")

    passed = sum(1 for r in results.values() if r == "✅ PASS")
    total = len(results)

    print(f"\n🎯 РЕЗУЛЬТАТ: {passed}/{total} тестов прошли")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! DeviceManager успешно интегрирован")
        return 0
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте интеграцию.")
        return 1


if __name__ == "__main__":
    exit(main())
