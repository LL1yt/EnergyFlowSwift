"""
Тест интеграции Dynamic Configuration System с основным ConfigManager
"""

import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_dynamic_config_standalone():
    """Тест автономной работы Dynamic Configuration System"""
    print("🧪 Testing standalone Dynamic Configuration System...")

    try:
        from utils.config_manager.dynamic_config import DynamicConfigManager

        # Создание менеджера
        manager = DynamicConfigManager()

        # Тест автоопределения режима
        auto_config = manager.create_config_for_mode("auto")
        print(f"[OK] Auto-detected mode: {auto_config['_metadata']['mode']}")

        # Тест всех режимов
        for mode in ["development", "research", "validation"]:
            config = manager.create_config_for_mode(mode)
            lattice = config["lattice"]

            # Проверка округлений
            assert isinstance(
                lattice["xs"], int
            ), f"xs must be int, got {type(lattice['xs'])}"
            assert isinstance(
                lattice["ys"], int
            ), f"ys must be int, got {type(lattice['ys'])}"
            assert isinstance(
                lattice["zs"], int
            ), f"zs must be int, got {type(lattice['zs'])}"

            print(f"[OK] {mode.upper()} mode:")
            print(f"   Lattice: {lattice['xs']}x{lattice['ys']}x{lattice['zs']}")
            print(f"   Neurons: {lattice['total_neurons']:,}")
            print(f"   Embedding dim: {config['embeddings']['embedding_dim']:,}")
            print(f"   Batch size: {config['training']['batch_size']}")

        print("[OK] Standalone Dynamic Configuration System works!")
        return True

    except Exception as e:
        print(f"[ERROR] Standalone test failed: {e}")
        return False


def test_config_manager_integration():
    """Тест интеграции с основным ConfigManager"""
    print("\n🧪 Testing ConfigManager integration...")

    try:
        from utils.config_manager.config_manager import (
            ConfigManager,
            ConfigManagerSettings,
        )

        # Создание настроек с включенной динамической конфигурацией
        settings = ConfigManagerSettings(
            enable_dynamic_config=True,
            dynamic_config_mode="research",  # Фиксированный режим для тестирования
            auto_hardware_detection=False,
            enable_hot_reload=False,  # Отключаем для тестирования
        )

        # Создание ConfigManager
        config_manager = ConfigManager(settings)

        # Проверка наличия динамической конфигурации
        dynamic_info = config_manager.get_dynamic_config_info()
        if dynamic_info:
            print(f"[OK] Dynamic config info: {dynamic_info}")
        else:
            print("[WARNING] Dynamic config info not found")

        # Проверка загруженных секций
        lattice_config = config_manager.get_config("lattice")
        if lattice_config:
            print(f"[OK] Lattice config loaded:")
            print(
                f"   Size: {lattice_config.get('xs', 'N/A')}x{lattice_config.get('ys', 'N/A')}x{lattice_config.get('zs', 'N/A')}"
            )
            print(f"   Total neurons: {lattice_config.get('total_neurons', 'N/A'):,}")

        embeddings_config = config_manager.get_config("embeddings")
        if embeddings_config:
            print(f"[OK] Embeddings config loaded:")
            print(f"   Embedding dim: {embeddings_config.get('embedding_dim', 'N/A')}")

        # Тест регенерации
        success = config_manager.regenerate_dynamic_config("development")
        print(f"[OK] Regeneration success: {success}")

        # Проверка изменений после регенерации
        new_lattice_config = config_manager.get_config("lattice")
        if new_lattice_config:
            print(f"[OK] After regeneration:")
            print(
                f"   Size: {new_lattice_config.get('xs', 'N/A')}x{new_lattice_config.get('ys', 'N/A')}x{new_lattice_config.get('zs', 'N/A')}"
            )

        print("[OK] ConfigManager integration works!")
        return True

    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hardware_detection():
    """Тест автоопределения железа"""
    print("\n🧪 Testing hardware detection...")

    try:
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        generator = DynamicConfigGenerator()
        detected_mode = generator.detect_hardware_mode()

        print(f"[OK] Detected hardware mode: {detected_mode}")

        # Генерируем конфигурацию для автоопределенного режима
        config = generator.generate_config(detected_mode)
        lattice = config["lattice"]

        print(f"[OK] Config for detected mode:")
        print(f"   Lattice: {lattice['xs']}x{lattice['ys']}x{lattice['zs']}")
        print(f"   Scale factor: {lattice['scale_factor']}")

        return True

    except Exception as e:
        print(f"[ERROR] Hardware detection test failed: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("[START] Testing Dynamic Configuration System Integration")
    print("=" * 60)

    success_count = 0
    total_tests = 3

    # Тест 1: Автономная работа
    if test_dynamic_config_standalone():
        success_count += 1

    # Тест 2: Интеграция с ConfigManager
    if test_config_manager_integration():
        success_count += 1

    # Тест 3: Автоопределение железа
    if test_hardware_detection():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"[TARGET] Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("[SUCCESS] All tests passed! Dynamic Configuration System is ready to use.")
    else:
        print("[WARNING] Some tests failed. Check the implementation.")

    return success_count == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
