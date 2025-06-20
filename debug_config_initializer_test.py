#!/usr/bin/env python3
"""
Отладка config_initializer теста
"""

import tempfile
import yaml
from pathlib import Path


def test_config_initializer_debug():
    """Подробный тест config_initializer с логами"""
    print("🔍 ПОДРОБНЫЙ ДЕБАГ config_initializer")
    print("=" * 60)

    try:
        from smart_resume_training.core.config_initializer import ConfigInitializer
        from utils.config_manager.dynamic_config import DynamicConfigGenerator

        # Шаг 1: Генерируем конфигурацию
        print("1️⃣ Генерация конфигурации...")
        generator = DynamicConfigGenerator()
        test_config = generator.generate_config("development")

        # Проверяем что генератор создал
        print("🔍 Сгенерированная конфигурация:")
        architecture_debug = test_config.get("architecture", {})
        emergent_debug = test_config.get("emergent_training", {})
        print(f"   architecture.hybrid_mode: {architecture_debug.get('hybrid_mode')}")
        print(
            f"   architecture.neuron_architecture: {architecture_debug.get('neuron_architecture')}"
        )
        print(
            f"   emergent_training.cell_architecture: {emergent_debug.get('cell_architecture')}"
        )

        # Шаг 2: Сохраняем в файл
        print("\n2️⃣ Сохранение в файл...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name

        print(f"   Файл: {temp_path}")

        # Проверяем содержимое файла
        print("🔍 Содержимое файла:")
        with open(temp_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # Ищем ключевые строки
        if "hybrid_mode: true" in file_content:
            print("   ✅ hybrid_mode: true найден")
        elif "hybrid_mode: false" in file_content:
            print("   ❌ hybrid_mode: false найден")
        else:
            print("   ❓ hybrid_mode не найден")

        if "cell_architecture: nca" in file_content:
            print("   ✅ cell_architecture: nca найден")
        elif "cell_architecture: gmlp" in file_content:
            print("   ❌ cell_architecture: gmlp найден")
        else:
            print("   ❓ cell_architecture не найден")

        # Шаг 3: Загружаем через ConfigInitializer
        print("\n3️⃣ Загрузка через ConfigInitializer...")
        initializer = ConfigInitializer(temp_path)
        config = initializer.config

        print("🔍 Загруженная конфигурация:")
        architecture_loaded = config.get("architecture", {})
        emergent_loaded = config.get("emergent_training", {})
        print(f"   architecture.hybrid_mode: {architecture_loaded.get('hybrid_mode')}")
        print(
            f"   emergent_training.cell_architecture: {emergent_loaded.get('cell_architecture')}"
        )

        # Шаг 4: Проверяем логику теста
        print("\n4️⃣ Проверка логики теста...")
        hybrid_mode = architecture_loaded.get("hybrid_mode", False)
        cell_architecture = emergent_loaded.get("cell_architecture", "gmlp")

        print(f"   hybrid_mode = {hybrid_mode}")
        print(f"   cell_architecture = {cell_architecture}")
        print(f"   Условие: hybrid_mode={hybrid_mode} AND cell_architecture='nca'")

        if hybrid_mode and cell_architecture == "nca":
            print("   ✅ Условие выполнено!")
            success = True
        else:
            print("   ❌ Условие НЕ выполнено!")
            success = False

        # Шаг 5: Дополнительная диагностика
        print("\n5️⃣ Дополнительная диагностика...")

        # Проверяем NCA конфигурацию
        nca_config = emergent_loaded.get("nca_config", {})
        minimal_nca = config.get("minimal_nca_cell", {})

        print(f"   emergent_training.nca_config: {bool(nca_config)}")
        print(f"   minimal_nca_cell: {bool(minimal_nca)}")

        if nca_config:
            print(f"   nca_config.state_size: {nca_config.get('state_size')}")
        if minimal_nca:
            print(f"   minimal_nca.state_size: {minimal_nca.get('state_size')}")

        # Удаляем временный файл
        Path(temp_path).unlink()

        print(f"\n🎯 РЕЗУЛЬТАТ: {'✅ УСПЕХ' if success else '❌ НЕУДАЧА'}")
        return success

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_config_initializer_debug()
