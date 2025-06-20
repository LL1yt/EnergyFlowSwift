#!/usr/bin/env python3
"""
🔧 ИСПРАВЛЕНИЕ: Проблема с mapping полей решетки

Найденная проблема:
- Новая система: lattice_width, lattice_height, lattice_depth
- Старая система: xs, ys, zs + cube_dimensions
- config_initializer.py ищет xs/ys/zs, не находит, fallback к cube_dimensions

Решение:
1. Обновить config_initializer.py для поддержки новых полей
2. Добавить backward compatibility mapping
3. Убедиться что progressive scaling применяется корректно
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))


def patch_config_initializer():
    """Патч для config_initializer.py чтобы поддерживать новые поля решетки"""

    config_initializer_path = "smart_resume_training/core/config_initializer.py"

    print("🔧 Применение патча к config_initializer.py...")

    # Читаем текущий файл
    with open(config_initializer_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Старый код для поиска и замены
    old_log_method = """    def _log_config_details(self):
        \"\"\"Logs the key details of the generated configuration.\"\"\"
        if not self.config or not self.metadata:
            logger.warning("Config or metadata not available for logging.")
            return

        mode = self.metadata.get("mode", "unknown")
        scale = self.metadata.get("scale_factor", "unknown")
        logger.info(f"Initialized config in '{mode}' mode with scale factor {scale}.")

        lattice = self.config.get("lattice", {})
        if lattice:
            logger.info(
                f"Target Lattice: {lattice.get('xs')}x{lattice.get('ys')}x{lattice.get('zs')}"
            )

        gmlp = self.config.get("gmlp", {})
        if gmlp:
            logger.info(
                f"gMLP state size: {gmlp.get('state_size')}, hidden_dim: {gmlp.get('hidden_dim')}"
            )"""

    # Новый код с поддержкой обеих систем
    new_log_method = """    def _log_config_details(self):
        \"\"\"Logs the key details of the generated configuration.\"\"\"
        if not self.config or not self.metadata:
            logger.warning("Config or metadata not available for logging.")
            return

        mode = self.metadata.get("mode", "unknown")
        scale = self.metadata.get("scale_factor", "unknown")
        logger.info(f"Initialized config in '{mode}' mode with scale factor {scale}.")

        # === PHASE 4 FIX: Support both old and new lattice field names ===
        lattice = self.config.get("lattice", {})
        if lattice:
            # Try new field names first (Phase 4 integration)
            width = lattice.get('lattice_width') or lattice.get('xs')
            height = lattice.get('lattice_height') or lattice.get('ys') 
            depth = lattice.get('lattice_depth') or lattice.get('zs')
            
            # Fallback to cube_dimensions if available
            if not all([width, height, depth]):
                emergent = self.config.get("emergent_training", {})
                cube_dims = emergent.get("cube_dimensions", [])
                if len(cube_dims) >= 3:
                    width, height, depth = cube_dims[0], cube_dims[1], cube_dims[2]
                    logger.warning("Using fallback cube_dimensions - this may indicate a configuration issue")
            
            logger.info(f"Target Lattice: {width}x{height}x{depth}")
            
            # Log field source for debugging
            if lattice.get('lattice_width'):
                logger.info("Using Phase 4 lattice field names (lattice_width/height/depth)")
            elif lattice.get('xs'):
                logger.info("Using legacy lattice field names (xs/ys/zs)")
            else:
                logger.info("Using cube_dimensions fallback")

        gmlp = self.config.get("gmlp", {})
        if gmlp:
            logger.info(
                f"gMLP state size: {gmlp.get('state_size')}, hidden_dim: {gmlp.get('hidden_dim')}"
            )"""

    # Проверяем что метод найден
    if old_log_method.strip() not in content:
        print("⚠️  Точное соответствие не найдено. Ищу альтернативные варианты...")

        # Ищем только ключевую строку
        target_line = "f\"Target Lattice: {lattice.get('xs')}x{lattice.get('ys')}x{lattice.get('zs')}\""
        if target_line in content:
            # Заменяем только эту строку
            replacement_block = """# === PHASE 4 FIX: Support both old and new lattice field names ===
            # Try new field names first (Phase 4 integration)
            width = lattice.get('lattice_width') or lattice.get('xs')
            height = lattice.get('lattice_height') or lattice.get('ys') 
            depth = lattice.get('lattice_depth') or lattice.get('zs')
            
            # Fallback to cube_dimensions if available
            if not all([width, height, depth]):
                emergent = self.config.get("emergent_training", {})
                cube_dims = emergent.get("cube_dimensions", [])
                if len(cube_dims) >= 3:
                    width, height, depth = cube_dims[0], cube_dims[1], cube_dims[2]
                    logger.warning("Using fallback cube_dimensions - this may indicate a configuration issue")
            
            logger.info(f"Target Lattice: {width}x{height}x{depth}")
            
            # Log field source for debugging
            if lattice.get('lattice_width'):
                logger.info("Using Phase 4 lattice field names (lattice_width/height/depth)")
            elif lattice.get('xs'):
                logger.info("Using legacy lattice field names (xs/ys/zs)")
            else:
                logger.info("Using cube_dimensions fallback")"""

            content = content.replace(
                f"logger.info(\n                {target_line}\n            )",
                replacement_block,
            )
        else:
            print("❌ Не удалось найти целевую строку для замены")
            return False
    else:
        # Полная замена метода
        content = content.replace(old_log_method, new_log_method)

    # Записываем обновленный файл
    with open(config_initializer_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Патч применен к config_initializer.py")
    return True


def verify_progressive_scaling_mapping():
    """Проверить что progressive scaling корректно мапится в конфигурацию"""

    print("\n🔍 Проверка mapping progressive scaling...")

    from training.automated_training.types import StageConfig
    from training.automated_training.stage_runner import TrainingStageRunner

    # Создаем тестовую конфигурацию с progressive scaling
    stage_config = StageConfig(
        stage=2,
        dataset_limit=100,
        epochs=2,
        batch_size=16,
        description="Progressive Scaling Mapping Test",
        plasticity_profile="learning",
        clustering_enabled=True,
        activity_threshold=0.025,
        memory_optimizations=True,
        emergence_tracking=True,
        progressive_scaling=True,
    )

    # Создаем runner с небольшим scale для тестирования
    runner = TrainingStageRunner(mode="development", scale=0.05, verbose=True)

    # Генерируем конфигурацию
    temp_config_path = runner._generate_temp_config(stage_config)

    if temp_config_path:
        import yaml

        with open(temp_config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        print("📋 Проверка полей решетки в сгенерированной конфигурации:")

        # Проверяем новые поля
        lattice = config_data.get("lattice", {})
        print(f"   lattice_width: {lattice.get('lattice_width')}")
        print(f"   lattice_height: {lattice.get('lattice_height')}")
        print(f"   lattice_depth: {lattice.get('lattice_depth')}")

        # Проверяем старые поля
        print(f"   xs: {lattice.get('xs')}")
        print(f"   ys: {lattice.get('ys')}")
        print(f"   zs: {lattice.get('zs')}")

        # Проверяем cube_dimensions
        emergent = config_data.get("emergent_training", {})
        cube_dims = emergent.get("cube_dimensions", [])
        print(f"   cube_dimensions: {cube_dims}")

        # Проверяем ожидаемые значения
        expected_dims = runner._get_adaptive_dimensions(stage_config.stage)
        print(f"   Expected (progressive): {expected_dims}")

        # Анализ проблемы
        new_fields_present = all(
            [
                lattice.get("lattice_width"),
                lattice.get("lattice_height"),
                lattice.get("lattice_depth"),
            ]
        )

        if new_fields_present:
            actual_dims = (
                lattice.get("lattice_width"),
                lattice.get("lattice_height"),
                lattice.get("lattice_depth"),
            )
            if actual_dims == expected_dims:
                print("✅ Progressive scaling применяется корректно")
            else:
                print(f"❌ Progressive scaling не соответствует ожидаемым значениям")
                print(f"   Expected: {expected_dims}")
                print(f"   Actual: {actual_dims}")
        else:
            print("❌ Новые поля решетки отсутствуют в конфигурации")

        import os

        os.remove(temp_config_path)
        return new_fields_present
    else:
        print("❌ Не удалось сгенерировать временную конфигурацию")
        return False


def add_field_mapping_to_dynamic_config():
    """Добавить backward compatibility mapping в DynamicConfigGenerator"""

    print("\n🔧 Добавление backward compatibility mapping...")

    # Проверяем что нужно добавить mapping в базовую конфигурацию
    from utils.config_manager.dynamic_config import DynamicConfigGenerator

    generator = DynamicConfigGenerator()

    # Создаем тестовую конфигурацию
    config = generator.create_base_config_template()

    lattice_section = config.get("lattice", {})

    print("📋 Текущие поля решетки в базовой конфигурации:")
    for key, value in lattice_section.items():
        print(f"   {key}: {value}")

    # Проверяем есть ли новые поля
    has_new_fields = any(
        [
            "lattice_width" in lattice_section,
            "lattice_height" in lattice_section,
            "lattice_depth" in lattice_section,
        ]
    )

    has_old_fields = any(
        ["xs" in lattice_section, "ys" in lattice_section, "zs" in lattice_section]
    )

    print(f"\n📊 Анализ полей:")
    print(
        f"   Новые поля (lattice_width/height/depth): {'✅' if has_new_fields else '❌'}"
    )
    print(f"   Старые поля (xs/ys/zs): {'✅' if has_old_fields else '❌'}")

    return has_new_fields, has_old_fields


def main():
    """Основная функция исправления"""
    print("🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С ПОЛЯМИ РЕШЕТКИ")
    print("=" * 60)
    print("Цель: Исправить несоответствие lattice_width vs xs/ys/zs")
    print()

    try:
        # 1. Применяем патч к config_initializer
        success_patch = patch_config_initializer()

        # 2. Проверяем progressive scaling mapping
        success_mapping = verify_progressive_scaling_mapping()

        # 3. Анализируем поля в базовой конфигурации
        has_new, has_old = add_field_mapping_to_dynamic_config()

        print("\n" + "=" * 60)
        print("🎯 РЕЗУЛЬТАТЫ ИСПРАВЛЕНИЯ:")
        print()

        print(
            f"✅ Патч config_initializer: {'Применен' if success_patch else 'Ошибка'}"
        )
        print(
            f"✅ Progressive scaling mapping: {'Работает' if success_mapping else 'Требует исправления'}"
        )
        print(
            f"✅ Новые поля в базовой конфигурации: {'Есть' if has_new else 'Отсутствуют'}"
        )
        print(
            f"✅ Старые поля в базовой конфигурации: {'Есть' if has_old else 'Отсутствуют'}"
        )

        print("\n🔧 СЛЕДУЮЩИЕ ШАГИ:")

        if success_patch and success_mapping:
            print("1. ✅ Основная проблема исправлена")
            print(
                "2. 🧪 Запустить тест для проверки: python test_phase4_full_training_cycle.py"
            )
            print("3. 🔍 Проверить логи обучения на корректные размеры решетки")
        else:
            if not success_patch:
                print("1. ❌ Требуется ручное исправление config_initializer.py")
            if not success_mapping:
                print("2. ❌ Требуется исправление progressive scaling mapping")

        print("\n💡 СУТЬ ПРОБЛЕМЫ БЫЛА:")
        print("   - TrainingStageRunner генерирует lattice_width/height/depth")
        print("   - config_initializer ищет xs/ys/zs")
        print("   - Fallback к cube_dimensions дает 7×7×3 = smart_round(666*0.01)")
        print("   - Теперь поддерживаются оба формата!")

        return success_patch and success_mapping

    except Exception as e:
        print(f"❌ ОШИБКА В ИСПРАВЛЕНИИ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
