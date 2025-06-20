#!/usr/bin/env python3
"""
🧪 ТЕСТ ИНТЕГРАЦИИ ФАЗЫ 4: Базовая проверка пластичности и оптимизаций

Этот тест проверяет корректность интеграции новых возможностей Фазы 4:
- Расширенная структура StageConfig
- Профили пластичности в ProgressiveConfigManager
- Генерация секций пластичности и оптимизации
- Интеграция в TrainingStageRunner

Цель: Убедиться, что все изменения работают корректно перед масштабированием
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))

from training.automated_training.types import StageConfig, StageResult
from training.automated_training.progressive_config import ProgressiveConfigManager
from training.automated_training.stage_runner import TrainingStageRunner
from utils.config_manager.dynamic_config import DynamicConfigGenerator


def test_phase4_types():
    """Тест новых полей в StageConfig"""
    print("🔬 Тестирование новых типов StageConfig...")

    # Создаем конфигурацию с новыми полями Фазы 4
    config = StageConfig(
        stage=3,
        dataset_limit=1000,
        epochs=5,
        batch_size=32,
        description="Test Phase 4 Integration",
        # === PHASE 4 FIELDS ===
        plasticity_profile="learning",
        clustering_enabled=True,
        activity_threshold=0.03,
        memory_optimizations=True,
        emergence_tracking=True,
        sparse_connection_ratio=0.2,
    )

    assert config.plasticity_profile == "learning"
    assert config.clustering_enabled == True
    assert config.activity_threshold == 0.03
    assert config.memory_optimizations == True
    assert config.emergence_tracking == True
    assert config.sparse_connection_ratio == 0.2

    print("✅ StageConfig расширен корректно")


def test_progressive_config_plasticity():
    """Тест профилей пластичности в ProgressiveConfigManager"""
    print("🧠 Тестирование профилей пластичности...")

    config_manager = ProgressiveConfigManager()

    # Тест стадии 1 (discovery)
    stage1_config = config_manager.get_stage_config(1)
    assert stage1_config.plasticity_profile == "discovery"
    assert stage1_config.clustering_enabled == False
    assert stage1_config.activity_threshold == 0.01
    assert stage1_config.memory_optimizations == True

    # Тест стадии 3 (learning + clustering)
    stage3_config = config_manager.get_stage_config(3)
    assert stage3_config.plasticity_profile == "learning"
    assert stage3_config.clustering_enabled == True
    assert stage3_config.activity_threshold == 0.03

    # Тест стадии 5 (consolidation + advanced)
    stage5_config = config_manager.get_stage_config(5)
    assert stage5_config.plasticity_profile == "consolidation"
    assert stage5_config.progressive_scaling == True
    assert stage5_config.decoder_monitoring == True
    assert stage5_config.sparse_connection_ratio == 0.3

    print("✅ Профили пластичности настроены корректно")


def test_dynamic_config_generation():
    """Тест генерации секций пластичности и оптимизации"""
    print("⚙️ Тестирование генерации конфигураций...")

    generator = DynamicConfigGenerator()

    # Контекст для тестирования
    stage_context = {
        "plasticity_profile": "learning",
        "clustering_enabled": True,
        "activity_threshold": 0.03,
        "memory_optimizations": True,
        "emergence_tracking": True,
        "sparse_connection_ratio": 0.2,
    }

    # Тест генерации пластичности
    plasticity_config = generator.generate_plasticity_section(stage_context)
    assert plasticity_config["enable_plasticity"] == True
    assert plasticity_config["plasticity_rule"] == "combined"
    assert plasticity_config["activity_threshold"] == 0.03
    assert plasticity_config["profile"] == "learning"
    assert "functional_clustering" in plasticity_config
    assert "emergence_detection" in plasticity_config

    # Тест генерации оптимизации
    optimization_config = generator.generate_optimization_section(stage_context)
    assert "mixed_precision" in optimization_config
    assert "gradient_checkpointing" in optimization_config
    assert "sparse_connections" in optimization_config
    assert optimization_config["sparse_connections"]["sparsity_ratio"] == 0.2

    print("✅ Генерация конфигураций работает корректно")


def test_stage_runner_integration():
    """Тест интеграции в TrainingStageRunner"""
    print("🏃‍♂️ Тестирование интеграции TrainingStageRunner...")

    runner = TrainingStageRunner(mode="development", verbose=True)

    # Создаем тестовую конфигурацию стадии
    stage_config = StageConfig(
        stage=2,
        dataset_limit=100,  # Маленький датасет для теста
        epochs=1,
        batch_size=16,
        description="Phase 4 Integration Test",
        plasticity_profile="learning",
        clustering_enabled=False,
        activity_threshold=0.02,
        memory_optimizations=True,
        emergence_tracking=True,
    )

    # Тест генерации временной конфигурации
    temp_config_path = runner._generate_temp_config(stage_config)
    assert temp_config_path is not None

    # Проверяем, что файл создался и содержимое корректно
    with open(temp_config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Проверяем наличие секций Фазы 4
    assert "plasticity" in config_data
    assert config_data["plasticity"]["enable_plasticity"] == True
    assert config_data["plasticity"]["profile"] == "learning"

    assert "optimization" in config_data
    assert config_data["optimization"]["mixed_precision"]["enable"] == True

    # Очищаем временный файл
    import os

    os.remove(temp_config_path)

    print("✅ TrainingStageRunner интеграция работает корректно")


def test_adaptive_scaling():
    """Тест адаптивного масштабирования размеров решетки"""
    print("📐 Тестирование адаптивного масштабирования...")

    runner = TrainingStageRunner()

    # Тест разных стадий
    dims_stage1 = runner._get_adaptive_dimensions(1)
    assert dims_stage1 == (16, 16, 16)

    dims_stage3 = runner._get_adaptive_dimensions(3)
    assert dims_stage3 == (24, 24, 24)

    dims_stage5 = runner._get_adaptive_dimensions(5)
    assert dims_stage5 == (40, 40, 30)

    # Тест большой стадии (должна возвращать значение по умолчанию)
    dims_large = runner._get_adaptive_dimensions(100)
    assert dims_large == (40, 40, 30)  # Default to stage 5

    print("✅ Адаптивное масштабирование работает корректно")


def main():
    """Основная функция тестирования"""
    print("🎯 ТЕСТ ИНТЕГРАЦИИ ФАЗЫ 4: Начинаем проверку...")
    print("=" * 60)

    try:
        test_phase4_types()
        test_progressive_config_plasticity()
        test_dynamic_config_generation()
        test_stage_runner_integration()
        test_adaptive_scaling()

        print("=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ФАЗЫ 4 ПРОШЛИ УСПЕШНО!")
        print()
        print("✅ Готово к следующему шагу:")
        print("   - StageConfig расширен")
        print("   - Профили пластичности настроены")
        print("   - Генерация конфигураций работает")
        print("   - TrainingStageRunner интегрирован")
        print("   - Адаптивное масштабирование готово")
        print()
        print("🚀 Можно переходить к тестированию на малых решетках!")

        return True

    except Exception as e:
        print(f"❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
