#!/usr/bin/env python3
"""
🔍 ДИАГНОСТИКА: Проблема с размерами решетки

Расследуем почему получается 7×7×3 вместо ожидаемых 16×16×16+

Проверяем:
1. Scale factor и его влияние на размеры
2. Progressive scaling применение
3. Expression evaluation в dynamic config
4. Передача параметров через stage runner
5. Конфигурация решетки на разных этапах
"""

import sys
import os
import tempfile
import yaml
import json
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))

from training.automated_training.types import StageConfig
from training.automated_training.progressive_config import ProgressiveConfigManager
from training.automated_training.stage_runner import TrainingStageRunner
from utils.config_manager.dynamic_config import (
    DynamicConfigGenerator,
    DynamicConfigManager,
)


def test_scale_factor_impact():
    """Тест влияния scale factor на размеры решетки"""
    print("🔍 Анализ влияния scale factor на размеры решетки...")

    scale_factors = [0.001, 0.01, 0.1, 1.0]

    for scale in scale_factors:
        print(f"\n--- Scale Factor: {scale} ---")

        # Создаем dynamic config manager с разными scale
        dynamic_manager = DynamicConfigManager()

        # Устанавливаем кастомный scale
        setattr(dynamic_manager.generator.scale_settings, "development", scale)

        # Генерируем конфигурацию
        config_data = dynamic_manager.create_config_for_mode("development")

        # Проверяем размеры решетки
        lattice_config = config_data.get("lattice", {})
        width = lattice_config.get("lattice_width", "unknown")
        height = lattice_config.get("lattice_height", "unknown")
        depth = lattice_config.get("lattice_depth", "unknown")

        print(f"   Lattice dimensions: {width}×{height}×{depth}")

        # Проверяем базовые размеры и биологические константы
        bio_constants = dynamic_manager.generator.bio_constants
        print(
            f"   Base dimensions: {bio_constants.base_width}×{bio_constants.base_height}×{bio_constants.base_depth}"
        )

        # Вычисляем ожидаемые размеры
        expected_width = int(bio_constants.base_width * scale)
        expected_height = int(bio_constants.base_height * scale)
        expected_depth = int(bio_constants.base_depth * scale)

        print(f"   Expected: {expected_width}×{expected_height}×{expected_depth}")

        # Проверяем есть ли expression evaluation
        if isinstance(width, str) and width.startswith("{"):
            print(f"   ⚠️  Width is expression: {width}")
        if isinstance(height, str) and height.startswith("{"):
            print(f"   ⚠️  Height is expression: {height}")
        if isinstance(depth, str) and depth.startswith("{"):
            print(f"   ⚠️  Depth is expression: {depth}")


def test_progressive_scaling_integration():
    """Тест интеграции progressive scaling"""
    print("\n🔍 Анализ интеграции progressive scaling...")

    # Создаем stage configs с progressive scaling
    for stage in [1, 2, 3, 4, 5]:
        print(f"\n--- Stage {stage} Analysis ---")

        # Создаем конфигурацию стадии
        stage_config = StageConfig(
            stage=stage,
            dataset_limit=100,
            epochs=2,
            batch_size=16,
            description=f"Debug Stage {stage}",
            plasticity_profile="learning",
            clustering_enabled=True,
            activity_threshold=0.03,
            memory_optimizations=True,
            emergence_tracking=True,
            progressive_scaling=True,  # Ключевая опция
        )

        # Создаем runner
        runner = TrainingStageRunner(mode="development", scale=0.01, verbose=True)

        # Проверяем адаптивные размеры
        adaptive_dims = runner._get_adaptive_dimensions(stage)
        print(
            f"   Adaptive dimensions: {adaptive_dims[0]}×{adaptive_dims[1]}×{adaptive_dims[2]}"
        )

        # Генерируем временную конфигурацию
        temp_config_path = runner._generate_temp_config(stage_config)

        if temp_config_path:
            with open(temp_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            lattice_config = config_data.get("lattice", {})
            actual_width = lattice_config.get("lattice_width", "unknown")
            actual_height = lattice_config.get("lattice_height", "unknown")
            actual_depth = lattice_config.get("lattice_depth", "unknown")

            print(f"   Generated config: {actual_width}×{actual_height}×{actual_depth}")

            # Проверяем соответствие
            if (actual_width, actual_height, actual_depth) == adaptive_dims:
                print("   ✅ Progressive scaling применен корректно")
            else:
                print("   ❌ Progressive scaling НЕ применен или работает неправильно")
                print(f"      Expected: {adaptive_dims}")
                print(
                    f"      Actual: ({actual_width}, {actual_height}, {actual_depth})"
                )

            # Проверяем другие секции
            if "plasticity" in config_data:
                print("   ✅ Plasticity section present")
            else:
                print("   ❌ Plasticity section missing")

            if "optimization" in config_data:
                print("   ✅ Optimization section present")
            else:
                print("   ❌ Optimization section missing")

            os.remove(temp_config_path)
        else:
            print("   ❌ Failed to generate config")


def test_expression_evaluation():
    """Тест evaluation выражений в dynamic config"""
    print("\n🔍 Анализ expression evaluation...")

    generator = DynamicConfigGenerator()

    # Тест простых выражений
    test_expressions = [
        "{smart_round(666 * 0.01)}",
        "{int(666 * 0.01)}",
        "{666 * 0.01}",
        "{lattice_width}",
        "{smart_round(lattice_width * 0.5)}",
    ]

    # Создаем тестовый контекст
    test_context = {
        "lattice_width": 666,
        "lattice_height": 666,
        "lattice_depth": 333,
        "scale": 0.01,
    }

    print("Testing expression evaluation:")
    for expr in test_expressions:
        try:
            result = generator.evaluator.evaluate_expression(expr, test_context)
            print(f"   {expr} → {result} (type: {type(result).__name__})")
        except Exception as e:
            print(f"   {expr} → ERROR: {e}")


def test_full_pipeline_tracing():
    """Полная трассировка pipeline генерации конфигурации"""
    print("\n🔍 Полная трассировка pipeline...")

    # Создаем stage config с progressive scaling
    stage_config = StageConfig(
        stage=2,
        dataset_limit=100,
        epochs=2,
        batch_size=16,
        description="Full Pipeline Debug",
        plasticity_profile="learning",
        clustering_enabled=True,
        activity_threshold=0.025,
        memory_optimizations=True,
        emergence_tracking=True,
        progressive_scaling=True,
    )

    print("1. Stage Config Created:")
    print(f"   Stage: {stage_config.stage}")
    print(f"   Progressive scaling: {stage_config.progressive_scaling}")

    # Создаем runner
    runner = TrainingStageRunner(mode="development", scale=0.01, verbose=True)

    print("\n2. TrainingStageRunner Created:")
    print(f"   Mode: {runner.mode}")
    print(f"   Scale: {runner.scale}")

    # Проверяем адаптивные размеры
    adaptive_dims = runner._get_adaptive_dimensions(stage_config.stage)
    print(f"\n3. Adaptive Dimensions:")
    print(f"   Expected: {adaptive_dims[0]}×{adaptive_dims[1]}×{adaptive_dims[2]}")

    # Создаем dynamic manager
    dynamic_manager = DynamicConfigManager()

    print("\n4. Dynamic Config Manager:")
    print(f"   Scale settings: {dynamic_manager.generator.scale_settings.development}")

    # Устанавливаем scale если нужно
    if runner.scale is not None:
        setattr(dynamic_manager.generator.scale_settings, runner.mode, runner.scale)
        print(f"   Applied custom scale: {runner.scale}")

    # Генерируем базовую конфигурацию
    config_data = dynamic_manager.create_config_for_mode(runner.mode)

    print("\n5. Base Config Generated:")
    lattice_config = config_data.get("lattice", {})
    print(
        f"   Base lattice: {lattice_config.get('lattice_width')}×{lattice_config.get('lattice_height')}×{lattice_config.get('lattice_depth')}"
    )

    # Применяем оптимизации
    optimized_config = runner._prepare_config_with_optimizations(
        config_data, stage_config
    )

    print("\n6. Optimized Config Applied:")
    opt_lattice_config = optimized_config.get("lattice", {})
    print(
        f"   Final lattice: {opt_lattice_config.get('lattice_width')}×{opt_lattice_config.get('lattice_height')}×{opt_lattice_config.get('lattice_depth')}"
    )

    # Проверяем что progressive scaling применился
    if stage_config.progressive_scaling:
        expected_width, expected_height, expected_depth = adaptive_dims
        actual_width = opt_lattice_config.get("lattice_width")
        actual_height = opt_lattice_config.get("lattice_height")
        actual_depth = opt_lattice_config.get("lattice_depth")

        print("\n7. Progressive Scaling Check:")
        print(f"   Expected: {expected_width}×{expected_height}×{expected_depth}")
        print(f"   Actual: {actual_width}×{actual_height}×{actual_depth}")

        if (actual_width, actual_height, actual_depth) == (
            expected_width,
            expected_height,
            expected_depth,
        ):
            print("   ✅ Progressive scaling applied correctly")
        else:
            print("   ❌ Progressive scaling NOT applied correctly")
            print("   🔍 This is likely the source of the 7×7×3 issue!")

    return optimized_config


def diagnose_smart_resume_integration():
    """Диагностика интеграции с smart_resume_training"""
    print("\n🔍 Диагностика интеграции smart_resume_training...")

    # Проверяем как smart_resume_training получает конфигурацию
    print("Checking smart_resume_training config flow:")
    print("1. TrainingStageRunner generates temp config")
    print("2. smart_resume_training.py loads temp config")
    print("3. Config initializer processes loaded config")

    # Возможные точки сбоя:
    problems = [
        "Temp config не содержит правильные размеры",
        "smart_resume_training перезаписывает размеры решетки",
        "Config initializer неправильно интерпретирует размеры",
        "Scale factor применяется дважды (compound scaling)",
        "Expression evaluation происходит некорректно",
    ]

    print("\nВозможные точки сбоя:")
    for i, problem in enumerate(problems, 1):
        print(f"   {i}. {problem}")

    print("\nРекомендации для дальнейшего расследования:")
    print("1. Добавить логи в smart_resume_training.py для размеров решетки")
    print("2. Проверить config_initializer.py на перезапись размеров")
    print("3. Добавить детальные логи в _prepare_config_with_optimizations")
    print("4. Проверить правильность expression evaluation")


def main():
    """Основная функция диагностики"""
    print("🔍 ДИАГНОСТИКА ПРОБЛЕМЫ С РАЗМЕРАМИ РЕШЕТКИ")
    print("=" * 70)
    print("Цель: Понять почему получается 7×7×3 вместо 16×16×16+")
    print()

    try:
        # 1. Тест влияния scale factor
        test_scale_factor_impact()

        # 2. Тест progressive scaling
        test_progressive_scaling_integration()

        # 3. Тест expression evaluation
        test_expression_evaluation()

        # 4. Полная трассировка
        final_config = test_full_pipeline_tracing()

        # 5. Диагностика smart_resume интеграции
        diagnose_smart_resume_integration()

        print("\n" + "=" * 70)
        print("🎯 ДИАГНОСТИКА ЗАВЕРШЕНА")
        print()
        print("📊 НАЙДЕННЫЕ ПРОБЛЕМЫ:")

        # Сохраняем финальную конфигурацию для анализа
        debug_config_path = "debug_final_config.yaml"
        with open(debug_config_path, "w", encoding="utf-8") as f:
            yaml.dump(final_config, f, allow_unicode=True, indent=2)

        print(f"💾 Финальная конфигурация сохранена: {debug_config_path}")
        print("📋 Проверьте логи выше для выявления проблем")
        print()
        print("🔧 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Исправить обнаруженные проблемы")
        print("2. Добавить дополнительные логи в критичные места")
        print("3. Повторить тестирование с исправлениями")

        return True

    except Exception as e:
        print(f"❌ ОШИБКА В ДИАГНОСТИКЕ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
