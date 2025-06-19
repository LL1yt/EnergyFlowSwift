#!/usr/bin/env python3
"""
Test NCA Integration with Automated Training
Тест интеграции NCA клеток с автоматизированным обучением

Проверяет:
- Совместимость MinimalNCACell с системой обучения
- Правильность работы логирования
- Корректность конфигурации
- Интеграцию с новой архитектурой

Автор: 3D Cellular Neural Network Project
"""

import sys
import time
import logging
from pathlib import Path

# Добавляем пути для импорта
sys.path.append(str(Path(__file__).parent))


def test_logging_system():
    """Тестирует новую систему логирования"""
    print("🔧 [TEST] Testing new logging system...")

    try:
        from training.automated_training.logging_config import (
            setup_automated_training_logging,
            get_training_logger,
            get_metrics_logger,
            log_stage_start,
            log_stage_complete,
        )

        # Настраиваем логирование
        setup_automated_training_logging(verbose=True, quiet=False)

        # Тестируем различные логгеры
        trainer_logger = get_training_logger("test.trainer")
        session_logger = get_training_logger("test.session")
        metrics_logger = get_metrics_logger("test_session")

        # Тестируем логирование
        trainer_logger.info("Test trainer message")
        session_logger.info("Test session message")
        metrics_logger.log_performance("test_operation", 1.5, {"status": "success"})

        # Тестируем специальные функции
        log_stage_start(
            1,
            {
                "description": "Test stage",
                "dataset_limit": 100,
                "epochs": 2,
                "batch_size": 8,
            },
        )

        log_stage_complete(
            1, {"success": True, "actual_time_minutes": 5.0, "final_similarity": 0.85}
        )

        print("✅ [TEST] Logging system works correctly!")
        return True

    except Exception as e:
        print(f"❌ [TEST] Logging system failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_nca_cell_creation():
    """Тестирует создание NCA клеток"""
    print("🧠 [TEST] Testing NCA cell creation...")

    try:
        from core.cell_prototype.architectures.minimal_nca_cell import (
            MinimalNCACell,
            create_nca_cell_from_config,
            create_compatible_nca_cell,
        )

        # Тест 1: Базовое создание клетки
        cell = MinimalNCACell(
            state_size=8, neighbor_count=6, hidden_dim=4, target_params=150
        )

        # Проверяем параметры
        total_params = sum(p.numel() for p in cell.parameters())
        print(f"   Basic NCA cell: {total_params} parameters")

        # Тест 2: Создание из конфигурации
        config = {
            "state_size": 8,
            "neighbor_count": 6,
            "hidden_dim": 4,
            "target_params": 300,
        }

        cell_from_config = create_nca_cell_from_config(config)
        config_params = sum(p.numel() for p in cell_from_config.parameters())
        print(f"   Config NCA cell: {config_params} parameters")

        # Тест 3: Совместимое создание
        compat_cell = create_compatible_nca_cell(
            state_size=8, neighbor_count=6, target_params=200
        )
        compat_params = sum(p.numel() for p in compat_cell.parameters())
        print(f"   Compatible NCA cell: {compat_params} parameters")

        print("✅ [TEST] NCA cell creation works correctly!")
        return True

    except Exception as e:
        print(f"❌ [TEST] NCA cell creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_nca_cell_forward():
    """Тестирует forward pass NCA клеток"""
    print("⚡ [TEST] Testing NCA cell forward pass...")

    try:
        import torch
        from core.cell_prototype.architectures.minimal_nca_cell import MinimalNCACell

        # Создаем клетку
        cell = MinimalNCACell(
            state_size=8,
            neighbor_count=6,
            hidden_dim=4,
            external_input_size=2,
            target_params=150,
        )

        # Подготавливаем тестовые данные
        batch_size = 4
        neighbor_states = torch.randn(
            batch_size, 6, 8
        )  # [batch, neighbors, state_size]
        own_state = torch.randn(batch_size, 8)  # [batch, state_size]
        external_input = torch.randn(batch_size, 2)  # [batch, external_input_size]

        # Тестируем forward pass
        start_time = time.time()
        new_state = cell(neighbor_states, own_state, external_input)
        forward_time = time.time() - start_time

        # Проверяем результат
        assert new_state.shape == (
            batch_size,
            8,
        ), f"Wrong output shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values in output"
        assert torch.isfinite(new_state).all(), "Infinite values in output"

        print(f"   Forward pass: {forward_time*1000:.2f}ms")
        print(f"   Output shape: {new_state.shape}")
        print(
            f"   Output range: [{new_state.min().item():.3f}, {new_state.max().item():.3f}]"
        )

        # Тестируем без external input
        new_state_no_ext = cell(neighbor_states, own_state, None)
        assert new_state_no_ext.shape == (
            batch_size,
            8,
        ), "Wrong output shape without external input"

        print("✅ [TEST] NCA cell forward pass works correctly!")
        return True

    except Exception as e:
        print(f"❌ [TEST] NCA cell forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_automated_training_config():
    """Тестирует конфигурацию автоматизированного обучения"""
    print("⚙️ [TEST] Testing automated training configuration...")

    try:
        from training.automated_training import (
            AutomatedTrainer,
            ProgressiveConfigManager,
            AUTOMATED_TRAINING_AVAILABLE,
        )

        if not AUTOMATED_TRAINING_AVAILABLE:
            print("❌ [TEST] Automated training not available!")
            return False

        # Создаем конфигурационный менеджер
        config_manager = ProgressiveConfigManager(
            dataset_limit_override=50,  # Маленькое значение для тестирования
            batch_size_override=4,
        )

        # Тестируем получение конфигурации стадий
        stage_1_config = config_manager.get_stage_config(1)
        stage_2_config = config_manager.get_stage_config(2)

        print(
            f"   Stage 1: {stage_1_config.dataset_limit} examples, {stage_1_config.epochs} epochs"
        )
        print(
            f"   Stage 2: {stage_2_config.dataset_limit} examples, {stage_2_config.epochs} epochs"
        )

        # Тестируем валидацию
        is_valid_1 = config_manager.validate_stage_config(stage_1_config)
        is_valid_2 = config_manager.validate_stage_config(stage_2_config)

        assert is_valid_1, "Stage 1 config is invalid"
        assert is_valid_2, "Stage 2 config is invalid"

        # Тестируем оценку времени
        estimated_time_1 = config_manager.estimate_stage_time(
            stage_1_config, "development"
        )
        estimated_time_2 = config_manager.estimate_stage_time(
            stage_2_config, "development"
        )

        print(f"   Estimated time stage 1: {estimated_time_1:.1f} minutes")
        print(f"   Estimated time stage 2: {estimated_time_2:.1f} minutes")

        # Создаем тренер
        trainer = AutomatedTrainer(
            mode="development",
            max_total_time_hours=0.5,  # 30 минут для тестирования
            dataset_limit_override=50,
            batch_size_override=4,
            timeout_multiplier=1.5,
        )

        # Тестируем предварительный просмотр стадий
        stages_preview = trainer.get_stages_preview()
        total_time = trainer.estimate_total_time()
        can_fit = trainer.can_fit_in_time_limit()

        print(f"   Total stages: {len(stages_preview)}")
        print(f"   Total estimated time: {total_time:.1f} hours")
        print(f"   Can fit in time limit: {can_fit}")

        print("✅ [TEST] Automated training configuration works correctly!")
        return True

    except Exception as e:
        print(f"❌ [TEST] Automated training configuration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration_compatibility():
    """Тестирует совместимость компонентов"""
    print("🔗 [TEST] Testing integration compatibility...")

    try:
        # Тестируем импорты
        from training.automated_training import (
            AutomatedTrainer,
            setup_automated_training_logging,
            get_training_logger,
            AUTOMATED_TRAINING_AVAILABLE,
        )

        from core.cell_prototype.architectures.minimal_nca_cell import (
            MinimalNCACell,
            create_compatible_nca_cell,
        )

        # Настраиваем логирование
        setup_automated_training_logging(verbose=False, quiet=True)
        logger = get_training_logger("integration_test")

        # Создаем NCA клетку
        nca_cell = create_compatible_nca_cell(
            state_size=8, neighbor_count=6, target_params=200
        )

        # Логируем создание клетки
        total_params = sum(p.numel() for p in nca_cell.parameters())
        logger.info(f"Created NCA cell with {total_params} parameters")

        # Создаем тренер с минимальными настройками
        trainer = AutomatedTrainer(
            mode="development",
            max_total_time_hours=0.1,  # 6 минут
            dataset_limit_override=20,
            batch_size_override=2,
            timeout_multiplier=1.0,
        )

        # Проверяем совместимость
        can_fit = trainer.can_fit_in_time_limit()
        stages_info = trainer.get_stages_preview()

        logger.info(f"Training can fit in time limit: {can_fit}")
        logger.info(f"Total stages configured: {len(stages_info)}")

        print("✅ [TEST] Integration compatibility works correctly!")
        return True

    except Exception as e:
        print(f"❌ [TEST] Integration compatibility failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Главная функция тестирования"""
    print("🧪 [TEST] Starting NCA + Automated Training Integration Tests")
    print("=" * 60)

    tests = [
        ("Logging System", test_logging_system),
        ("NCA Cell Creation", test_nca_cell_creation),
        ("NCA Cell Forward Pass", test_nca_cell_forward),
        ("Automated Training Config", test_automated_training_config),
        ("Integration Compatibility", test_integration_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)

        success = test_func()
        results.append((test_name, success))

        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")

    print("\n" + "=" * 60)
    print("🏁 [RESULTS] Test Summary:")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All tests passed! NCA + Automated Training integration is working correctly."
        )
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
