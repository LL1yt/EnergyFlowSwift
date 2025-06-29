#!/usr/bin/env python3
"""
[BOT] Automated Long Training Script (Refactored)
Автоматизированное долгое обучение с прогрессивным увеличением сложности

НОВАЯ МОДУЛЬНАЯ АРХИТЕКТУРА:
- Разделено на компоненты для лучшей читаемости и поддерживаемости
- Каждый компонент отвечает за свою область ответственности
- Сохранена обратная совместимость с оригинальным API

Компоненты:
- ProgressiveConfigManager: управление конфигурациями стадий
- TrainingStageRunner: выполнение тренировочных процессов
- SessionManager: управление сессиями и логированием
- AutomatedTrainer: интеграция всех компонентов
- CLIInterface: интерфейс командной строки

Автор: 3D Cellular Neural Network Project
Версия: v2.0.0 (Refactored)
"""

import logging
import sys

# Настройка логирования (базовая, будет переопределена CLI)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    # Импортируем новую модульную архитектуру
    from training.automated_training import (
        AutomatedTrainer,
        ProgressiveConfigManager,
        TrainingStageRunner,
        SessionManager,
        CLIInterface,
        AUTOMATED_TRAINING_AVAILABLE,
    )

    if not AUTOMATED_TRAINING_AVAILABLE:
        raise ImportError(
            "Automated training components not available - check training.automated_training.__init__.py"
        )

    # Создаем алиасы для обратной совместимости
    # Если кто-то импортировал классы напрямую из automated_training.py
    __all__ = [
        "AutomatedTrainer",
        "ProgressiveConfigManager",
        "TrainingStageRunner",
        "SessionManager",
        "main",
    ]

except ImportError as e:
    # Fallback: если новые модули недоступны, показываем детальную диагностику
    print(f"[ERROR] Failed to import refactored automated training modules: {e}")
    print("\n[DIAGNOSTIC] Checking component availability:")

    # Проверяем каждый компонент отдельно
    missing_components = []
    try:
        from training.automated_training.automated_trainer import AutomatedTrainer

        print("   ✅ AutomatedTrainer - OK")
    except ImportError as comp_e:
        print(f"   ❌ AutomatedTrainer - FAILED: {comp_e}")
        missing_components.append("AutomatedTrainer")

    try:
        from training.automated_training.progressive_config import (
            ProgressiveConfigManager,
        )

        print("   ✅ ProgressiveConfigManager - OK")
    except ImportError as comp_e:
        print(f"   ❌ ProgressiveConfigManager - FAILED: {comp_e}")
        missing_components.append("ProgressiveConfigManager")

    try:
        from training.automated_training.stage_runner import TrainingStageRunner

        print("   ✅ TrainingStageRunner - OK")
    except ImportError as comp_e:
        print(f"   ❌ TrainingStageRunner - FAILED: {comp_e}")
        missing_components.append("TrainingStageRunner")

    try:
        from training.automated_training.session_manager import SessionManager

        print("   ✅ SessionManager - OK")
    except ImportError as comp_e:
        print(f"   ❌ SessionManager - FAILED: {comp_e}")
        missing_components.append("SessionManager")

    try:
        from training.automated_training.cli_interface import CLIInterface

        print("   ✅ CLIInterface - OK")
    except ImportError as comp_e:
        print(f"   ❌ CLIInterface - FAILED: {comp_e}")
        missing_components.append("CLIInterface")

    print(f"\n[INFO] Missing components: {missing_components}")
    print("[INFO] Проверьте наличие всех файлов в training/automated_training/:")
    print("   - __init__.py")
    print("   - automated_trainer.py")
    print("   - progressive_config.py")
    print("   - stage_runner.py")
    print("   - session_manager.py")
    print("   - cli_interface.py")
    print("   - logging_config.py")
    print("   - process_runner.py")
    print("   - types.py")

    print("\n[FALLBACK] Проверьте также зависимости subprocess модулей:")
    print("   - smart_resume_training.py должен существовать")
    print("   - real_llama_training_production.py должен поддерживать CLI")
    print("   - production_training/ модули должны быть доступны")

    # Здесь можно добавить fallback на оригинальную реализацию
    # Но для демонстрации рефакторинга показываем ошибку
    print(f"\n[CRITICAL] Cannot proceed without automated training components.")
    print(
        f"[SUGGESTION] Run: python -m training.automated_training.cli_interface --help"
    )
    print(f"             Or: python smart_resume_training.py --help")
    sys.exit(1)


def main():
    """
    Главная функция - точка входа для скрипта

    Использует новый модульный CLI интерфейс
    """
    try:
        cli = CLIInterface()
        return cli.main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"[ERROR] Failed to start automated training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
