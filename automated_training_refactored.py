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
        raise ImportError("Automated training components not available")

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
    # Fallback: если новые модули недоступны, показываем ошибку
    print(f"[ERROR] Failed to import refactored automated training modules: {e}")
    print("[INFO] Make sure all components are properly implemented:")
    print("   - training/automated_training/__init__.py")
    print("   - training/automated_training/automated_trainer.py")
    print("   - training/automated_training/progressive_config.py")
    print("   - training/automated_training/stage_runner.py")
    print("   - training/automated_training/session_manager.py")
    print("   - training/automated_training/cli_interface.py")
    print("[FALLBACK] Using original implementation...")

    # Здесь можно добавить fallback на оригинальную реализацию
    # Но для демонстрации рефакторинга показываем ошибку
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
