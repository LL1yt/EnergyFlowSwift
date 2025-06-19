"""
Automated Training Module - Система автоматизированного обучения

Этот модуль реализует систему автоматизированного долгосрочного обучения
с прогрессивным увеличением сложности для 3D Cellular Neural Network.

Основные компоненты:
- AutomatedTrainer: главный класс для управления обучением
- ProgressiveConfigManager: менеджер конфигураций для разных стадий
- TrainingStageRunner: выполнитель тренировочных стадий
- SessionManager: управление сессиями и логированием
- CLIInterface: интерфейс командной строки

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0
Дата: 2025
"""

__version__ = "1.0.0"
__author__ = "3D Cellular Neural Network Project"

# Импорты компонентов модуля
try:
    from .automated_trainer import AutomatedTrainer
    from .progressive_config import ProgressiveConfigManager
    from .stage_runner import TrainingStageRunner
    from .session_manager import SessionManager
    from .cli_interface import CLIInterface

    AUTOMATED_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Automated Training components not available: {e}")
    AUTOMATED_TRAINING_AVAILABLE = False

# Публичные компоненты
__all__ = [
    "AutomatedTrainer",
    "ProgressiveConfigManager",
    "TrainingStageRunner",
    "SessionManager",
    "CLIInterface",
    "AUTOMATED_TRAINING_AVAILABLE",
]
