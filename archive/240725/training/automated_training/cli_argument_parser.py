"""
Parser for CLI arguments - Парсер для CLI аргументов
"""

import argparse


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Создает и настраивает парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Automated Long Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Стандартное обучение в development режиме на 8 часов
  python automated_training.py
  
  # Быстрое тестирование с маленьким датасетом
  python automated_training.py --dataset-limit 100 --batch-size 16 --max-hours 1
  
  # Продакшн обучение на 24 часа
  python automated_training.py --mode production --max-hours 24
  
  # Показать конфигурации стадий без запуска
  python automated_training.py --test-config
  
  # Исследовательский режим с custom scale
  python automated_training.py --mode research --scale 1.5 --max-hours 12
        """,
    )

    _add_main_arguments(parser)
    _add_testing_arguments(parser)
    _add_technical_arguments(parser)
    _add_logging_arguments(parser)

    return parser


def _add_main_arguments(parser: argparse.ArgumentParser):
    """Добавляет основные параметры."""
    group = parser.add_argument_group("Основные параметры")
    group.add_argument(
        "--mode",
        choices=["development", "research", "validation", "production"],
        default="development",
        help="Configuration mode (default: development)",
    )
    group.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Custom scale factor for dynamic configuration",
    )
    group.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="Maximum training time in hours (default: 8.0)",
    )


def _add_testing_arguments(parser: argparse.ArgumentParser):
    """Добавляет параметры для тестирования."""
    group = parser.add_argument_group("Параметры тестирования")
    group.add_argument(
        "--test-config",
        action="store_true",
        help="Show training stages configuration and exit",
    )
    group.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Override dataset_limit for all stages (useful for quick testing)",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size for all stages (useful for faster training)",
    )


def _add_technical_arguments(parser: argparse.ArgumentParser):
    """Добавляет технические параметры."""
    group = parser.add_argument_group("Технические параметры")
    group.add_argument(
        "--timeout-multiplier",
        type=float,
        default=2.0,
        help="Timeout multiplier for training processes (default: 2.0)",
    )


def _add_logging_arguments(parser: argparse.ArgumentParser):
    """Добавляет параметры логирования."""
    group = parser.add_argument_group("Параметры логирования")
    group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress most output"
    )
