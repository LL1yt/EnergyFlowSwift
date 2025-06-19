"""
CLI Interface - Интерфейс командной строки для автоматизированного обучения

Этот модуль предоставляет CLI интерфейс для запуска автоматизированного
обучения с различными параметрами и конфигурациями.
"""

import argparse
import sys
import logging
from typing import Optional

from .automated_trainer import AutomatedTrainer
from .logging_config import setup_automated_training_logging, get_training_logger

logger = logging.getLogger(__name__)


class CLIInterface:
    """CLI интерфейс для автоматизированного обучения"""

    def __init__(self):
        """Инициализация CLI интерфейса"""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Создает парсер аргументов командной строки"""
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

        # Основные параметры
        parser.add_argument(
            "--mode",
            choices=["development", "research", "validation", "production"],
            default="development",
            help="Configuration mode (default: development)",
        )

        parser.add_argument(
            "--scale",
            type=float,
            default=None,
            help="Custom scale factor for dynamic configuration",
        )

        parser.add_argument(
            "--max-hours",
            type=float,
            default=8.0,
            help="Maximum training time in hours (default: 8.0)",
        )

        # Параметры тестирования
        parser.add_argument(
            "--test-config",
            action="store_true",
            help="Show training stages configuration and exit",
        )

        # Override параметры для тестирования
        parser.add_argument(
            "--dataset-limit",
            type=int,
            default=None,
            help="Override dataset_limit for all stages (useful for quick testing)",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Override batch_size for all stages (useful for faster training)",
        )

        # Технические параметры
        parser.add_argument(
            "--timeout-multiplier",
            type=float,
            default=2.0,
            help="Timeout multiplier for training processes (default: 2.0)",
        )

        # Логирование
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )

        parser.add_argument(
            "--quiet", "-q", action="store_true", help="Suppress most output"
        )

        return parser

    def parse_args(self, args: Optional[list] = None) -> argparse.Namespace:
        """
        Парсит аргументы командной строки

        Args:
            args: Список аргументов (для тестирования), None для sys.argv

        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)

    def setup_logging(self, args: argparse.Namespace):
        """
        Настраивает логирование на основе CLI аргументов

        Args:
            args: Parsed CLI arguments
        """
        # Используем новую централизованную систему логирования
        setup_automated_training_logging(verbose=args.verbose, quiet=args.quiet)

    def validate_args(self, args: argparse.Namespace) -> bool:
        """
        Валидирует аргументы командной строки

        Args:
            args: Parsed CLI arguments

        Returns:
            bool: True если все аргументы валидны
        """
        # Валидация max_hours
        if args.max_hours <= 0:
            logger.error(f"Invalid max-hours: {args.max_hours}. Must be positive.")
            return False

        if args.max_hours > 72:  # Разумное ограничение в 72 часа
            logger.warning(
                f"Large max-hours value: {args.max_hours}. Consider if this is intentional."
            )

        # Валидация dataset_limit
        if args.dataset_limit is not None and args.dataset_limit <= 0:
            logger.error(
                f"Invalid dataset-limit: {args.dataset_limit}. Must be positive."
            )
            return False

        # Валидация batch_size
        if args.batch_size is not None and args.batch_size <= 0:
            logger.error(f"Invalid batch-size: {args.batch_size}. Must be positive.")
            return False

        # Валидация timeout_multiplier
        if args.timeout_multiplier <= 0:
            logger.error(
                f"Invalid timeout-multiplier: {args.timeout_multiplier}. Must be positive."
            )
            return False

        # Валидация scale
        if args.scale is not None and args.scale <= 0:
            logger.error(f"Invalid scale: {args.scale}. Must be positive.")
            return False

        return True

    def show_config_test(self, args: argparse.Namespace):
        """
        Показывает конфигурацию стадий без запуска обучения

        Args:
            args: Parsed CLI arguments
        """
        try:
            trainer = AutomatedTrainer(
                mode=args.mode,
                scale=args.scale,
                max_total_time_hours=args.max_hours,
                dataset_limit_override=args.dataset_limit,
                batch_size_override=args.batch_size,
                timeout_multiplier=args.timeout_multiplier,
            )

            # Показываем конфигурацию стадий (компактно)
            logger.warning(f"⚙️ Training Configuration Preview:")
            stages_info = trainer.config_manager.get_all_stages_info()

            total_estimated_time = 0
            for stage, info in stages_info.items():
                config = info["config"]
                estimated_time = info["estimated_time_minutes"]
                total_estimated_time += estimated_time

                logger.warning(
                    f"   Stage {stage}: {config.dataset_limit:,} samples, {config.epochs}e, {estimated_time:.0f}min"
                )

            logger.warning(
                f"Total: {total_estimated_time:.0f}min ({total_estimated_time/60:.1f}h)"
            )

            fits = total_estimated_time / 60 <= args.max_hours
            logger.warning(f"Fits in {args.max_hours}h: {'✅' if fits else '❌'}")

        except Exception as e:
            logger.error(f"❌ Failed to show config: {e}")
            sys.exit(1)

    def run_automated_training(self, args: argparse.Namespace):
        """
        Запускает автоматизированное обучение

        Args:
            args: Parsed CLI arguments
        """
        try:
            trainer = AutomatedTrainer(
                mode=args.mode,
                scale=args.scale,
                max_total_time_hours=args.max_hours,
                dataset_limit_override=args.dataset_limit,
                batch_size_override=args.batch_size,
                timeout_multiplier=args.timeout_multiplier,
            )

            # Запускаем автоматизированное обучение
            trainer.run_automated_training()

        except KeyboardInterrupt:
            logger.warning("⏹️ Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def main(self, args: Optional[list] = None) -> int:
        """
        Главная функция CLI интерфейса

        Args:
            args: Аргументы командной строки (для тестирования)

        Returns:
            int: Код выхода (0 - успех, 1 - ошибка)
        """
        try:
            # Парсим аргументы
            parsed_args = self.parse_args(args)

            # Настраиваем логирование
            self.setup_logging(parsed_args)

            # Валидируем аргументы
            if not self.validate_args(parsed_args):
                return 1

            # Показываем конфигурацию если запрошено
            if parsed_args.test_config:
                self.show_config_test(parsed_args)
                return 0

            # Запускаем обучение
            self.run_automated_training(parsed_args)
            return 0

        except Exception as e:
            logger.error(f"[ERROR] CLI interface failed: {e}")
            return 1


def main():
    """Точка входа для CLI"""
    cli = CLIInterface()
    return cli.main()
