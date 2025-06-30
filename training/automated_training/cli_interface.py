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
from .logging_config import setup_automated_training_logging
from .cli_argument_parser import create_cli_parser
from .cli_handler import handle_show_config_test, handle_run_automated_training

logger = logging.getLogger(__name__)


class CLIInterface:
    """CLI интерфейс для автоматизированного обучения"""

    def __init__(self):
        """Инициализация CLI интерфейса"""
        self.parser = create_cli_parser()

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

    def main(self, args: Optional[list] = None) -> int:
        """
        Главный метод для запуска CLI

        Args:
            args: Список аргументов для тестирования

        Returns:
            Exit code (0 on success)
        """
        parsed_args = self.parse_args(args)

        self.setup_logging(parsed_args)

        if not self.validate_args(parsed_args):
            return 1

        logger.info(f"🚀 Starting Automated Training CLI...")
        logger.info(f"CLI Arguments: {vars(parsed_args)}")

        if parsed_args.test_config:
            handle_show_config_test(parsed_args)
        else:
            handle_run_automated_training(parsed_args)

        logger.info("✅ Automated Training process finished.")
        return 0


def main():
    """Точка входа для CLI"""
    cli = CLIInterface()
    return cli.main()
