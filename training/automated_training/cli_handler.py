"""
CLI Command Handlers - Обработчики команд CLI

Этот модуль содержит логику для выполнения команд, полученных от CLI,
таких как запуск обучения или отображение конфигурации.
"""

import logging
import sys
import argparse
import traceback

from .automated_trainer import AutomatedTrainer

logger = logging.getLogger(__name__)


def handle_show_config_test(args: argparse.Namespace):
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
            verbose=args.verbose,
        )

        logger.warning("⚙️ Training Configuration Preview:")
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
        logger.warning(f"Fits in {args.max_hours}h: {'[OK]' if fits else '[ERROR]'}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to show config: {e}")
        sys.exit(1)


def handle_run_automated_training(args: argparse.Namespace):
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
            verbose=args.verbose,
        )

        trainer.run_automated_training()

    except KeyboardInterrupt:
        logger.warning("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
