"""
Automated Trainer - Главный класс для автоматизированного обучения

Этот класс интегрирует все компоненты системы автоматизированного обучения:
- ProgressiveConfigManager для управления конфигурациями стадий
- TrainingStageRunner для выполнения тренировочных процессов
- SessionManager для управления сессиями и логированием

Предоставляет упрощенный интерфейс для запуска долгосрочного обучения
с прогрессивным увеличением сложности.
"""

import time
import logging
from typing import Optional

from .progressive_config import ProgressiveConfigManager
from .stage_runner import TrainingStageRunner
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class AutomatedTrainer:
    """Автоматизированный тренер для долгого обучения"""

    def __init__(
        self,
        mode: str = "development",
        scale: Optional[float] = None,
        max_total_time_hours: float = 8.0,
        dataset_limit_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        timeout_multiplier: float = 2.0,
    ):
        """
        Args:
            mode: Режим конфигурации (development, research, etc.)
            scale: Custom scale factor
            max_total_time_hours: Максимальное время обучения в часах
            dataset_limit_override: Переопределить dataset_limit для всех стадий (для тестирования)
            batch_size_override: Переопределить batch_size для всех стадий (для ускорения)
            timeout_multiplier: Multiplier for the timeout
        """
        self.mode = mode
        self.scale = scale
        self.max_total_time_hours = max_total_time_hours

        # Инициализация компонентов
        self.config_manager = ProgressiveConfigManager(
            dataset_limit_override=dataset_limit_override,
            batch_size_override=batch_size_override,
        )

        self.stage_runner = TrainingStageRunner(
            mode=mode, scale=scale, timeout_multiplier=timeout_multiplier
        )

        self.session_manager = SessionManager(
            mode=mode, scale=scale, max_total_time_hours=max_total_time_hours
        )

        logger.info(f"[BOT] AutomatedTrainer initialized")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   Max time: {max_total_time_hours} hours")
        if dataset_limit_override:
            logger.info(f"   Dataset limit override: {dataset_limit_override}")
        if batch_size_override:
            logger.info(f"   Batch size override: {batch_size_override}")
        logger.info(f"   Timeout multiplier: {timeout_multiplier}")

    def run_automated_training(self):
        """Запускает автоматизированное обучение"""
        # Логируем начало сессии
        self.session_manager.log_session_start()

        stage = 1

        while self.session_manager.should_continue_session():
            # Получаем конфигурацию стадии
            stage_config = self.config_manager.get_stage_config(stage)

            # Валидируем конфигурацию
            if not self.config_manager.validate_stage_config(stage_config):
                logger.error(f"[ERROR] Invalid configuration for stage {stage}")
                break

            # Оцениваем время выполнения
            estimated_time = self.config_manager.estimate_stage_time(
                stage_config, self.mode
            )

            # Проверяем, хватит ли времени для этой стадии
            remaining_hours = self.session_manager.get_remaining_time_hours()
            estimated_time_hours = estimated_time / 60

            if estimated_time_hours > remaining_hours:
                logger.info(f"[TIME] Not enough time for stage {stage}")
                logger.info(
                    f"   Estimated: {estimated_time_hours:.1f}h, Remaining: {remaining_hours:.1f}h"
                )
                break

            # Логируем начало стадии
            self.session_manager.log_stage_start(stage)
            stage_start_time = time.time()

            # Запускаем стадию
            result = self.stage_runner.run_stage(stage_config, estimated_time)

            stage_end_time = time.time()
            stage_duration = (stage_end_time - stage_start_time) / 60

            if result is None or not result.success:
                logger.error(
                    f"[ERROR] Stage {stage} failed after {stage_duration:.1f} minutes, stopping automated training"
                )
                break

            # Добавляем результат в сессию
            self.session_manager.add_stage_result(result)

            # Логируем завершение стадии
            self.session_manager.log_stage_completion(result, stage_duration)

            stage += 1

            logger.info(f"[NEXT] Preparing for stage {stage}...")

            # Небольшая пауза между стадиями
            logger.info("   [PAUSE] 10 second break between stages...")
            time.sleep(10)

        # Финальная сводка
        self.session_manager.log_final_summary()

    def get_stages_preview(self):
        """
        Возвращает предварительную информацию о всех стадиях

        Returns:
            Dict с информацией о стадиях
        """
        return self.config_manager.get_all_stages_info()

    def estimate_total_time(self) -> float:
        """
        Оценивает общее время выполнения всех стадий

        Returns:
            float: Общее время в часах
        """
        stages_info = self.get_stages_preview()
        total_minutes = sum(
            info["estimated_time_minutes"] for info in stages_info.values()
        )
        return total_minutes / 60

    def can_fit_in_time_limit(self) -> bool:
        """
        Проверяет, помещается ли полное обучение в лимит времени

        Returns:
            bool: True если помещается
        """
        estimated_total = self.estimate_total_time()
        return estimated_total <= self.max_total_time_hours
