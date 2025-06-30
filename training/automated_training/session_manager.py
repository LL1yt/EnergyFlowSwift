"""
Session Manager - Управление сессиями автоматизированного обучения

Этот модуль отвечает за управление сессиями обучения, логирование прогресса,
сохранение истории и генерацию сводок по результатам.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from .types import StageResult, SessionSummary
from . import session_persistence

logger = logging.getLogger(__name__)


class SessionManager:
    """Менеджер сессий автоматизированного обучения"""

    def __init__(
        self,
        mode: str = "development",
        scale: Optional[float] = None,
        max_total_time_hours: float = 8.0,
    ):
        """
        Args:
            mode: Режим конфигурации
            scale: Custom scale factor
            max_total_time_hours: Максимальное время обучения в часах
        """
        self.mode = mode
        self.scale = scale
        self.max_total_time_hours = max_total_time_hours
        self.start_time = datetime.now()

        # История обучения
        self.training_history: List[StageResult] = []

        # Создаем директорию для логов
        self.log_dir = Path("logs/automated_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем директорию для основных логов
        Path("logs").mkdir(exist_ok=True)

        # Файл лога сессии
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.log_dir / f"automated_session_{timestamp}.json"

        # Минимальное логирование инициализации
        if max_total_time_hours > 4:  # Логируем только длительные сессии
            logger.warning(
                f"[SESSION] Long training session: {self.max_total_time_hours}h"
            )

    def add_stage_result(self, result: StageResult):
        """Добавляет результат стадии в историю"""
        self.training_history.append(result)
        self.save_session()

    def get_session_summary(self) -> SessionSummary:
        """Генерирует сводку по текущей сессии"""
        successful_stages = [h for h in self.training_history if h.success]

        if not successful_stages:
            return SessionSummary(
                total_stages=0,
                total_time_minutes=0.0,
                best_similarity=None,
                avg_similarity=None,
                similarity_trend=[],
            )

        total_time = sum(h.actual_time_minutes for h in successful_stages)
        similarities = [
            h.final_similarity
            for h in successful_stages
            if h.final_similarity is not None
        ]

        return SessionSummary(
            total_stages=len(successful_stages),
            total_time_minutes=total_time,
            best_similarity=max(similarities) if similarities else None,
            avg_similarity=(
                sum(similarities) / len(similarities) if similarities else None
            ),
            similarity_trend=(
                similarities[-3:] if len(similarities) >= 3 else similarities
            ),
        )

    def should_continue_session(self) -> bool:
        """Проверяет, стоит ли продолжать сессию обучения"""
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        if elapsed_hours >= self.max_total_time_hours:
            logger.warning(f"[ALARM] Time limit reached: {elapsed_hours:.1f}h")
            return False

        return True

    def get_remaining_time_hours(self) -> float:
        """Возвращает оставшееся время сессии в часах"""
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        return max(0, self.max_total_time_hours - elapsed_hours)

    def get_elapsed_time_hours(self) -> float:
        """Возвращает прошедшее время сессии в часах"""
        return (datetime.now() - self.start_time).total_seconds() / 3600

    def log_session_start(self):
        """Логирует начало сессии обучения (минимально)"""
        logger.warning(f"[TARGET] AUTOMATED TRAINING SESSION STARTED")
        logger.warning(f"   Duration: {self.max_total_time_hours}h | Mode: {self.mode}")
        if self.scale:
            logger.warning(f"   Scale: {self.scale}")

    def log_stage_start(self, stage: int):
        """Логирует начало стадии (убрано - используется глобальная функция)"""
        pass

    def log_stage_completion(self, result: StageResult, stage_duration: float):
        """Логирует завершение стадии с критическим прогрессом"""
        summary = self.get_session_summary()
        elapsed_total = self.get_elapsed_time_hours()
        remaining_time = self.get_remaining_time_hours()

        # Логируем только каждую 3-ю стадию или важные моменты
        if result.stage % 3 == 0 or not result.success or remaining_time < 1.0:
            logger.warning(
                f"[DATA] Progress: Stage {result.stage} | {elapsed_total:.1f}h/{self.max_total_time_hours}h"
            )
            if summary.best_similarity:
                logger.warning(f"   Best similarity: {summary.best_similarity:.3f}")
            if remaining_time < 1.0:
                logger.warning(f"   [WARN] Less than 1h remaining!")

    def log_final_summary(self):
        """Выводит финальную сводку по сессии"""
        summary = self.get_session_summary()
        elapsed_hours = self.get_elapsed_time_hours()

        logger.warning(f"[START] TRAINING SESSION COMPLETED")
        logger.warning(
            f"   Duration: {elapsed_hours:.1f}h | Stages: {summary.total_stages}"
        )
        if summary.best_similarity:
            logger.warning(f"   Best similarity: {summary.best_similarity:.3f}")
        logger.warning(f"   Session log: {self.session_log}")

    def save_session(self):
        """Сохраняет лог сессии в JSON файл"""
        session_persistence.save_session_log(
            session_log_file=self.session_log,
            mode=self.mode,
            scale=self.scale,
            max_hours=self.max_total_time_hours,
            start_time=self.start_time.isoformat(),
            elapsed_hours=self.get_elapsed_time_hours(),
            history=self.training_history,
            summary=self.get_session_summary(),
        )

    def load_session_history(self, session_file: Path) -> bool:
        """
        Загружает историю сессии из файла

        Args:
            session_file: Путь к файлу сессии

        Returns:
            bool: True если загрузка успешна
        """
        history = session_persistence.load_session_history(session_file)
        if history:
            self.training_history = history
            # Potentially restore other state here if needed
            logger.info(f"[SESSION] Loaded session history from {session_file}")
            return True
        return False
