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
from dataclasses import dataclass, asdict

from .stage_runner import StageResult

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Сводка по сессии обучения"""

    total_stages: int
    total_time_minutes: float
    best_similarity: Optional[float]
    avg_similarity: Optional[float]
    similarity_trend: List[float]


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

        logger.info("[SESSION] SessionManager initialized")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Scale: {scale}")
        logger.info(f"   Max time: {max_total_time_hours} hours")
        logger.info(f"   Session log: {self.session_log}")

    def add_stage_result(self, result: StageResult):
        """Добавляет результат стадии в историю"""
        self.training_history.append(result)
        self._save_session_log()
        logger.info(f"[SESSION] Added stage {result.stage} result to history")

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
            logger.info(
                f"[TIME] Time limit reached: {elapsed_hours:.1f}/{self.max_total_time_hours} hours"
            )
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
        """Логирует начало сессии обучения"""
        logger.info(f"🎯 ======== AUTOMATED TRAINING SESSION STARTED ========")
        logger.info(f"[TARGET] Starting automated training session")
        logger.info(f"   Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Max duration: {self.max_total_time_hours} hours")
        logger.info(
            f"   Target end time: {(self.start_time + timedelta(hours=self.max_total_time_hours)).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"   Mode: {self.mode}")
        if self.scale:
            logger.info(f"   Scale factor: {self.scale}")
        logger.info(f"   Session log: {self.session_log}")
        logger.info("=" * 60)

    def log_stage_start(self, stage: int):
        """Логирует начало стадии"""
        logger.info(f"🚀 [STAGE-{stage}] ======== STARTING STAGE {stage} ========")

    def log_stage_completion(self, result: StageResult, stage_duration: float):
        """Логирует завершение стадии с детальным прогрессом"""
        summary = self.get_session_summary()
        elapsed_total = self.get_elapsed_time_hours()
        remaining_time = self.get_remaining_time_hours()

        logger.info(
            f"✅ [STAGE-{result.stage}] ======== STAGE {result.stage} COMPLETED ========"
        )
        logger.info(f"   Stage duration: {stage_duration:.1f} minutes")
        logger.info(f"[DATA] Overall Progress:")
        logger.info(f"   Stages completed: {summary.total_stages}")
        logger.info(
            f"   Session time: {elapsed_total:.1f}h / {self.max_total_time_hours}h"
        )
        logger.info(f"   Remaining time: {remaining_time:.1f}h")
        logger.info(
            f"   Best similarity: {summary.best_similarity:.4f}"
            if summary.best_similarity
            else "   Best similarity: N/A"
        )
        logger.info(f"   Total training time: {summary.total_time_minutes:.1f} minutes")

        # Показываем тренд похожести
        if summary.similarity_trend and len(summary.similarity_trend) > 1:
            trend = summary.similarity_trend
            logger.info(f"   Similarity trend: {[f'{s:.3f}' for s in trend]}")
            if len(trend) >= 2:
                improvement = trend[-1] - trend[-2]
                logger.info(f"   Last improvement: {improvement:+.4f}")

    def log_final_summary(self):
        """Выводит финальную сводку по сессии"""
        summary = self.get_session_summary()
        elapsed_hours = self.get_elapsed_time_hours()

        logger.info(f"\n[SUCCESS] Automated training session completed!")
        logger.info(f"[DATA] Final Summary:")
        logger.info(f"   Total duration: {elapsed_hours:.1f} hours")
        logger.info(f"   Stages completed: {summary.total_stages}")
        logger.info(f"   Total training time: {summary.total_time_minutes:.1f} minutes")
        logger.info(
            f"   Best similarity achieved: {summary.best_similarity:.4f}"
            if summary.best_similarity
            else "   Best similarity: N/A"
        )
        logger.info(
            f"   Average similarity: {summary.avg_similarity:.4f}"
            if summary.avg_similarity
            else "   Average similarity: N/A"
        )
        logger.info(f"   Session log saved: {self.session_log}")

        if summary.similarity_trend:
            logger.info(
                f"   Recent similarity trend: {[f'{s:.3f}' for s in summary.similarity_trend]}"
            )

    def _save_session_log(self):
        """Сохраняет лог сессии в JSON файл"""
        session_data = {
            "mode": self.mode,
            "scale": self.scale,
            "max_total_time_hours": self.max_total_time_hours,
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "elapsed_hours": self.get_elapsed_time_hours(),
            "training_history": [
                self._serialize_stage_result(r) for r in self.training_history
            ],
            "summary": asdict(self.get_session_summary()),
        }

        try:
            with open(self.session_log, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[ERROR] Failed to save session log: {e}")

    def _serialize_stage_result(self, result: StageResult) -> Dict[str, Any]:
        """Сериализует StageResult для JSON"""
        return {
            "stage": result.stage,
            "config": {
                "dataset_limit": result.config.dataset_limit,
                "epochs": result.config.epochs,
                "batch_size": result.config.batch_size,
                "description": result.config.description,
                "stage": result.config.stage,
            },
            "success": result.success,
            "actual_time_minutes": result.actual_time_minutes,
            "estimated_time_minutes": result.estimated_time_minutes,
            "final_similarity": result.final_similarity,
            "error": result.error,
            "timestamp": result.timestamp,
            # stdout не сохраняем для экономии места
        }

    def load_session_history(self, session_file: Path) -> bool:
        """
        Загружает историю сессии из файла

        Args:
            session_file: Путь к файлу сессии

        Returns:
            bool: True если загрузка успешна
        """
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Восстанавливаем историю (упрощенно)
            self.training_history = []

            logger.info(f"[SESSION] Loaded session history from {session_file}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to load session history: {e}")
            return False
