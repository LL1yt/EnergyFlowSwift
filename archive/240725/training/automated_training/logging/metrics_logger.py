"""
Specialized logger for metrics and performance.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MetricsLogger:
    """Специализированный логгер для метрик и производительности (оптимизированный)"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(f"metrics.{self.session_id}")
        self.metrics_file = (
            Path("logs/automated_training") / f"metrics_{self.session_id}.json"
        )

        # Только критические метрики в лог
        self.logger.setLevel(logging.WARNING)

    def log_stage_metrics(self, stage: int, metrics: Dict[str, Any]):
        """Логирует метрики стадии (только важные)"""
        # Логируем только успешные завершения или ошибки
        if metrics.get("status") == "completed":
            if metrics.get("success"):
                self.logger.info(
                    f"✅ Stage {stage}: {metrics.get('actual_time_minutes', 0):.1f}min"
                )
            else:
                self.logger.error(f"❌ Stage {stage}: FAILED")

        # JSON метрики сохраняем всегда для анализа
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            data[f"stage_{stage}"] = metrics

            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to write metrics to JSON file: {e}")

    def log_performance(
        self, operation: str, duration: float, details: Dict[str, Any] = None
    ):
        """Логирует данные о производительности (редко)"""
        if duration > 10:  # Логируем только длительные операции
            self.logger.info(
                f"PERF::{operation} took {duration:.2f}s", extra={"details": details}
            )
