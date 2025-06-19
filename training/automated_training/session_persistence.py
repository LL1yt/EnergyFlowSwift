"""
Session Persistence - Сохранение и загрузка сессий
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from datetime import datetime

from .types import StageResult, SessionSummary, StageConfig

logger = logging.getLogger(__name__)


def save_session_log(
    session_log_file: Path,
    mode: str,
    scale: float,
    max_hours: float,
    start_time: str,
    elapsed_hours: float,
    history: List[StageResult],
    summary: SessionSummary,
):
    """Сохраняет лог сессии в JSON файл."""
    session_data = {
        "mode": mode,
        "scale": scale,
        "max_total_time_hours": max_hours,
        "start_time": start_time,
        "current_time": datetime.now().isoformat(),
        "elapsed_hours": elapsed_hours,
        "training_history": [_serialize_stage_result(r) for r in history],
        "summary": asdict(summary),
    }

    try:
        with open(session_log_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save session log to {session_log_file}: {e}")


def _serialize_stage_result(result: StageResult) -> Dict[str, Any]:
    """Сериализует StageResult для JSON."""
    return asdict(result)


def load_session_history(session_file: Path) -> List[StageResult]:
    """
    Загружает историю сессии из файла.
    """
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = []
        for r in data.get("training_history", []):
            config_data = r.get("config", {})
            config = StageConfig(**config_data)
            r["config"] = config
            history.append(StageResult(**r))

        logger.info(f"Successfully loaded {len(history)} results from {session_file}")
        return history
    except Exception as e:
        logger.error(f"Failed to load session history from {session_file}: {e}")
        return []
