"""
Data Types for Automated Training
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


# This was in progressive_config.py
@dataclass
class StageConfig:
    """Конфигурация для одной стадии обучения"""

    stage: int
    dataset_limit: int
    epochs: int
    batch_size: int
    description: str


# This was in stage_runner.py
@dataclass
class StageResult:
    """Результат выполнения стадии обучения"""

    stage: int
    config: StageConfig
    success: bool
    actual_time_minutes: float
    estimated_time_minutes: float
    final_similarity: Optional[float] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# This was in session_manager.py
@dataclass
class SessionSummary:
    """Сводка по сессии обучения"""

    total_stages: int
    total_time_minutes: float
    best_similarity: Optional[float]
    avg_similarity: Optional[float]
    similarity_trend: List[float]
