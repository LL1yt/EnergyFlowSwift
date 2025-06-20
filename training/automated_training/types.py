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

    # === PHASE 4 INTEGRATION: Plasticity & Optimization ===
    plasticity_profile: str = "balanced"  # discovery/learning/consolidation/freeze
    clustering_enabled: bool = False  # Функциональная кластеризация
    activity_threshold: float = 0.05  # Порог активности для пластичности
    memory_optimizations: bool = False  # Mixed precision + checkpointing
    emergence_tracking: bool = False  # Emergent morphology detection
    sparse_connection_ratio: float = 0.0  # Sparse connections (0.0 = no pruning)

    # Advanced features (TIER 2-3)
    progressive_scaling: bool = False  # Enable progressive dimension scaling
    decoder_monitoring: bool = False  # Real-time decoder monitoring
    transfer_learning: bool = False  # Transfer weights between stages


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

    # === PHASE 4 INTEGRATION: Enhanced Metrics ===
    memory_peak_mb: Optional[float] = None  # Peak memory usage
    emergent_patterns_detected: Optional[int] = None  # Number of emergent structures
    clustering_quality: Optional[float] = None  # Clustering effectiveness score
    decoder_quality_score: Optional[float] = None  # Real-time decoder assessment


# This was in session_manager.py
@dataclass
class SessionSummary:
    """Сводка по сессии обучения"""

    total_stages: int
    total_time_minutes: float
    best_similarity: Optional[float]
    avg_similarity: Optional[float]
    similarity_trend: List[float]

    # === PHASE 4 INTEGRATION: Enhanced Session Analytics ===
    memory_efficiency_improvement: Optional[float] = None  # % memory reduction achieved
    emergent_behavior_evolution: List[int] = field(
        default_factory=list
    )  # Patterns per stage
    plasticity_progression: List[str] = field(default_factory=list)  # Profile evolution
