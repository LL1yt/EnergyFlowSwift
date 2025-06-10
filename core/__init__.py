"""
Основные компоненты 3D Клеточной Нейронной Сети

Этот пакет содержит базовые компоненты для построения и работы
с 3D клеточной нейронной сетью.

Модули:
    cell_prototype: Прототип "умной клетки" ✅ ГОТОВ
    lattice_3d: 3D решетка клеток ✅ ГОТОВ
    signal_propagation: Распространение сигналов ✅ ГОТОВ
    🆕 embedding_processor: Центральный процессор эмбедингов (Phase 2.5) 🚀 НОВЫЙ
"""

# Импорт готовых модулей
from .cell_prototype import (
    CellPrototype,
    create_cell_from_config,
)

from .lattice_3d import (
    Lattice3D,
    LatticeConfig,
    Position3D,
    NeighborTopology,
    BoundaryCondition,
    Face,
    load_lattice_config,
    create_lattice_from_config,
    validate_lattice_config,
    Coordinates3D,
    Dimensions3D,
)

from .signal_propagation import (
    TimeManager,
    TimeConfig,
    SignalPropagator,
    PropagationConfig,
    PropagationPatterns,
    PatternAnalyzer,
    ConvergenceDetector,
    ConvergenceConfig,
)

# 🆕 PHASE 2.5: EmbeddingProcessor - центральный процессор эмбедингов (условный импорт)
try:
    from .embedding_processor import (
        EmbeddingProcessor,
        EmbeddingConfig,
        ProcessingMode,
        ProcessingMetrics,
        create_autoencoder_config,
        create_generator_config,
        create_dialogue_config,
        calculate_processing_quality,
        create_test_embedding_batch,
        validate_processor_output,
        benchmark_processing_speed,
        run_comprehensive_test,
        create_quality_report,
        export_processing_results,
    )

    EMBEDDING_PROCESSOR_AVAILABLE = True
except ImportError:
    # EmbeddingProcessor недоступен - заглушки
    EMBEDDING_PROCESSOR_AVAILABLE = False
    EmbeddingProcessor = None
    EmbeddingConfig = None
    ProcessingMode = None
    ProcessingMetrics = None
    create_autoencoder_config = None
    create_generator_config = None
    create_dialogue_config = None
    calculate_processing_quality = None
    create_test_embedding_batch = None
    validate_processor_output = None
    benchmark_processing_speed = None
    run_comprehensive_test = None
    create_quality_report = None
    export_processing_results = None

__all__ = [
    # Cell Prototype модуль
    "CellPrototype",
    "create_cell_from_config",
    # Lattice 3D модуль
    "Lattice3D",
    "LatticeConfig",
    "Position3D",
    "NeighborTopology",
    "BoundaryCondition",
    "Face",
    "load_lattice_config",
    "create_lattice_from_config",
    "validate_lattice_config",
    "Coordinates3D",
    "Dimensions3D",
    # Signal Propagation модуль
    "TimeManager",
    "TimeConfig",
    "SignalPropagator",
    "PropagationConfig",
    "PropagationPatterns",
    "PatternAnalyzer",
    "ConvergenceDetector",
    "ConvergenceConfig",
    # 🆕 PHASE 2.5: EmbeddingProcessor модуль
    "EmbeddingProcessor",
    "EmbeddingConfig",
    "ProcessingMode",
    "ProcessingMetrics",
    "create_autoencoder_config",
    "create_generator_config",
    "create_dialogue_config",
    "calculate_processing_quality",
    "create_test_embedding_batch",
    "validate_processor_output",
    "benchmark_processing_speed",
    "run_comprehensive_test",
    "create_quality_report",
    "export_processing_results",
]

__version__ = "2.5.0"  # Обновлена для Phase 2.5
