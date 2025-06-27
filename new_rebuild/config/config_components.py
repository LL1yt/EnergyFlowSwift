#!/usr/bin/env python3
"""
Компоненты конфигурации для Clean 3D Cellular Neural Network
===========================================================

Модульные компоненты конфигурации, используемые через композицию
вместо глубокой вложенности dataclass'ов.

Принцип: Простота > Сложность. Каждый компонент независим и переиспользуем.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import torch


# === ОСНОВНЫЕ КОМПОНЕНТЫ ===

@dataclass 
class LatticeSettings:
    """Базовые настройки решетки"""
    dimensions: Tuple[int, int, int] = (5, 5, 5)
    
    @property
    def total_cells(self) -> int:
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]


@dataclass
class ModelSettings:
    """Настройки модели GNN"""
    state_size: int = 32
    message_dim: int = 16
    hidden_dim: int = 32
    neighbor_count: int = 26
    external_input_size: int = 8
    target_params: int = 8000
    activation: str = "gelu"
    use_attention: bool = True
    aggregation: str = "attention"
    num_layers: int = 1


@dataclass
class TrainingSettings:
    """Настройки обучения"""
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 1000
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 50
    validation_split: float = 0.2
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100


@dataclass
class CNFSettings:
    """Настройки Continuous Normalizing Flows"""
    enabled: bool = True
    functional_connections: bool = True
    distant_connections: bool = True
    integration_steps: int = 3
    adaptive_step_size: bool = True
    target_params_per_connection: int = 3000
    batch_processing_mode: str = "ADAPTIVE_BATCH"
    max_batch_size: int = 1024
    adaptive_method: str = "LIPSCHITZ_BASED"


@dataclass
class EulerSettings:
    """Настройки Euler Solver"""
    adaptive_method: str = "LIPSCHITZ_BASED"
    base_dt: float = 0.1
    min_dt: float = 0.001
    max_dt: float = 0.5
    lipschitz_safety_factor: float = 0.8
    stability_threshold: float = 10.0
    memory_efficient: bool = True
    max_batch_size: int = 1000
    error_tolerance: float = 1e-3
    enable_profiling: bool = True


@dataclass
class CacheSettings:
    """Настройки кэширования"""
    enabled: bool = True
    enable_performance_monitoring: bool = True
    enable_detailed_stats: bool = False
    auto_enable_threshold: int = 3000
    auto_disable_threshold: int = 1000
    small_lattice_fallback: bool = True
    use_gpu_acceleration: bool = True
    gpu_batch_size: int = 10000
    prefer_gpu_for_large_lattices: bool = True
    gpu_memory_fraction: float = 0.8


@dataclass
class SpatialSettings:
    """Настройки пространственной оптимизации"""
    chunk_size: int = 64
    chunk_overlap: int = 8
    max_chunks_in_memory: int = 4
    memory_pool_size_gb: float = 12.0
    garbage_collect_frequency: int = 100
    prefetch_chunks: bool = True
    levels: int = 3
    min_cells_per_node: int = 1000
    max_search_radius: float = 50.0
    num_worker_threads: int = 4
    batch_size_per_thread: int = 10000
    enable_async_processing: bool = True
    enable_profiling: bool = True
    log_memory_usage: bool = True


@dataclass
class VectorizedSettings:
    """Настройки векторизации"""
    enabled: bool = True
    force_vectorized: bool = True
    chunk_size: int = 1000
    parallel_processing: bool = True
    use_compiled_kernels: bool = True
    memory_efficient_mode: bool = True
    batch_norm_enabled: bool = True
    dropout_rate: float = 0.1


@dataclass
class DeviceSettings:
    """Настройки устройства"""
    prefer_cuda: bool = True
    debug_mode: bool = False
    fallback_cpu: bool = True
    memory_fraction: float = 0.9
    allow_tf32: bool = True
    deterministic: bool = False


@dataclass
class LoggingSettings:
    """Настройки логирования"""
    level: str = "INFO"
    debug_mode: bool = False
    log_to_file: bool = True
    log_file: str = "logs/cnf_debug.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_profiling: bool = True
    performance_tracking: bool = True


@dataclass
class MemorySettings:
    """Настройки управления памятью"""
    optimal_batch_size: int = 1000
    memory_per_cell_base: int = 64
    memory_overhead_factor: float = 1.3
    max_history: int = 1000
    min_available_memory_mb: float = 500.0
    cuda_fallback_available_mb: float = 8000.0
    cpu_fallback_available_mb: float = 16000.0
    safe_memory_buffer: float = 0.8
    max_concurrent_chunks: int = 4
    max_chunks_in_memory: int = 4
    min_chunk_size: int = 8
    chunk_size_fallback_div: int = 8


# === СПЕЦИАЛИЗИРОВАННЫЕ КОМПОНЕНТЫ ===

@dataclass
class ExperimentSettings:
    """Настройки для экспериментов"""
    experiment_name: str = "default"
    save_results: bool = True
    results_dir: str = "results"
    random_seed: int = 42
    reproducible: bool = True
    track_metrics: bool = True
    save_checkpoints: bool = True
    visualize_results: bool = False


@dataclass
class PerformanceSettings:
    """Настройки производительности"""
    enable_jit: bool = False
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    dataloader_workers: int = 4
    pin_memory: bool = True
    non_blocking: bool = True
    benchmark_mode: bool = False
    profiling_enabled: bool = False


@dataclass
class ValidationSettings:
    """Настройки валидации"""
    validate_config: bool = True
    strict_validation: bool = False
    warn_on_deprecated: bool = True
    auto_fix_conflicts: bool = True
    check_memory_requirements: bool = True
    estimate_compute_time: bool = False


# === ФУНКЦИИ КОМПОЗИЦИИ ===

def create_basic_config() -> Dict[str, Any]:
    """Создает базовую конфигурацию с минимальными настройками"""
    return {
        'lattice': LatticeSettings(),
        'model': ModelSettings(),
        'training': TrainingSettings(),
        'device': DeviceSettings(),
        'logging': LoggingSettings(),
    }


def create_research_config() -> Dict[str, Any]:
    """Создает конфигурацию для исследований"""
    config = create_basic_config()
    config.update({
        'cnf': CNFSettings(),
        'euler': EulerSettings(),
        'cache': CacheSettings(),
        'spatial': SpatialSettings(),
        'vectorized': VectorizedSettings(),
        'experiment': ExperimentSettings(),
        'performance': PerformanceSettings(),
    })
    return config


def create_production_config() -> Dict[str, Any]:
    """Создает конфигурацию для production"""
    config = create_research_config()
    
    # Отключаем debug режимы
    config['device'].debug_mode = False
    config['logging'].debug_mode = False
    config['logging'].level = "WARNING"
    
    # Оптимизируем для производительности
    config['performance'].enable_jit = True
    config['performance'].use_mixed_precision = True
    config['performance'].benchmark_mode = True
    
    # Ограничиваем ресурсы
    config['cache'].enable_detailed_stats = False
    config['logging'].performance_tracking = False
    
    return config


def validate_config_components(config: Dict[str, Any]) -> bool:
    """Валидирует компоненты конфигурации на совместимость"""
    try:
        # Проверяем базовые требования
        if 'lattice' not in config or 'model' not in config:
            return False
            
        lattice = config['lattice']
        model = config['model']
        
        # Проверяем размеры
        if lattice.total_cells > 10000 and not config.get('cache', CacheSettings()).enabled:
            print("WARNING: Large lattice without cache may be slow")
            
        # Проверяем совместимость модели
        if model.state_size < 8:
            print("WARNING: Very small state_size may limit model capacity")
            
        return True
        
    except Exception as e:
        print(f"Config validation error: {e}")
        return False


def merge_config_updates(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Безопасно объединяет обновления конфигурации"""
    result = base_config.copy()
    
    for key, value in updates.items():
        if key in result:
            if hasattr(result[key], '__dict__') and hasattr(value, '__dict__'):
                # Обновляем dataclass
                for attr, new_val in value.__dict__.items():
                    if hasattr(result[key], attr):
                        setattr(result[key], attr, new_val)
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result
