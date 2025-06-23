#!/usr/bin/env python3
"""
Централизованная конфигурация для Clean 3D Cellular Neural Network
=================================================================

Единственный источник истины для всей архитектуры.
Основано на принципах из utils/centralized_config.py, но упрощено для clean архитектуры.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import logging
import torch

# Для spatial optimization - используем Tuple вместо Coordinates3D чтобы избежать circular import
from typing import Tuple


@dataclass
class ChunkInfo:
    """Информация о chunk'е решетки для spatial optimization"""

    chunk_id: int
    start_coords: Tuple[int, int, int]  # (x, y, z)
    end_coords: Tuple[int, int, int]  # (x, y, z)
    cell_indices: List[int]
    neighbor_chunks: List[int]  # ID соседних chunk'ов
    memory_size_mb: float
    processing_time_ms: float = 0.0


@dataclass
class ProjectConfig:
    """
    Единственный источник истины для всей архитектуры

    Принципы:
    1. Все параметры в одном месте
    2. Биологическая корректность
    3. Эмерджентность через простоту
    4. Centralised logging
    """

    # === АРХИТЕКТУРА ===
    architecture_type: str = "moe"  # moe | gnn (deprecated: nca, gmlp, hybrid)

    # === 3D РЕШЕТКА ===
    # Начинаем с малой для тестов, масштабируем до цели
    lattice_dimensions: Tuple[int, int, int] = (27, 27, 27)  # MoE тестирование
    # lattice_dimensions: Tuple[int, int, int] = (6, 6, 6)  # отладка
    # lattice_dimensions: Tuple[int, int, int] = (16, 16, 16)  # test
    # target_dimensions: Tuple[int, int, int] = (666, 666, 333)  # научные опыты

    # === GNN СВЯЗИ (замена gMLP) - оптимизированная коммуникация ===
    gnn_state_size: int = 32  # размер состояния клетки
    gnn_message_dim: int = 16  # размер сообщений между клетками
    gnn_hidden_dim: int = 32  # скрытый слой для обработки
    gnn_neighbor_count: int = 26  # синхронизация с NCA
    gnn_external_input_size: int = 8  # внешний вход
    gnn_target_params: int = 8000  # намного меньше чем gMLP (113k → 8k)
    gnn_activation: str = "gelu"  # современная активация
    gnn_use_attention: bool = True  # attention mechanism для селективной агрегации
    gnn_aggregation: str = "attention"  # тип агрегации сообщений
    gnn_num_layers: int = 1  # количество слоев GNN (начинаем с 1)

    # === ОБУЧЕНИЕ ===
    learning_rate: float = 0.001
    batch_size: int = 4
    device: str = "auto"  # auto-detect cuda/cpu

    # === ЭМБЕДДИНГИ ===
    embedding_dim: int = 768  # from DistilBERT
    phrase_based_training: bool = True  # целые фразы, не токены

    # === БИОЛОГИЧЕСКИЕ ПРИНЦИПЫ ===
    shared_weights: bool = True  # клетки как нейроны с общими весами
    tissue_simulation: bool = True  # решетка как нервная ткань
    receptor_coverage: float = 1.0  # рецепторная стратегия (100% покрытия)
    signal_propagation: bool = True  # сигналы как нервные импульсы
    # === ТОПОЛОГИЯ СОСЕДСТВА (динамическая, зависит от размера решетки) ===
    # Обновленные пропорции: 10/55/35 для увеличения CNF влияния
    neighbors: int = 26  # Базовое значение (legacy совместимость)
    max_neighbors: int = 20000  # Биологический максимум (10k связей)
    neighbor_finding_strategy: str = "tiered"
    dynamic_neighbor_count: bool = True  # Автоматический расчет на основе решетки

    # === ADAPTIVE RADIUS (пространственный поиск соседей) ===
    adaptive_radius_enabled: bool = True  # Включить адаптивный радиус
    adaptive_radius_ratio: float = 0.3  # 30% от максимального размера решетки
    adaptive_radius_max: float = 500.0  # Максимальный радиус (биологический лимит)
    adaptive_radius_min: float = 5  # Минимальный радиус (локальные соседи)

    # neighbor_strategy_config:
    local_tier: float = 0.1  # 10% локальные (минимум для стабильности)
    functional_tier: float = 0.55  # 55% функциональные (уменьшено для CNF)
    distant_tier: float = 0.35  # 35% дальние (увеличено для CNF)
    local_grid_cell_size: int = 8  # Размер spatial hash bins

    # === ПЛАСТИЧНОСТЬ ===
    enable_plasticity: bool = True
    plasticity_rule: str = "combined"  # STDP + BCM + competitive
    enable_competitive_learning: bool = True
    enable_metaplasticity: bool = True
    enable_clustering: bool = False  # пока отключено

    # === MoE ARCHITECTURE (ОСНОВНАЯ АРХИТЕКТУРА) ===
    enable_moe: bool = True  # Mixture of Experts - основная архитектура

    # === GATING NETWORK (заменяет NCA нейрон) ===
    gating_state_size: int = 32  # размер состояния для gating (= gnn_state_size)
    gating_params: int = 808  # точно по спецификации
    gating_num_experts: int = 3  # количество экспертов
    gating_activation: str = "gelu"  # активация для gating network
    gating_hidden_dim: int = 11  # скрытый слой для достижения 808 параметров

    # === LOCAL EXPERT PARAMETERS ===
    local_expert_alpha: float = 0.1  # adaptive weight mixing parameter
    local_expert_beta: float = 0.9  # residual connection weight

    # === ЭКСПЕРТЫ И ИХ ПАРАМЕТРЫ ===
    # Local Expert (SimpleLinear) - рефлексы
    local_expert_params: int = 2059  # точно по спецификации
    local_expert_type: str = "linear"  # тип эксперта

    # Functional Expert (GNN) - основная обработка
    functional_expert_params: int = 8233  # верхняя граница
    functional_expert_type: str = "gnn"  # только GNN (без CNF в functional)

    # Distant Expert (CNF) - долгосрочная память
    distant_expert_params: int = 4000  # верхняя граница для LightweightCNF
    distant_expert_type: str = "cnf"  # только CNF

    # === РАСПРЕДЕЛЕНИЕ СВЯЗЕЙ ПО ЭКСПЕРТАМ ===
    local_connections_ratio: float = 0.10  # 10% связей - Local Expert
    functional_connections_ratio: float = 0.55  # 55% связей - Functional Expert
    distant_connections_ratio: float = 0.35  # 35% связей - Distant Expert

    # Пороги для классификации связей
    local_distance_threshold: float = 1.5  # расстояние для local connections
    functional_similarity_threshold: float = 0.3  # порог функциональной схожести
    distant_distance_threshold: float = (
        local_distance_threshold * 3.0
    )  # расстояние для distant connections

    # === DEPRECATED: HYBRID GNN+CNF EXPERT ===
    # hybrid_gnn_cnf_expert_params: int = 12233  # DEPRECATED - слишком сложно
    # cnf_expert_params: int = 3000  # DEPRECATED - используйте distant_expert_params

    # === PHASE 4: LIGHTWEIGHT CNF ===
    enable_cnf: bool = True  # Включаем CNF для MoE
    cnf_functional_connections: bool = True  # CNF для functional (55%)
    cnf_distant_connections: bool = True  # CNF для distant (35%)
    cnf_integration_steps: int = 3  # 3-step Euler (вместо 10 RK4)
    cnf_adaptive_step_size: bool = True  # Адаптивный шаг интеграции
    cnf_target_params_per_connection: int = 3000  # Базовое значение для CNF

    # === SPATIAL OPTIMIZATION (перенесено из spatial_optimization/config.py) ===
    # Chunking parameters
    spatial_chunk_size: int = 64  # Размер chunk'а (64×64×64 = 262k клеток)
    spatial_chunk_overlap: int = 8  # Перекрытие между chunk'ами для соседства
    spatial_max_chunks_in_memory: int = 4  # Максимум chunk'ов в GPU памяти одновременно

    # Memory management
    spatial_memory_pool_size_gb: float = 12.0  # Размер memory pool (75% от 16GB)
    spatial_garbage_collect_frequency: int = 100  # GC каждые N операций
    spatial_prefetch_chunks: bool = True  # Предзагрузка следующих chunk'ов

    # Hierarchical indexing
    spatial_levels: int = 3  # Количество уровней пространственного индекса
    spatial_min_cells_per_node: int = 1000  # Минимум клеток в узле индекса
    spatial_max_search_radius: float = 50.0  # Максимальный радиус поиска соседей

    # Parallel processing
    spatial_num_worker_threads: int = 4  # Количество worker потоков
    spatial_batch_size_per_thread: int = 10000  # Размер batch'а на поток
    spatial_enable_async_processing: bool = True  # Асинхронная обработка

    # Performance monitoring
    spatial_enable_profiling: bool = True  # Профилирование производительности
    spatial_log_memory_usage: bool = True  # Логирование использования памяти

    # === ОПТИМИЗАЦИЯ ПАМЯТИ ===
    memory_efficient: bool = True
    use_checkpointing: bool = True
    mixed_precision: bool = True

    # === ЛОГИРОВАНИЕ ===
    debug_mode: bool = True  # максимум логов для отладки
    enable_logging: bool = True
    log_level: str = "INFO"

    # === ИНИЦИАЛИЗАЦИЯ ===
    seed: int = 42
    initialization_method: str = "xavier"

    def __post_init__(self):
        """Валидация и автоматическая настройка после инициализации"""
        # Автоопределение устройства
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DEPRECATED: NCA синхронизация больше не нужна в MoE архитектуре
        # В MoE архитектуре NCA заменен на GatingNetwork

        # Подсчет общего количества клеток
        self.total_cells = (
            self.lattice_dimensions[0]
            * self.lattice_dimensions[1]
            * self.lattice_dimensions[2]
        )

        self.max_neighbors = self.calculate_dynamic_neighbors()

        if self.debug_mode:
            logging.info(f"🔧 ProjectConfig initialized:")
            logging.info(f"   Architecture: {self.architecture_type}")
            logging.info(
                f"   Lattice: {self.lattice_dimensions} = {self.total_cells} cells"
            )
            logging.info(f"   Device: {self.device}")
            logging.info(f"   MoE Gating params target: {self.gating_params}")
            logging.info(
                f"   GNN Expert params target: {self.functional_expert_params}"
            )
            logging.info(f"   CNF Expert params target: {self.distant_expert_params}")

    # === МЕТОДЫ ДОСТУПА (совместимость с Legacy) ===
    def get_nca_config(self) -> Dict[str, Any]:
        """DEPRECATED: NCA заменен на GatingNetwork в MoE"""
        # Возвращаем конфигурацию GatingNetwork для совместимости
        return self.get_gating_config()

    def get_gnn_config(self) -> Dict[str, Any]:
        """Получить полную GNN конфигурацию (замена gMLP)"""
        return {
            "state_size": self.gnn_state_size,
            "neighbor_count": self.gnn_neighbor_count,
            "message_dim": self.gnn_message_dim,
            "hidden_dim": self.gnn_hidden_dim,
            "external_input_size": self.gnn_external_input_size,
            "target_params": self.gnn_target_params,
            "activation": self.gnn_activation,
            "use_attention": self.gnn_use_attention,
            "aggregation": self.gnn_aggregation,
            "num_layers": self.gnn_num_layers,
            "dropout": 0.0,  # отключено для clean архитектуры
        }

    def get_gmlp_config(self) -> Dict[str, Any]:
        """DEPRECATED: Получить полную gMLP конфигурацию (Legacy совместимость)"""
        # Возвращаем GNN конфигурацию для обратной совместимости
        gnn_config = self.get_gnn_config()
        # Маппинг для совместимости
        return {
            "state_size": gnn_config["state_size"],
            "neighbor_count": gnn_config["neighbor_count"],
            "hidden_dim": gnn_config["hidden_dim"],
            "external_input_size": gnn_config["external_input_size"],
            "target_params": gnn_config["target_params"],
            "activation": gnn_config["activation"],
            "dropout": 0.0,  # отключено для clean архитектуры
            "use_memory": False,  # память отключена (shared weights)
            # НЕ ИСПОЛЬЗУЕМ bottleneck для полноценной архитектуры
            "bottleneck_dim": None,
        }

    def get_lattice_config(self) -> Dict[str, Any]:
        """Получить конфигурацию 3D решетки"""
        return {
            "dimensions": self.lattice_dimensions,
            "total_cells": self.total_cells,
            # "neighbors": self.effective_neighbors,  # используем динамические соседи
            "device": self.device,
            "enable_logging": self.enable_logging,
            "seed": self.seed,
            "initialization_method": self.initialization_method,
        }

    def get_cnf_config(self) -> Dict[str, Any]:
        """Получить конфигурацию Lightweight CNF"""
        return {
            "enable_cnf": self.enable_cnf,
            "functional_connections": self.cnf_functional_connections,
            "distant_connections": self.cnf_distant_connections,
            "integration_steps": self.cnf_integration_steps,
            "adaptive_step_size": self.cnf_adaptive_step_size,
            "target_params_per_connection": self.cnf_target_params_per_connection,
        }

    def get_neighbor_strategy_config(self) -> Dict[str, Any]:
        """Получить конфигурацию стратегии соседства (для CNF классификации)"""
        return {
            "local_tier": self.local_tier,
            "functional_tier": self.functional_tier,
            "distant_tier": self.distant_tier,
            "local_grid_cell_size": self.local_grid_cell_size,
            # Добавляем adaptive radius конфигурацию
            "adaptive_radius_enabled": self.adaptive_radius_enabled,
            "adaptive_radius_ratio": self.adaptive_radius_ratio,
            "adaptive_radius_max": self.adaptive_radius_max,
            "adaptive_radius_min": self.adaptive_radius_min,
            "adaptive_radius": self.calculate_adaptive_radius(),  # Вычисленное значение
        }

    def get_gating_config(self) -> Dict[str, Any]:
        """Получить конфигурацию GatingNetwork (замена NCA)"""
        return {
            "state_size": self.gating_state_size,
            "num_experts": self.gating_num_experts,
            "target_params": self.gating_params,
            "activation": self.gating_activation,
            "hidden_dim": self.gating_hidden_dim,
        }

    def get_local_expert_config(self) -> Dict[str, Any]:
        """Получить конфигурацию Local Expert"""
        return {
            "type": self.local_expert_type,
            "params": self.local_expert_params,
            "alpha": self.local_expert_alpha,
            "beta": self.local_expert_beta,
        }

    def get_moe_config(self) -> Dict[str, Any]:
        """Получить полную конфигурацию MoE архитектуры"""
        return {
            "enable_moe": self.enable_moe,
            "gating_config": self.get_gating_config(),
            "experts": {
                "local": {
                    "type": self.local_expert_type,
                    "params": self.local_expert_params,
                    "ratio": self.local_connections_ratio,
                },
                "functional": {
                    "type": self.functional_expert_type,
                    "params": self.functional_expert_params,
                    "ratio": self.functional_connections_ratio,
                },
                "distant": {
                    "type": self.distant_expert_type,
                    "params": self.distant_expert_params,
                    "ratio": self.distant_connections_ratio,
                },
            },
            "thresholds": {
                "local_distance": self.local_distance_threshold,
                "functional_similarity": self.functional_similarity_threshold,
            },
        }

    def get_spatial_optim_config(self) -> Dict[str, Any]:
        """Получить конфигурацию spatial optimization"""
        # Адаптивные настройки на основе размера решетки
        total_cells = self.total_cells
        max_dim = max(self.lattice_dimensions)

        return {
            # Chunking адаптируется под размер решетки
            "chunk_size": (
                min(self.spatial_chunk_size, max_dim // 2)
                if total_cells > 100_000
                else 32
            ),
            "chunk_overlap": self.spatial_chunk_overlap if total_cells > 50_000 else 4,
            "max_chunks_in_memory": (
                self.spatial_max_chunks_in_memory if total_cells > 100_000 else 2
            ),
            # Memory management на основе конфигурации
            "memory_pool_size_gb": (
                self.spatial_memory_pool_size_gb if self.device == "cuda" else 4.0
            ),
            "garbage_collect_frequency": self.spatial_garbage_collect_frequency,
            "prefetch_chunks": self.spatial_prefetch_chunks,
            # Hierarchical indexing
            "spatial_levels": self.spatial_levels,
            "min_cells_per_node": self.spatial_min_cells_per_node,
            "max_search_radius": self.calculate_adaptive_radius(),  # Используем adaptive radius
            # Parallel processing
            "num_worker_threads": (
                self.spatial_num_worker_threads if self.device == "cuda" else 2
            ),
            "batch_size_per_thread": self.spatial_batch_size_per_thread,
            "enable_async_processing": self.spatial_enable_async_processing
            and self.device == "cuda",
            # Performance monitoring
            "enable_profiling": self.spatial_enable_profiling,
            "log_memory_usage": self.spatial_log_memory_usage,
        }

    @property
    def total_target_params(self) -> int:
        """Общее количество целевых параметров MoE"""
        return (
            self.gating_params
            + self.local_expert_params
            + self.functional_expert_params
            + self.distant_expert_params
        )

    def calculate_dynamic_neighbors(self) -> int:
        """
        Динамический расчет количества соседей на основе размера решетки

        Принцип: стремимся к биологическим ~10000 связей на нейрон
        Биологическая аналогия: в плотной ткани больше связей
        """
        if not self.dynamic_neighbor_count:
            return self.neighbors  # Возвращаем фиксированное значение

        # Расчет на основе размера решетки с биологическим обоснованием
        total_cells = self.total_cells

        if total_cells <= 216:  # 6x6x6
            return 26  # Базовое фиксированное соседство (минимум)
        elif total_cells <= 4096:  # 16x16x16
            return 500  # Средний размер для тестирования
        elif total_cells <= 19683:  # 27x27x27
            return 5000  # Приближение к биологическим 10k для MoE
        elif total_cells <= 262144:  # 64x64x64
            return 10000  # Большие решетки
        else:  # Крупные решетки (200x200x1000)
            return min(self.max_neighbors, total_cells)  # Биологический максимум

    """DEPRECATED
    @property
    def effective_neighbors(self) -> int:
        # DEPRECATED: Используйте calculate_adaptive_radius() для биологически правдоподобного поиска по радиусу
        return self.calculate_dynamic_neighbors()
    """

    def calculate_adaptive_radius(self) -> float:
        """
        Вычисляет адаптивный радиус поиска соседей

        Returns:
            Радиус поиска на основе размера решетки и настроек
        """
        if not self.adaptive_radius_enabled:
            return self.adaptive_radius_max  # Возвращаем фиксированный радиус

        # Получаем максимальный размер решетки
        max_dimension = max(self.lattice_dimensions)

        # Вычисляем адаптивный радиус как процент от размера
        adaptive_radius = max_dimension * self.adaptive_radius_ratio

        # Ограничиваем минимальными и максимальными значениями
        adaptive_radius = max(self.adaptive_radius_min, adaptive_radius)
        adaptive_radius = min(self.adaptive_radius_max, adaptive_radius)

        return adaptive_radius


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ SPATIAL OPTIMIZATION ===
def create_spatial_config_for_lattice(
    dimensions: Tuple[int, int, int],
) -> Dict[str, Any]:
    """
    Создает оптимальную конфигурацию для заданного размера решетки

    РЕКОМЕНДУЕТСЯ: Используйте get_project_config().get_spatial_optim_config() для
    интеграции с централизованным ProjectConfig

    Args:
        dimensions: размеры решетки (x, y, z)

    Returns:
        dict с оптимальными настройками spatial optimization
    """

    # Приоритет: если размеры совпадают с ProjectConfig, используем централизованную конфигурацию
    try:
        project_config = get_project_config()
        current_dims = project_config.lattice_dimensions

        if (
            current_dims[0] == dimensions[0]
            and current_dims[1] == dimensions[1]
            and current_dims[2] == dimensions[2]
        ):
            return project_config.get_spatial_optim_config()
    except Exception:
        # Fallback на legacy логику если ProjectConfig недоступен
        pass

    # Legacy логика для обратной совместимости
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    max_dim = max(dimensions)

    # Адаптивная настройка параметров
    if total_cells < 50_000:  # Малые решетки (< 50k клеток)
        return {
            "chunk_size": 32,
            "chunk_overlap": 4,
            "max_chunks_in_memory": 2,
            "memory_pool_size_gb": 2.0,
            "num_worker_threads": 2,
            "batch_size_per_thread": 5000,
            "max_search_radius": min(20.0, max_dim * 0.5),
            "enable_async_processing": False,
        }
    elif total_cells < 500_000:  # Средние решетки (50k - 500k клеток)
        return {
            "chunk_size": 48,
            "chunk_overlap": 6,
            "max_chunks_in_memory": 3,
            "memory_pool_size_gb": 4.0,
            "num_worker_threads": 4,
            "batch_size_per_thread": 10000,
            "max_search_radius": min(30.0, max_dim * 0.3),
            "enable_async_processing": True,
        }
    else:  # Большие решетки (> 500k клеток)
        return {
            "chunk_size": 64,
            "chunk_overlap": 8,
            "max_chunks_in_memory": 4,
            "memory_pool_size_gb": 12.0,
            "num_worker_threads": 6,
            "batch_size_per_thread": 15000,
            "max_search_radius": min(50.0, max_dim * 0.2),
            "enable_async_processing": True,
        }


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===
_global_config: Optional[ProjectConfig] = None


def get_project_config() -> ProjectConfig:
    """
    Получить глобальный экземпляр конфигурации

    Singleton pattern для единственного источника истины
    """
    global _global_config
    if _global_config is None:
        _global_config = ProjectConfig()
    return _global_config


def set_project_config(config: ProjectConfig):
    """Установить новую глобальную конфигурацию"""
    global _global_config
    _global_config = config
