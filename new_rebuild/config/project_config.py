#!/usr/bin/env python3
"""
Централизованная конфигурация для Clean 3D Cellular Neural Network
=================================================================

Единственный источник истины для всей архитектуры.
Основано на принципах из utils/centralized_config.py, но упрощено для clean архитектуры.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import logging
import torch


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

    # === DEPRECATED: NCA НЕЙРОНЫ (заменены на MoE Gating) ===
    # nca_state_size: int = 4  # DEPRECATED - используйте gnn_state_size
    # nca_hidden_dim: int = 3  # DEPRECATED
    # nca_neighbor_count: int = 26  # DEPRECATED - используйте effective_neighbors
    # nca_external_input_size: int = 1  # DEPRECATED
    # nca_target_params: int = 69  # DEPRECATED
    # nca_activation: str = "tanh"  # DEPRECATED

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

    # === HYBRID ИНТЕГРАЦИЯ ===
    # deprecated hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    # deprecated hybrid_gnn_weight: float = 0.9  # 90% влияние связей (было gmlp_weight)

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
    neighbors: int = 26  # Базовое значение (3D Moore neighborhood)
    max_neighbors: int = 10000  # Биологический максимум (10k связей)
    neighbor_finding_strategy: str = "tiered"
    dynamic_neighbor_count: bool = True  # Автоматический расчет на основе решетки
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

        # Синхронизация neighbor_count между NCA и GNN
        if self.nca_neighbor_count != self.gnn_neighbor_count:
            logging.warning(
                f"⚠️ NCA neighbor_count ({self.nca_neighbor_count}) != GNN neighbor_count ({self.gnn_neighbor_count})"
            )
            # Принудительно синхронизируем
            self.gnn_neighbor_count = self.nca_neighbor_count

        # Подсчет общего количества клеток
        self.total_cells = (
            self.lattice_dimensions[0]
            * self.lattice_dimensions[1]
            * self.lattice_dimensions[2]
        )

        if self.debug_mode:
            logging.info(f"🔧 ProjectConfig initialized:")
            logging.info(f"   Architecture: {self.architecture_type}")
            logging.info(
                f"   Lattice: {self.lattice_dimensions} = {self.total_cells} cells"
            )
            logging.info(f"   Device: {self.device}")
            logging.info(f"   NCA params target: {self.nca_target_params}")
            logging.info(f"   GNN params target: {self.gnn_target_params}")

    # === МЕТОДЫ ДОСТУПА (совместимость с Legacy) ===
    def get_nca_config(self) -> Dict[str, Any]:
        """Получить полную NCA конфигурацию"""
        return {
            "state_size": self.nca_state_size,
            "hidden_dim": self.nca_hidden_dim,
            "external_input_size": self.nca_external_input_size,
            "neighbor_count": self.nca_neighbor_count,
            "target_params": self.nca_target_params,
            "activation": self.nca_activation,
            "dropout": 0.0,
            "use_memory": False,
            "enable_lattice_scaling": False,
        }

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
            "neighbors": self.nca_neighbor_count,  # синхронизировано
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
        }

    @property
    def total_target_params(self) -> int:
        """Общее количество целевых параметров"""
        return self.nca_target_params + self.gnn_target_params

    @property
    def neighbor_strategy_config(self) -> Dict[str, Any]:
        """Свойство для совместимости с кодом, который ожидает neighbor_strategy_config"""
        return self.get_neighbor_strategy_config()

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
            return 26  # Стандартное 3D Moore neighborhood (минимум)
        elif total_cells <= 4096:  # 16x16x16
            return 500  # Средний размер для тестирования
        elif total_cells <= 19683:  # 27x27x27
            return 2000  # Приближение к биологическим 10k для MoE
        elif total_cells <= 262144:  # 64x64x64
            return 5000  # Большие решетки
        else:  # Крупные решетки (666x666x333)
            return min(self.max_neighbors, 10000)  # Биологический максимум

    @property
    def effective_neighbors(self) -> int:
        """Эффективное количество соседей (динамическое или фиксированное)"""
        return self.calculate_dynamic_neighbors()


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
