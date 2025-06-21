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
    architecture_type: str = "hybrid"  # nca | gnn | gmlp (deprecated) | hybrid

    # === 3D РЕШЕТКА ===
    # Начинаем с малой для тестов, масштабируем до цели
    lattice_dimensions: Tuple[int, int, int] = (6, 6, 6)  # отладка
    # lattice_dimensions: Tuple[int, int, int] = (16, 16, 16)  # test
    # target_dimensions: Tuple[int, int, int] = (666, 666, 333)  # научные опыты

    # === NCA НЕЙРОНЫ (биологический аналог) ===
    nca_state_size: int = 4  # состояние нейрона
    nca_hidden_dim: int = 3  # внутренняя обработка
    nca_neighbor_count: int = 26  # 3D Moore neighborhood
    nca_external_input_size: int = 1  # минимальный внешний вход
    nca_target_params: int = 69  # ~60 параметров как в биологии
    nca_activation: str = "tanh"  # стабильная для NCA

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
    hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    hybrid_gnn_weight: float = 0.9  # 90% влияние связей (было gmlp_weight)

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
    # === ТОПОЛОГИЯ СОСЕДСТВА (оптимизированная для эмерджентности) ===
    # Пропорции из GNN анализа: 10/60/30 для максимизации эмерджентности
    neighbors: int = 26  # 3D соседство
    neighbor_finding_strategy: str = "tiered"
    # neighbor_strategy_config:
    local_tier: float = 0.1  # 10% локальные (минимум для стабильности)
    functional_tier: float = 0.6  # 60% функциональные (ЯДРО эмерджентности)
    distant_tier: float = 0.3  # 30% дальние (глобальная координация)
    local_grid_cell_size: int = 8  # Размер spatial hash bins

    # === ПЛАСТИЧНОСТЬ ===
    enable_plasticity: bool = True
    plasticity_rule: str = "combined"  # STDP + BCM + competitive
    enable_competitive_learning: bool = True
    enable_metaplasticity: bool = True
    enable_clustering: bool = False  # пока отключено

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
        """Получить полную NCA конфигурацию (совместимость с Legacy)"""
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

    @property
    def total_target_params(self) -> int:
        """Общее количество целевых параметров"""
        return self.nca_target_params + self.gnn_target_params


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
