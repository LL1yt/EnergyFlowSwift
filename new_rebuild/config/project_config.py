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
    architecture_type: str = "hybrid"  # nca | gmlp | hybrid

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

    # === gMLP СВЯЗИ (межнейронные соединения) - БЕЗ BOTTLENECK ===
    gmlp_state_size: int = 32  # полноценная архитектура (было 36 в Legacy)
    gmlp_hidden_dim: int = 64  # увеличено от bottleneck 32
    gmlp_neighbor_count: int = 26  # синхронизация с NCA
    gmlp_external_input_size: int = 8  # полноценный external input
    gmlp_target_params: int = (
        80000  # ~10k связей как в биологии, но учитывая ограничения архитектуры
    )
    gmlp_activation: str = "gelu"  # современная активация
    gmlp_use_memory: bool = False  # память отключена (shared weights)

    # === HYBRID ИНТЕГРАЦИЯ ===
    hybrid_nca_weight: float = 0.1  # 10% влияние нейронов
    hybrid_gmlp_weight: float = 0.9  # 90% влияние связей

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

        # Синхронизация neighbor_count между NCA и gMLP
        if self.nca_neighbor_count != self.gmlp_neighbor_count:
            logging.warning(
                f"⚠️ NCA neighbor_count ({self.nca_neighbor_count}) != gMLP neighbor_count ({self.gmlp_neighbor_count})"
            )
            # Принудительно синхронизируем
            self.gmlp_neighbor_count = self.nca_neighbor_count

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
            logging.info(f"   gMLP params target: {self.gmlp_target_params}")

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

    def get_gmlp_config(self) -> Dict[str, Any]:
        """Получить полную gMLP конфигурацию (совместимость с Legacy)"""
        return {
            "state_size": self.gmlp_state_size,
            "neighbor_count": self.gmlp_neighbor_count,
            "hidden_dim": self.gmlp_hidden_dim,
            "external_input_size": self.gmlp_external_input_size,
            "target_params": self.gmlp_target_params,
            "activation": self.gmlp_activation,
            "dropout": 0.0,  # отключено для clean архитектуры
            "use_memory": self.gmlp_use_memory,
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
        return self.nca_target_params + self.gmlp_target_params


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
