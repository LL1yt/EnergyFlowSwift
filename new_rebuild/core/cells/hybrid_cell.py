#!/usr/bin/env python3
"""
Hybrid Cell - объединение NCA и GNN архитектур
============================================

Гибридная клетка, сочетающая:
1. NCA (Neural Cellular Automata) - 10% влияния - локальная динамика
2. GNN (Graph Neural Networks) - 90% влияния - глобальная коммуникация

Принцип: NCA обеспечивает стабильную локальную динамику,
         GNN - богатую коммуникацию между клетками
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from .base_cell import BaseCell
from .nca_cell import NCACell
from .gnn_cell import GNNCell
from ...config import get_project_config
from ...utils.logging import (
    get_logger,
    log_cell_init,
    log_cell_forward,
    log_cell_component_params,
)

logger = get_logger(__name__)


class StateAligner(nn.Module):
    """
    Выравнивание состояний между NCA и GNN архитектурами

    NCA работает с малыми состояниями (4D), GNN - с большими (32D)
    Нужен адаптер для согласования размерностей
    """

    def __init__(self, nca_state_size: int, gnn_state_size: int):
        super().__init__()

        self.nca_state_size = nca_state_size
        self.gnn_state_size = gnn_state_size

        # Проекция из малого NCA состояния в большое GNN состояние
        self.nca_to_gnn = nn.Linear(nca_state_size, gnn_state_size, bias=True)

        # Проекция из большого GNN состояния в малое NCA состояние
        self.gnn_to_nca = nn.Linear(gnn_state_size, nca_state_size, bias=True)

        # Инициализация весов для стабильности
        nn.init.xavier_uniform_(self.nca_to_gnn.weight, gain=0.1)
        nn.init.xavier_uniform_(self.gnn_to_nca.weight, gain=0.1)
        nn.init.zeros_(self.nca_to_gnn.bias)
        nn.init.zeros_(self.gnn_to_nca.bias)

    def expand_nca_to_gnn(self, nca_state: torch.Tensor) -> torch.Tensor:
        """Расширение NCA состояния для GNN обработки"""
        return self.nca_to_gnn(nca_state)

    def compress_gnn_to_nca(self, gnn_state: torch.Tensor) -> torch.Tensor:
        """Сжатие GNN состояния для NCA обработки"""
        return self.gnn_to_nca(gnn_state)


class HybridCell(BaseCell):
    """
    Гибридная клетка: NCA + GNN

    АРХИТЕКТУРА:
    1. NCA компонент (10% влияние) - локальная стабильная динамика
    2. GNN компонент (90% влияние) - глобальная коммуникация
    3. StateAligner - согласование размерностей состояний
    4. Weighted combination - взвешенное объединение результатов

    БИОЛОГИЧЕСКОЕ ОБОСНОВАНИЕ:
    - NCA = локальные биохимические процессы в нейроне
    - GNN = электрические синапсы и нейромедиаторы между нейронами
    - Баланс 10/90 = преобладание коммуникации над локальной динамикой
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        external_input_size: Optional[int] = None,
        activation: Optional[str] = None,
        nca_weight: Optional[float] = None,
        gnn_weight: Optional[float] = None,
        **kwargs,
    ):
        """
        Hybrid клетка с параметрами из ProjectConfig

        Args:
            Все параметры опциональны - берутся из ProjectConfig если не указаны
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_project_config()
        nca_config = config.get_nca_config()
        gnn_config = config.get_gnn_config()

        # Основные параметры из GNN (доминирующая архитектура)
        self.state_size = state_size or gnn_config["state_size"]
        self.neighbor_count = neighbor_count or gnn_config["neighbor_count"]
        self.external_input_size = (
            external_input_size or gnn_config["external_input_size"]
        )

        # Веса влияния компонентов
        self.nca_weight = nca_weight or config.hybrid_nca_weight
        self.gnn_weight = gnn_weight or config.hybrid_gnn_weight

        # Нормализация весов (должны в сумме давать 1.0)
        total_weight = self.nca_weight + self.gnn_weight
        self.nca_weight = self.nca_weight / total_weight
        self.gnn_weight = self.gnn_weight / total_weight

        # Размеры состояний для каждого компонента
        self.nca_state_size = nca_config["state_size"]
        self.gnn_state_size = gnn_config["state_size"]

        # === КОМПОНЕНТЫ АРХИТЕКТУРЫ ===

        # 1. NCA компонент - локальная динамика
        self.nca_cell = NCACell(
            state_size=self.nca_state_size,
            neighbor_count=self.neighbor_count,
            hidden_dim=nca_config["hidden_dim"],
            external_input_size=nca_config["external_input_size"],
            activation=nca_config["activation"],
            target_params=nca_config["target_params"],
        )

        # 2. GNN компонент - глобальная коммуникация
        self.gnn_cell = GNNCell(
            state_size=self.gnn_state_size,
            neighbor_count=self.neighbor_count,
            message_dim=gnn_config["message_dim"],
            hidden_dim=gnn_config["hidden_dim"],
            external_input_size=gnn_config["external_input_size"],
            activation=gnn_config["activation"],
            target_params=gnn_config["target_params"],
            use_attention=gnn_config["use_attention"],
        )

        # 3. Выравнивание состояний между компонентами
        self.state_aligner = StateAligner(self.nca_state_size, self.gnn_state_size)

        # 4. Целевые параметры
        self.target_params = nca_config["target_params"] + gnn_config["target_params"]

        # Логирование
        if config.debug_mode:
            self._log_parameter_count()

    def _log_parameter_count(self):
        """Логирование параметров гибридной архитектуры"""
        total_params = sum(p.numel() for p in self.parameters())
        nca_params = sum(p.numel() for p in self.nca_cell.parameters())
        gnn_params = sum(p.numel() for p in self.gnn_cell.parameters())
        aligner_params = sum(p.numel() for p in self.state_aligner.parameters())

        log_cell_init(
            cell_type="HYBRID",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            neighbor_count=self.neighbor_count,
            external_input_size=self.external_input_size,
            additional_info={
                "nca_params": nca_params,
                "gnn_params": gnn_params,
                "aligner_params": aligner_params,
                "nca_weight": self.nca_weight,
                "gnn_weight": self.gnn_weight,
                "nca_state_size": self.nca_state_size,
                "gnn_state_size": self.gnn_state_size,
            },
        )

        # Детализация по компонентам
        config = get_project_config()
        if config.debug_mode:
            component_params = {
                "nca_component": nca_params,
                "gnn_component": gnn_params,
                "state_aligner": aligner_params,
            }
            log_cell_component_params(component_params, total_params)

    def _prepare_nca_inputs(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Подготовка входов для NCA компонента

        Преобразует большие GNN состояния в малые NCA состояния
        """
        # Сжимаем собственное состояние
        nca_own_state = self.state_aligner.compress_gnn_to_nca(own_state)

        # Сжимаем состояния соседей
        batch_size, num_neighbors, _ = neighbor_states.shape
        neighbor_states_flat = neighbor_states.view(-1, self.gnn_state_size)
        nca_neighbor_states_flat = self.state_aligner.compress_gnn_to_nca(
            neighbor_states_flat
        )
        nca_neighbor_states = nca_neighbor_states_flat.view(
            batch_size, num_neighbors, self.nca_state_size
        )

        # Внешний вход для NCA (берем только первую часть)
        nca_external_input = None
        if external_input is not None:
            # NCA использует только 1 измерение из external_input
            nca_external_input = external_input[:, : self.nca_cell.external_input_size]

        return nca_neighbor_states, nca_own_state, nca_external_input

    def forward(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Hybrid forward pass: NCA + GNN

        Args:
            neighbor_states: [batch, neighbor_count, gnn_state_size] - состояния соседей
            own_state: [batch, gnn_state_size] - собственное состояние
            external_input: [batch, external_input_size] - внешний вход

        Returns:
            new_state: [batch, gnn_state_size] - новое гибридное состояние
        """
        batch_size = own_state.shape[0]

        # Логирование forward pass
        config = get_project_config()
        if config.debug_mode:
            input_shapes = {
                "neighbor_states": neighbor_states.shape,
                "own_state": own_state.shape,
            }
            if external_input is not None:
                input_shapes["external_input"] = external_input.shape

            log_cell_forward("HYBRID", input_shapes)

        # === STEP 1: GNN КОМПОНЕНТ (90% влияние) ===

        # GNN работает с полными состояниями
        gnn_new_state = self.gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            **kwargs,
        )

        # === STEP 2: NCA КОМПОНЕНТ (10% влияние) ===

        # Подготавливаем входы для NCA (сжатые состояния)
        nca_neighbor_states, nca_own_state, nca_external_input = (
            self._prepare_nca_inputs(neighbor_states, own_state, external_input)
        )

        # NCA forward pass
        nca_new_state_small = self.nca_cell(
            neighbor_states=nca_neighbor_states,
            own_state=nca_own_state,
            external_input=nca_external_input,
        )

        # Расширяем NCA результат до размера GNN
        nca_new_state = self.state_aligner.expand_nca_to_gnn(nca_new_state_small)

        # === STEP 3: WEIGHTED COMBINATION ===

        # Взвешенное объединение результатов
        hybrid_new_state = (
            self.gnn_weight * gnn_new_state + self.nca_weight * nca_new_state
        )

        return hybrid_new_state

    def get_component_states(
        self,
        neighbor_states: torch.Tensor,
        own_state: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Получить состояния отдельных компонентов для анализа

        Полезно для отладки и понимания вклада каждого компонента
        """
        # GNN компонент
        gnn_new_state = self.gnn_cell(
            neighbor_states=neighbor_states,
            own_state=own_state,
            external_input=external_input,
            **kwargs,
        )

        # NCA компонент
        nca_neighbor_states, nca_own_state, nca_external_input = (
            self._prepare_nca_inputs(neighbor_states, own_state, external_input)
        )

        nca_new_state_small = self.nca_cell(
            neighbor_states=nca_neighbor_states,
            own_state=nca_own_state,
            external_input=nca_external_input,
        )
        nca_new_state = self.state_aligner.expand_nca_to_gnn(nca_new_state_small)

        # Итоговое состояние
        hybrid_new_state = (
            self.gnn_weight * gnn_new_state + self.nca_weight * nca_new_state
        )

        return {
            "gnn_state": gnn_new_state,
            "nca_state": nca_new_state,
            "nca_state_small": nca_new_state_small,
            "hybrid_state": hybrid_new_state,
            "gnn_weight": self.gnn_weight,
            "nca_weight": self.nca_weight,
        }

    def reset_memory(self):
        """Сброс памяти в обоих компонентах"""
        self.nca_cell.reset_memory()
        self.gnn_cell.reset_memory()

    def get_info(self) -> Dict[str, Any]:
        """Расширенная информация о гибридной клетке"""
        base_info = super().get_info()

        nca_params = sum(p.numel() for p in self.nca_cell.parameters())
        gnn_params = sum(p.numel() for p in self.gnn_cell.parameters())
        aligner_params = sum(p.numel() for p in self.state_aligner.parameters())

        base_info.update(
            {
                "architecture": "hybrid",
                "nca_params": nca_params,
                "gnn_params": gnn_params,
                "aligner_params": aligner_params,
                "nca_weight": self.nca_weight,
                "gnn_weight": self.gnn_weight,
                "nca_state_size": self.nca_state_size,
                "gnn_state_size": self.gnn_state_size,
            }
        )

        return base_info
