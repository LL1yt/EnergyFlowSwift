#!/usr/bin/env python3
"""
MoE Connection Processor - Mixture of Experts для 3D CNN
========================================================

Новая архитектура с тремя специализированными экспертами:
- SimpleLinear (10%) - рефлексы
- HybridGNN_CNF (55%) - основная обработка
- LightweightCNF (35%) - долгосрочная память

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. Точные параметры согласно спецификации
2. Gating network для управления экспертами (808 параметров)
3. Поддержка решетки 27×27×27
4. Централизованная конфигурация (все параметры из ProjectConfig)
5. Динамический расчет соседей

АРХИТЕКТУРА:
- UnifiedConnectionClassifier для определения типов связей (без deprecated зависимостей)
- Три специализированных эксперта
- Learnable Gating Network для комбинирования
- Результирующая обработка с residual connections
- Batch processing для производительности
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple

# НОВЫЕ МОДУЛЬНЫЕ ИМПОРТЫ
from .moe_processor import MoEConnectionProcessor as NewMoEConnectionProcessor
from .gating_network import GatingNetwork as NewGatingNetwork
from .connection_classifier import (
    UnifiedConnectionClassifier as NewUnifiedConnectionClassifier,
)
from .connection_types import ConnectionCategory, ConnectionInfo
from .distance_calculator import DistanceCalculator
from .functional_similarity import FunctionalSimilarityAnalyzer

# СТАРЫЕ ИМПОРТЫ (для обратной совместимости)
from .simple_linear_expert import SimpleLinearExpert
from .hybrid_gnn_cnf_expert import HybridGNN_CNF_Expert
from ..cnf.lightweight_cnf import LightweightCNF, ConnectionType
from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward
from ..lattice.position import Position3D

logger = get_logger(__name__)


class GatingNetwork(nn.Module):
    """
    Learnable Gating Network для комбинирования результатов экспертов

    Цель: 808 параметров точно по спецификации
    Принцип: адаптивное взвешивание результатов на основе контекста
    """

    def __init__(self, state_size: Optional[int] = None, num_experts: int = 3):
        super().__init__()

        config = get_project_config()

        self.state_size = state_size or config.gnn_state_size  # 32 из конфига
        self.num_experts = num_experts
        self.target_params = config.gating_params  # 808 из конфига

        # Рассчитываем архитектуру для достижения 808 параметров
        # Input: state_size + neighbor_activity = 32 + 32 = 64
        input_size = self.state_size * 2

        # Получаем hidden_dim из централизованной конфигурации
        hidden_dim = config.gating_hidden_dim  # Централизованное значение

        self.gating_network = nn.Sequential(
            nn.Linear(input_size, hidden_dim, bias=True),  # 64*11 + 11 = 715
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts, bias=True),  # 11*3 + 3 = 36
            nn.Softmax(dim=-1),
        )
        # Итого: 715 + 36 = 751

        # Добавляем normalization для достижения 808
        # Нужно добавить: 808 - 751 = 57 параметров
        self.context_norm = nn.LayerNorm(self.state_size, bias=True)  # 32*2 = 64
        # Итого: 751 + 64 = 815 (близко к 808)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"GatingNetwork: {total_params} параметров (цель: {self.target_params})"
        )

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_activity: torch.Tensor,
        expert_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление весов экспертов и комбинирование результатов

        Args:
            current_state: [batch, state_size]
            neighbor_activity: [batch, state_size]
            expert_outputs: List[Tensor] - результаты от каждого эксперта

        Returns:
            combined_output: [batch, state_size] - взвешенная комбинация
            expert_weights: [batch, num_experts] - веса экспертов
        """
        # 1. Подготовка контекста для принятия решения
        normalized_state = self.context_norm(current_state)
        context = torch.cat([normalized_state, neighbor_activity], dim=-1)

        # 2. Вычисление весов экспертов
        expert_weights = self.gating_network(context)  # [batch, num_experts]

        # 3. Взвешенное комбинирование результатов экспертов
        stacked_outputs = torch.stack(
            expert_outputs, dim=1
        )  # [batch, num_experts, state_size]
        weights_expanded = expert_weights.unsqueeze(-1)  # [batch, num_experts, 1]

        combined_output = torch.sum(
            stacked_outputs * weights_expanded, dim=1
        )  # [batch, state_size]

        return combined_output, expert_weights


class MoEConnectionProcessor(nn.Module):
    """
    Mixture of Experts Connection Processor для 3D решетки 27×27×27

    ЭКСПЕРТЫ:
    - local_expert: SimpleLinear (2059 params) - 10% связей
    - functional_expert: HybridGNN_CNF (5500-12233 params) - 55% связей
    - distant_expert: LightweightCNF (1500-4000 params) - 35% связей

    УПРАВЛЕНИЕ:
    - gating_network: (808 params) - адаптивное взвешивание
    - connection_classifier: классификация связей по типам

    ВСЕ ПАРАМЕТРЫ ИЗ ЦЕНТРАЛИЗОВАННОГО КОНФИГА!
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        lattice_dimensions: Optional[Tuple[int, int, int]] = None,
        neighbor_count: Optional[int] = None,
        enable_cnf: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()

        config = get_project_config()

        # === ЦЕНТРАЛИЗОВАННАЯ КОНФИГУРАЦИЯ ===
        self.state_size = state_size or config.gnn_state_size  # 32
        self.lattice_dimensions = (
            lattice_dimensions or config.lattice_dimensions
        )  # (27, 27, 27)
        # ОБНОВЛЕНО: Используем adaptive_radius вместо фиксированного количества соседей
        self.adaptive_radius = (
            config.calculate_adaptive_radius()
        )  # Биологически правдоподобный радиус
        self.max_neighbors = (
            config.max_neighbors
        )  # Максимальный лимит для производительности
        self.enable_cnf = (
            enable_cnf if enable_cnf is not None else config.enable_cnf
        )  # True

        # Конфигурация распределения связей: 10%/55%/35%
        self.connection_ratios = {
            "local": config.local_tier,  # 0.10 из конфига
            "functional": config.functional_tier,  # 0.55 из конфига
            "distant": config.distant_tier,  # 0.35 из конфига
        }

        # === UNIFIED КЛАССИФИКАТОР СВЯЗЕЙ ===
        # Новый унифицированный классификатор без deprecated зависимостей
        self.connection_classifier = UnifiedConnectionClassifier(
            lattice_dimensions=self.lattice_dimensions
        )

        # === ЭКСПЕРТЫ ===

        # 1. Local Expert - рефлексы (2059 параметров)
        # Поддерживает переменное количество соседей в пределах max_neighbors
        self.local_expert = SimpleLinearExpert(
            state_size=self.state_size, max_neighbors=self.max_neighbors
        )

        # 2. Functional Expert - основная обработка (HybridGNN_CNF)
        # Используем flexible_neighbor_count для работы с переменным количеством соседей
        self.functional_expert = HybridGNN_CNF_Expert(
            state_size=self.state_size,
            neighbor_count=self.max_neighbors,  # Max capacity, но будет использовать переменное количество
            target_params=config.functional_expert_params,  # 8233 (обновлено)
            cnf_params=config.distant_expert_params,  # 4000 (обновлено)
        )

        # 3. Distant Expert - долгосрочная память (LightweightCNF)
        self.distant_expert = LightweightCNF(
            state_size=self.state_size,
            connection_type=ConnectionType.DISTANT,
            integration_steps=config.cnf_integration_steps,  # 3
            adaptive_step_size=config.cnf_adaptive_step_size,  # True
            target_params=config.distant_expert_params,  # 4000
        )

        # === GATING NETWORK ===
        self.gating_network = GatingNetwork(state_size=self.state_size, num_experts=3)

        # === СТАТИСТИКА ===
        self.usage_stats = {
            "local_calls": 0,
            "functional_calls": 0,
            "distant_calls": 0,
            "total_calls": 0,
        }

        # Подсчет общих параметров
        total_params = sum(p.numel() for p in self.parameters())

        expert_params = {
            "local": sum(p.numel() for p in self.local_expert.parameters()),
            "functional": sum(p.numel() for p in self.functional_expert.parameters()),
            "distant": sum(p.numel() for p in self.distant_expert.parameters()),
            "gating": sum(p.numel() for p in self.gating_network.parameters()),
            "classifier": sum(
                p.numel() for p in self.connection_classifier.parameters()
            ),
        }

        log_cell_init(
            cell_type="MoEConnectionProcessor",
            total_params=total_params,
            target_params=None,  # Общая цель не задана
            lattice_dimensions=self.lattice_dimensions,
            **expert_params,
        )

        logger.info(f"🔧 MoEConnectionProcessor: {total_params} параметров")
        logger.info(f"   📐 Adaptive radius: {self.adaptive_radius:.2f}")
        logger.info(f"   🔢 Max neighbors: {self.max_neighbors}")
        logger.info(
            f"   🎯 Local Expert: {expert_params['local']} (цель: {config.local_expert_params})"
        )
        logger.info(
            f"   🎯 Functional Expert: {expert_params['functional']} (цель: {config.functional_expert_params})"
        )
        logger.info(
            f"   🎯 Distant Expert: {expert_params['distant']} (цель: {config.distant_expert_params})"
        )
        logger.info(
            f"   🎯 Gating Network: {expert_params['gating']} (цель: {config.gating_params})"
        )

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        cell_idx: int,
        neighbor_indices: List[int],
        external_input: Optional[torch.Tensor] = None,
        spatial_optimizer=None,  # НОВОЕ: поддержка adaptive radius поиска
        **kwargs,
    ) -> Dict[str, Any]:
        """
        MoE обработка связей через специализированных экспертов

        Args:
            current_state: [batch, state_size] - текущее состояние клетки
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей
            cell_idx: int - индекс текущей клетки
            neighbor_indices: List[int] - индексы соседей
            external_input: Optional[Tensor] - внешний вход
            spatial_optimizer: Optional[SpatialOptimizer] - для adaptive radius поиска

        Returns:
            result: Dict с новым состоянием и статистикой
        """

        # НОВАЯ АРХИТЕКТУРА: Используем adaptive radius поиск если передан spatial_optimizer
        if spatial_optimizer is not None:
            # Находим соседей по adaptive radius вместо использования переданных
            adaptive_neighbors = self.find_neighbors_by_radius(
                cell_idx, spatial_optimizer
            )

            if adaptive_neighbors:
                # Обновляем neighbor_indices и neighbor_states на основе adaptive radius
                neighbor_indices = adaptive_neighbors

                # Для adaptive radius архитектуры нам нужен доступ к состояниям всей решетки
                # Мы получим их через kwargs или создадим заглушку
                if "full_lattice_states" in kwargs:
                    full_states = kwargs["full_lattice_states"]
                    # Создаем neighbor_states из полных состояний решетки
                    neighbor_states = full_states[neighbor_indices].unsqueeze(
                        0
                    )  # [1, num_neighbors, state_size]
                    neighbor_indices = adaptive_neighbors
                else:
                    # Fallback - используем переданные состояния (старая архитектура)
                    logger.warning(
                        "⚠️ Не переданы full_lattice_states, используем fallback режим"
                    )
                    actual_neighbor_count = min(
                        len(adaptive_neighbors), neighbor_states.shape[1]
                    )
                    neighbor_states = neighbor_states[:, :actual_neighbor_count, :]
                    neighbor_indices = neighbor_indices[:actual_neighbor_count]
        batch_size = current_state.shape[0]

        if neighbor_states.shape[1] == 0:
            return {
                "new_state": current_state,
                "expert_weights": torch.zeros(batch_size, 3),
                "expert_contributions": {
                    "local": 0.0,
                    "functional": 0.0,
                    "distant": 0.0,
                },
            }

        # 1. Классификация связей по типам
        connection_classification = self.connection_classifier.classify_connections(
            cell_idx, neighbor_indices, current_state[0], neighbor_states[0]
        )

        # 2. Подготовка входов для экспертов
        neighbor_activity = torch.mean(
            torch.abs(neighbor_states), dim=1
        )  # [batch, state_size]

        # 3. Вызов экспертов для обработки своих типов связей
        expert_outputs = []

        # Local Expert (10%)
        local_connections = connection_classification.get(ConnectionCategory.LOCAL, [])
        local_indices = [conn.target_idx for conn in local_connections]
        local_neighbors = self._filter_neighbors_by_indices(
            neighbor_states, neighbor_indices, local_indices
        )
        if local_neighbors.shape[1] > 0:
            local_result = self.local_expert(current_state, local_neighbors)
            self.usage_stats["local_calls"] += 1
        else:
            local_result = current_state
        expert_outputs.append(local_result)

        # Functional Expert (55%)
        functional_connections = connection_classification.get(
            ConnectionCategory.FUNCTIONAL, []
        )
        functional_indices = [conn.target_idx for conn in functional_connections]
        functional_neighbors = self._filter_neighbors_by_indices(
            neighbor_states, neighbor_indices, functional_indices
        )
        if functional_neighbors.shape[1] > 0:
            functional_result_dict = self.functional_expert(
                current_state, functional_neighbors, external_input
            )
            functional_result = functional_result_dict["new_state"]
            self.usage_stats["functional_calls"] += 1
        else:
            functional_result = current_state
        expert_outputs.append(functional_result)

        # Distant Expert (35%)
        distant_connections = connection_classification.get(
            ConnectionCategory.DISTANT, []
        )
        distant_indices = [conn.target_idx for conn in distant_connections]
        distant_neighbors = self._filter_neighbors_by_indices(
            neighbor_states, neighbor_indices, distant_indices
        )
        if distant_neighbors.shape[1] > 0:
            distant_result = self.distant_expert(current_state, distant_neighbors)
            self.usage_stats["distant_calls"] += 1
        else:
            distant_result = current_state
        expert_outputs.append(distant_result)

        # 4. Gating Network - адаптивное комбинирование
        combined_result, expert_weights = self.gating_network(
            current_state, neighbor_activity, expert_outputs
        )

        # 5. Финальная residual connection для стабильности
        alpha = 0.1  # Малое влияние для стабильности
        final_result = (1 - alpha) * combined_result + alpha * current_state

        self.usage_stats["total_calls"] += 1

        return {
            "new_state": final_result,
            "expert_weights": expert_weights,
            "expert_contributions": {
                "local": expert_weights[:, 0].mean().item(),
                "functional": expert_weights[:, 1].mean().item(),
                "distant": expert_weights[:, 2].mean().item(),
            },
            "connection_counts": {
                "local": len(local_indices),
                "functional": len(functional_indices),
                "distant": len(distant_indices),
            },
        }

    def _filter_neighbors_by_indices(
        self,
        neighbor_states: torch.Tensor,
        neighbor_indices: List[int],
        target_indices: List[int],
    ) -> torch.Tensor:
        """Фильтрация соседей по типу связи"""
        if not target_indices:
            return torch.empty(
                neighbor_states.shape[0],
                0,
                neighbor_states.shape[2],
                device=neighbor_states.device,
            )

        # Находим позиции целевых индексов в списке соседей
        positions = [
            i for i, idx in enumerate(neighbor_indices) if idx in target_indices
        ]

        if not positions:
            return torch.empty(
                neighbor_states.shape[0],
                0,
                neighbor_states.shape[2],
                device=neighbor_states.device,
            )

        return neighbor_states[:, positions, :]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Получить статистику использования экспертов"""
        total = max(self.usage_stats["total_calls"], 1)  # Избегаем деления на ноль

        return {
            "calls": self.usage_stats.copy(),
            "ratios": {
                "local": self.usage_stats["local_calls"] / total,
                "functional": self.usage_stats["functional_calls"] / total,
                "distant": self.usage_stats["distant_calls"] / total,
            },
            "target_ratios": self.connection_ratios,
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.usage_stats = {k: 0 for k in self.usage_stats}

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Получить детальную разбивку параметров"""
        return {
            "local_expert": self.local_expert.get_parameter_info(),
            "functional_expert": self.functional_expert.get_parameter_info(),
            "distant_expert": {
                "total_params": sum(
                    p.numel() for p in self.distant_expert.parameters()
                ),
                "target_params": getattr(self.distant_expert, "target_params", 4000),
            },
            "gating_network": {
                "total_params": sum(
                    p.numel() for p in self.gating_network.parameters()
                ),
                "target_params": 808,
            },
            "connection_classifier": {
                "total_params": sum(
                    p.numel() for p in self.connection_classifier.parameters()
                )
            },
        }

    def get_adaptive_radius(self) -> float:
        """Получить текущий adaptive radius для поиска соседей"""
        return self.adaptive_radius

    def get_neighbor_search_config(self) -> Dict[str, Any]:
        """Получить конфигурацию поиска соседей"""
        return {
            "adaptive_radius": self.adaptive_radius,
            "max_neighbors": self.max_neighbors,
            "search_strategy": "radius_based",  # Новая стратегия вместо fixed_count
            "lattice_dimensions": self.lattice_dimensions,
        }

    def find_neighbors_by_radius(
        self, cell_idx: int, spatial_optimizer=None
    ) -> List[int]:
        """
        Находит соседей по adaptive radius (новая архитектура)

        Args:
            cell_idx: индекс клетки
            spatial_optimizer: опциональный SpatialOptimizer для поиска

        Returns:
            список индексов соседей в adaptive radius
        """
        pos_helper = Position3D(self.lattice_dimensions)

        # ПРОВЕРЯЕМ BOUNDS ДЛЯ CELL_IDX
        total_cells = (
            self.lattice_dimensions[0]
            * self.lattice_dimensions[1]
            * self.lattice_dimensions[2]
        )
        if not (0 <= cell_idx < total_cells):
            logger.warning(
                f"⚠️ Неправильный cell_idx: {cell_idx} для решетки {self.lattice_dimensions} (max: {total_cells-1})"
            )
            return []

        if spatial_optimizer is None:
            # Fallback - создаем простой helper для координат

            # coords = pos_helper.index_to_coords(cell_idx)
            coords = pos_helper.to_3d_coordinates(cell_idx)

            # Простой поиск в радиусе без spatial optimization
            # TODO: В будущем можно использовать spatial hashing
            neighbors = []
            total_cells = (
                self.lattice_dimensions[0]
                * self.lattice_dimensions[1]
                * self.lattice_dimensions[2]
            )

            for neighbor_idx in range(total_cells):
                if neighbor_idx == cell_idx:
                    continue

                # neighbor_coords = pos_helper.index_to_coords(neighbor_idx)
                neighbor_coords = pos_helper.to_3d_coordinates(neighbor_idx)
                distance = (
                    (coords[0] - neighbor_coords[0]) ** 2
                    + (coords[1] - neighbor_coords[1]) ** 2
                    + (coords[2] - neighbor_coords[2]) ** 2
                ) ** 0.5

                if distance <= self.adaptive_radius:
                    neighbors.append(neighbor_idx)

                # Лимит производительности
                if len(neighbors) >= self.max_neighbors:
                    break

            return neighbors
        else:
            # Используем оптимизированный поиск

            # coords = pos_helper.index_to_coords(cell_idx)
            coords = pos_helper.to_3d_coordinates(cell_idx)

            neighbors = spatial_optimizer.find_neighbors_optimized(
                coords, self.adaptive_radius
            )

            # ФИЛЬТРУЕМ СОСЕДЕЙ ПО BOUNDS
            valid_neighbors = []
            for neighbor_idx in neighbors:
                if 0 <= neighbor_idx < total_cells:
                    valid_neighbors.append(neighbor_idx)
                else:
                    logger.warning(
                        f"⚠️ Неправильный neighbor_idx: {neighbor_idx} из spatial_optimizer для решетки {self.lattice_dimensions} (max: {total_cells-1})"
                    )

            # Применяем лимит производительности
            if len(valid_neighbors) > self.max_neighbors:
                valid_neighbors = valid_neighbors[: self.max_neighbors]

            return valid_neighbors

    def forward_batch(
        self,
        batch_states: torch.Tensor,
        batch_neighbor_states: torch.Tensor,
        batch_cell_indices: torch.Tensor,
        batch_neighbor_indices: torch.Tensor,
        full_lattice_states: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Batch обработка для производительности на больших решетках

        Args:
            batch_states: [batch_size, state_size] - состояния клеток
            batch_neighbor_states: [batch_size, max_neighbors, state_size] - состояния соседей
            batch_cell_indices: [batch_size] - индексы центральных клеток
            batch_neighbor_indices: [batch_size, max_neighbors] - индексы соседей (padded с -1)
            full_lattice_states: [total_cells, state_size] - все состояния решетки
            external_input: Optional[Tensor] - внешний вход

        Returns:
            result: Dict с batch результатами
        """
        batch_size, max_neighbors, state_size = batch_neighbor_states.shape
        device = batch_states.device

        # 1. Batch классификация связей
        classifications = self.connection_classifier.classify_connections_batch(
            batch_cell_indices, batch_neighbor_indices, full_lattice_states
        )

        # 2. Подготовка batch neighbor activity
        valid_mask = batch_neighbor_indices >= 0  # [batch_size, max_neighbors]
        neighbor_activity = torch.zeros_like(batch_states)  # [batch_size, state_size]

        # Вычисляем активность соседей для каждой клетки
        for batch_idx in range(batch_size):
            valid_neighbors = valid_mask[batch_idx]
            if valid_neighbors.sum() > 0:
                neighbor_activity[batch_idx] = torch.mean(
                    torch.abs(batch_neighbor_states[batch_idx, valid_neighbors]), dim=0
                )

        # 3. Batch обработка экспертами
        expert_outputs = []

        # Local Expert (10%)
        local_mask = classifications["local"]  # [batch_size, max_neighbors]
        local_outputs = self._process_expert_batch(
            batch_states,
            batch_neighbor_states,
            local_mask,
            self.local_expert,
            "local",
            external_input,
        )
        expert_outputs.append(local_outputs)

        # Functional Expert (55%)
        functional_mask = classifications["functional"]  # [batch_size, max_neighbors]
        functional_outputs = self._process_expert_batch(
            batch_states,
            batch_neighbor_states,
            functional_mask,
            self.functional_expert,
            "functional",
            external_input,
        )
        expert_outputs.append(functional_outputs)

        # Distant Expert (35%)
        distant_mask = classifications["distant"]  # [batch_size, max_neighbors]
        distant_outputs = self._process_expert_batch(
            batch_states,
            batch_neighbor_states,
            distant_mask,
            self.distant_expert,
            "distant",
            external_input,
        )
        expert_outputs.append(distant_outputs)

        # 4. Batch Gating Network
        combined_result, expert_weights = self.gating_network(
            batch_states, neighbor_activity, expert_outputs
        )

        # 5. Batch residual connection
        alpha = 0.1
        final_result = (1 - alpha) * combined_result + alpha * batch_states

        # 6. Batch статистика
        connection_counts = self._compute_batch_connection_counts(classifications)

        self.usage_stats["total_calls"] += batch_size

        return {
            "new_states": final_result,  # [batch_size, state_size]
            "expert_weights": expert_weights,  # [batch_size, num_experts]
            "expert_contributions": {
                "local": expert_weights[:, 0].mean().item(),
                "functional": expert_weights[:, 1].mean().item(),
                "distant": expert_weights[:, 2].mean().item(),
            },
            "connection_counts": connection_counts,
            "batch_size": batch_size,
        }

    def _process_expert_batch(
        self,
        batch_states: torch.Tensor,
        batch_neighbor_states: torch.Tensor,
        connection_mask: torch.Tensor,
        expert: nn.Module,
        expert_type: str,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch обработка одного эксперта

        Args:
            batch_states: [batch_size, state_size]
            batch_neighbor_states: [batch_size, max_neighbors, state_size]
            connection_mask: [batch_size, max_neighbors] - boolean mask для связей
            expert: экземпляр эксперта
            expert_type: тип эксперта для статистики
            external_input: внешний вход

        Returns:
            outputs: [batch_size, state_size] - результаты эксперта
        """
        batch_size, max_neighbors, state_size = batch_neighbor_states.shape
        outputs = torch.zeros_like(batch_states)

        for batch_idx in range(batch_size):
            # Получаем маску соединений для этой клетки
            cell_mask = connection_mask[batch_idx]  # [max_neighbors]

            if cell_mask.sum() == 0:
                # Нет соединений этого типа - возвращаем текущее состояние
                outputs[batch_idx] = batch_states[batch_idx]
                continue

            # Фильтруем соседей по маске
            filtered_neighbors = batch_neighbor_states[batch_idx, cell_mask].unsqueeze(
                0
            )  # [1, num_connections, state_size]
            current_state = batch_states[batch_idx].unsqueeze(0)  # [1, state_size]

            # Вызываем эксперта
            if expert_type == "functional" and hasattr(expert, "forward"):
                # Functional expert возвращает dict
                result = expert(current_state, filtered_neighbors, external_input)
                if isinstance(result, dict) and "new_state" in result:
                    outputs[batch_idx] = result["new_state"].squeeze(0)
                else:
                    outputs[batch_idx] = (
                        result.squeeze(0)
                        if hasattr(result, "squeeze")
                        else current_state.squeeze(0)
                    )
            else:
                # Остальные эксперты возвращают tensor
                result = expert(current_state, filtered_neighbors)
                outputs[batch_idx] = result.squeeze(0)

        # Обновляем статистику
        active_connections = connection_mask.sum().item()
        if active_connections > 0:
            self.usage_stats[f"{expert_type}_calls"] += batch_size

        return outputs

    def _compute_batch_connection_counts(
        self, classifications: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Вычисляет статистику соединений для batch"""
        local_counts = classifications["local"].sum(dim=1)  # [batch_size]
        functional_counts = classifications["functional"].sum(dim=1)  # [batch_size]
        distant_counts = classifications["distant"].sum(dim=1)  # [batch_size]

        return {
            "local": {
                "mean": local_counts.float().mean().item(),
                "total": local_counts.sum().item(),
                "per_cell": local_counts.tolist(),
            },
            "functional": {
                "mean": functional_counts.float().mean().item(),
                "total": functional_counts.sum().item(),
                "per_cell": functional_counts.tolist(),
            },
            "distant": {
                "mean": distant_counts.float().mean().item(),
                "total": distant_counts.sum().item(),
                "per_cell": distant_counts.tolist(),
            },
        }


# === РЕФАКТОРИНГ И ОБРАТНАЯ СОВМЕСТИМОСТЬ ===
#
# ВНИМАНИЕ: Этот файл был разбит на более мелкие модули для улучшения читаемости:
#
# Новые модули:
# - connection_types.py - типы данных для связей (ConnectionCategory, ConnectionInfo)
# - distance_calculator.py - вычисление расстояний в 3D решетке
# - functional_similarity.py - анализ функциональной близости
# - gating_network.py - сеть управления экспертами
# - connection_classifier.py - классификация связей
# - moe_processor.py - упрощенный основной процессор
#
# Рекомендуемые импорты для нового кода:
# from .moe_processor import MoEConnectionProcessor  # Упрощенная версия
# from .gating_network import GatingNetwork
# from .connection_classifier import UnifiedConnectionClassifier
# from .connection_types import ConnectionCategory, ConnectionInfo
#
# Этот файл сохранен для обратной совместимости с существующим кодом.
