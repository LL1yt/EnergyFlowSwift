#!/usr/bin/env python3
"""
MoE Processor - упрощенный Mixture of Experts процессор
=====================================================

Основной MoE процессор, использующий модульную архитектуру
для эффективной обработки связей в 3D нейронной решетке.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from .gating_network import GatingNetwork
from .connection_classifier import UnifiedConnectionClassifier
from .connection_types import ConnectionCategory
from .simple_linear_expert import SimpleLinearExpert
from .hybrid_gnn_cnf_expert import HybridGNN_CNF_Expert
from ..cnf.gpu_enhanced_cnf import GPUEnhancedCNF, ConnectionType
from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_forward
from ...utils.device_manager import get_device_manager
from ..lattice.position import Position3D

logger = get_logger(__name__)


class MoEConnectionProcessor(nn.Module):
    """
    Упрощенный Mixture of Experts Connection Processor для 3D решетки 27×27×27

    ЭКСПЕРТЫ:
    - local_expert: SimpleLinear (2059 params) - 10% связей
    - functional_expert: HybridGNN_CNF (5500-12233 params) - 55% связей
    - distant_expert: LightweightCNF (1500-4000 params) - 35% связей

    УПРАВЛЕНИЕ:
    - gating_network: (808 params) - адаптивное взвешивание
    - connection_classifier: классификация связей по типам
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

        # === DEVICE MANAGEMENT ===
        self.device_manager = config.get_device_manager()
        self.device = self.device_manager.get_device()

        # === ЦЕНТРАЛИЗОВАННАЯ КОНФИГУРАЦИЯ ===
        self.state_size = state_size or config.gnn.state_size
        self.lattice_dimensions = lattice_dimensions or config.lattice.dimensions
        self.adaptive_radius = config.calculate_adaptive_radius()
        self.max_neighbors = config.max_neighbors
        self.enable_cnf = enable_cnf if enable_cnf is not None else config.cnf.enabled

        # Конфигурация распределения связей: 10%/55%/35%
        self.connection_ratios = {
            "local": config.neighbors.local_tier,
            "functional": config.neighbors.functional_tier,
            "distant": config.neighbors.distant_tier,
        }

        # === КЛАССИФИКАТОР СВЯЗЕЙ ===
        self.connection_classifier = UnifiedConnectionClassifier(
            lattice_dimensions=self.lattice_dimensions
        )

        # === ЭКСПЕРТЫ ===
        self.local_expert = SimpleLinearExpert(state_size=self.state_size)

        self.functional_expert = HybridGNN_CNF_Expert(
            state_size=self.state_size,
            neighbor_count=self.max_neighbors,
            target_params=config.expert.functional.params,
            cnf_params=config.expert.distant.params,
        )

        # 3. Distant Expert - долгосрочная память (LightweightCNF)
        if self.enable_cnf:
            self.distant_expert = GPUEnhancedCNF(
                state_size=self.state_size,
                connection_type=ConnectionType.DISTANT,
                integration_steps=config.cnf.integration_steps,
                batch_processing_mode=config.cnf.batch_processing_mode,
                max_batch_size=config.cnf.max_batch_size,
                adaptive_method=config.cnf.adaptive_method
            )
        else:
            # Fallback к простому linear если CNF отключен
            self.distant_expert = SimpleLinearExpert(state_size=self.state_size)

        # === GATING NETWORK ===
        self.gating_network = GatingNetwork(state_size=self.state_size, num_experts=3)

        # === СТАТИСТИКА ===
        self.reset_stats()

        # Подсчет общих параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MoEConnectionProcessor: {total_params} параметров всего")

        # Перенос модели на правильное устройство
        self.device_manager.transfer_module(self)
        logger.info(f"MoEConnectionProcessor перенесен на устройство: {self.device}")

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        cell_idx: int,
        neighbor_indices: List[int],
        external_input: Optional[torch.Tensor] = None,
        spatial_optimizer=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Основной forward pass с упрощенной логикой

        Args:
            current_state: [state_size] - текущее состояние клетки
            neighbor_states: [num_neighbors, state_size] - состояния соседей
            cell_idx: индекс текущей клетки
            neighbor_indices: индексы соседних клеток
            external_input: внешний вход (опционально)
            spatial_optimizer: опциональный spatial optimizer для adaptive radius поиска

        Returns:
            Dict с результатами обработки
        """
        # === ОСНОВНОЙ МЕТОД: Adaptive Radius Neighbor Search ===
        if spatial_optimizer is not None and hasattr(
            spatial_optimizer, "find_neighbors_by_radius_safe"
        ):
            # ВСЕГДА используем spatial optimizer для безопасного поиска соседей
            adaptive_neighbors = spatial_optimizer.find_neighbors_by_radius_safe(
                cell_idx
            )

            if adaptive_neighbors and "full_lattice_states" in kwargs:
                # Получаем состояния соседей из полной решетки
                full_states = kwargs["full_lattice_states"]

                # ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ индексов перед использованием
                max_idx = full_states.shape[0] - 1
                valid_neighbors = [
                    idx for idx in adaptive_neighbors if 0 <= idx <= max_idx
                ]

                if len(valid_neighbors) != len(adaptive_neighbors):
                    logger.warning(
                        f"⚠️ Отфильтровано {len(adaptive_neighbors) - len(valid_neighbors)} невалидных индексов для клетки {cell_idx}"
                    )

                if valid_neighbors:
                    neighbor_indices = valid_neighbors
                    neighbor_states = full_states[neighbor_indices]

                    logger.debug(
                        f"🔍 ОСНОВНОЙ РЕЖИМ: spatial_optimizer для клетки {cell_idx}: найдено {len(neighbor_indices)} валидных соседей"
                    )
                else:
                    neighbor_indices = []
                    neighbor_states = torch.empty(
                        0, full_states.shape[1], device=full_states.device
                    )
            else:
                logger.warning(
                    f"⚠️ spatial_optimizer передан, но full_lattice_states отсутствует - fallback к переданным соседям"
                )
        else:
            # Fallback только если spatial_optimizer не передан
            if len(neighbor_indices) > 0:
                logger.debug(
                    f"🔄 FALLBACK РЕЖИМ: используем переданные соседи для клетки {cell_idx}: {len(neighbor_indices)} соседей"
                )
            else:
                logger.warning(
                    f"⚠️ Ни spatial_optimizer, ни neighbor_indices не переданы для клетки {cell_idx}"
                )

        if len(neighbor_indices) == 0:
            return self._empty_forward_result(current_state)

        batch_size = 1
        device = current_state.device

        # Убеждаемся что все tensor'ы на правильном устройстве
        current_state = self.device_manager.ensure_device(current_state)
        neighbor_states = self.device_manager.ensure_device(neighbor_states)
        if external_input is not None:
            external_input = self.device_manager.ensure_device(external_input)

        # === 1. КЛАССИФИКАЦИЯ СВЯЗЕЙ ===
        classifications = self.connection_classifier.classify_connections(
            cell_idx=cell_idx,
            neighbor_indices=neighbor_indices,
            cell_state=current_state,
            neighbor_states=neighbor_states,
        )

        # === 2. ОБРАБОТКА КАЖДЫМ ЭКСПЕРТОМ ===
        expert_outputs = []

        # Local Expert
        local_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.LOCAL]
        ]
        if local_neighbors:
            local_neighbor_states = neighbor_states[
                [neighbor_indices.index(idx) for idx in local_neighbors]
            ]
            local_result = self.local_expert(
                current_state.unsqueeze(0), local_neighbor_states.unsqueeze(0)
            )
            # Извлекаем tensor из результата (может быть dict или tensor)
            if isinstance(local_result, dict):
                local_output = local_result.get(
                    "output", local_result.get("new_state", current_state.unsqueeze(0))
                )
            else:
                local_output = local_result
        else:
            local_output = self.device_manager.allocate_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
        expert_outputs.append(local_output.squeeze(0))

        # Functional Expert
        functional_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.FUNCTIONAL]
        ]
        if functional_neighbors:
            functional_neighbor_states = neighbor_states[
                [neighbor_indices.index(idx) for idx in functional_neighbors]
            ]
            functional_result = self.functional_expert(
                current_state.unsqueeze(0), functional_neighbor_states.unsqueeze(0)
            )
            # Извлекаем tensor из результата (может быть dict или tensor)
            if isinstance(functional_result, dict):
                functional_output = functional_result.get(
                    "output",
                    functional_result.get("new_state", current_state.unsqueeze(0)),
                )
            else:
                functional_output = functional_result
        else:
            functional_output = self.device_manager.allocate_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
        expert_outputs.append(functional_output.squeeze(0))

        # Distant Expert
        distant_neighbors = [
            conn.target_idx for conn in classifications[ConnectionCategory.DISTANT]
        ]
        if distant_neighbors:
            distant_neighbor_states = neighbor_states[
                [neighbor_indices.index(idx) for idx in distant_neighbors]
            ]
            if self.enable_cnf:
                distant_result = self.distant_expert(
                    current_state.unsqueeze(0), distant_neighbor_states.unsqueeze(0)
                )
            else:
                distant_result = self.distant_expert(
                    current_state.unsqueeze(0), distant_neighbor_states.unsqueeze(0)
                )
            # Извлекаем tensor из результата (может быть dict или tensor)
            if isinstance(distant_result, dict):
                distant_output = distant_result.get(
                    "output",
                    distant_result.get("new_state", current_state.unsqueeze(0)),
                )
            else:
                distant_output = distant_result
        else:
            distant_output = self.device_manager.allocate_tensor(
                (1, self.state_size), dtype=current_state.dtype
            )
        expert_outputs.append(distant_output.squeeze(0))

        # === 3. GATING NETWORK ===
        neighbor_activity = neighbor_states.mean(dim=0)  # Среднее состояние соседей
        combined_output, expert_weights = self.gating_network(
            current_state.unsqueeze(0),
            neighbor_activity.unsqueeze(0),
            [out.unsqueeze(0) for out in expert_outputs],
        )

        # === 4. РЕЗУЛЬТАТ ===
        self._update_stats(classifications, expert_weights.squeeze(0))

        return {
            "new_state": combined_output.squeeze(0),
            "expert_weights": expert_weights.squeeze(0),
            "classifications": classifications,
            "expert_outputs": expert_outputs,
            "neighbor_count": len(neighbor_indices),
        }

    def _empty_forward_result(self, current_state: torch.Tensor) -> Dict[str, Any]:
        """Результат для случая без соседей"""
        return {
            "new_state": current_state,
            "expert_weights": torch.tensor(
                [1.0, 0.0, 0.0], device=current_state.device
            ),
            "classifications": {cat: [] for cat in ConnectionCategory},
            "expert_outputs": [
                current_state,
                torch.zeros_like(current_state),
                torch.zeros_like(current_state),
            ],
            "neighbor_count": 0,
        }

    def _update_stats(self, classifications: Dict, expert_weights: torch.Tensor):
        """Обновление статистики использования"""
        local_count = len(classifications[ConnectionCategory.LOCAL])
        functional_count = len(classifications[ConnectionCategory.FUNCTIONAL])
        distant_count = len(classifications[ConnectionCategory.DISTANT])

        self.usage_stats["local_connections"] += local_count
        self.usage_stats["functional_connections"] += functional_count
        self.usage_stats["distant_connections"] += distant_count
        self.usage_stats["total_forward_calls"] += 1

        # Статистика весов экспертов
        weights = expert_weights.detach()
        self.usage_stats["expert_weights"]["local"] += weights[0].item()
        self.usage_stats["expert_weights"]["functional"] += weights[1].item()
        self.usage_stats["expert_weights"]["distant"] += weights[2].item()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Получить статистику использования"""
        total_connections = max(
            1,
            self.usage_stats["local_connections"]
            + self.usage_stats["functional_connections"]
            + self.usage_stats["distant_connections"],
        )
        total_calls = max(1, self.usage_stats["total_forward_calls"])

        return {
            "connection_distribution": {
                "local_ratio": self.usage_stats["local_connections"]
                / total_connections,
                "functional_ratio": self.usage_stats["functional_connections"]
                / total_connections,
                "distant_ratio": self.usage_stats["distant_connections"]
                / total_connections,
            },
            "expert_usage": {
                "local_avg_weight": self.usage_stats["expert_weights"]["local"]
                / total_calls,
                "functional_avg_weight": self.usage_stats["expert_weights"][
                    "functional"
                ]
                / total_calls,
                "distant_avg_weight": self.usage_stats["expert_weights"]["distant"]
                / total_calls,
            },
            "total_forward_calls": total_calls,
            "total_connections": total_connections,
        }

    def reset_stats(self):
        """Сброс статистики"""
        self.usage_stats = {
            "local_connections": 0,
            "functional_connections": 0,
            "distant_connections": 0,
            "total_forward_calls": 0,
            "expert_weights": {"local": 0.0, "functional": 0.0, "distant": 0.0},
        }

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Получить разбивку параметров по компонентам"""
        return {
            "local_expert": sum(p.numel() for p in self.local_expert.parameters()),
            "functional_expert": sum(
                p.numel() for p in self.functional_expert.parameters()
            ),
            "distant_expert": sum(p.numel() for p in self.distant_expert.parameters()),
            "gating_network": sum(p.numel() for p in self.gating_network.parameters()),
            "connection_classifier": sum(
                p.numel() for p in self.connection_classifier.parameters()
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
