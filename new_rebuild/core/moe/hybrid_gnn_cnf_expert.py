#!/usr/bin/env python3
"""
Hybrid GNN+CNF Expert - для функциональных связей (55%)
=====================================================

Гибридный эксперт, комбинирующий GNN и CNF для обработки функциональных связей.
Основная рабочая лошадка архитектуры - обрабатывает большинство связей.

АРХИТЕКТУРА:
- GNN компонент для дискретной обработки сообщений
- CNF компонент для continuous dynamics
- Adaptive gating между GNN и CNF
- Параметры: 5500-12233 (настраиваемые)

ПРИНЦИПЫ:
1. Лучшее из двух миров: дискретная + continuous обработка
2. Адаптивное переключение на основе активности
3. Максимальная выразительность для основных связей
4. Централизованная конфигурация (все параметры из ProjectConfig)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from ..cnf.gpu_enhanced_cnf import GPUEnhancedCNF, ConnectionType
from ..cells.vectorized_gnn_cell import VectorizedGNNCell
from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward

logger = get_logger(__name__)


class AdaptiveGatingNetwork(nn.Module):
    """
    Адаптивное переключение между GNN и CNF на основе активности

    Идея: для разных паттернов активности оптимальны разные методы обработки
    - Высокая активность → GNN (дискретная обработка)
    - Низкая активность → CNF (continuous evolution)
    """

    def __init__(self, state_size: int):
        super().__init__()

        # Сеть для принятия решения GNN vs CNF
        self.gating_network = nn.Sequential(
            nn.Linear(
                state_size * 2, state_size // 2, bias=True
            ),  # state + neighbor_activity
            nn.GELU(),
            nn.Linear(state_size // 2, 1, bias=True),  # Один скаляр для решения
            nn.Sigmoid(),  # [0, 1]: 0 = GNN, 1 = CNF
        )

    def forward(
        self, current_state: torch.Tensor, neighbor_activity: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление gating weights для GNN vs CNF

        Args:
            current_state: [batch, state_size]
            neighbor_activity: [batch, state_size] - агрегированная активность соседей

        Returns:
            gating_weight: [batch, 1] - 0=GNN, 1=CNF
        """
        combined_input = torch.cat([current_state, neighbor_activity], dim=-1)
        gating_weight = self.gating_network(combined_input)
        return gating_weight


class HybridGNN_CNF_Expert(nn.Module):
    """
    Гибридный эксперт для functional connections (55%)

    Комбинирует GNN и CNF для максимальной выразительности.
    Адаптивно переключается между дискретной и continuous обработкой.

    ВСЕ ПАРАМЕТРЫ ИЗ ЦЕНТРАЛИЗОВАННОГО КОНФИГА!
    """

    def __init__(
        self,
        state_size: Optional[int] = None,
        neighbor_count: Optional[int] = None,
        target_params: Optional[int] = None,
        cnf_params: Optional[int] = None,
    ):
        super().__init__()

        config = get_project_config()

        # === ЦЕНТРАЛИЗОВАННАЯ КОНФИГУРАЦИЯ ===
        self.state_size = state_size or config.model.state_size
        self.neighbor_count = (
            neighbor_count
            if neighbor_count is not None
            else config.neighbors.max_neighbors
        )
        self.target_params = target_params or config.expert.functional.params
        self.cnf_params = cnf_params or config.expert.distant.params

        # === КОМПОНЕНТЫ ГИБРИДНОГО ЭКСПЕРТА ===

        # 1. GNN компонент (примерно 60% от общих параметров)
        # Используем параметры из настроек функционального эксперта
        self.gnn_component = VectorizedGNNCell(
            state_size=self.state_size,
            neighbor_count=self.neighbor_count,
            message_dim=config.expert.functional.message_dim,
            hidden_dim=config.expert.functional.hidden_dim,
            external_input_size=config.model.external_input_size,
            use_attention=config.expert.functional.use_attention,
        )

        # 2. CNF компонент (примерно 25% от общих параметров)
        self.cnf_component = GPUEnhancedCNF(
            state_size=self.state_size,
            connection_type=ConnectionType.FUNCTIONAL,
            integration_steps=config.cnf.integration_steps,
            batch_processing_mode=config.cnf.batch_processing_mode,
            max_batch_size=config.cnf.max_batch_size,
            adaptive_method=config.cnf.adaptive_method,
        )

        # 3. Adaptive gating network (примерно 15% от общих параметров)
        self.adaptive_gating = AdaptiveGatingNetwork(self.state_size)

        # 4. Результирующая обработка для комбинирования GNN и CNF
        self.result_processor = nn.Sequential(
            nn.Linear(
                self.state_size * 2, self.state_size, bias=True
            ),  # GNN_result + CNF_result
            nn.LayerNorm(self.state_size),
            nn.GELU(),
        )

        # Подсчет общего количества параметров
        total_params = sum(p.numel() for p in self.parameters())

        log_cell_init(
            cell_type="HybridGNN_CNF_Expert",
            total_params=total_params,
            target_params=self.target_params,
            state_size=self.state_size,
            neighbor_count=self.neighbor_count,
            gnn_params=sum(p.numel() for p in self.gnn_component.parameters()),
            cnf_params=sum(p.numel() for p in self.cnf_component.parameters()),
            gating_params=sum(p.numel() for p in self.adaptive_gating.parameters()),
        )

        # Получаем пороги из конфига для информативности
        try:
            from ...config import get_project_config
            config = get_project_config()
            local_pct = config.lattice.local_distance_ratio * 100
            functional_pct = config.lattice.functional_distance_ratio * 100
        except:
            local_pct, functional_pct = 10, 65  # fallback
            
        logger.info(f"[Functional Expert] HybridGNN_CNF инициализирован:")
        logger.info(f"   Параметров: {total_params} (target: {self.target_params})")
        logger.info(f"   GNN компонент: {sum(p.numel() for p in self.gnn_component.parameters())} параметров")
        logger.info(f"   CNF компонент: {sum(p.numel() for p in self.cnf_component.parameters())} параметров")
        logger.info(f"   Neighbor count: {'dynamic' if self.neighbor_count == -1 else self.neighbor_count}")
        logger.info(f"   Обрабатывает FUNCTIONAL соседей ({local_pct:.0f}%-{functional_pct:.0f}% от adaptive_radius)")

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Гибридная обработка через GNN + CNF

        Args:
            current_state: [batch, state_size] - текущее состояние
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей
            external_input: [batch, external_input_size] - внешний вход

        Returns:
            result: Dict с результатами и метриками
        """
        batch_size = current_state.shape[0]
        config = get_project_config()

        if neighbor_states.shape[1] == 0:
            return {
                "new_state": current_state,
                "gnn_contribution": 0.0,
                "cnf_contribution": 0.0,
                "gating_weight": 0.5,
            }

        # 1. Вычисляем активность соседей для gating
        if neighbor_states.dim() == 3:
            # [batch, num_neighbors, state_size] -> [batch, state_size]
            neighbor_activity = torch.mean(
                torch.abs(neighbor_states), dim=1
            )  # [batch, state_size]
        else:
            # [num_neighbors, state_size] -> [1, state_size]
            neighbor_activity = torch.mean(
                torch.abs(neighbor_states), dim=0, keepdim=True
            )  # [1, state_size]

        # 2. Адаптивное переключение GNN vs CNF
        gating_weight = self.adaptive_gating(
            current_state, neighbor_activity
        )  # [batch, 1]

        # 3. GNN обработка
        if external_input is None:
            external_input = torch.zeros(
                batch_size,
                config.model.external_input_size,
                device=current_state.device,
            )

        gnn_result = self.gnn_component(
            neighbor_states=neighbor_states,
            own_state=current_state,
            external_input=external_input,
            flexible_neighbor_count=True,
        )

        # 4. CNF обработка
        cnf_result_dict = self.cnf_component(
            current_state=current_state, neighbor_states=neighbor_states
        )
        cnf_result = cnf_result_dict["new_state"]

        # 5. Комбинирование результатов на основе gating
        # gating_weight: 0 = только GNN, 1 = только CNF
        gnn_contribution = (1 - gating_weight) * gnn_result
        cnf_contribution = gating_weight * cnf_result

        # 6. Финальная обработка комбинированного результата
        combined_input = torch.cat([gnn_contribution, cnf_contribution], dim=-1)
        final_result = self.result_processor(combined_input)

        return {
            "new_state": final_result,
            "gnn_result": gnn_result,
            "cnf_result": cnf_result,
            "gnn_contribution": (1 - gating_weight).mean().item(),
            "cnf_contribution": gating_weight.mean().item(),
            "gating_weight": gating_weight.mean().item(),
        }

    def get_parameter_info(self) -> Dict[str, Any]:
        """Получить детальную информацию о параметрах"""
        param_breakdown = {
            "gnn_component": sum(p.numel() for p in self.gnn_component.parameters()),
            "cnf_component": sum(p.numel() for p in self.cnf_component.parameters()),
            "adaptive_gating": sum(
                p.numel() for p in self.adaptive_gating.parameters()
            ),
            "result_processor": sum(
                p.numel() for p in self.result_processor.parameters()
            ),
        }

        total = sum(param_breakdown.values())

        return {
            "total_params": total,
            "target_params": self.target_params,
            "breakdown": param_breakdown,
            "gnn_ratio": f"{param_breakdown['gnn_component']/total:.1%}",
            "cnf_ratio": f"{param_breakdown['cnf_component']/total:.1%}",
            "gating_ratio": f"{param_breakdown['adaptive_gating']/total:.1%}",
            "efficiency": (
                f"{total/self.target_params:.1%}" if self.target_params > 0 else "N/A"
            ),
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Получить статистику обработки (для анализа)"""
        return {
            "component_type": "HybridGNN_CNF",
            "supports_continuous": True,
            "supports_discrete": True,
            "adaptive_routing": True,
            "connection_types": ["functional"],
            "integration_steps": (
                self.cnf_component.integration_steps
                if hasattr(self.cnf_component, "integration_steps")
                else 3
            ),
        }
