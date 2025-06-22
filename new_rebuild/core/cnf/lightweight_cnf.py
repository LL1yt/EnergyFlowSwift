#!/usr/bin/env python3
"""
Lightweight CNF - Continuous Normalizing Flow для 3D CNN
========================================================

Основной класс для continuous dynamics в functional и distant connections.
Использует Neural ODE с 3-step Euler solver для эффективной эволюции состояний.

ПРИНЦИПЫ:
1. Neural ODE только для non-local connections (90% связей)
2. 3-step Euler solver (7x быстрее чем 10-step RK4)
3. Адаптивный размер шага на основе активности
4. ~500 параметров на связь (биологически правдоподобно)

ЭМЕРДЖЕНТНЫЕ СВОЙСТВА:
- Естественные траектории в пространстве состояний
- Автоматическая стабилизация через аттракторы
- Спонтанные осцилляции и волновые паттерны
- Критические переходы между режимами
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from enum import Enum

from ...config import get_project_config
from ...utils.logging import get_logger, log_cell_init, log_cell_forward

logger = get_logger(__name__)


class ConnectionType(Enum):
    """Типы связей для CNF обработки"""

    FUNCTIONAL = "functional"  # 60% связей - средние расстояния
    DISTANT = "distant"  # 30% связей - дальние расстояния


class NeuralODE(nn.Module):
    """
    Neural ODE network для вычисления производной dx/dt

    Моделирует continuous dynamics: dx/dt = f(x, neighbors, t)
    """

    def __init__(
        self,
        state_size: int,
        connection_type: ConnectionType,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.state_size = state_size
        self.connection_type = connection_type
        self.hidden_dim = hidden_dim or max(16, state_size // 2)

        # Размер входа: собственное состояние + агрегированные соседи
        input_size = state_size * 2

        # Компактная архитектура для эффективности
        self.ode_network = nn.Sequential(
            nn.Linear(input_size, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, state_size, bias=True),
        )

        # Damping term для стабилизации (предотвращает взрывы)
        self.damping_strength = nn.Parameter(torch.tensor(0.1))

        total_params = sum(p.numel() for p in self.parameters())
        log_cell_init(
            cell_type="NeuralODE",
            total_params=total_params,
            target_params=1500,  # Оптимизировано для ~3k общих параметров CNF
            state_size=state_size,
            connection_type=connection_type.value,
            hidden_dim=self.hidden_dim,
            input_size=input_size,
        )

    def forward(
        self,
        t: torch.Tensor,
        current_state: torch.Tensor,
        neighbor_influence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисление производной dx/dt = f(x, neighbors, t)

        Args:
            t: время (не используется в простой версии)
            current_state: [batch, state_size] - текущее состояние
            neighbor_influence: [batch, state_size] - влияние соседей

        Returns:
            derivative: [batch, state_size] - dx/dt
        """
        # Комбинируем собственное состояние и влияние соседей
        combined_input = torch.cat([current_state, neighbor_influence], dim=-1)

        # Вычисляем сырую производную
        raw_derivative = self.ode_network(combined_input)

        # Добавляем damping для стабильности: -α * x
        damping_effect = -self.damping_strength * current_state

        # Финальная производная
        derivative = raw_derivative + damping_effect

        return derivative


class LightweightCNF(nn.Module):
    """
    Lightweight Continuous Normalizing Flow

    Continuous dynamics для functional и distant connections.
    Заменяет дискретные обновления на плавную эволюцию состояний.

    АРХИТЕКТУРА:
    1. NeuralODE для вычисления dx/dt
    2. EulerSolver для интеграции (3 шага)
    3. Адаптивный размер шага
    4. Стабилизация через damping
    """

    def __init__(
        self,
        state_size: int,
        connection_type: ConnectionType,
        integration_steps: Optional[int] = None,
        adaptive_step_size: Optional[bool] = None,
        target_params: Optional[int] = None,
    ):
        super().__init__()

        config = get_project_config()

        self.state_size = state_size
        self.connection_type = connection_type
        self.integration_steps = integration_steps or config.cnf_integration_steps
        self.adaptive_step_size = adaptive_step_size or config.cnf_adaptive_step_size
        self.target_params = target_params or config.cnf_target_params_per_connection

        # Neural ODE для вычисления производной
        # Оптимальный hidden_dim для достижения ~3000 параметров
        hidden_dim = max(32, state_size)  # Баланс между сложностью и эффективностью
        self.neural_ode = NeuralODE(
            state_size=state_size,
            connection_type=connection_type,
            hidden_dim=hidden_dim,
        )

        # Параметры интеграции
        self.base_dt = nn.Parameter(torch.tensor(0.1))  # Базовый размер шага
        self.min_dt = 0.01
        self.max_dt = 0.3

        total_params = sum(p.numel() for p in self.parameters())
        log_cell_init(
            cell_type="LightweightCNF",
            total_params=total_params,
            target_params=self.target_params,
            state_size=state_size,
            connection_type=connection_type.value,
            integration_steps=self.integration_steps,
            adaptive_step_size=self.adaptive_step_size,
        )

    def _compute_adaptive_dt(
        self, current_state: torch.Tensor, neighbor_influence: torch.Tensor
    ) -> float:
        """
        Адаптивный размер шага на основе активности

        Принцип: высокая активность → меньший шаг (больше точности)
                 низкая активность → больший шаг (больше эффективности)
        """
        if not self.adaptive_step_size:
            return self.base_dt.item()

        # Вычисляем уровень активности
        state_magnitude = torch.norm(current_state, dim=-1).mean()
        neighbor_magnitude = torch.norm(neighbor_influence, dim=-1).mean()

        total_activity = state_magnitude + neighbor_magnitude

        # Обратная зависимость: больше активности → меньше шаг
        activity_factor = 1.0 / (1.0 + total_activity.item())

        # Адаптивный dt в разумных пределах
        adaptive_dt = self.base_dt.item() * activity_factor
        adaptive_dt = max(self.min_dt, min(self.max_dt, adaptive_dt))

        return adaptive_dt

    def _euler_step(
        self,
        current_state: torch.Tensor,
        neighbor_influence: torch.Tensor,
        dt: float,
        t: float = 0.0,
    ) -> torch.Tensor:
        """
        Один шаг Euler интеграции: x_{n+1} = x_n + dt * f(x_n)
        """
        t_tensor = torch.tensor(t, device=current_state.device)
        derivative = self.neural_ode(t_tensor, current_state, neighbor_influence)

        next_state = current_state + dt * derivative
        return next_state

    def evolve_state(
        self,
        initial_state: torch.Tensor,
        neighbor_influence: torch.Tensor,
        integration_time: float = 1.0,
    ) -> torch.Tensor:
        """
        Эволюция состояния через continuous dynamics

        Args:
            initial_state: [batch, state_size] - начальное состояние
            neighbor_influence: [batch, state_size] - влияние соседей
            integration_time: общее время интеграции

        Returns:
            final_state: [batch, state_size] - финальное состояние
        """
        current_state = initial_state

        # Адаптивный размер шага
        dt = self._compute_adaptive_dt(current_state, neighbor_influence)
        actual_steps = max(1, int(integration_time / dt))
        dt = integration_time / actual_steps  # Корректируем для точного времени

        # Интеграция методом Эйлера
        for step in range(min(actual_steps, self.integration_steps)):
            t = step * dt
            current_state = self._euler_step(current_state, neighbor_influence, dt, t)

        return current_state

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        connection_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass для CNF эволюции

        Args:
            current_state: [batch, state_size] - текущее состояние клетки
            neighbor_states: [batch, num_neighbors, state_size] - состояния соседей
            connection_weights: [batch, num_neighbors] - веса связей (STDP)

        Returns:
            new_state: [batch, state_size] - новое состояние после эволюции
        """
        batch_size = current_state.shape[0]

        # Агрегация влияния соседей
        if neighbor_states.numel() > 0:
            if connection_weights is not None:
                # Модуляция через STDP веса
                weighted_neighbors = neighbor_states * connection_weights.unsqueeze(-1)
                neighbor_influence = weighted_neighbors.mean(dim=1)
            else:
                neighbor_influence = neighbor_states.mean(dim=1)
        else:
            neighbor_influence = torch.zeros_like(current_state)

        # Continuous evolution
        new_state = self.evolve_state(
            initial_state=current_state,
            neighbor_influence=neighbor_influence,
            integration_time=1.0,
        )

        log_cell_forward(
            f"LightweightCNF-{self.connection_type.value}",
            input_shapes={
                "current_state": current_state.shape,
                "neighbor_states": neighbor_states.shape,
                "connection_weights": (
                    connection_weights.shape if connection_weights is not None else None
                ),
            },
            output_shape=new_state.shape,
        )

        return new_state

    def get_dynamics_info(self) -> Dict[str, Any]:
        """Информация о dynamics параметрах"""
        return {
            "connection_type": self.connection_type.value,
            "integration_steps": self.integration_steps,
            "adaptive_step_size": self.adaptive_step_size,
            "base_dt": self.base_dt.item(),
            "damping_strength": self.neural_ode.damping_strength.item(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "target_parameters": self.target_params,
        }
