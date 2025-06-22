#!/usr/bin/env python3
"""
Euler Solver - быстрая интеграция для CNF
========================================

3-step Euler solver как эффективная альтернатива 10-step RK4.
Обеспечивает ~7x ускорение при достаточной точности для эмерджентной динамики.

ПРИНЦИПЫ:
1. Простота > точность (для эмерджентных систем)
2. Адаптивный размер шага на основе активности
3. Стабилизация через damping
4. Vectorized operations для GPU эффективности
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Dict, Any

from ...utils.logging import get_logger

logger = get_logger(__name__)


class EulerSolver(nn.Module):
    """
    3-step Euler solver для Neural ODE интеграции

    Альтернатива медленному RK4 solver'у для CNF dynamics.
    Оптимизирован для эмерджентных систем где стабильность важнее точности.
    """

    def __init__(
        self,
        adaptive_step_size: bool = True,
        base_dt: float = 0.1,
        min_dt: float = 0.01,
        max_dt: float = 0.3,
        stability_threshold: float = 10.0,
    ):
        super().__init__()

        self.adaptive_step_size = adaptive_step_size
        self.base_dt = nn.Parameter(torch.tensor(base_dt))
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.stability_threshold = stability_threshold

        # Статистика интеграции
        self.integration_stats = {
            "total_steps": 0,
            "adaptive_adjustments": 0,
            "stability_violations": 0,
        }

        logger.info(
            f"EulerSolver initialized: "
            f"adaptive={adaptive_step_size}, "
            f"base_dt={base_dt}, "
            f"dt_range=({min_dt}, {max_dt})"
        )

    def _compute_adaptive_dt(
        self, state: torch.Tensor, derivative: torch.Tensor, base_dt: float
    ) -> float:
        """
        Адаптивный размер шага на основе активности и стабильности

        Args:
            state: текущее состояние
            derivative: производная dx/dt
            base_dt: базовый размер шага

        Returns:
            adaptive_dt: адаптированный размер шага
        """
        if not self.adaptive_step_size:
            return base_dt

        # Оценка уровня активности
        state_magnitude = torch.norm(state, dim=-1).mean().item()
        derivative_magnitude = torch.norm(derivative, dim=-1).mean().item()

        # Критерий стабильности: derivative не должна быть слишком большой
        if derivative_magnitude > self.stability_threshold:
            # Уменьшаем шаг для стабильности
            stability_factor = self.stability_threshold / max(
                derivative_magnitude, 1e-8
            )
            self.integration_stats["stability_violations"] += 1
        else:
            stability_factor = 1.0

        # Активность влияет на размер шага
        activity_factor = 1.0 / (1.0 + state_magnitude + derivative_magnitude)

        # Комбинированный adaptive фактор
        adaptive_factor = min(stability_factor, activity_factor)
        adaptive_dt = base_dt * adaptive_factor

        # Ограничиваем в разумных пределах
        adaptive_dt = max(self.min_dt, min(self.max_dt, adaptive_dt))

        if abs(adaptive_dt - base_dt) > 0.01:
            self.integration_stats["adaptive_adjustments"] += 1

        return adaptive_dt

    def euler_step(
        self,
        derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Один шаг Euler интеграции: x_{n+1} = x_n + dt * f(x_n, t)

        Args:
            derivative_fn: функция для вычисления dx/dt
            state: текущее состояние
            t: текущее время
            dt: размер шага
            *args, **kwargs: дополнительные аргументы для derivative_fn

        Returns:
            next_state: состояние после интеграции
        """
        t_tensor = torch.tensor(t, device=state.device, dtype=state.dtype)
        derivative = derivative_fn(t_tensor, state, *args, **kwargs)

        # Проверка на NaN/Inf
        if torch.isnan(derivative).any() or torch.isinf(derivative).any():
            logger.warning("NaN/Inf detected in derivative, returning original state")
            return state

        next_state = state + dt * derivative

        # Проверка результата на стабильность
        if torch.isnan(next_state).any() or torch.isinf(next_state).any():
            logger.warning("NaN/Inf detected in next_state, returning original state")
            return state

        self.integration_stats["total_steps"] += 1

        return next_state

    def integrate(
        self,
        derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        integration_time: float = 1.0,
        num_steps: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Полная интеграция с multiple Euler steps

        Args:
            derivative_fn: функция для вычисления dx/dt
            initial_state: начальное состояние [batch, state_size]
            integration_time: общее время интеграции
            num_steps: количество шагов интеграции
            *args, **kwargs: аргументы для derivative_fn

        Returns:
            final_state: финальное состояние после интеграции
        """
        current_state = initial_state
        dt = integration_time / num_steps

        for step in range(num_steps):
            t = step * dt

            # Адаптивный размер шага для каждого step
            if self.adaptive_step_size and step == 0:
                # Пробная производная для оценки адаптивного dt
                t_tensor = torch.tensor(t, device=current_state.device)
                probe_derivative = derivative_fn(
                    t_tensor, current_state, *args, **kwargs
                )
                adaptive_dt = self._compute_adaptive_dt(
                    current_state, probe_derivative, dt
                )
                # Пересчитываем количество шагов
                actual_steps = max(1, int(integration_time / adaptive_dt))
                dt = integration_time / actual_steps
                num_steps = min(
                    actual_steps, num_steps
                )  # Не превышаем исходное количество

            # Выполняем Euler шаг
            current_state = self.euler_step(
                derivative_fn, current_state, t, dt, *args, **kwargs
            )

        return current_state

    def integrate_adaptive(
        self,
        derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        integration_time: float = 1.0,
        target_error: float = 1e-3,
        max_steps: int = 10,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Адаптивная интеграция с контролем ошибки

        Автоматически подбирает размер шага для достижения целевой точности.
        Более медленный но точный вариант для критических случаев.

        Args:
            derivative_fn: функция для вычисления dx/dt
            initial_state: начальное состояние
            integration_time: общее время интеграции
            target_error: целевая ошибка
            max_steps: максимальное количество шагов

        Returns:
            final_state: финальное состояние
            info: информация об интеграции
        """
        current_state = initial_state
        current_time = 0.0
        dt = self.base_dt.item()
        step_count = 0

        info = {
            "steps_taken": 0,
            "final_dt": dt,
            "error_estimate": 0.0,
            "success": True,
        }

        while current_time < integration_time and step_count < max_steps:
            remaining_time = integration_time - current_time
            dt = min(dt, remaining_time)

            # Полный шаг
            full_step_state = self.euler_step(
                derivative_fn, current_state, current_time, dt, *args, **kwargs
            )

            # Два половинных шага для оценки ошибки
            half_dt = dt / 2
            half_step1 = self.euler_step(
                derivative_fn, current_state, current_time, half_dt, *args, **kwargs
            )
            half_step2 = self.euler_step(
                derivative_fn,
                half_step1,
                current_time + half_dt,
                half_dt,
                *args,
                **kwargs,
            )

            # Оценка локальной ошибки
            error = torch.norm(full_step_state - half_step2, dim=-1).mean().item()
            info["error_estimate"] = error

            if error <= target_error or dt <= self.min_dt:
                # Принимаем шаг
                current_state = full_step_state
                current_time += dt
                step_count += 1

                # Увеличиваем dt если ошибка мала
                if error < target_error / 2:
                    dt = min(dt * 1.2, self.max_dt)
            else:
                # Уменьшаем dt и повторяем
                dt = max(dt * 0.5, self.min_dt)
                if dt == self.min_dt:
                    # Принудительно принимаем шаг с минимальным dt
                    current_state = full_step_state
                    current_time += dt
                    step_count += 1

        info["steps_taken"] = step_count
        info["final_dt"] = dt
        info["success"] = current_time >= integration_time * 0.99  # 99% покрытие

        return current_state, info

    def reset_stats(self):
        """Сброс статистики интеграции"""
        self.integration_stats = {
            "total_steps": 0,
            "adaptive_adjustments": 0,
            "stability_violations": 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return {
            "integration_stats": self.integration_stats.copy(),
            "solver_config": {
                "adaptive_step_size": self.adaptive_step_size,
                "base_dt": self.base_dt.item(),
                "min_dt": self.min_dt,
                "max_dt": self.max_dt,
                "stability_threshold": self.stability_threshold,
            },
        }


# Utility functions для удобства использования


def euler_solve(
    derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    integration_time: float = 1.0,
    num_steps: int = 3,
    adaptive: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Удобная функция для быстрого Euler solving

    Args:
        derivative_fn: функция dx/dt
        initial_state: начальное состояние
        integration_time: время интеграции
        num_steps: количество шагов
        adaptive: использовать адаптивный размер шага

    Returns:
        final_state: результат интеграции
    """
    solver = EulerSolver(adaptive_step_size=adaptive)
    return solver.integrate(
        derivative_fn, initial_state, integration_time, num_steps, *args, **kwargs
    )


def adaptive_euler_solve(
    derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    integration_time: float = 1.0,
    target_error: float = 1e-3,
    max_steps: int = 10,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Адаптивное Euler solving с контролем ошибки

    Returns:
        final_state: результат интеграции
        info: информация об интеграции
    """
    solver = EulerSolver(adaptive_step_size=True)
    return solver.integrate_adaptive(
        derivative_fn,
        initial_state,
        integration_time,
        target_error,
        max_steps,
        *args,
        **kwargs,
    )
