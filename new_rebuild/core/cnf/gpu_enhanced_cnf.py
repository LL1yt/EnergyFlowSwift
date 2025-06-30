#!/usr/bin/env python3
"""
GPU Enhanced CNF - Улучшенный CNF с GPU-оптимизированной интеграцией
===================================================================

Расширенная версия LightweightCNF с интеграцией GPU Optimized Euler Solver:
- Batch processing для множественных connections
- Vectorized Neural ODE operations
- Adaptive Lipschitz-based integration
- Memory-efficient batch operations
- Real-time performance monitoring

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
1. Batch CNF processing - обработка множественных связей одновременно
2. GPU-accelerated Neural ODE с vectorized operations
3. Adaptive step size на основе математически обоснованной Lipschitz константы
4. Memory pooling для эффективного использования GPU памяти
5. Enhanced stability analysis и error control

Автор: 3D Cellular Neural Network Project
Версия: 2.0.0 (2024-12-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List, Union
from enum import Enum
import time

try:
    from ...config import get_project_config
    from ...utils.logging import get_logger, log_cell_init, log_cell_forward
    from ...utils.device_manager import get_device_manager
    from .gpu_optimized_euler_solver import (
        GPUOptimizedEulerSolver,
        AdaptiveMethod,
        create_gpu_optimized_euler_solver,
    )

    # from .lightweight_cnf import ConnectionType  # Импорт из legacy версии
except ImportError:
    # Fallback для прямого запуска
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from config import get_project_config
    from utils.logging import get_logger, log_cell_init, log_cell_forward
    from utils.device_manager import get_device_manager

logger = get_logger(__name__)


class BatchProcessingMode(Enum):
    """Режимы batch processing для CNF"""

    SINGLE = "single"  # Одна связь за раз (legacy)
    CONNECTION_BATCH = "batch"  # Batch по связям
    ADAPTIVE_BATCH = "adaptive"  # Adaptive batch size на основе памяти


class ConnectionType(Enum):
    """Типы связей для CNF обработки"""

    FUNCTIONAL = "functional"  # 60% связей - средние расстояния
    DISTANT = "distant"  # 30% связей - дальние расстояния


class VectorizedNeuralODE(nn.Module):
    """
    Vectorized Neural ODE для GPU-эффективной обработки

    Поддерживает batch processing множественных connections
    с shared parameters но разными входными данными.
    """

    def __init__(
        self,
        state_size: int,
        connection_type: ConnectionType,
        hidden_dim: Optional[int] = None,
        batch_size: int = 100,
    ):
        super().__init__()

        self.state_size = state_size
        self.connection_type = connection_type
        self.hidden_dim = hidden_dim or max(16, state_size // 2)
        self.max_batch_size = batch_size

        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Размер входа: собственное состояние + агрегированные соседи
        input_size = state_size * 2

        # Компактная но мощная архитектура для vectorized operations
        self.ode_network = nn.Sequential(
            nn.Linear(input_size, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(0.1),  # Небольшая regularization
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, state_size, bias=True),
        )

        # Learnable damping для стабилизации
        self.damping_strength = nn.Parameter(torch.tensor(0.1))

        # Time embedding для time-dependent dynamics
        self.time_embedding = nn.Linear(1, self.hidden_dim // 4, bias=False)

        # Normalization layers
        self.input_norm = nn.LayerNorm(input_size)
        self.output_norm = nn.LayerNorm(state_size)

        total_params = sum(p.numel() for p in self.parameters())
        log_cell_init(
            cell_type="VectorizedNeuralODE",
            total_params=total_params,
            target_params=3000,  # Оптимизировано для GPU efficiency
            state_size=state_size,
            connection_type=connection_type.value,
            hidden_dim=self.hidden_dim,
            input_size=input_size,
            max_batch_size=batch_size,
        )
        self.to(self.device)

    def forward(
        self,
        t: torch.Tensor,
        current_states: torch.Tensor,
        neighbor_influences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized вычисление производной dx/dt для batch connections

        Args:
            t: время [batch] или scalar
            current_states: [batch, state_size] - текущие состояния
            neighbor_influences: [batch, state_size] - влияние соседей

        Returns:
            derivatives: [batch, state_size] - dx/dt для каждой связи
        """
        batch_size = current_states.shape[0]

        # Убеждаемся что все на правильном устройстве
        current_states = self.device_manager.ensure_device(current_states)
        neighbor_influences = self.device_manager.ensure_device(neighbor_influences)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=current_states.dtype)
        elif t.device != self.device:
            t = t.to(self.device)

        # Ensure t has batch dimension
        if t.dim() == 0:
            t = t.expand(batch_size)
        elif t.shape[0] != batch_size:
            t = t.expand(batch_size)

        # Объединяем входы
        combined_input = torch.cat([current_states, neighbor_influences], dim=-1)

        # Input normalization для стабильности
        normalized_input = self.input_norm(combined_input)

        # Time embedding
        time_features = self.time_embedding(t.unsqueeze(-1))  # [batch, time_features]

        # Основная ODE network
        ode_output = self.ode_network(normalized_input)

        # Добавляем time features через residual connection
        if time_features.shape[-1] == ode_output.shape[-1]:
            ode_output = ode_output + time_features
        elif time_features.shape[-1] < ode_output.shape[-1]:
            # Pad time features
            padding = ode_output.shape[-1] - time_features.shape[-1]
            time_features_padded = F.pad(time_features, (0, padding))
            ode_output = ode_output + time_features_padded

        # Output normalization
        normalized_output = self.output_norm(ode_output)

        # Damping term для стабилизации: -λ * x
        damping_term = -self.damping_strength.abs() * current_states

        # Итоговая производная
        derivatives = normalized_output + damping_term

        return derivatives


class GPUEnhancedCNF(nn.Module):
    """
    GPU Enhanced CNF - Улучшенная версия LightweightCNF

    Ключевые улучшения:
    - Batch processing для множественных connections
    - GPU-оптимизированная интеграция с Lipschitz-based adaptive stepping
    - Vectorized Neural ODE operations
    - Memory-efficient batch operations
    - Real-time performance monitoring
    """

    def __init__(
        self,
        state_size: int,
        connection_type: ConnectionType = ConnectionType.DISTANT,
        integration_steps: int = 3,
        batch_processing_mode: BatchProcessingMode = BatchProcessingMode.ADAPTIVE_BATCH,
        max_batch_size: int = 100,
        adaptive_method: AdaptiveMethod = AdaptiveMethod.LIPSCHITZ_BASED,
    ):
        super().__init__()

        self.state_size = state_size
        self.connection_type = connection_type
        self.integration_steps = integration_steps
        self.batch_processing_mode = batch_processing_mode
        self.max_batch_size = max_batch_size

        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()

        # Vectorized Neural ODE
        self.neural_ode = VectorizedNeuralODE(
            state_size=state_size,
            connection_type=connection_type,
            batch_size=max_batch_size,
        )

        # GPU Optimized Euler Solver
        euler_cfg = get_project_config().euler
        self.solver = GPUOptimizedEulerSolver(config=euler_cfg)

        # Performance tracking
        self.performance_stats = {
            "total_forward_passes": 0,
            "total_connections_processed": 0,
            "avg_batch_size": 0.0,
            "avg_processing_time_ms": 0.0,
            "batch_efficiency": 0.0,
            "gpu_memory_usage_mb": 0.0,
        }

        total_params = sum(p.numel() for p in self.parameters())

        log_cell_init(
            cell_type="GPUEnhancedCNF",
            total_params=total_params,
            target_params=get_project_config().cnf.target_params_per_connection,
            state_size=state_size,
            connection_type=(
                connection_type.value
                if isinstance(connection_type, Enum)
                else connection_type
            ),
            integration_steps=integration_steps,
            batch_mode=(
                batch_processing_mode.value
                if isinstance(batch_processing_mode, Enum)
                else batch_processing_mode
            ),
            max_batch_size=max_batch_size,
            adaptive_method=(
                adaptive_method.value
                if isinstance(adaptive_method, Enum)
                else adaptive_method
            ),
        )

    def _create_derivative_function(self, neighbor_influences: torch.Tensor):
        """Создает derivative function для solver'а"""

        def derivative_fn(
            t: torch.Tensor, states: torch.Tensor, *args, **kwargs
        ) -> torch.Tensor:
            return self.neural_ode(t, states, neighbor_influences)

        return derivative_fn

    def _process_single_connection(
        self, current_state: torch.Tensor, neighbor_states: torch.Tensor
    ) -> torch.Tensor:
        """Обработка одной связи (один-ко-многим)"""
        start_time = time.time()

        # 1. Агрегируем влияние соседей
        if neighbor_states.dim() == 3:
            # Batched input: [batch, num_neighbors, state_size] -> [batch, state_size]
            aggregated_influence = torch.mean(neighbor_states, dim=1)
        else:
            # Non-batched: [num_neighbors, state_size] -> [state_size]
            aggregated_influence = torch.mean(neighbor_states, dim=0)

        # 2. Подготавливаем начальное состояние для интегратора
        if current_state.dim() == 1:
            initial_state = current_state.unsqueeze(0)  # [1, state_size]
        else:
            initial_state = current_state  # Already [batch, state_size]

        if aggregated_influence.dim() == 1:
            aggregated_influence = aggregated_influence.unsqueeze(0)  # Ensure batch dim

        # 3. Создаем функцию производной с зафиксированным влиянием соседей
        derivative_fn = self._create_derivative_function(
            aggregated_influence  # Shape: [batch, state_size]
        )

        # 4. Выполняем интегрирование
        result = self.solver.batch_integrate(
            derivative_fn,
            initial_state,
            t_span=(0.0, 1.0),
            num_steps=self.integration_steps,
            return_trajectory=True,
            adaptive_method=self.solver.config.adaptive_method,
        )

        # Возвращаем последнее состояние из траектории
        final_state = result.trajectory[-1]

        processing_time_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Processed single connection in {processing_time_ms:.2f}ms. "
            f"Input shape: {current_state.shape}, Neighbor shape: {neighbor_states.shape}"
        )

        return final_state

    def _process_connection_batch(
        self, current_states: torch.Tensor, neighbor_states_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Batch processing для множественных connections"""
        batch_size = len(neighbor_states_list)

        # Агрегируем neighbor influences для всего batch'а
        neighbor_influences = []
        for neighbor_states in neighbor_states_list:
            if neighbor_states.shape[0] > 0:
                neighbor_influence = neighbor_states.mean(dim=0)
            else:
                neighbor_influence = torch.zeros(self.state_size, device=self.device)
            neighbor_influences.append(neighbor_influence)

        neighbor_influences = torch.stack(neighbor_influences)  # [batch, state_size]

        # Создаем derivative function для batch
        derivative_fn = self._create_derivative_function(neighbor_influences)

        # Batch интеграция
        result = self.solver.batch_integrate(
            derivative_fn,
            current_states,
            t_span=(0.0, 1.0),
            num_steps=self.integration_steps,
        )

        return result

    def _determine_optimal_batch_size(self, total_connections: int) -> int:
        """Определяет оптимальный размер batch'а на основе доступной памяти"""
        if self.batch_processing_mode == BatchProcessingMode.SINGLE:
            return 1
        elif self.batch_processing_mode == BatchProcessingMode.CONNECTION_BATCH:
            return min(total_connections, self.max_batch_size)
        else:  # ADAPTIVE_BATCH
            # Оценка доступной памяти
            device_stats = self.device_manager.get_memory_stats()
            available_mb = device_stats.get("available_mb", 1000)

            # Эвристическая оценка memory per connection
            memory_per_connection = (
                self.state_size * 4 * 10 / (1024**2)
            )  # очень грубая оценка

            max_affordable = int(
                available_mb * 0.5 / memory_per_connection
            )  # 50% от доступной памяти
            optimal_batch_size = min(
                total_connections, self.max_batch_size, max_affordable
            )

            return max(1, optimal_batch_size)

    def forward(
        self,
        current_state: torch.Tensor,
        neighbor_states: torch.Tensor,
        cell_idx: Optional[int] = None,
        neighbor_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass с поддержкой различных batch modes

        Args:
            current_state: [1, state_size] или [batch, state_size] - текущие состояния
            neighbor_states: [num_neighbors, state_size] или List[Tensor] для batch mode
            cell_idx: индекс клетки (для совместимости)
            neighbor_indices: индексы соседей (для совместимости)

        Returns:
            dict с new_state и дополнительной информацией
        """
        start_time = time.time()

        # Убеждаемся что inputs на правильном устройстве
        current_state = self.device_manager.ensure_device(current_state)

        # Определяем режим обработки
        if isinstance(neighbor_states, list) or (
            isinstance(neighbor_states, torch.Tensor) and neighbor_states.dim() == 3
        ):

            if isinstance(neighbor_states, torch.Tensor):
                # Для single-mode, но с batched-like tensor
                batch_size = neighbor_states.shape[0]
            else:
                # Для batch mode
                batch_size = len(neighbor_states)

            if current_state.dim() == 1:
                current_state = current_state.unsqueeze(0)

            if current_state.shape[0] == 1 and batch_size > 1:
                current_state = current_state.expand(batch_size, -1)

            optimal_batch_size = self._determine_optimal_batch_size(batch_size)

            if optimal_batch_size >= batch_size:
                # Обрабатываем весь batch сразу
                integration_result = self._process_connection_batch(
                    current_state, neighbor_states
                )
                new_states = integration_result.final_state

            else:
                # Chunked processing
                new_states = []
                for i in range(0, batch_size, optimal_batch_size):
                    end_idx = min(i + optimal_batch_size, batch_size)
                    batch_current = current_state[i:end_idx]
                    batch_neighbors = (
                        neighbor_states[i:end_idx]
                        if isinstance(neighbor_states, list)
                        else neighbor_states[:, i:end_idx]
                    )

                    batch_result = self._process_connection_batch(
                        batch_current, batch_neighbors
                    )
                    new_states.append(batch_result.final_state)

                new_states = torch.cat(new_states, dim=0)

            # Для batch mode возвращаем весь batch
            result_state = new_states

        else:
            # Single connection mode (legacy compatibility)
            neighbor_states = self.device_manager.ensure_device(neighbor_states)
            result_state = self._process_single_connection(
                current_state, neighbor_states
            )

        processing_time_ms = (time.time() - start_time) * 1000

        # Обновляем статистику
        self._update_performance_stats(processing_time_ms, current_state.shape[0])

        # Логирование
        if hasattr(self, "_log_forward_passes") and self._log_forward_passes:
            log_cell_forward(
                cell_type="GPUEnhancedCNF",
                input_shape=current_state.shape,
                output_shape=result_state.shape,
                processing_time_ms=processing_time_ms,
                connection_type=self.connection_type.value,
                batch_size=current_state.shape[0],
            )

        return {
            "new_state": result_state,
            "processing_time_ms": processing_time_ms,
            "connection_type": self.connection_type.value,
            "batch_size": current_state.shape[0],
        }

    def _update_performance_stats(self, processing_time_ms: float, batch_size: int):
        """Обновляет статистику производительности"""
        self.performance_stats["total_forward_passes"] += 1
        self.performance_stats["total_connections_processed"] += batch_size

        # Обновляем средние значения
        total_passes = self.performance_stats["total_forward_passes"]
        old_avg_time = self.performance_stats["avg_processing_time_ms"]
        new_avg_time = (
            old_avg_time * (total_passes - 1) + processing_time_ms
        ) / total_passes
        self.performance_stats["avg_processing_time_ms"] = new_avg_time

        # Средний размер batch'а
        total_connections = self.performance_stats["total_connections_processed"]
        self.performance_stats["avg_batch_size"] = total_connections / total_passes

        # Batch efficiency (connections per second)
        self.performance_stats["batch_efficiency"] = batch_size / (
            processing_time_ms / 1000
        )

        # GPU memory usage
        device_stats = self.device_manager.get_memory_stats()
        self.performance_stats["gpu_memory_usage_mb"] = device_stats.get(
            "allocated_mb", 0
        )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Получить полную статистику производительности"""
        solver_stats = self.solver.get_comprehensive_stats()

        return {
            "cnf_config": {
                "state_size": self.state_size,
                "connection_type": self.connection_type.value,
                "integration_steps": self.integration_steps,
                "batch_mode": self.batch_processing_mode.value,
                "max_batch_size": self.max_batch_size,
            },
            "cnf_performance": self.performance_stats.copy(),
            "solver_stats": solver_stats,
            "neural_ode": {
                "total_params": sum(p.numel() for p in self.neural_ode.parameters()),
                "hidden_dim": self.neural_ode.hidden_dim,
                "max_batch_size": self.neural_ode.max_batch_size,
            },
        }

    def optimize_performance(self):
        """Принудительная оптимизация производительности"""
        logger.info("🔧 Оптимизация GPU Enhanced CNF")

        # Оптимизируем solver
        self.solver.optimize_performance()

        # Сбрасываем некоторую статистику
        self.performance_stats["gpu_memory_usage_mb"] = 0.0

        logger.info("✅ CNF оптимизация завершена")

    def cleanup(self):
        """Освобождение ресурсов"""
        logger.info("🛑 Cleanup GPU Enhanced CNF")
        self.solver.cleanup()


# === FACTORY FUNCTIONS ===


def create_gpu_enhanced_cnf(
    state_size: int,
    connection_type: ConnectionType = ConnectionType.DISTANT,
    batch_processing_mode: BatchProcessingMode = BatchProcessingMode.ADAPTIVE_BATCH,
    max_batch_size: int = 100,
    adaptive_method: AdaptiveMethod = AdaptiveMethod.LIPSCHITZ_BASED,
) -> GPUEnhancedCNF:
    """
    Фабричная функция для создания GPU Enhanced CNF

    Args:
        state_size: размер состояния
        connection_type: тип связей
        batch_processing_mode: режим batch processing
        max_batch_size: максимальный размер batch'а
        adaptive_method: метод адаптации solver'а

    Returns:
        Настроенный GPUEnhancedCNF
    """
    config = get_project_config()

    return GPUEnhancedCNF(
        state_size=state_size,
        connection_type=connection_type,
        integration_steps=config.cnf_integration_steps,
        batch_processing_mode=batch_processing_mode,
        max_batch_size=max_batch_size,
        adaptive_method=adaptive_method,
    )


def benchmark_cnf_performance(
    state_sizes: List[int] = [16, 32, 64],
    batch_sizes: List[int] = [1, 10, 50, 100],
    num_trials: int = 5,
) -> Dict[str, Any]:
    """
    Бенчмарк производительности GPU Enhanced CNF

    Returns:
        Детальная статистика производительности
    """
    results = {}

    for state_size in state_sizes:
        for batch_size in batch_sizes:
            logger.info(
                f"🧪 Бенчмарк CNF: state_size={state_size}, batch_size={batch_size}"
            )

            trial_results = []

            for trial in range(num_trials):
                # Создаем CNF
                cnf = create_gpu_enhanced_cnf(
                    state_size=state_size,
                    batch_processing_mode=BatchProcessingMode.CONNECTION_BATCH,
                    max_batch_size=max(100, batch_size),
                )

                # Тестовые данные
                current_states = torch.randn(batch_size, state_size)
                neighbor_states_list = [
                    torch.randn(torch.randint(5, 20, (1,)).item(), state_size)
                    for _ in range(batch_size)
                ]

                # Выполняем forward pass
                start_time = time.time()
                result = cnf(current_states, neighbor_states_list)
                wall_time = time.time() - start_time

                trial_results.append(
                    {
                        "wall_time_s": wall_time,
                        "processing_time_ms": result["processing_time_ms"],
                        "batch_size": result["batch_size"],
                    }
                )

                cnf.cleanup()

            # Агрегируем результаты
            avg_wall_time = sum(r["wall_time_s"] for r in trial_results) / num_trials
            avg_processing_time = (
                sum(r["processing_time_ms"] for r in trial_results) / num_trials
            )

            key = f"state_{state_size}_batch_{batch_size}"
            results[key] = {
                "state_size": state_size,
                "batch_size": batch_size,
                "avg_wall_time_s": avg_wall_time,
                "avg_processing_time_ms": avg_processing_time,
                "throughput_connections_per_second": batch_size / avg_wall_time,
            }

    return results
