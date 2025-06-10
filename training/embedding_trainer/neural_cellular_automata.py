#!/usr/bin/env python3
"""
🧠 Phase 3 Task 3.1: Neural Cellular Automata Patterns

Реализация emergent behavior preservation во время GPU-optimized training.

Ключевые принципы NCA для 3D Cellular Neural Network:
1. Stochastic Cell Updates - избежание глобальной синхронизации
2. Residual Update Rules - маленькие, стабильные модификации
3. Pattern Formation Metrics - количественная оценка emergence
4. Emergent Behavior Preservation - сохранение паттернов при оптимизации

Исследовательская база:
"Emergent Training Architecture for 3D Cellular Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import random
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NCAConfig:
    """Конфигурация Neural Cellular Automata паттернов"""

    # Stochastic update settings
    update_probability: float = 0.7  # Вероятность обновления клетки
    stochastic_scheduling: bool = True
    synchronization_avoidance: bool = True

    # Residual update settings
    residual_learning_rate: float = 0.1  # Для residual updates
    stability_threshold: float = 0.01  # Пороговое значение для stability
    max_update_magnitude: float = 0.5  # Максимальная величина изменения

    # Pattern formation settings
    pattern_detection_enabled: bool = True
    spatial_coherence_weight: float = 0.3
    temporal_consistency_weight: float = 0.2

    # Emergent metrics settings
    track_emergent_specialization: bool = True
    specialization_threshold: float = 0.15
    diversity_preservation_weight: float = 0.25


class StochasticCellUpdater(nn.Module):
    """
    Stochastic cell updating система для избежания глобальной синхронизации

    Принципы:
    - Не все клетки обновляются одновременно
    - Асинхронный update schedule
    - Preservation emergent patterns
    """

    def __init__(self, config: NCAConfig, cube_dimensions: Tuple[int, int, int]):
        super().__init__()
        self.config = config
        self.cube_dimensions = cube_dimensions
        self.total_cells = cube_dimensions[0] * cube_dimensions[1] * cube_dimensions[2]

        # Stochastic update mask generation
        self.update_probability = config.update_probability

        # Register update schedule tracking
        self.register_buffer(
            "update_history", torch.zeros(self.total_cells, dtype=torch.float32)
        )
        self.register_buffer(
            "last_update_step", torch.zeros(self.total_cells, dtype=torch.long)
        )

        logger.info(
            f"🎲 StochasticCellUpdater initialized: {self.total_cells} cells, p={self.update_probability}"
        )

    def generate_update_mask(
        self, batch_size: int, current_step: int = 0
    ) -> torch.Tensor:
        """
        Генерирует стохастическую маску для обновления клеток

        Args:
            batch_size: размер batch
            current_step: текущий шаг обучения

        Returns:
            update_mask: [batch, depth, height, width] Boolean mask
        """
        # CRITICAL FIX: cube_dimensions from EmergentCubeTrainer is (width, height, depth)
        # But tensors are in format [batch, depth, height, width, state_size]
        # So we need to map correctly: cube_dimensions[2] = depth, cube_dimensions[1] = height, cube_dimensions[0] = width
        width, height, depth = self.cube_dimensions
        # Tensor format: [batch, depth, height, width, state_size]
        # cube_dimensions: (width=15, height=15, depth=11)

        if self.config.stochastic_scheduling:
            # Stochastic mask с балансировкой для equal opportunity updates
            prob_matrix = torch.full(
                (batch_size, depth, height, width),
                self.update_probability,
                dtype=torch.float32,
                device=self.update_history.device,
            )

            # Adjust probability based on update history (encourage underused cells)
            if current_step > 0:
                update_ratios = self.update_history / max(current_step, 1)
                # Boost probability for cells that haven't been updated recently
                boost_factor = 1.0 - update_ratios.view(depth, height, width)
                prob_matrix = prob_matrix * (1.0 + 0.3 * boost_factor.unsqueeze(0))
                prob_matrix = torch.clamp(
                    prob_matrix, 0.1, 0.9
                )  # Keep reasonable bounds

            update_mask = torch.bernoulli(prob_matrix).bool()
        else:
            # Fallback: deterministic mask
            update_mask = torch.ones(
                batch_size,
                depth,
                height,
                width,
                dtype=torch.bool,
                device=self.update_history.device,
            )

        # Update tracking
        if current_step > 0:
            mask_sum = update_mask[0].float()  # Use first batch element for tracking
            self.update_history += mask_sum.view(-1)
            self.last_update_step[update_mask[0].view(-1)] = current_step

        return update_mask

    def forward(
        self, cell_states: torch.Tensor, new_states: torch.Tensor, current_step: int = 0
    ) -> torch.Tensor:
        """
        Применяет stochastic update к клеткам

        Args:
            cell_states: текущие состояния клеток [batch, depth, height, width, state_size]
            new_states: новые computed состояния [batch, depth, height, width, state_size]
            current_step: текущий шаг для tracking

        Returns:
            updated_states: обновленные состояния с stochastic masking
        """
        batch_size = cell_states.size(0)

        # Generate update mask
        update_mask = self.generate_update_mask(batch_size, current_step)

        # Apply stochastic updates
        # update_mask: [batch, depth, height, width] -> [batch, depth, height, width, 1]
        update_mask_expanded = update_mask.unsqueeze(-1).expand_as(cell_states)

        updated_states = torch.where(update_mask_expanded, new_states, cell_states)

        return updated_states

    def get_update_statistics(self) -> Dict[str, float]:
        """Получить статистику обновлений для monitoring"""
        if self.update_history.sum() == 0:
            return {"avg_updates": 0.0, "min_updates": 0.0, "max_updates": 0.0}

        return {
            "avg_updates": self.update_history.mean().item(),
            "min_updates": self.update_history.min().item(),
            "max_updates": self.update_history.max().item(),
            "update_variance": self.update_history.var().item(),
        }


class ResidualUpdateRules(nn.Module):
    """
    Residual update rules для стабильных модификаций состояний клеток

    Принципы:
    - Маленькие, контролируемые изменения
    - Stability preservation
    - Gradient flow optimization
    """

    def __init__(self, config: NCAConfig, state_size: int = 32):
        super().__init__()
        self.config = config
        self.state_size = state_size

        # Residual update parameters
        self.residual_lr = config.residual_learning_rate
        self.stability_threshold = config.stability_threshold
        self.max_magnitude = config.max_update_magnitude

        # Learnable residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_lr))

        # Stability gate for controlling update magnitudes
        self.stability_gate = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.GELU(),
            nn.Linear(state_size // 2, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"🔄 ResidualUpdateRules initialized: state_size={state_size}, lr={self.residual_lr}"
        )

    def forward(
        self, current_states: torch.Tensor, raw_updates: torch.Tensor
    ) -> torch.Tensor:
        """
        Применяет residual update rules

        Args:
            current_states: текущие состояния [batch, ..., state_size]
            raw_updates: сырые обновления от gMLP [batch, ..., state_size]

        Returns:
            refined_states: состояния с residual updates
        """
        # Compute residual delta
        residual_delta = raw_updates - current_states

        # Apply stability gating
        stability_scores = self.stability_gate(current_states)  # [batch, ..., 1]

        # Scale residual based on stability and learnable parameter
        scaled_residual = residual_delta * self.residual_scale * stability_scores

        # Clamp magnitude для предотвращения unstable updates
        residual_magnitude = torch.norm(scaled_residual, dim=-1, keepdim=True)
        magnitude_mask = residual_magnitude > self.max_magnitude

        if magnitude_mask.any():
            # Normalize large updates
            scaling_factor = self.max_magnitude / (residual_magnitude + 1e-8)
            scaling_factor = torch.where(
                magnitude_mask, scaling_factor, torch.ones_like(scaling_factor)
            )
            scaled_residual = scaled_residual * scaling_factor

        # Apply residual update
        refined_states = current_states + scaled_residual

        return refined_states

    def get_update_magnitude_stats(
        self, current_states: torch.Tensor, updates: torch.Tensor
    ) -> Dict[str, float]:
        """Статистика magnitude обновлений для monitoring"""
        with torch.no_grad():
            residual_delta = updates - current_states
            magnitudes = torch.norm(residual_delta, dim=-1)

            return {
                "mean_magnitude": magnitudes.mean().item(),
                "max_magnitude": magnitudes.max().item(),
                "std_magnitude": magnitudes.std().item(),
                "stability_score": self.stability_gate(current_states).mean().item(),
            }


class PatternFormationMetrics(nn.Module):
    """
    Метрики для обнаружения и количественной оценки emergent patterns

    Отслеживает:
    - Spatial coherence patterns
    - Temporal consistency
    - Emergent specialization
    - Pattern diversity
    """

    def __init__(self, config: NCAConfig, cube_dimensions: Tuple[int, int, int]):
        super().__init__()
        self.config = config
        self.cube_dimensions = cube_dimensions
        # FIXED: cube_dimensions from EmergentCubeTrainer is (width, height, depth)
        self.width, self.height, self.depth = cube_dimensions

        # Pattern detection settings
        self.track_specialization = config.track_emergent_specialization
        self.specialization_threshold = config.specialization_threshold

        # Running statistics для temporal tracking
        self.register_buffer(
            "pattern_history", torch.zeros(100, dtype=torch.float32)
        )  # Last 100 steps
        self.register_buffer(
            "specialization_history", torch.zeros(100, dtype=torch.float32)
        )
        self.step_counter = 0

        logger.info(f"📊 PatternFormationMetrics initialized: {cube_dimensions} cube")

    def compute_spatial_coherence(self, cube_states: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет spatial coherence patterns в кубе

        Args:
            cube_states: [batch, depth, height, width, state_size]

        Returns:
            coherence_score: spatial coherence metric
        """
        batch_size = cube_states.size(0)

        # Compute pairwise similarities between neighboring cells
        coherence_scores = []

        # Check all 6-connectivity neighbors
        directions = [
            (1, 0, 0),
            (-1, 0, 0),  # depth
            (0, 1, 0),
            (0, -1, 0),  # height
            (0, 0, 1),
            (0, 0, -1),  # width
        ]

        for dz, dy, dx in directions:
            # Extract shifted versions для neighbor comparison
            if (
                0 <= dz + self.depth - 1 < self.depth
                and 0 <= dy + self.height - 1 < self.height
                and 0 <= dx + self.width - 1 < self.width
            ):

                # Create neighbor slices
                if dz != 0:
                    if dz > 0:
                        current_slice = cube_states[:, :-dz, :, :, :]
                        neighbor_slice = cube_states[:, dz:, :, :, :]
                    else:
                        current_slice = cube_states[:, -dz:, :, :, :]
                        neighbor_slice = cube_states[:, :dz, :, :, :]
                elif dy != 0:
                    if dy > 0:
                        current_slice = cube_states[:, :, :-dy, :, :]
                        neighbor_slice = cube_states[:, :, dy:, :, :]
                    else:
                        current_slice = cube_states[:, :, -dy:, :, :]
                        neighbor_slice = cube_states[:, :, :dy, :, :]
                else:  # dx != 0
                    if dx > 0:
                        current_slice = cube_states[:, :, :, :-dx, :]
                        neighbor_slice = cube_states[:, :, :, dx:, :]
                    else:
                        current_slice = cube_states[:, :, :, -dx:, :]
                        neighbor_slice = cube_states[:, :, :, :dx, :]

                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    current_slice.flatten(start_dim=1, end_dim=-2),
                    neighbor_slice.flatten(start_dim=1, end_dim=-2),
                    dim=-1,
                )
                coherence_scores.append(similarity.mean())

        # Average coherence across all directions
        if coherence_scores:
            spatial_coherence = torch.stack(coherence_scores).mean()
        else:
            # Fallback: если нет valid neighbors, возвращаем нейтральное значение
            logger.warning(
                "⚠️ [NCA] No valid neighbors found for spatial coherence computation"
            )
            spatial_coherence = torch.tensor(
                0.5, device=cube_states.device
            )  # Нейтральное значение

        return spatial_coherence

    def compute_emergent_specialization(
        self, cube_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет emergent specialization score

        Высокая специализация = клетки в разных регионах имеют distinct patterns
        """
        batch_size = cube_states.size(0)

        # Разделяем куб на regions для analysis
        mid_depth, mid_height, mid_width = (
            self.depth // 2,
            self.height // 2,
            self.width // 2,
        )

        regions = {
            "front": cube_states[:, :mid_depth, :, :, :],
            "back": cube_states[:, mid_depth:, :, :, :],
            "top": cube_states[:, :, :mid_height, :, :],
            "bottom": cube_states[:, :, mid_height:, :, :],
            "left": cube_states[:, :, :, :mid_width, :],
            "right": cube_states[:, :, :, mid_width:, :],
        }

        # Compute region signatures (mean activations)
        region_signatures = {}
        for region_name, region_states in regions.items():
            region_signatures[region_name] = region_states.mean(
                dim=(1, 2, 3)
            )  # [batch, state_size]

        # Compute pairwise distances между regions
        specialization_scores = []
        region_names = list(region_signatures.keys())

        for i in range(len(region_names)):
            for j in range(i + 1, len(region_names)):
                sig1 = region_signatures[region_names[i]]
                sig2 = region_signatures[region_names[j]]

                # Cosine distance (1 - cosine similarity)
                distance = 1.0 - F.cosine_similarity(sig1, sig2, dim=-1).mean()
                specialization_scores.append(distance)

        # Average specialization
        if specialization_scores:
            specialization = torch.stack(specialization_scores).mean()
        else:
            # Fallback: если нет valid regions, возвращаем нейтральное значение
            logger.warning(
                "⚠️ [NCA] No valid regions found for specialization computation"
            )
            specialization = torch.tensor(
                0.1, device=cube_states.device
            )  # Низкая специализация по умолчанию

        return specialization

    def forward(self, cube_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all pattern formation metrics

        Args:
            cube_states: [batch, depth, height, width, state_size]

        Returns:
            metrics: Dictionary of pattern metrics
        """
        metrics = {}

        # Spatial coherence
        if self.config.spatial_coherence_weight > 0:
            spatial_coherence = self.compute_spatial_coherence(cube_states)
            metrics["spatial_coherence"] = spatial_coherence

        # Emergent specialization
        if self.track_specialization:
            specialization = self.compute_emergent_specialization(cube_states)
            metrics["emergent_specialization"] = specialization

            # Update history для temporal tracking
            if self.step_counter < 100:
                self.specialization_history[self.step_counter] = specialization.item()
            else:
                # Rolling window
                self.specialization_history[:-1] = self.specialization_history[1:]
                self.specialization_history[-1] = specialization.item()

        # Temporal consistency (based on history)
        if self.step_counter > 5:  # Need some history
            recent_specialization = self.specialization_history[
                max(0, self.step_counter - 5) : self.step_counter
            ]
            temporal_consistency = 1.0 - torch.std(recent_specialization) / (
                torch.mean(recent_specialization) + 1e-8
            )
            metrics["temporal_consistency"] = temporal_consistency

        self.step_counter += 1

        return metrics

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Получить summary pattern statistics для monitoring"""
        if self.step_counter == 0:
            return {"status": "no_data"}

        recent_steps = min(self.step_counter, 20)
        recent_specialization = self.specialization_history[
            max(0, self.step_counter - recent_steps) : self.step_counter
        ]

        # Безопасный расчет std() - защита от warning
        if len(recent_specialization) <= 1:
            specialization_stability = (
                1.0  # Максимальная стабильность при недостатке данных
            )
        else:
            specialization_std = recent_specialization.std().item()
            specialization_stability = 1.0 / (1.0 + specialization_std)

        return {
            "current_specialization": (
                self.specialization_history[self.step_counter - 1].item()
                if self.step_counter > 0
                else 0.0
            ),
            "avg_specialization": recent_specialization.mean().item(),
            "specialization_trend": (
                (
                    recent_specialization[-5:].mean() - recent_specialization[:5].mean()
                ).item()
                if recent_steps >= 10
                else 0.0
            ),
            "specialization_stability": specialization_stability,
            "steps_tracked": self.step_counter,
        }


class NeuralCellularAutomata(nn.Module):
    """
    Главный модуль Neural Cellular Automata для Phase 3 Task 3.1

    Интегрирует:
    - StochasticCellUpdater для асинхронных обновлений
    - ResidualUpdateRules для стабильности
    - PatternFormationMetrics для мониторинга emergence
    """

    def __init__(
        self,
        config: NCAConfig,
        cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
        state_size: int = 32,
    ):
        super().__init__()
        self.config = config
        self.cube_dimensions = cube_dimensions
        self.state_size = state_size

        # Core NCA components
        self.stochastic_updater = StochasticCellUpdater(config, cube_dimensions)
        self.residual_rules = ResidualUpdateRules(config, state_size)

        # Pattern analysis
        if config.pattern_detection_enabled:
            self.pattern_metrics = PatternFormationMetrics(config, cube_dimensions)
        else:
            self.pattern_metrics = None

        # Training step counter
        self.training_step = 0

        logger.info(
            f"🧠 NeuralCellularAutomata initialized: {cube_dimensions} cube, {state_size}D states"
        )

    def forward(
        self,
        current_states: torch.Tensor,
        raw_updates: torch.Tensor,
        enable_stochastic: bool = True,
        enable_residual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply NCA rules to cell state updates

        Args:
            current_states: текущие состояния [batch, depth, height, width, state_size]
            raw_updates: raw updates from gMLP cells [batch, depth, height, width, state_size]
            enable_stochastic: применять stochastic updates
            enable_residual: применять residual rules

        Returns:
            results: Dictionary with updated states and metrics
        """
        # КРИТИЧЕСКАЯ ПРОВЕРКА: Защита от None inputs
        if current_states is None:
            logger.error("❌ [NCA] current_states is None!")
            raise ValueError("NCA received None for current_states")

        if raw_updates is None:
            logger.error("❌ [NCA] raw_updates is None!")
            raise ValueError("NCA received None for raw_updates")

        # ДИАГНОСТИКА: Проверяем размеры
        logger.debug(f"🔍 [NCA] current_states: {current_states.shape}")
        logger.debug(f"🔍 [NCA] raw_updates: {raw_updates.shape}")

        if current_states.shape != raw_updates.shape:
            logger.error(
                f"❌ [NCA] Shape mismatch: current_states {current_states.shape} vs raw_updates {raw_updates.shape}"
            )
            raise ValueError(
                f"Shape mismatch in NCA inputs: {current_states.shape} vs {raw_updates.shape}"
            )

        results = {}

        # 1. Apply residual update rules (если enabled)
        if enable_residual:
            refined_updates = self.residual_rules(current_states, raw_updates)
            results["residual_stats"] = self.residual_rules.get_update_magnitude_stats(
                current_states, raw_updates
            )
        else:
            refined_updates = raw_updates

        # 2. Apply stochastic cell updates (если enabled)
        if enable_stochastic:
            final_states = self.stochastic_updater(
                current_states, refined_updates, self.training_step
            )
            results["update_stats"] = self.stochastic_updater.get_update_statistics()
        else:
            final_states = refined_updates

        # 3. Compute pattern formation metrics (если enabled)
        if self.pattern_metrics is not None:
            pattern_metrics = self.pattern_metrics(final_states)
            results["pattern_metrics"] = pattern_metrics
            results["pattern_summary"] = self.pattern_metrics.get_pattern_summary()

        # 4. Main result
        results["updated_states"] = final_states
        results["state_change_magnitude"] = torch.norm(
            final_states - current_states, dim=-1
        ).mean()

        self.training_step += 1

        return results

    def reset_tracking(self):
        """Reset tracking statistics (для нового training run)"""
        self.training_step = 0
        if hasattr(self.stochastic_updater, "update_history"):
            self.stochastic_updater.update_history.zero_()
            self.stochastic_updater.last_update_step.zero_()

        if self.pattern_metrics is not None:
            self.pattern_metrics.pattern_history.zero_()
            self.pattern_metrics.specialization_history.zero_()
            self.pattern_metrics.step_counter = 0

    def get_nca_summary(self) -> Dict[str, Any]:
        """Получить comprehensive summary NCA статистики"""
        summary = {
            "training_step": self.training_step,
            "config": {
                "update_probability": self.config.update_probability,
                "residual_lr": self.config.residual_learning_rate,
                "pattern_detection": self.config.pattern_detection_enabled,
            },
        }

        # Add component summaries
        if hasattr(self.stochastic_updater, "get_update_statistics"):
            summary["stochastic_stats"] = (
                self.stochastic_updater.get_update_statistics()
            )

        if self.pattern_metrics is not None:
            summary["pattern_analysis"] = self.pattern_metrics.get_pattern_summary()

        return summary


def create_nca_config(
    update_probability: float = 0.7,
    residual_learning_rate: float = 0.1,
    enable_pattern_detection: bool = True,
) -> NCAConfig:
    """
    Создать NCA configuration с популярными настройками

    Args:
        update_probability: вероятность stochastic update (0.5-0.9)
        residual_learning_rate: learning rate для residual updates (0.05-0.2)
        enable_pattern_detection: включить pattern analysis

    Returns:
        NCAConfig object
    """
    return NCAConfig(
        update_probability=update_probability,
        stochastic_scheduling=True,
        synchronization_avoidance=True,
        residual_learning_rate=residual_learning_rate,
        stability_threshold=0.01,
        max_update_magnitude=0.5,
        pattern_detection_enabled=enable_pattern_detection,
        spatial_coherence_weight=0.3,
        temporal_consistency_weight=0.2,
        track_emergent_specialization=True,
        specialization_threshold=0.15,
        diversity_preservation_weight=0.25,
    )


def test_nca_basic() -> bool:
    """Базовый тест NCA functionality"""
    try:
        print("🧪 Testing Neural Cellular Automata...")

        # Create config and NCA
        config = create_nca_config()
        nca = NeuralCellularAutomata(config, cube_dimensions=(8, 8, 6), state_size=16)

        # Test data - ИСПРАВЛЕНО: правильный порядок размерностей
        batch_size = 2
        # Формат: [batch, depth, height, width, state_size]
        # cube_dimensions = (width=8, height=8, depth=6)
        # Поэтому tensor shape = [batch, 6, 8, 8, 16]
        current_states = torch.randn(
            batch_size, 6, 8, 8, 16
        )  # [batch, depth, height, width, state_size]
        raw_updates = torch.randn(batch_size, 6, 8, 8, 16)

        # Forward pass
        results = nca(current_states, raw_updates)

        # Validate results
        assert "updated_states" in results
        assert results["updated_states"].shape == current_states.shape

        print("✅ NCA basic functionality works!")
        return True

    except Exception as e:
        print(f"❌ NCA test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic test
    test_nca_basic()
