"""
Централизованная система нормализации для энергетической архитектуры
===================================================================

Управляет нормализацией всех ключевых величин в системе:
- Координаты (X, Y, Z) с учетом размеров решетки
- Энергия (преобразование между [0,1] и [-1,1])
- Вероятности spawn'ов
- Flow age для прогрессивных bias'ов

Принципы:
- Единая точка истины для всех правил нормализации
- Fail-fast при некорректных значениях (без fallback'ов)
- Четкое разделение нормализованных и реальных значений
"""

import torch
import torch.nn as nn
from typing import Tuple, Union
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizationRanges:
    """Диапазоны для нормализации всех величин"""
    
    # Координаты (зависят от размеров решетки)
    x_range: Tuple[float, float]  # (0, width-1)
    y_range: Tuple[float, float]  # (0, height-1) 
    z_range: Tuple[float, float]  # (0, depth*2-1) для выхода за пределы
    
    # Энергия (преобразование между диапазонами)
    energy_raw_range: Tuple[float, float] = (0.0, 1.0)      # Исходная энергия [0,1]
    energy_normalized_range: Tuple[float, float] = (-1.0, 1.0)  # Нормализованная [-1,1]
    
    # Вероятности (уже в правильном диапазоне)
    spawn_prob_range: Tuple[float, float] = (0.0, 1.0)     # Sigmoid выход
    
    # Flow age (если будет использоваться)
    flow_age_max: float = 100.0  # Максимальный возраст потока
    
    def __post_init__(self):
        """Валидация диапазонов"""
        # Координаты должны быть неотрицательными
        assert self.x_range[0] >= 0 and self.x_range[1] > self.x_range[0], f"Некорректный X диапазон: {self.x_range}"
        assert self.y_range[0] >= 0 and self.y_range[1] > self.y_range[0], f"Некорректный Y диапазон: {self.y_range}"
        assert self.z_range[0] >= 0 and self.z_range[1] > self.z_range[0], f"Некорректный Z диапазон: {self.z_range}"
        
        # Энергия должна быть в ожидаемых диапазонах
        assert self.energy_raw_range == (0.0, 1.0), f"Исходная энергия должна быть [0,1], получено: {self.energy_raw_range}"
        assert self.energy_normalized_range == (-1.0, 1.0), f"Нормализованная энергия должна быть [-1,1], получено: {self.energy_normalized_range}"
        
        logger.info(f"NormalizationRanges: X{self.x_range}, Y{self.y_range}, Z{self.z_range}")


class NormalizationManager:
    """
    Централизованный менеджер нормализации
    
    Предоставляет методы для нормализации/денормализации всех величин
    и правильные activation functions для каждого типа данных.
    """
    
    def __init__(self, ranges: NormalizationRanges):
        self.ranges = ranges
        logger.info("NormalizationManager инициализирован")
    
    # ========== КООРДИНАТЫ ==========
    
    def normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Нормализует координаты [batch, 3] в диапазон [-1, 1]
        
        Args:
            coords: [batch, 3] - реальные координаты (X, Y, Z)
            
        Returns:
            normalized: [batch, 3] - нормализованные координаты [-1, 1]
        """
        batch_size = coords.shape[0]
        assert coords.shape == (batch_size, 3), f"Ожидался тензор [batch, 3], получен: {coords.shape}"
        
        normalized = torch.zeros_like(coords)
        
        # X координата: [0, width-1] → [-1, 1]
        normalized[:, 0] = self._normalize_to_range(
            coords[:, 0], self.ranges.x_range[0], self.ranges.x_range[1]
        )
        
        # Y координата: [0, height-1] → [-1, 1]  
        normalized[:, 1] = self._normalize_to_range(
            coords[:, 1], self.ranges.y_range[0], self.ranges.y_range[1]
        )
        
        # Z координата: [0, depth*2-1] → [-1, 1]
        normalized[:, 2] = self._normalize_to_range(
            coords[:, 2], self.ranges.z_range[0], self.ranges.z_range[1]
        )
        
        return normalized
    
    def denormalize_coordinates(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Денормализует координаты из [-1, 1] в реальные значения
        
        Args:
            normalized: [batch, 3] - нормализованные координаты [-1, 1]
            
        Returns:
            coords: [batch, 3] - реальные координаты
        """
        batch_size = normalized.shape[0]
        assert normalized.shape == (batch_size, 3), f"Ожидался тензор [batch, 3], получен: {normalized.shape}"
        
        # Проверяем, что значения в ожидаемом диапазоне
        assert torch.all(normalized >= -1.0) and torch.all(normalized <= 1.0), \
            f"Нормализованные координаты должны быть в [-1,1], получен диапазон: [{normalized.min():.3f}, {normalized.max():.3f}]"
        
        coords = torch.zeros_like(normalized)
        
        # X координата: [-1, 1] → [0, width-1]
        coords[:, 0] = self._denormalize_from_range(
            normalized[:, 0], self.ranges.x_range[0], self.ranges.x_range[1]
        )
        
        # Y координата: [-1, 1] → [0, height-1]
        coords[:, 1] = self._denormalize_from_range(
            normalized[:, 1], self.ranges.y_range[0], self.ranges.y_range[1]
        )
        
        # Z координата: [-1, 1] → [0, depth*2-1]
        coords[:, 2] = self._denormalize_from_range(
            normalized[:, 2], self.ranges.z_range[0], self.ranges.z_range[1]
        )
        
        return coords
    
    # ========== ЭНЕРГИЯ ==========
    
    def normalize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Нормализует энергию из [0, 1] в [-1, 1]
        
        Args:
            energy: [batch, 1] - исходная энергия [0, 1]
            
        Returns:
            normalized: [batch, 1] - нормализованная энергия [-1, 1]
        """
        assert energy.shape[-1] == 1, f"Энергия должна быть [batch, 1], получено: {energy.shape}"
        
        # Проверяем исходный диапазон
        assert torch.all(energy >= 0.0) and torch.all(energy <= 1.0), \
            f"Исходная энергия должна быть в [0,1], получен диапазон: [{energy.min():.3f}, {energy.max():.3f}]"
        
        # Преобразование [0, 1] → [-1, 1]
        normalized = 2.0 * energy - 1.0
        
        return normalized
    
    def denormalize_energy(self, normalized_energy: torch.Tensor) -> torch.Tensor:
        """
        Денормализует энергию из [-1, 1] в [0, 1]
        
        Args:
            normalized_energy: [batch, 1] - нормализованная энергия [-1, 1]
            
        Returns:
            energy: [batch, 1] - реальная энергия [0, 1]
        """
        assert normalized_energy.shape[-1] == 1, f"Энергия должна быть [batch, 1], получено: {normalized_energy.shape}"
        
        # Проверяем нормализованный диапазон
        assert torch.all(normalized_energy >= -1.0) and torch.all(normalized_energy <= 1.0), \
            f"Нормализованная энергия должна быть в [-1,1], получен диапазон: [{normalized_energy.min():.3f}, {normalized_energy.max():.3f}]"
        
        # Преобразование [-1, 1] → [0, 1]
        energy = (normalized_energy + 1.0) / 2.0
        
        return energy
    
    # ========== FLOW AGE ==========
    
    def normalize_flow_age(self, age: torch.Tensor) -> torch.Tensor:
        """
        Нормализует возраст потока в [0, 1]
        
        Args:
            age: [batch] - возраст потока [0, max_age]
            
        Returns:
            normalized: [batch] - нормализованный возраст [0, 1]
        """
        assert torch.all(age >= 0), "Возраст потока не может быть отрицательным"
        
        # Ограничиваем максимальным возрастом и нормализуем
        clamped_age = torch.clamp(age, 0, self.ranges.flow_age_max)
        normalized = clamped_age / self.ranges.flow_age_max
        
        return normalized
    
    # ========== ACTIVATION FUNCTIONS ==========
    
    def get_coordinate_activation(self) -> nn.Module:
        """Возвращает правильную активацию для координат"""
        return nn.Tanh()  # Координаты всегда в [-1, 1]
    
    def get_energy_activation(self) -> nn.Module:
        """Возвращает правильную активацию для энергии"""
        return nn.Tanh()  # Энергия в нормализованном виде [-1, 1]
    
    def get_spawn_probability_activation(self) -> nn.Module:
        """Возвращает правильную активацию для spawn вероятностей"""
        return nn.Sigmoid()  # Вероятности всегда [0, 1]
    
    def get_flow_age_activation(self) -> nn.Module:
        """Возвращает правильную активацию для возраста потоков"""
        return nn.Sigmoid()  # Возраст нормализуется в [0, 1]
    
    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========
    
    def _normalize_to_range(self, values: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Нормализует значения из [min_val, max_val] в [-1, 1]"""
        return 2.0 * (values - min_val) / (max_val - min_val) - 1.0
    
    def _denormalize_from_range(self, normalized: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Денормализует значения из [-1, 1] в [min_val, max_val]"""
        return min_val + (normalized + 1.0) / 2.0 * (max_val - min_val)
    
    # ========== УТИЛИТЫ ДЛЯ ПОРОГОВ ==========
    
    def check_energy_threshold(self, normalized_energy: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Проверяет энергию против порога (threshold в диапазоне [0,1])
        
        Args:
            normalized_energy: [batch, 1] - нормализованная энергия [-1, 1]
            threshold: float - порог в диапазоне [0, 1]
            
        Returns:
            is_alive: [batch] - маска активных потоков
        """
        assert 0.0 <= threshold <= 1.0, f"Порог энергии должен быть в [0,1], получен: {threshold}"
        
        # Денормализуем энергию для сравнения с порогом
        real_energy = self.denormalize_energy(normalized_energy)
        energy_magnitude = torch.abs(real_energy.squeeze(-1))  # [batch]
        
        return energy_magnitude > threshold
    
    def check_spawn_threshold(self, spawn_prob: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Проверяет spawn вероятность против порога
        
        Args:
            spawn_prob: [batch] - вероятность spawn'а [0, 1]
            threshold: float - порог в диапазоне [0, 1]
            
        Returns:
            should_spawn: [batch] - маска для spawn'а
        """
        assert 0.0 <= threshold <= 1.0, f"Порог spawn'а должен быть в [0,1], получен: {threshold}"
        assert torch.all(spawn_prob >= 0.0) and torch.all(spawn_prob <= 1.0), \
            f"Spawn вероятность должна быть в [0,1], получен диапазон: [{spawn_prob.min():.3f}, {spawn_prob.max():.3f}]"
        
        return spawn_prob > threshold


def create_normalization_manager(lattice_width: int, lattice_height: int, lattice_depth: int) -> NormalizationManager:
    """
    Фабричная функция для создания NormalizationManager
    
    Args:
        lattice_width: Ширина решетки
        lattice_height: Высота решетки  
        lattice_depth: Глубина решетки
        
    Returns:
        NormalizationManager с правильными диапазонами
    """
    ranges = NormalizationRanges(
        x_range=(0.0, float(lattice_width - 1)),
        y_range=(0.0, float(lattice_height - 1)),
        z_range=(0.0, float(lattice_depth * 2 - 1))  # Позволяем выход за пределы
    )
    
    return NormalizationManager(ranges)