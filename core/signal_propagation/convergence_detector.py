"""
Convergence Detector - детекция сходимости системы

Этот модуль определяет:
- Стабилизацию состояний
- Критерии сходимости
- Различные метрики сходимости
- Автоматическую остановку симуляции
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from enum import Enum

class ConvergenceMode(Enum):
    """Режимы детекции сходимости"""
    ABSOLUTE = "absolute"  # Абсолютная разность состояний
    RELATIVE = "relative"  # Относительная разность
    ENERGY = "energy"  # На основе энергии системы
    GRADIENT = "gradient"  # На основе градиентов
    STATISTICAL = "statistical"  # Статистические критерии
    COMBINED = "combined"  # Комбинированный подход

@dataclass
class ConvergenceConfig:
    """Конфигурация детектора сходимости"""
    mode: ConvergenceMode = ConvergenceMode.COMBINED
    tolerance: float = 1e-6  # Допустимая погрешность
    patience: int = 5  # Количество шагов для подтверждения сходимости
    min_steps: int = 10  # Минимальное количество шагов перед проверкой
    max_steps: int = 1000  # Максимальное количество шагов
    
    # Параметры для разных режимов
    relative_threshold: float = 1e-4  # Порог для относительной разности
    energy_threshold: float = 1e-5  # Порог для изменения энергии
    gradient_threshold: float = 1e-5  # Порог для градиентов
    
    # Статистические параметры
    variance_threshold: float = 1e-6  # Порог для дисперсии
    correlation_threshold: float = 0.99  # Порог корреляции между состояниями
    
    # Дополнительные настройки
    check_interval: int = 1  # Интервал проверки сходимости
    early_stopping: bool = True  # Ранняя остановка при сходимости
    
    def __post_init__(self):
        """Проверка корректности конфигурации"""
        if self.tolerance <= 0:
            raise ValueError("tolerance должен быть положительным")
        if self.patience < 1:
            raise ValueError("patience должен быть >= 1")
        if self.min_steps < 0:
            raise ValueError("min_steps не может быть отрицательным")

class ConvergenceDetector:
    """
    Детектор сходимости системы
    
    Основные функции:
    - Мониторинг состояний системы
    - Детекция различных типов сходимости
    - Предоставление метрик сходимости
    - Автоматическая остановка симуляции
    """
    
    def __init__(self, config: ConvergenceConfig):
        """
        Инициализация детектора сходимости
        
        Args:
            config: Конфигурация детектора
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # История состояний для анализа
        self.state_history = []
        self.convergence_history = []
        
        # Счетчики и флаги
        self.step_count = 0
        self.convergence_count = 0  # Количество последовательных "сходящихся" шагов
        self.is_converged = False
        self.convergence_step = None
        
        # Метрики сходимости
        self.metrics = {
            'absolute_difference': [],
            'relative_difference': [],
            'energy_change': [],
            'gradient_magnitude': [],
            'variance': [],
            'correlation': []
        }
        
        self.logger.info(f"ConvergenceDetector инициализирован с режимом: {config.mode.value}")
    
    def add_state(self, state: torch.Tensor) -> bool:
        """
        Добавление нового состояния и проверка сходимости
        
        Args:
            state: Новое состояние системы
            
        Returns:
            bool: True если система сошлась
        """
        self.state_history.append(state.clone())
        self.step_count += 1
        
        # Проверяем сходимость только если достаточно истории
        if self.step_count < self.config.min_steps:
            return False
        
        # Проверяем с заданным интервалом
        if self.step_count % self.config.check_interval != 0:
            return False
        
        # Вычисляем метрики сходимости
        self._calculate_metrics()
        
        # Проверяем сходимость по выбранному режиму
        converged = self._check_convergence()
        
        if converged:
            self.convergence_count += 1
            if self.convergence_count >= self.config.patience:
                if not self.is_converged:
                    self.is_converged = True
                    self.convergence_step = self.step_count
                    self.logger.info(f"Сходимость достигнута на шаге {self.step_count}")
                return True
        else:
            self.convergence_count = 0
        
        # Проверяем максимальное количество шагов
        if self.step_count >= self.config.max_steps:
            self.logger.warning(f"Достигнут лимит шагов {self.config.max_steps}")
            return True if self.config.early_stopping else False
        
        return False
    
    def _calculate_metrics(self):
        """Расчет метрик сходимости"""
        if len(self.state_history) < 2:
            return
        
        current_state = self.state_history[-1]
        previous_state = self.state_history[-2]
        
        # Абсолютная разность
        abs_diff = torch.abs(current_state - previous_state).mean().item()
        self.metrics['absolute_difference'].append(abs_diff)
        
        # Относительная разность
        prev_magnitude = previous_state.abs().mean().item()
        if prev_magnitude > 0:
            rel_diff = abs_diff / prev_magnitude
        else:
            rel_diff = abs_diff
        self.metrics['relative_difference'].append(rel_diff)
        
        # Изменение энергии
        current_energy = (current_state ** 2).sum().item()
        previous_energy = (previous_state ** 2).sum().item()
        energy_change = abs(current_energy - previous_energy) / max(previous_energy, 1e-10)
        self.metrics['energy_change'].append(energy_change)
        
        # Магнитуда градиента
        gradient_mag = self._calculate_gradient_magnitude(current_state)
        self.metrics['gradient_magnitude'].append(gradient_mag)
        
        # Дисперсия последних состояний
        if len(self.state_history) >= 3:
            recent_states = torch.stack(self.state_history[-3:])
            variance = recent_states.var(dim=0).mean().item()
            self.metrics['variance'].append(variance)
        
        # Корреляция между состояниями
        correlation = self._calculate_state_correlation(current_state, previous_state)
        self.metrics['correlation'].append(correlation)
    
    def _check_convergence(self) -> bool:
        """Проверка сходимости по выбранному режиму"""
        if self.config.mode == ConvergenceMode.ABSOLUTE:
            return self._check_absolute_convergence()
        elif self.config.mode == ConvergenceMode.RELATIVE:
            return self._check_relative_convergence()
        elif self.config.mode == ConvergenceMode.ENERGY:
            return self._check_energy_convergence()
        elif self.config.mode == ConvergenceMode.GRADIENT:
            return self._check_gradient_convergence()
        elif self.config.mode == ConvergenceMode.STATISTICAL:
            return self._check_statistical_convergence()
        elif self.config.mode == ConvergenceMode.COMBINED:
            return self._check_combined_convergence()
        else:
            return False
    
    def _check_absolute_convergence(self) -> bool:
        """Проверка абсолютной сходимости"""
        if not self.metrics['absolute_difference']:
            return False
        
        return self.metrics['absolute_difference'][-1] < self.config.tolerance
    
    def _check_relative_convergence(self) -> bool:
        """Проверка относительной сходимости"""
        if not self.metrics['relative_difference']:
            return False
        
        return self.metrics['relative_difference'][-1] < self.config.relative_threshold
    
    def _check_energy_convergence(self) -> bool:
        """Проверка сходимости по энергии"""
        if not self.metrics['energy_change']:
            return False
        
        return self.metrics['energy_change'][-1] < self.config.energy_threshold
    
    def _check_gradient_convergence(self) -> bool:
        """Проверка сходимости по градиентам"""
        if not self.metrics['gradient_magnitude']:
            return False
        
        return self.metrics['gradient_magnitude'][-1] < self.config.gradient_threshold
    
    def _check_statistical_convergence(self) -> bool:
        """Проверка статистической сходимости"""
        # Проверяем дисперсию
        variance_ok = (self.metrics['variance'] and 
                      self.metrics['variance'][-1] < self.config.variance_threshold)
        
        # Проверяем корреляцию
        correlation_ok = (self.metrics['correlation'] and 
                         self.metrics['correlation'][-1] > self.config.correlation_threshold)
        
        return variance_ok and correlation_ok
    
    def _check_combined_convergence(self) -> bool:
        """Комбинированная проверка сходимости"""
        checks = []
        
        # Абсолютная разность
        if self.metrics['absolute_difference']:
            checks.append(self.metrics['absolute_difference'][-1] < self.config.tolerance)
        
        # Относительная разность
        if self.metrics['relative_difference']:
            checks.append(self.metrics['relative_difference'][-1] < self.config.relative_threshold)
        
        # Изменение энергии
        if self.metrics['energy_change']:
            checks.append(self.metrics['energy_change'][-1] < self.config.energy_threshold)
        
        # Градиенты
        if self.metrics['gradient_magnitude']:
            checks.append(self.metrics['gradient_magnitude'][-1] < self.config.gradient_threshold)
        
        # Требуем выполнения минимум 2 из критериев
        return sum(checks) >= min(2, len(checks))
    
    def _calculate_gradient_magnitude(self, state: torch.Tensor) -> float:
        """Расчет магнитуды градиента состояния"""
        # Суммируем по последней размерности (состояние клетки)
        intensity = state.abs().sum(dim=-1)
        
        # Вычисляем градиенты по пространственным измерениям
        gradients = []
        
        if intensity.dim() >= 1:
            gradients.append(torch.diff(intensity, dim=0).abs().mean())
        if intensity.dim() >= 2:
            gradients.append(torch.diff(intensity, dim=1).abs().mean())
        if intensity.dim() >= 3:
            gradients.append(torch.diff(intensity, dim=2).abs().mean())
        
        if gradients:
            # Евклидова норма градиента
            magnitude = torch.sqrt(sum(g**2 for g in gradients))
            return magnitude.item()
        else:
            return 0.0
    
    def _calculate_state_correlation(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Расчет корреляции между состояниями"""
        # Плоские векторы состояний
        flat1 = state1.flatten()
        flat2 = state2.flatten()
        
        # Вычисляем корреляцию
        if len(flat1) > 1:
            correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
            return correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            return 1.0 if torch.allclose(flat1, flat2) else 0.0
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Получение информации о сходимости"""
        return {
            'is_converged': self.is_converged,
            'convergence_step': self.convergence_step,
            'current_step': self.step_count,
            'convergence_count': self.convergence_count,
            'required_patience': self.config.patience,
            'current_metrics': {
                key: values[-1] if values else None 
                for key, values in self.metrics.items()
            },
            'convergence_mode': self.config.mode.value
        }
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Получение истории метрик сходимости"""
        return self.metrics.copy()
    
    def plot_convergence_metrics(self) -> Dict[str, Any]:
        """Подготовка данных для визуализации метрик сходимости"""
        plot_data = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                plot_data[metric_name] = {
                    'values': values,
                    'steps': list(range(len(values))),
                    'threshold': self._get_threshold_for_metric(metric_name)
                }
        
        return plot_data
    
    def _get_threshold_for_metric(self, metric_name: str) -> Optional[float]:
        """Получение порога для метрики"""
        thresholds = {
            'absolute_difference': self.config.tolerance,
            'relative_difference': self.config.relative_threshold,
            'energy_change': self.config.energy_threshold,
            'gradient_magnitude': self.config.gradient_threshold,
            'variance': self.config.variance_threshold,
            'correlation': self.config.correlation_threshold
        }
        return thresholds.get(metric_name)
    
    def reset(self):
        """Сброс состояния детектора"""
        self.state_history.clear()
        self.convergence_history.clear()
        self.step_count = 0
        self.convergence_count = 0
        self.is_converged = False
        self.convergence_step = None
        
        for metric_list in self.metrics.values():
            metric_list.clear()
        
        self.logger.info("ConvergenceDetector сброшен")
    
    def set_tolerance(self, new_tolerance: float):
        """Изменение допустимой погрешности"""
        if new_tolerance <= 0:
            raise ValueError("tolerance должен быть положительным")
        
        old_tolerance = self.config.tolerance
        self.config.tolerance = new_tolerance
        
        self.logger.info(f"Tolerance изменен с {old_tolerance} на {new_tolerance}")
    
    def set_patience(self, new_patience: int):
        """Изменение терпения (количества подтверждающих шагов)"""
        if new_patience < 1:
            raise ValueError("patience должен быть >= 1")
        
        old_patience = self.config.patience
        self.config.patience = new_patience
        
        self.logger.info(f"Patience изменен с {old_patience} на {new_patience}")
    
    def __repr__(self) -> str:
        return (f"ConvergenceDetector(mode={self.config.mode.value}, "
                f"step={self.step_count}, "
                f"converged={self.is_converged})")

# Вспомогательные функции
def create_convergence_detector(config_dict: Dict[str, Any]) -> ConvergenceDetector:
    """
    Создание ConvergenceDetector из словаря конфигурации
    
    Args:
        config_dict: Словарь с настройками
        
    Returns:
        ConvergenceDetector: Настроенный детектор
    """
    # Преобразование строки в enum если нужно
    if 'mode' in config_dict and isinstance(config_dict['mode'], str):
        config_dict['mode'] = ConvergenceMode(config_dict['mode'])
    
    config = ConvergenceConfig(**config_dict)
    return ConvergenceDetector(config)

def auto_detect_convergence(
    signal_history: List[torch.Tensor],
    tolerance: float = 1e-6,
    patience: int = 3
) -> Tuple[bool, Dict[str, Any]]:
    """
    Автоматическая детекция сходимости для истории сигналов
    
    Args:
        signal_history: История состояний
        tolerance: Допустимая погрешность
        patience: Количество подтверждающих шагов
        
    Returns:
        Tuple[bool, Dict]: (сошлась ли система, информация о сходимости)
    """
    config = ConvergenceConfig(
        mode=ConvergenceMode.COMBINED,
        tolerance=tolerance,
        patience=patience,
        min_steps=2
    )
    
    detector = ConvergenceDetector(config)
    
    # Добавляем все состояния
    converged = False
    for state in signal_history:
        converged = detector.add_state(state)
        if converged:
            break
    
    return converged, detector.get_convergence_info()

class AdaptiveConvergenceDetector(ConvergenceDetector):
    """
    Адаптивный детектор сходимости
    
    Автоматически корректирует пороги на основе поведения системы
    """
    
    def __init__(self, config: ConvergenceConfig, adaptation_rate: float = 0.1):
        """
        Инициализация адаптивного детектора
        
        Args:
            config: Базовая конфигурация
            adaptation_rate: Скорость адаптации порогов
        """
        super().__init__(config)
        self.adaptation_rate = adaptation_rate
        self.initial_tolerance = config.tolerance
        self.adaptation_history = []
        
        self.logger.info("Инициализирован адаптивный детектор сходимости")
    
    def _adapt_tolerance(self):
        """Адаптация порога допустимости"""
        if len(self.metrics['absolute_difference']) < 10:
            return
        
        # Анализируем тенденцию последних изменений
        recent_diffs = self.metrics['absolute_difference'][-10:]
        trend = np.mean(np.diff(recent_diffs))
        
        # Если тенденция к уменьшению - можем ужесточить пороги
        if trend < 0:
            new_tolerance = self.config.tolerance * (1 - self.adaptation_rate)
            new_tolerance = max(new_tolerance, self.initial_tolerance * 0.1)  # Не менее 10% от исходного
        # Если тенденция к увеличению - ослабляем пороги
        elif trend > 0:
            new_tolerance = self.config.tolerance * (1 + self.adaptation_rate)
            new_tolerance = min(new_tolerance, self.initial_tolerance * 10)  # Не более 10x от исходного
        else:
            return  # Без изменений
        
        if abs(new_tolerance - self.config.tolerance) > self.initial_tolerance * 0.01:
            self.adaptation_history.append({
                'step': self.step_count,
                'old_tolerance': self.config.tolerance,
                'new_tolerance': new_tolerance,
                'trend': trend
            })
            
            self.config.tolerance = new_tolerance
            self.logger.debug(f"Адаптация tolerance: {new_tolerance:.2e} (тренд: {trend:.2e})")
    
    def add_state(self, state: torch.Tensor) -> bool:
        """Переопределение с адаптацией"""
        result = super().add_state(state)
        
        # Адаптируем пороги каждые 20 шагов
        if self.step_count % 20 == 0:
            self._adapt_tolerance()
        
        return result
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """Информация об адаптации"""
        return {
            'initial_tolerance': self.initial_tolerance,
            'current_tolerance': self.config.tolerance,
            'adaptation_rate': self.adaptation_rate,
            'adaptation_history': self.adaptation_history.copy()
        } 