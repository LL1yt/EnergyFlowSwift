"""
Signal Propagator - основной класс для управления распространением сигналов

Этот модуль координирует:
- Интеграцию TimeManager и Lattice3D
- Распространение сигналов по решетке
- Контроль динамики системы
- Анализ состояний сети
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from enum import Enum

# Импорты наших модулей
from .time_manager import TimeManager, TimeConfig, TimeMode
from ..lattice_3d import Lattice3D, LatticeConfig
from ..cell_prototype import CellPrototype

class PropagationMode(Enum):
    """Режимы распространения сигнала"""
    WAVE = "wave"  # Волновое распространение
    DIFFUSION = "diffusion"  # Диффузионное распространение
    DIRECTIONAL = "directional"  # Направленное распространение
    CUSTOM = "custom"  # Пользовательский режим

@dataclass
class PropagationConfig:
    """Конфигурация распространения сигналов"""
    mode: PropagationMode = PropagationMode.WAVE
    signal_strength: float = 1.0  # Сила сигнала
    decay_rate: float = 0.1  # Скорость затухания
    noise_level: float = 0.0  # Уровень шума
    boundary_condition: str = "reflective"  # Граничные условия
    
    # Параметры для разных режимов
    wave_speed: float = 1.0  # Скорость волны
    diffusion_coefficient: float = 0.5  # Коэффициент диффузии
    direction_vector: Optional[Tuple[float, float, float]] = None  # Вектор направления
    
    # Дополнительные параметры
    max_signal_amplitude: float = 10.0  # Максимальная амплитуда
    min_signal_threshold: float = 1e-6  # Минимальный порог сигнала
    
    def __post_init__(self):
        """Проверка корректности конфигурации"""
        if self.signal_strength <= 0:
            raise ValueError("signal_strength должно быть положительным")
        if self.decay_rate < 0:
            raise ValueError("decay_rate не может быть отрицательным")
        if self.noise_level < 0:
            raise ValueError("noise_level не может быть отрицательным")

class SignalPropagator:
    """
    Основной класс для управления распространением сигналов
    
    Интегрирует TimeManager и Lattice3D для создания динамической системы
    """
    
    def __init__(
        self,
        lattice: Lattice3D,
        time_manager: TimeManager,
        propagation_config: PropagationConfig
    ):
        """
        Инициализация распространителя сигналов
        
        Args:
            lattice: 3D решетка клеток
            time_manager: Менеджер времени
            propagation_config: Конфигурация распространения
        """
        self.lattice = lattice
        self.time_manager = time_manager
        self.config = propagation_config
        
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        
        # Состояние системы
        self.current_signals = None
        self.signal_history = []
        self.is_active = False
        
        # Статистика
        self.stats = {
            'total_propagations': 0,
            'average_signal_strength': 0.0,
            'max_signal_reached': 0.0,
            'convergence_time': None,
            'active_cells_count': 0
        }
        
        self.logger.info(f"SignalPropagator инициализирован с режимом: {propagation_config.mode.value}")
    
    def initialize_signals(self, input_signals: torch.Tensor, input_face: str = "front"):
        """
        Инициализация сигналов на входной грани
        
        Args:
            input_signals: Входные сигналы
            input_face: Входная грань ("front", "back", "left", "right", "top", "bottom")
        """
        self.logger.info(f"Инициализация сигналов на грани: {input_face}")
        
        # Получаем размеры решетки
        lattice_size = self.lattice.get_size()
        
        # Создаем тензор для всех сигналов решетки
        signal_shape = (lattice_size[0], lattice_size[1], lattice_size[2], 
                       self.lattice.get_cell_state_size())
        self.current_signals = torch.zeros(signal_shape)
        
        # Размещаем входные сигналы на указанной грани
        self._place_signals_on_face(input_signals, input_face)
        
        # Сохраняем в истории
        self.signal_history.append(self.current_signals.clone())
        
        self.is_active = True
        self.logger.info("Сигналы инициализированы")
    
    def _place_signals_on_face(self, signals: torch.Tensor, face: str):
        """
        Размещение сигналов на указанной грани решетки
        
        Args:
            signals: Сигналы для размещения
            face: Грань решетки
        """
        x_size, y_size, z_size, _ = self.current_signals.shape
        
        if face == "front":
            # Размещаем на передней грани (z=0)
            self.current_signals[:, :, 0, :] = signals
        elif face == "back":
            # Размещаем на задней грани (z=z_size-1)
            self.current_signals[:, :, z_size-1, :] = signals
        elif face == "left":
            # Размещаем на левой грани (x=0)
            self.current_signals[0, :, :, :] = signals
        elif face == "right":
            # Размещаем на правой грани (x=x_size-1)
            self.current_signals[x_size-1, :, :, :] = signals
        elif face == "top":
            # Размещаем на верхней грани (y=0)
            self.current_signals[:, 0, :, :] = signals
        elif face == "bottom":
            # Размещаем на нижней грани (y=y_size-1)
            self.current_signals[:, y_size-1, :, :] = signals
        else:
            raise ValueError(f"Неизвестная грань: {face}")
    
    def propagate_step(self) -> torch.Tensor:
        """
        Выполнение одного шага распространения
        
        Returns:
            torch.Tensor: Новое состояние сигналов
        """
        if not self.is_active:
            raise RuntimeError("Сигналы не инициализированы")
        
        # Выполняем шаг через TimeManager
        self.current_signals = self.time_manager.step(self.current_signals)
        
        # Применяем логику распространения
        self.current_signals = self._apply_propagation_logic(self.current_signals)
        
        # Обновляем статистику
        self._update_stats()
        
        # Сохраняем в истории
        self.signal_history.append(self.current_signals.clone())
        
        return self.current_signals
    
    def _apply_propagation_logic(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Применение логики распространения в зависимости от режима
        
        Args:
            signals: Текущие сигналы
            
        Returns:
            torch.Tensor: Обновленные сигналы
        """
        if self.config.mode == PropagationMode.WAVE:
            return self._wave_propagation(signals)
        elif self.config.mode == PropagationMode.DIFFUSION:
            return self._diffusion_propagation(signals)
        elif self.config.mode == PropagationMode.DIRECTIONAL:
            return self._directional_propagation(signals)
        else:
            # Пользовательский режим или по умолчанию
            return self._default_propagation(signals)
    
    def _wave_propagation(self, signals: torch.Tensor) -> torch.Tensor:
        """Волновое распространение"""
        # Интегрируем с lattice для обновления состояний клеток
        updated_signals = self.lattice.forward(signals)
        
        # Применяем волновую динамику
        wave_effect = self._calculate_wave_effect(updated_signals)
        
        return updated_signals + wave_effect
    
    def _diffusion_propagation(self, signals: torch.Tensor) -> torch.Tensor:
        """Диффузионное распространение"""
        updated_signals = self.lattice.forward(signals)
        
        # Применяем диффузию
        diffusion_effect = self._calculate_diffusion_effect(updated_signals)
        
        return updated_signals + diffusion_effect
    
    def _directional_propagation(self, signals: torch.Tensor) -> torch.Tensor:
        """Направленное распространение"""
        updated_signals = self.lattice.forward(signals)
        
        # Применяем направленный эффект
        if self.config.direction_vector:
            directional_effect = self._calculate_directional_effect(updated_signals)
            return updated_signals + directional_effect
        
        return updated_signals
    
    def _default_propagation(self, signals: torch.Tensor) -> torch.Tensor:
        """Стандартное распространение через решетку"""
        return self.lattice.forward(signals)
    
    def _calculate_wave_effect(self, signals: torch.Tensor) -> torch.Tensor:
        """Расчет волнового эффекта"""
        # Простая реализация волнового эффекта
        wave_effect = torch.zeros_like(signals)
        
        # Добавляем небольшое волновое усиление
        wave_amplitude = self.config.signal_strength * self.config.wave_speed
        wave_effect = wave_amplitude * torch.sin(self.time_manager.current_step * 0.1) * signals
        
        return wave_effect * 0.1  # Небольшой коэффициент
    
    def _calculate_diffusion_effect(self, signals: torch.Tensor) -> torch.Tensor:
        """Расчет диффузионного эффекта"""
        # Простая реализация диффузии
        diffusion_effect = torch.zeros_like(signals)
        
        # Сглаживание с соседями (упрощенная версия)
        x_size, y_size, z_size, state_size = signals.shape
        
        for x in range(1, x_size - 1):
            for y in range(1, y_size - 1):
                for z in range(1, z_size - 1):
                    # Среднее значение соседей
                    neighbors_mean = (
                        signals[x-1, y, z, :] + signals[x+1, y, z, :] +
                        signals[x, y-1, z, :] + signals[x, y+1, z, :] +
                        signals[x, y, z-1, :] + signals[x, y, z+1, :]
                    ) / 6.0
                    
                    # Диффузионный эффект
                    diffusion_effect[x, y, z, :] = self.config.diffusion_coefficient * (
                        neighbors_mean - signals[x, y, z, :]
                    )
        
        return diffusion_effect
    
    def _calculate_directional_effect(self, signals: torch.Tensor) -> torch.Tensor:
        """Расчет направленного эффекта"""
        # Простая реализация направленного распространения
        directional_effect = torch.zeros_like(signals)
        
        # Применяем направленный сдвиг
        dx, dy, dz = self.config.direction_vector
        
        # Простое смещение в указанном направлении
        if abs(dx) > 0.1:
            shift = 1 if dx > 0 else -1
            directional_effect[shift:, :, :, :] += signals[:-shift, :, :, :] * abs(dx)
        
        return directional_effect * 0.1
    
    def _update_stats(self):
        """Обновление статистики"""
        if self.current_signals is not None:
            self.stats['total_propagations'] += 1
            self.stats['average_signal_strength'] = self.current_signals.mean().item()
            self.stats['max_signal_reached'] = max(
                self.stats['max_signal_reached'],
                self.current_signals.max().item()
            )
            self.stats['active_cells_count'] = (
                self.current_signals.abs() > self.config.min_signal_threshold
            ).sum().item()
    
    def run_simulation(self, max_steps: Optional[int] = None) -> List[torch.Tensor]:
        """
        Запуск полной симуляции
        
        Args:
            max_steps: Максимальное количество шагов (если None - используется из конфигурации)
            
        Returns:
            List[torch.Tensor]: История состояний
        """
        if not self.is_active:
            raise RuntimeError("Сигналы не инициализированы")
        
        self.logger.info("Запуск симуляции распространения сигналов")
        
        # Настройка количества шагов
        if max_steps:
            original_max_steps = self.time_manager.config.max_time_steps
            self.time_manager.config.max_time_steps = max_steps
        
        # Запуск симуляции
        self.time_manager.start_simulation()
        
        simulation_history = []
        
        try:
            while self.time_manager.is_running:
                # Выполняем шаг
                current_state = self.propagate_step()
                simulation_history.append(current_state.clone())
                
                # Проверяем сходимость
                if self.time_manager.is_converged():
                    self.logger.info("Симуляция завершена - достигнута сходимость")
                    break
                
                # Проверяем критерий остановки
                if self._should_stop_simulation():
                    self.logger.info("Симуляция остановлена по критерию")
                    break
                    
        except Exception as e:
            self.logger.error(f"Ошибка в симуляции: {e}")
            raise
        finally:
            # Восстанавливаем настройки
            if max_steps:
                self.time_manager.config.max_time_steps = original_max_steps
        
        self.logger.info(f"Симуляция завершена. Выполнено шагов: {len(simulation_history)}")
        return simulation_history
    
    def _should_stop_simulation(self) -> bool:
        """Проверка критериев остановки симуляции"""
        # Критерий: сигнал стал слишком слабым
        if self.stats['average_signal_strength'] < self.config.min_signal_threshold:
            return True
        
        # Критерий: нет активных клеток
        if self.stats['active_cells_count'] == 0:
            return True
        
        return False
    
    def get_output_signals(self, output_face: str = "back") -> torch.Tensor:
        """
        Получение выходных сигналов с указанной грани
        
        Args:
            output_face: Выходная грань
            
        Returns:
            torch.Tensor: Выходные сигналы
        """
        if self.current_signals is None:
            raise RuntimeError("Сигналы не инициализированы")
        
        x_size, y_size, z_size, _ = self.current_signals.shape
        
        if output_face == "front":
            return self.current_signals[:, :, 0, :]
        elif output_face == "back":
            return self.current_signals[:, :, z_size-1, :]
        elif output_face == "left":
            return self.current_signals[0, :, :, :]
        elif output_face == "right":
            return self.current_signals[x_size-1, :, :, :]
        elif output_face == "top":
            return self.current_signals[:, 0, :, :]
        elif output_face == "bottom":
            return self.current_signals[:, y_size-1, :, :]
        else:
            raise ValueError(f"Неизвестная грань: {output_face}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики распространения"""
        return {
            **self.stats,
            'time_manager_stats': self.time_manager.get_stats(),
            'lattice_stats': self.lattice.get_stats() if hasattr(self.lattice, 'get_stats') else {},
            'current_step': self.time_manager.current_step,
            'is_active': self.is_active
        }
    
    def reset(self):
        """Сброс состояния распространителя"""
        self.time_manager.reset()
        self.current_signals = None
        self.signal_history.clear()
        self.is_active = False
        self.stats = {
            'total_propagations': 0,
            'average_signal_strength': 0.0,
            'max_signal_reached': 0.0,
            'convergence_time': None,
            'active_cells_count': 0
        }
        self.logger.info("SignalPropagator сброшен")
    
    def __repr__(self) -> str:
        return (f"SignalPropagator(mode={self.config.mode.value}, "
                f"step={self.time_manager.current_step}, "
                f"active={self.is_active})")

# Вспомогательные функции
def create_signal_propagator(
    lattice_config: Dict[str, Any],
    time_config: Dict[str, Any],
    propagation_config: Dict[str, Any]
) -> SignalPropagator:
    """
    Создание SignalPropagator из конфигураций
    
    Args:
        lattice_config: Конфигурация решетки
        time_config: Конфигурация времени
        propagation_config: Конфигурация распространения
        
    Returns:
        SignalPropagator: Настроенный распространитель
    """
    from .time_manager import create_time_manager
    from ..lattice_3d import create_lattice
    
    # Создаем компоненты
    lattice = create_lattice(lattice_config)
    time_manager = create_time_manager(time_config)
    
    # Преобразуем режим если нужно
    if 'mode' in propagation_config and isinstance(propagation_config['mode'], str):
        propagation_config['mode'] = PropagationMode(propagation_config['mode'])
    
    prop_config = PropagationConfig(**propagation_config)
    
    return SignalPropagator(lattice, time_manager, prop_config) 