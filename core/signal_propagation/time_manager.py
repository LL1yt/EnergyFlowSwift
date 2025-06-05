"""
Time Manager для управления временной динамикой системы

Этот модуль управляет:
- Временными шагами
- Синхронным/асинхронным режимами
- Контролем скорости симуляции
- Историей состояний
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

class TimeMode(Enum):
    """Режимы временного обновления"""
    SYNCHRONOUS = "synchronous"  # Все клетки обновляются одновременно
    ASYNCHRONOUS = "asynchronous"  # Клетки обновляются по очереди
    RANDOM = "random"  # Случайный порядок обновления

@dataclass
class TimeConfig:
    """Конфигурация временного менеджера"""
    max_time_steps: int = 100
    time_mode: TimeMode = TimeMode.SYNCHRONOUS
    step_delay: float = 0.0  # Задержка между шагами (сек)
    save_history: bool = True
    history_length: int = 50  # Максимальная длина истории
    convergence_check_interval: int = 10  # Интервал проверки сходимости
    
    def __post_init__(self):
        """Проверка корректности конфигурации"""
        if self.max_time_steps <= 0:
            raise ValueError("max_time_steps должно быть положительным")
        if self.step_delay < 0:
            raise ValueError("step_delay не может быть отрицательным")
        if self.history_length <= 0:
            raise ValueError("history_length должно быть положительным")

class TimeManager:
    """
    Менеджер времени для управления динамикой системы
    
    Основные функции:
    - Выполнение временных шагов
    - Управление режимами обновления
    - Сохранение истории состояний
    - Контроль сходимости
    """
    
    def __init__(self, config: TimeConfig):
        """
        Инициализация менеджера времени
        
        Args:
            config: Конфигурация временного менеджера
        """
        self.config = config
        self.current_step = 0
        self.is_running = False
        self.history = []  # История состояний системы
        self.step_times = []  # Время выполнения каждого шага
        
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        
        # Статистика
        self.stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'convergence_detected': False,
            'convergence_step': None
        }
        
        self.logger.info(f"TimeManager инициализирован с режимом: {config.time_mode.value}")
    
    def reset(self):
        """Сброс состояния менеджера времени"""
        self.current_step = 0
        self.is_running = False
        self.history.clear()
        self.step_times.clear()
        self.stats = {
            'total_steps': 0,
            'avg_step_time': 0.0,
            'convergence_detected': False,
            'convergence_step': None
        }
        self.logger.info("TimeManager сброшен")
    
    def start_simulation(self):
        """Запуск временной симуляции"""
        self.is_running = True
        self.logger.info("Симуляция запущена")
    
    def stop_simulation(self):
        """Остановка временной симуляции"""
        self.is_running = False
        self.logger.info(f"Симуляция остановлена на шаге {self.current_step}")
    
    def step(self, system_state: torch.Tensor) -> torch.Tensor:
        """
        Выполнение одного временного шага
        
        Args:
            system_state: Текущее состояние системы
            
        Returns:
            torch.Tensor: Новое состояние системы
        """
        if not self.is_running:
            raise RuntimeError("Симуляция не запущена. Вызовите start_simulation()")
        
        import time
        step_start_time = time.time()
        
        # Сохранение истории
        if self.config.save_history:
            self._save_to_history(system_state)
        
        # Выполнение шага (здесь будет логика обновления)
        new_state = self._execute_time_step(system_state)
        
        # Обновление статистики
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        self.current_step += 1
        self.stats['total_steps'] += 1
        
        # Обновление средней скорости
        if len(self.step_times) > 0:
            self.stats['avg_step_time'] = sum(self.step_times) / len(self.step_times)
        
        # Задержка если нужна
        if self.config.step_delay > 0:
            time.sleep(self.config.step_delay)
        
        # Проверка лимита шагов
        if self.current_step >= self.config.max_time_steps:
            self.stop_simulation()
            self.logger.info(f"Достигнут лимит шагов: {self.config.max_time_steps}")
        
        return new_state
    
    def _execute_time_step(self, system_state: torch.Tensor) -> torch.Tensor:
        """
        Выполнение логики временного шага
        
        Args:
            system_state: Текущее состояние
            
        Returns:
            torch.Tensor: Новое состояние
        """
        # Заглушка - здесь будет интеграция с другими модулями
        # Пока возвращаем то же состояние
        return system_state.clone()
    
    def _save_to_history(self, state: torch.Tensor):
        """Сохранение состояния в историю"""
        if len(self.history) >= self.config.history_length:
            self.history.pop(0)  # Удаляем самое старое состояние
        
        self.history.append({
            'step': self.current_step,
            'state': state.clone(),
            'timestamp': torch.tensor(time.time() if 'time' in globals() else 0.0)
        })
    
    def get_history(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение истории состояний
        
        Args:
            steps: Количество последних шагов (если None - вся история)
            
        Returns:
            Список состояний
        """
        if steps is None:
            return self.history.copy()
        else:
            return self.history[-steps:] if steps <= len(self.history) else self.history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики выполнения"""
        return self.stats.copy()
    
    def is_converged(self, tolerance: float = 1e-6) -> bool:
        """
        Проверка сходимости системы
        
        Args:
            tolerance: Допустимая погрешность для сходимости
            
        Returns:
            bool: True если система сошлась
        """
        if len(self.history) < 2:
            return False
        
        # Сравнение последних двух состояний
        last_state = self.history[-1]['state']
        prev_state = self.history[-2]['state']
        
        diff = torch.abs(last_state - prev_state).max().item()
        converged = diff < tolerance
        
        if converged and not self.stats['convergence_detected']:
            self.stats['convergence_detected'] = True
            self.stats['convergence_step'] = self.current_step
            self.logger.info(f"Сходимость достигнута на шаге {self.current_step}")
        
        return converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Информация о сходимости системы"""
        return {
            'is_converged': self.stats['convergence_detected'],
            'convergence_step': self.stats['convergence_step'],
            'current_step': self.current_step,
            'total_steps': self.stats['total_steps']
        }
    
    def __repr__(self) -> str:
        return (f"TimeManager(step={self.current_step}, "
                f"mode={self.config.time_mode.value}, "
                f"running={self.is_running})")

# Вспомогательные функции
def create_time_manager(config_dict: Dict[str, Any]) -> TimeManager:
    """
    Создание TimeManager из словаря конфигурации
    
    Args:
        config_dict: Словарь с настройками
        
    Returns:
        TimeManager: Настроенный менеджер времени
    """
    # Преобразование строки в enum если нужно  
    if 'time_mode' in config_dict and isinstance(config_dict['time_mode'], str):
        config_dict['time_mode'] = TimeMode(config_dict['time_mode'])
    
    config = TimeConfig(**config_dict)
    return TimeManager(config)

def load_time_config(config_path: Path) -> TimeConfig:
    """
    Загрузка конфигурации из файла
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        TimeConfig: Загруженная конфигурация
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Преобразование строки в enum
    if 'time_mode' in config_dict:
        config_dict['time_mode'] = TimeMode(config_dict['time_mode'])
    
    return TimeConfig(**config_dict) 