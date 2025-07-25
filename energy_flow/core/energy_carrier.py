"""
Energy Carrier - GRU-based энергетические потоки
================================================

GRU модель с ~10M параметров для представления энергии.
Общие веса для всех GRU потоков в решетке.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..config import get_energy_config, create_debug_config, set_energy_config

logger = get_logger(__name__)


@dataclass
class EnergyOutput:
    """Структурированный вывод EnergyCarrier"""
    energy_value: torch.Tensor      # Текущая энергия/эмбеддинг
    next_position: torch.Tensor     # Координаты следующей клетки
    spawn_energies: List[torch.Tensor]  # Энергии для новых потоков
    spawn_count: int               # Количество новых потоков


class EnergyCarrier(nn.Module):
    """
    GRU-based модель для представления энергетических потоков
    
    Принимает:
    - Выход SimpleNeuron
    - Часть входного эмбеддинга
    - Скрытое состояние GRU
    
    Выдает:
    - Структурированный вывод с энергией, позицией и новыми потоками
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: EnergyConfig с настройками. Если None - берется глобальный конфиг
        """
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Параметры из конфига
        self.hidden_size = config.carrier_hidden_size
        self.num_layers = config.carrier_num_layers
        self.dropout = config.carrier_dropout
        
        # Размерности
        self.neuron_output_dim = config.neuron_output_dim  # Выход SimpleNeuron (64)
        self.energy_dim = 1                                # Скалярная энергия от mapper'а
        self.input_dim = self.neuron_output_dim + self.energy_dim  # 64 + 1 = 65
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Projection heads для структурированного вывода
        # 1. Скалярная энергия (выход должен быть скаляром для consistency)
        self.energy_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.energy_dim),  # Выход: 1 скаляр
            nn.Tanh()  # Нормализация в [-1, 1]
        )
        
        # 2. Следующая позиция - предсказываем абсолютные координаты
        self.position_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3),  # x, y, z координаты (абсолютные)
            # Убираем Tanh - позволяем модели предсказывать любые координаты
        )
        
        # 3. Порождение новых потоков
        self.spawn_gate = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Вероятность порождения
        )
        
        # Проекция для энергии новых потоков (также скалярная)
        self.spawn_energy_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.energy_dim),
            nn.Tanh()  # Нормализация в [-1, 1]
        )
        
        # Инициализация весов
        self._init_weights()
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"EnergyCarrier initialized with {total_params:,} parameters")
        logger.debug(f"GRU: input={self.input_dim}, hidden={self.hidden_size}, layers={self.num_layers}")
    
    def _init_weights(self):
        """Инициализация весов"""
        # GRU уже имеет хорошую инициализацию по умолчанию
        
        # Инициализируем projection heads
        for module in [self.energy_projection, self.position_projection, 
                      self.spawn_gate, self.spawn_energy_projection]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                neuron_output: torch.Tensor,
                embedding_part: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                current_position: Optional[torch.Tensor] = None,
                flow_age: Optional[torch.Tensor] = None) -> Tuple[EnergyOutput, torch.Tensor]:
        """
        Прямой проход через EnergyCarrier
        
        Args:
            neuron_output: [batch, neuron_output_dim] - выход SimpleNeuron
            embedding_part: [batch, embedding_dim] - часть входного эмбеддинга
            hidden_state: [num_layers, batch, hidden_size] - скрытое состояние GRU
            current_position: [batch, 3] - текущая позиция (для расчета следующей)
            
        Returns:
            output: EnergyOutput - структурированный вывод
            new_hidden: [num_layers, batch, hidden_size] - новое скрытое состояние
        """
        batch_size = neuron_output.shape[0]
        
        # Объединяем входы
        combined_input = torch.cat([neuron_output, embedding_part], dim=-1)
        combined_input = combined_input.unsqueeze(1)  # [batch, 1, input_dim] для GRU
        
        # Проход через GRU
        gru_output, new_hidden = self.gru(combined_input, hidden_state)
        gru_output = gru_output.squeeze(1)  # [batch, hidden_size]
        
        # 1. Генерируем текущую энергию
        energy_value = self.energy_projection(gru_output)  # [batch, embedding_dim]
        
        # 2. Вычисляем следующую позицию
        predicted_position = self.position_projection(gru_output)  # [batch, 3]

        # Экспериментальная настройка для предварительного обучения
        if self.config.use_forward_movement_bias and self.config.initial_z_bias > 0:
            # Применяем экспериментальный progressive bias для Z координаты
            if (flow_age is not None):
                # Динамический bias = initial_bias + (age * multiplier)
                dynamic_z_bias = self.config.initial_z_bias + (flow_age * self.config.progressive_z_multiplier)
                predicted_position[:, 2] += dynamic_z_bias
        
        # Exploration noise для разнообразия путей
        if self.config.use_exploration_noise:
            noise = torch.randn_like(predicted_position) * self.config.exploration_noise
            predicted_position += noise
        
        
        # Применяем ограничения движения (только вперед по Z)
        if predicted_position is not None:
            next_position = self._compute_next_position(predicted_position)
        else:
            logger.error("Predicted position is None")
        
        # 3. Определяем порождение новых потоков
        spawn_prob = self.spawn_gate(gru_output).squeeze(-1)  # [batch]
        
        # Определяем количество новых потоков на основе вероятности и порогов
        spawn_decisions = spawn_prob > self.config.spawn_threshold
        spawn_counts = []
        spawn_energies = []
        
        for i in range(batch_size):
            if spawn_decisions[i] and spawn_prob[i].item() > self.config.spawn_threshold:
                # Количество потоков зависит от силы вероятности
                num_spawns = min(
                    int((spawn_prob[i].item() - self.config.spawn_threshold) / 
                        (1 - self.config.spawn_threshold) * self.config.max_spawn_per_step) + 1,
                    self.config.max_spawn_per_step
                )
                spawn_counts.append(num_spawns)
                
                # Генерируем энергии для новых потоков
                spawn_energy = self.spawn_energy_projection(gru_output[i])
                # Делим энергию между потоками
                for j in range(num_spawns):
                    energy_fraction = spawn_energy / (num_spawns + 1)  # +1 для родителя
                    # Убеждаемся что энергия на правильном устройстве
                    energy_fraction = energy_fraction.to(gru_output.device)
                    spawn_energies.append(energy_fraction)
            else:
                spawn_counts.append(0)
        
        # Создаем структурированный вывод
        output = EnergyOutput(
            energy_value=energy_value,
            next_position=next_position,
            spawn_energies=spawn_energies,
            spawn_count=sum(spawn_counts)
        )
        
        return output, new_hidden
    
    def _compute_next_position(self, 
                              predicted_position: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет следующую позицию согласно PLAN.md
        
        Логика движения потоков:
        1. Поток движется туда, куда указывает next_position (без принуждения)
        2. Проверка валидности движения происходит на уровне FlowProcessor/EnergyLattice
        3. Если поток не движется вперед по Z - он будет убит в FlowProcessor
        4. Если поток выходит за пределы по Z - он завершается и используется для выходного эмбеддинга
        
        Args:
            current_position: [batch, 3] - текущие координаты
            predicted_position: [batch, 3] - предсказанное смещение
            
        Returns:
            next_position: [batch, 3] - следующая позиция (целые координаты)
            
        TODO: При выходе за пределы решетки по X,Y направлять поток к ближайшей точке выходной стороны
        TODO: Реализовать умное усреднение для нескольких потоков в одной выходной точке
        TODO: Добавить более умную логику навигации через обучение после создания trainer
        """
        # Масштабируем дельту (EnergyCarrier определяет направление и дистанцию)
        scaled_delta = predicted_position # * 5.0   Максимальный прыжок ~5 клеток без прыжков пока.
        
        # Применяем смещение БЕЗ принуждения движения вперед
        next_position = scaled_delta
        
        # Только ограничиваем координаты размерами решетки (не убиваем поток здесь)
        # Логика убийства/завершения потоков будет в FlowProcessor
        next_position[:, 0] = torch.clamp(next_position[:, 0], 0, self.config.lattice_width - 1)
        next_position[:, 1] = torch.clamp(next_position[:, 1], 0, self.config.lattice_height - 1)
        # Z координату НЕ ограничиваем - позволяем выходить за пределы для детекции завершения
        
        # Округляем до целых значений для дискретной решетки
        next_position = torch.round(next_position)
        
        return next_position
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния GRU"""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=torch.float32
        )
    
    def check_energy_level(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Проверяет уровень энергии
        
        Args:
            energy: [batch, 1] - нормализованная энергия в диапазоне [-1, 1] (Tanh)
        
        Returns:
            is_alive: [batch] - маска активных потоков
        """
        # Преобразуем из [-1, 1] в [0, 1] для проверки порога
        energy_normalized = (energy + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        energy_magnitude = torch.abs(energy_normalized.squeeze(-1))  # [batch]
        return energy_magnitude > self.config.energy_threshold


def create_energy_carrier(config=None) -> EnergyCarrier:
    """Фабричная функция для создания EnergyCarrier"""
    return EnergyCarrier(config)