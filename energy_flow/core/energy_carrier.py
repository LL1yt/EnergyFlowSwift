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
    
    # Флаги завершения потоков (для обработки в FlowProcessor)
    is_terminated: torch.Tensor     # [batch] - маска завершенных потоков
    termination_reason: List[str]   # Причины завершения для каждого потока


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
        
        # 2. Следующая позиция - предсказываем нормализованные координаты
        self.position_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3),  # x, y, z координаты (нормализованные)
            self.config.normalization_manager.get_coordinate_activation()  # Tanh для [-1, 1]
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
        
        # 2. Вычисляем следующую позицию (нормализованную)
        predicted_position_normalized = self.position_projection(gru_output)  # [batch, 3] в [-1, 1]
        
        # Денормализуем для применения bias'ов и шума
        predicted_position = self.config.normalization_manager.denormalize_coordinates(
            predicted_position_normalized
        )

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
        
        # Применяем логику завершения потоков
        next_position, is_terminated, termination_reasons = self._compute_next_position(predicted_position)
        
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
            spawn_count=sum(spawn_counts),
            is_terminated=is_terminated,
            termination_reason=termination_reasons
        )
        
        return output, new_hidden
    
    def _compute_next_position(self, 
                              predicted_position: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Вычисляет следующую позицию и определяет завершенные потоки
        
        Новая логика движения потоков (без принудительного ограничения):
        1. Поток движется туда, куда указывает predicted_position
        2. Если поток выходит за пределы по X,Y - он завершается (нейросеть должна обучиться не делать этого)
        3. Если поток выходит за пределы по Z (depth*2-1) - он нормально завершается на выходной поверхности
        4. FlowProcessor обрабатывает завершенные потоки для сбора энергии
        
        Args:
            predicted_position: [batch, 3] - предсказанные координаты (реальные значения)
            
        Returns:
            next_position: [batch, 3] - следующая позиция (целые координаты)
            is_terminated: [batch] - маска завершенных потоков
            termination_reasons: List[str] - причины завершения для каждого потока
        """
        # Используем предсказанные координаты напрямую (уже денормализованы)
        next_position = predicted_position
        
        # Определяем завершенные потоки вместо принудительного ограничения
        batch_size = predicted_position.shape[0]
        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=predicted_position.device)
        termination_reasons = []
        
        # Проверяем выход за пределы по X и Y координатам
        out_of_bounds_x = (predicted_position[:, 0] < 0) | (predicted_position[:, 0] >= self.config.lattice_width)
        out_of_bounds_y = (predicted_position[:, 1] < 0) | (predicted_position[:, 1] >= self.config.lattice_height)
        out_of_bounds_xy = out_of_bounds_x | out_of_bounds_y
        
        # Правильная логика для Z координаты:
        # Z ∈ [0, depth-1] - активная зона
        # Z ∈ [depth, depth*2-1] - зона завершения (нормальное завершение)
        # Z ≥ depth*2 - выход за пределы (ошибка нейросети)
        
        depth = self.config.lattice_depth
        max_valid_z = depth * 2 - 1
        
        # Определяем типы завершения по Z координате
        reached_output_zone = (predicted_position[:, 2] >= depth) & (predicted_position[:, 2] <= max_valid_z)
        out_of_bounds_z = predicted_position[:, 2] > max_valid_z
        
        # Отмечаем завершенные потоки
        is_terminated = out_of_bounds_xy | reached_output_zone | out_of_bounds_z
        
        # Определяем причины завершения для каждого потока
        for i in range(batch_size):
            if out_of_bounds_xy[i]:
                termination_reasons.append("out_of_bounds_xy")
            elif out_of_bounds_z[i]:
                termination_reasons.append("out_of_bounds_z")  # Ошибка нейросети
            elif reached_output_zone[i]:
                termination_reasons.append("reached_output_surface")  # Нормальное завершение
            else:
                termination_reasons.append("active")  # Поток продолжает движение
        
        # Для активных потоков округляем координаты до целых значений
        next_position = torch.round(predicted_position.clone())
        
        # Для потоков в зоне завершения [depth, depth*2-1] - сопоставляем с выходной поверхностью
        # Сохраняем оригинальные X,Y но устанавливаем Z = depth для буферизации
        output_surface_mask = reached_output_zone
        if output_surface_mask.any():
            next_position[output_surface_mask, 2] = depth  # Сопоставляем с выходной поверхностью Z=depth
        
        return next_position, is_terminated, termination_reasons
    
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
        # Используем централизованную проверку энергии
        return self.config.normalization_manager.check_energy_threshold(
            energy, self.config.energy_threshold
        )


def create_energy_carrier(config=None) -> EnergyCarrier:
    """Фабричная функция для создания EnergyCarrier"""
    return EnergyCarrier(config)