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
from ..config import create_debug_config, set_energy_config

logger = get_logger(__name__)


@dataclass
class SpawnInfo:
    """Информация о новых потоках для одного batch элемента"""
    energies: List[torch.Tensor]    # Энергии новых потоков
    parent_batch_idx: int          # Индекс родительского потока


@dataclass
class EnergyOutput:
    """Структурированный вывод EnergyCarrier"""
    energy_value: torch.Tensor      # Текущая энергия/эмбеддинг
    next_position: torch.Tensor     # Координаты следующей клетки
    spawn_info: List[SpawnInfo]     # Структурированная информация о spawn'ах
    
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
        # Разделяем на части для диагностики
        self.position_projection_base = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3)  # x, y, z координаты (до активации)
        )
        self.position_activation = self.config.normalization_manager.get_coordinate_activation()  # Tanh для [-1, 1]
        
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
        logger.info(f"🎓 Curriculum settings: initial_z_bias={self.config.initial_z_bias}, "
                   f"use_forward_bias={self.config.use_forward_movement_bias}, "
                   f"decay_steps={getattr(self.config, 'bias_decay_steps', 'N/A')}")
    
    def _init_weights(self):
        """Инициализация весов с smart initialization для движения вперед"""
        # GRU уже имеет хорошую инициализацию по умолчанию
        
        # Инициализируем projection heads
        for module in [self.energy_projection, self.position_projection_base, 
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
        
        # SMART INITIALIZATION: положительный bias для Z-координаты
        # Помогает модели научиться движению вперед без жесткого кодирования
        with torch.no_grad():
            # position_projection_base[-1] это последний Linear слой перед активацией
            # Индекс 2 соответствует Z-координате в выходе [x, y, z]
            if hasattr(self.position_projection_base, '__getitem__') and len(self.position_projection_base) >= 2:
                final_linear = None
                # Находим последний Linear слой
                for i in range(len(self.position_projection_base) - 1, -1, -1):
                    if isinstance(self.position_projection_base[i], nn.Linear):
                        final_linear = self.position_projection_base[i]
                        break
                
                if final_linear is not None and final_linear.bias is not None:
                    # SMART INITIALIZATION: используем из config (теперь 0.0 для диагностики)
                    smart_init_bias = self.config.smart_init_bias
                    final_linear.bias[2] = smart_init_bias
                    logger.debug_init(f"🎩 SMART INITIALIZATION: Z-coordinate bias set to {smart_init_bias:.2f} (DISABLED for diagnostics)")
                    logger.debug_init(f"Full position_projection bias: {final_linear.bias.data}")
                else:
                    logger.warning("⚠️ Smart initialization FAILED: could not find final linear layer with bias!")
    
    def forward(self, 
                neuron_output: torch.Tensor,
                embedding_part: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None,
                current_position: Optional[torch.Tensor] = None,
                flow_age: Optional[torch.Tensor] = None,
                global_training_step: Optional[int] = None) -> Tuple[EnergyOutput, torch.Tensor]:
        """
        Прямой проход через EnergyCarrier
        
        Args:
            neuron_output: [batch, neuron_output_dim] - выход SimpleNeuron
            embedding_part: [batch, embedding_dim] - часть входного эмбеддинга
            hidden_state: [num_layers, batch, hidden_size] - скрытое состояние GRU
            current_position: [batch, 3] - текущая позиция (для расчета следующей)
            flow_age: [batch] - возраст потоков для progressive bias
            global_training_step: Глобальный шаг обучения для curriculum learning
            
        Returns:
            output: EnergyOutput - структурированный вывод
            new_hidden: [num_layers, batch, hidden_size] - новое скрытое состояние
        """
        batch_size = neuron_output.shape[0]
        
        # ДИАГНОСТИКА: логируем параметры входа
        if global_training_step is not None:
            logger.debug_energy(f"🔄 EnergyCarrier forward: batch={batch_size}, global_step={global_training_step}")
            if current_position is not None:
                current_z = current_position[:, 2]
                logger.debug_energy(f"📍 Current Z positions: min={current_z.min():.3f}, "
                           f"max={current_z.max():.3f}, mean={current_z.mean():.3f}")
        
        # Объединяем входы
        combined_input = torch.cat([neuron_output, embedding_part], dim=-1)
        combined_input = combined_input.unsqueeze(1)  # [batch, 1, input_dim] для GRU
        
        # Проход через GRU
        gru_output, new_hidden = self.gru(combined_input, hidden_state)
        gru_output = gru_output.squeeze(1)  # [batch, hidden_size]
        
        # 1. Генерируем текущую энергию
        energy_value = self.energy_projection(gru_output)  # [batch, embedding_dim]
        
        # 2. Вычисляем следующую позицию (нормализованную)
        # ДИАГНОСТИКА: логируем GRU выход перед position_projection
        if global_training_step is not None and global_training_step == 0:  # Только первый шаг
            logger.debug_forward(f"🧠 GRU output stats: min={gru_output.min():.3f}, max={gru_output.max():.3f}, "
                       f"mean={gru_output.mean():.3f}, std={gru_output.std():.3f}")
        
        # Получаем сырой выход (до активации)
        predicted_position_raw = self.position_projection_base(gru_output)  # [batch, 3] без ограничений
        
        # ДИАГНОСТИКА: логируем сырой выход модели (ДО Tanh)
        if global_training_step is not None and global_training_step == 0:  # Только первый шаг
            raw_z = predicted_position_raw[:, 2]
            logger.debug_forward(f"🔥 RAW model output (before Tanh): Z min={raw_z.min():.3f}, "
                       f"max={raw_z.max():.3f}, mean={raw_z.mean():.3f}, std={raw_z.std():.3f}")
        
        # Применяем активацию (Tanh)
        predicted_position_normalized = self.position_activation(predicted_position_raw)  # [batch, 3] в [-1, 1]
        
        # ДИАГНОСТИКА: логируем нормализованные координаты (ПОСЛЕ Tanh)
        norm_z = predicted_position_normalized[:, 2]
        logger.debug_energy(f"📊 Normalized Z (after Tanh): min={norm_z.min():.3f}, "
                       f"max={norm_z.max():.3f}, mean={norm_z.mean():.3f}")
        # Показываем Z-диапазон для нормализации
        z_range = self.config.normalization_manager.ranges.z_range
        logger.debug_energy(f"🔧 Z normalization range: {z_range} (depth={self.config.lattice_depth}, zones=[0,{self.config.lattice_depth-1}]|[{self.config.lattice_depth},{self.config.lattice_depth*2-1}]|{self.config.lattice_depth*2}+)")
        
        # Денормализуем для применения bias'ов и шума
        predicted_position = self.config.normalization_manager.denormalize_coordinates(
            predicted_position_normalized
        )
        
        # ДИАГНОСТИКА: логируем денормализованные координаты
        denorm_z = predicted_position[:, 2]
        logger.debug_energy(f"📊 Denormalized Z (before bias): min={denorm_z.min():.3f}, "
                       f"max={denorm_z.max():.3f}, mean={denorm_z.mean():.3f}")
        # Проверяем корректность денормализации
        expected_max_z = self.config.lattice_depth * 2 - 1  # Исправлено для трехзонной логики
        if denorm_z.max() > expected_max_z * 1.2:  # Допускаем небольшое превышение
            logger.error(f"🚫 DENORMALIZATION ERROR: Z > expected max ({expected_max_z})! "
                       f"Check normalization range: {self.config.normalization_manager.ranges.z_range}")

        # CURRICULUM LEARNING: прогрессивное уменьшение bias'а для движения вперед
        if self.config.use_forward_movement_bias and self.config.initial_z_bias > 0:
            if global_training_step is not None:
                # Прогрессивное уменьшение bias'а на основе глобального шага обучения
                bias_decay_factor = max(0.0, 1.0 - (global_training_step / self.config.bias_decay_steps))
                current_bias = self.config.initial_z_bias * bias_decay_factor
                
                # ДИАГНОСТИКА: логируем curriculum learning параметры
                logger.debug_energy(f"📓 Curriculum step {global_training_step}: "
                           f"decay_factor={bias_decay_factor:.4f}, current_bias={current_bias:.4f}")
                
                # Дополнительный progressive bias на основе возраста потока
                if flow_age is not None:
                    age_bonus = flow_age * self.config.progressive_z_multiplier * bias_decay_factor
                    total_bias = current_bias + age_bonus  # Может быть тензором [batch]
                else:
                    total_bias = current_bias  # Скаляр
                
                # Применяем bias только если он все еще значимый
                # Обрабатываем случай, когда total_bias может быть тензором
                
                # ДИАГНОСТИКА: логируем Z-координаты ДО применения bias'а
                if logger.isEnabledFor(10):  # DEBUG level
                    z_before = predicted_position[:, 2]
                    logger.debug(f"📊 Z-coords BEFORE bias: min={z_before.min():.3f}, "
                               f"max={z_before.max():.3f}, mean={z_before.mean():.3f}")
                
                if isinstance(total_bias, torch.Tensor):
                    # Векторизованное применение bias'а для каждого потока отдельно
                    valid_bias_mask = total_bias > 0.01
                    flows_with_bias = valid_bias_mask.sum().item()
                    if valid_bias_mask.any():
                        predicted_position[valid_bias_mask, 2] += total_bias[valid_bias_mask]
                        logger.debug(f"✅ Applied tensor bias to {flows_with_bias}/{batch_size} flows")
                else:
                    # total_bias - скаляр
                    if total_bias > 0.01:
                        predicted_position[:, 2] += total_bias
                        logger.debug(f"✅ Applied scalar bias {total_bias:.4f} to all {batch_size} flows")
                
                # ДИАГНОСТИКА: логируем Z-координаты ПОСЛЕ применения bias'а
                if logger.isEnabledFor(10):  # DEBUG level
                    z_after = predicted_position[:, 2]
                    logger.debug(f"📊 Z-coords AFTER bias: min={z_after.min():.3f}, "
                               f"max={z_after.max():.3f}, mean={z_after.mean():.3f}")
            
            elif flow_age is not None:
                # Fallback: используем только возраст потока без curriculum learning
                dynamic_z_bias = self.config.initial_z_bias + (flow_age * self.config.progressive_z_multiplier)
                predicted_position[:, 2] += dynamic_z_bias
                logger.debug(f"⚠️ Fallback: applied age-based bias without global_training_step")
        
        else:
            # ДИАГНОСТИКА: логируем, когда bias НЕ применяется
            if logger.isEnabledFor(10):  # DEBUG level
                reason = "disabled" if not self.config.use_forward_movement_bias else "zero_initial_bias"
                logger.debug(f"❌ NO BIAS applied: reason={reason}")
        
        # Exploration noise для разнообразия путей
        if self.config.use_exploration_noise:
            noise = torch.randn_like(predicted_position) * self.config.exploration_noise
            predicted_position += noise
            logger.debug(f"🎲 Added exploration noise: std={self.config.exploration_noise}")
        
        # ПРОВЕРКА ГРАНИЦ: убеждаемся что Z-координаты в разумных пределах
        if logger.isEnabledFor(10):  # DEBUG level
            z_coords = predicted_position[:, 2]
            max_expected_z = self.config.lattice_depth + 10  # Допускаем небольшой выход за пределы
            if torch.any(z_coords > max_expected_z):
                out_of_bounds_count = (z_coords > max_expected_z).sum().item()
                logger.error(f"🚫 Z-COORDINATE BOUNDS ERROR: {out_of_bounds_count}/{predicted_position.shape[0]} "
                           f"flows have Z > {max_expected_z} (max={z_coords.max():.2f})")
                logger.error(f"🔍 This indicates coordinate system malfunction!")
        
        # Применяем логику завершения потоков
        next_position, is_terminated, termination_reasons = self._compute_next_position(predicted_position)
        
        # ДИАГНОСТИКА: логируем результаты
        if logger.isEnabledFor(10):  # DEBUG level
            terminated_count = is_terminated.sum().item()
            logger.debug(f"🛡️ Termination: {terminated_count}/{batch_size} flows terminated")
            if terminated_count > 0:
                # Подсчитываем причины
                reason_counts = {}
                for reason in termination_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                logger.debug(f"📈 Termination reasons: {reason_counts}")
        
        # 3. Определяем порождение новых потоков
        spawn_prob = self.spawn_gate(gru_output).squeeze(-1)  # [batch]
        
        # Определяем количество новых потоков на основе вероятности и порогов
        spawn_decisions = spawn_prob > self.config.spawn_threshold
        spawn_info = []
        
        for i in range(batch_size):
            if spawn_decisions[i] and spawn_prob[i].item() > self.config.spawn_threshold:
                # Количество потоков зависит от силы вероятности
                num_spawns = min(
                    int((spawn_prob[i].item() - self.config.spawn_threshold) / 
                        (1 - self.config.spawn_threshold) * self.config.max_spawn_per_step) + 1,
                    self.config.max_spawn_per_step
                )
                
                # Генерируем энергии для новых потоков
                spawn_energy = self.spawn_energy_projection(gru_output[i])
                energies = []
                
                # Делим энергию между потоками
                for j in range(num_spawns):
                    energy_fraction = spawn_energy / (num_spawns + 1)  # +1 для родителя
                    energies.append(energy_fraction.to(gru_output.device))
                
                # Создаем SpawnInfo для текущего batch элемента
                spawn_info.append(SpawnInfo(
                    energies=energies,
                    parent_batch_idx=i
                ))
        
        # Создаем структурированный вывод
        output = EnergyOutput(
            energy_value=energy_value,
            next_position=next_position,
            spawn_info=spawn_info,
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