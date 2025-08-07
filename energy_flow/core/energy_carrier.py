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
        # УДАЛЕНО: dropout слои больше не используются в архитектуре относительных координат
        # Фильтрация потоков теперь основана на длине смещения, а не на dropout
        
        # Размерности
        self.neuron_output_dim = config.neuron_output_dim  # Выход SimpleNeuron (64)
        self.energy_dim = 1                                # Скалярная энергия от mapper'а
        self.input_dim = self.neuron_output_dim + self.energy_dim  # 64 + 1 = 65
        
        # GRU слои
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.0,  # Dropout отключен в новой архитектуре
            batch_first=True
        )
        
        # Projection heads для структурированного вывода
        # 1. Скалярная энергия (выход должен быть скаляром для consistency)
        self.energy_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            # Dropout слой удален
            nn.Linear(self.hidden_size // 2, self.energy_dim),  # Выход: 1 скаляр
            nn.Tanh()  # Нормализация в [-1, 1]
        )
        
        # 2. Смещения - предсказываем относительные смещения (Δx, Δy, Δz)
        self.displacement_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            # Dropout слой удален
            nn.Linear(64, 3)  # Δx, Δy, Δz смещения (до активации)
        )
        self.displacement_activation = self.config.normalization_manager.get_displacement_activation()  # Tanh для [-1, 1]
        
        # 3. УДАЛЕНО: spawn_gate и spawn_energy_projection
        # В архитектуре относительных координат spawn контролируется 
        # только на основе длины смещения в FlowProcessor
        
        # Инициализация весов
        self._init_weights()
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"EnergyCarrier initialized with {total_params:,} parameters")
        logger.debug(f"GRU: input={self.input_dim}, hidden={self.hidden_size}, layers={self.num_layers}")
        
    
    def _init_weights(self):
        """Инициализация весов с smart initialization для движения вперед"""
        # GRU уже имеет хорошую инициализацию по умолчанию
        
        # Инициализируем projection heads (spawn компоненты удалены)
        for module in [self.energy_projection, self.displacement_projection]:
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
        
        # Для новой архитектуры относительных координат:
        # - Нет smart initialization (смещения центрированы на 0)
        # - Нет bias для движения вперед (модель учится сама)
        logger.debug_init("🏗️ Relative coordinates architecture: no smart initialization, model learns naturally")
    
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
        
        # 2. Вычисляем смещения (относительные координаты)
        # ДИАГНОСТИКА: логируем GRU выход перед displacement_projection
        if global_training_step is not None and global_training_step == 0:  # Только первый шаг
            logger.debug_forward(f"🧠 GRU output stats: min={gru_output.min():.3f}, max={gru_output.max():.3f}, "
                       f"mean={gru_output.mean():.3f}, std={gru_output.std():.3f}")
            
            # ДИАГНОСТИКА: проверяем bias'ы в displacement_projection
            for i, module in enumerate(self.displacement_projection):
                if isinstance(module, nn.Linear) and module.bias is not None:
                    bias_stats = module.bias.data
                    logger.debug_forward(f"📊 displacement_projection[{i}] bias: "
                                       f"min={bias_stats.min():.4f}, max={bias_stats.max():.4f}, "
                                       f"mean={bias_stats.mean():.4f}, std={bias_stats.std():.4f}")
        
        # Получаем сырой выход смещений (до активации)
        displacement_raw = self.displacement_projection(gru_output)  # [batch, 3] без ограничений
        
        # ДИАГНОСТИКА: логируем сырой выход модели (ДО Clamp)
        if global_training_step is not None and global_training_step == 0:  # Только первый шаг
            raw_delta_z = displacement_raw[:, 2]
            logger.debug_forward(f"🔥 RAW displacement output (before Clamp): ΔZ min={raw_delta_z.min():.3f}, "
                       f"max={raw_delta_z.max():.3f}, mean={raw_delta_z.mean():.3f}, std={raw_delta_z.std():.3f}")
        
        # Применяем ограничение диапазона (Clamp вместо неэффективного Tanh)
        displacement_normalized = torch.clamp(displacement_raw, -1.0, 1.0)  # [batch, 3] в [-1, 1]
        
        # ДИАГНОСТИКА: логируем нормализованные смещения (ПОСЛЕ Clamp)
        norm_delta_z = displacement_normalized[:, 2]
        logger.debug_energy(f"📊 Normalized displacement (after Clamp): ΔZ min={norm_delta_z.min():.3f}, "
                       f"max={norm_delta_z.max():.3f}, mean={norm_delta_z.mean():.3f}")
        
        # Применяем нормализованные смещения к текущей позиции (все в [-1, 1] пространстве)
        if current_position is not None:
            next_position = current_position + displacement_normalized
        else:
            # Если текущая позиция не передана, используем смещения как абсолютные координаты
            logger.warning("⚠️ Current position is None, using displacement as absolute position")
            next_position = displacement_normalized
        
        # Exploration noise для разнообразия путей (в нормализованном пространстве)
        if self.config.use_exploration_noise:
            # Exploration noise тоже должен быть в нормализованном пространстве
            noise = torch.randn_like(displacement_normalized) * self.config.exploration_noise
            next_position += noise
            logger.debug(f"🎲 Added normalized exploration noise: std={self.config.exploration_noise}")
        
        # Применяем логику завершения потоков для новой трехплоскостной архитектуры
        next_position, is_terminated, termination_reasons = self._compute_next_position_relative(next_position)
        
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
        
        # 3. Spawn потоков теперь контролируется только movement_based_spawn в FlowProcessor
        # Устаревшая логика spawn на основе эмбеддингов удалена
        spawn_info = []  # Пустой список, spawn контролируется в FlowProcessor
        
        # Создаем структурированный вывод
        output = EnergyOutput(
            energy_value=energy_value,
            next_position=next_position,
            spawn_info=spawn_info,
            is_terminated=is_terminated,
            termination_reason=termination_reasons
        )
        
        return output, new_hidden
    
    def _compute_next_position_relative(self, 
                                   next_position: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Вычисляет следующую позицию для трехплоскостной архитектуры относительных координат
        
        Новая логика трехплоскостной архитектуры:
        1. Входная плоскость: Z = depth/2 (центр куба)
        2. Выходные плоскости: Z = 0 и Z = depth (края куба)
        3. X/Y границы: отражение обрабатывается в FlowProcessor
        4. Потоки завершаются при достижении Z ≤ 0 или Z ≥ depth
        
        Args:
            next_position: [batch, 3] - позиция после применения смещения
            
        Returns:
            next_position: [batch, 3] - следующая позиция (целые координаты)
            is_terminated: [batch] - маска завершенных потоков  
            termination_reasons: List[str] - причины завершения для каждого потока
        """
        batch_size = next_position.shape[0]
        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=next_position.device)
        termination_reasons = []
        
        depth = self.config.lattice_depth
        
        # Проверяем завершение по Z координате в трехплоскостной архитектуре
        # Z ≤ 0: достижение левой выходной плоскости
        # Z ≥ depth: достижение правой выходной плоскости
        reached_z0_plane = next_position[:, 2] <= 0
        reached_zdepth_plane = next_position[:, 2] >= depth
        reached_output_plane = reached_z0_plane | reached_zdepth_plane
        
        # Проверяем выход за пределы по X и Y (обрабатывается отражением в FlowProcessor)
        out_of_bounds_x = (next_position[:, 0] < 0) | (next_position[:, 0] >= self.config.lattice_width)
        out_of_bounds_y = (next_position[:, 1] < 0) | (next_position[:, 1] >= self.config.lattice_height)
        out_of_bounds_xy = out_of_bounds_x | out_of_bounds_y
        
        # В новой архитектуре X/Y границы НЕ завершают поток (отражение)
        # Завершение только при достижении выходных плоскостей по Z
        is_terminated = reached_output_plane
        
        # Определяем причины завершения для каждого потока
        for i in range(batch_size):
            if reached_z0_plane[i]:
                termination_reasons.append("reached_z0_plane")  # Левая выходная плоскость
            elif reached_zdepth_plane[i]:
                termination_reasons.append("reached_zdepth_plane")  # Правая выходная плоскость
            elif out_of_bounds_xy[i]:
                termination_reasons.append("xy_reflection_needed")  # Требуется отражение (но поток активен)
            else:
                termination_reasons.append("active")  # Поток продолжает движение
        
        # Для завершенных потоков проецируем на соответствующую выходную плоскость
        final_position = next_position.clone()
        
        # Проецирование на Z=0 плоскость
        if reached_z0_plane.any():
            final_position[reached_z0_plane, 2] = 0
        
        # Проецирование на Z=depth плоскость
        if reached_zdepth_plane.any():
            final_position[reached_zdepth_plane, 2] = depth
        
        # Округляем координаты для дискретной решетки
        final_position = torch.round(final_position)
        
        return final_position, is_terminated, termination_reasons
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния GRU"""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=torch.float32
        )
    
    # УДАЛЕН: check_energy_level() - в архитектуре относительных координат 
    # потоки не умирают от "недостатка энергии". Эмбеддинги - это данные, а не энергия.


def create_energy_carrier(config=None) -> EnergyCarrier:
    """Фабричная функция для создания EnergyCarrier"""
    return EnergyCarrier(config)