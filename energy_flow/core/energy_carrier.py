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
        
        # Память предыдущих позиций для улучшения контекста
        self.position_memory_size = 5  # Храним 5 предыдущих позиций
        self.position_memory = nn.Linear(
            3 * self.position_memory_size,  # 5 позиций * 3 координаты = 15
            self.hidden_size // 4
        )
        
        # Комбинирование истории позиций с GRU выходом
        self.history_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size // 4, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
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
                global_training_step: Optional[int] = None,
                position_history: Optional[torch.Tensor] = None) -> Tuple[EnergyOutput, torch.Tensor]:
        """
        Прямой проход через EnergyCarrier
        
        Args:
            neuron_output: [batch, neuron_output_dim] - выход SimpleNeuron
            embedding_part: [batch, embedding_dim] - часть входного эмбеддинга
            hidden_state: [num_layers, batch, hidden_size] - скрытое состояние GRU
            current_position: [batch, 3] - текущая позиция (для расчета следующей)
            flow_age: [batch] - возраст потоков для progressive bias
            global_training_step: Глобальный шаг обучения для curriculum learning
            position_history: [batch, memory_size, 3] - история позиций (опционально)
            
        Returns:
            output: EnergyOutput - структурированный вывод
            new_hidden: [num_layers, batch, hidden_size] - новое скрытое состояние
        """
        batch_size = neuron_output.shape[0]
        
        # ДИАГНОСТИКА основных параметров
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
        
        # Интеграция истории позиций для лучшего предсказания траекторий
        if position_history is not None and position_history.shape[1] > 0:
            # Flatten history: [batch, memory_size * 3]
            history_flat = position_history.view(batch_size, -1)
            
            # Дополняем нулями если история короче memory_size
            if history_flat.shape[1] < 3 * self.position_memory_size:
                padding_size = 3 * self.position_memory_size - history_flat.shape[1]
                padding = torch.zeros(batch_size, padding_size, device=history_flat.device)
                history_flat = torch.cat([history_flat, padding], dim=1)
            
            # Проецируем историю в features
            history_features = self.position_memory(history_flat)  # [batch, hidden_size // 4]
            
            # Объединяем с GRU выходом
            combined_features = torch.cat([gru_output, history_features], dim=-1)
            gru_output = self.history_fusion(combined_features)  # [batch, hidden_size]
            
            if global_training_step is not None and global_training_step <= 3:
                logger.debug_forward(f"📜 Position history integrated: shape={position_history.shape}, "
                                   f"history_features norm={history_features.norm(dim=-1).mean():.3f}")
        
        # 1. Генерируем текущую энергию
        energy_value = self.energy_projection(gru_output)  # [batch, embedding_dim]
        
        # 2. Вычисляем смещения (относительные координаты)
        # ДИАГНОСТИКА: логируем GRU выход перед displacement_projection
        if global_training_step is not None and global_training_step <= 3:  # Первые 3 шага
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
        if global_training_step is not None and global_training_step <= 3:  # Первые 3 шага
            raw_delta_z = displacement_raw[:, 2]
            logger.debug_forward(f"🔥 RAW displacement output (before Clamp): ΔZ min={raw_delta_z.min():.3f}, "
                       f"max={raw_delta_z.max():.3f}, mean={raw_delta_z.mean():.3f}, std={raw_delta_z.std():.3f}")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: правильное масштабирование с учетом границ
        # Сначала применяем масштабирование
        current_scale = self._calculate_displacement_scale(global_training_step)
        displacement_scaled = displacement_raw * current_scale
        
        # ОБЯЗАТЕЛЬНЫЙ clamp смещений ДО применения к позиции
        # Это гарантирует, что смещения не выведут позицию за границы [-1, 1]
        displacement_normalized = torch.clamp(displacement_scaled, -0.5, 0.5)  # Ограничиваем смещения
        
        if global_training_step is not None and global_training_step % self.config.displacement_scale_update_interval == 0:
            logger.debug_forward(f"🔧 DISPLACEMENT SCALING: step={global_training_step}, scale={current_scale:.3f}")
        
        # ДИАГНОСТИКА: логируем смещения после масштабирования (ДО финального clamp)
        norm_delta_z = displacement_normalized[:, 2]
        logger.debug_energy(f"📊 Scaled displacement (before final clamp): ΔZ min={norm_delta_z.min():.3f}, "
                       f"max={norm_delta_z.max():.3f}, mean={norm_delta_z.mean():.3f}")
        
        # ДИАГНОСТИКА смещений (только на первых шагах)
        if global_training_step is not None and global_training_step <= 3:
            depth = self.config.lattice_depth
            real_displacement_z = norm_delta_z * (depth / 2)  # Денормализуем смещения
            logger.debug_forward(f"🔍 Real world Z displacement: min={real_displacement_z.min():.3f}, "
                               f"max={real_displacement_z.max():.3f}, mean={real_displacement_z.mean():.3f} "
                               f"(depth={depth})")
        
        # Применяем нормализованные смещения к текущей позиции (все в [-1, 1] пространстве)
        if current_position is not None:
            # Сначала вычислим "сырую" следующую позицию БЕЗ clamp — нужна для корректной детекции выхода за границы X/Y
            raw_next_position = current_position + displacement_normalized
            # Затем сформируем безопасную позицию с clamp как базу дальнейшей обработки
            next_position = torch.clamp(raw_next_position, -1.0, 1.0)
            
            # ДИАГНОСТИКА Z-движения: детальный анализ с пояснениями
            if global_training_step is not None and global_training_step <= 3:
                z_current = current_position[:, 2]
                z_next = next_position[:, 2]
                z_delta = z_next - z_current
                
                # Расшифровка координат в понятном виде
                depth = self.config.lattice_depth
                current_real = self.config.normalization_manager.denormalize_coordinates(current_position)[:, 2]
                next_real = self.config.normalization_manager.denormalize_coordinates(next_position)[:, 2]
                
                logger.debug_forward(f"🎯 Z-POSITION ANALYSIS:")
                logger.debug_forward(f"  📍 Current normalized: [{z_current.min():.3f}, {z_current.max():.3f}] mean={z_current.mean():.3f}")
                logger.debug_forward(f"  📍 Current real: [{current_real.min():.1f}, {current_real.max():.1f}] mean={current_real.mean():.1f} (depth={depth})")
                logger.debug_forward(f"  📈 Delta normalized: [{z_delta.min():.3f}, {z_delta.max():.3f}] mean={z_delta.mean():.3f}")
                logger.debug_forward(f"  📍 Next normalized: [{z_next.min():.3f}, {z_next.max():.3f}] mean={z_next.mean():.3f}")
                logger.debug_forward(f"  📍 Next real: [{next_real.min():.1f}, {next_real.max():.1f}] mean={next_real.mean():.1f}")
                
                # Анализ направлений движения (ОБА направления валидны в dual output planes архитектуре!)
                positive_z_count = (z_delta > 0).sum().item()  # К Z=depth выходной плоскости
                negative_z_count = (z_delta < 0).sum().item() # К Z=0 выходной плоскости  
                neutral_count = (z_delta == 0).sum().item()
                logger.debug_forward(f"  🎯 Movement direction: to_zdepth_plane={positive_z_count}, to_z0_plane={negative_z_count}, neutral={neutral_count}")
                logger.debug_forward(f"  ℹ️  Both directions are valid - model chooses optimal output plane")
        else:
            # Если текущая позиция не передана, используем смещения как абсолютные координаты
            logger.warning("⚠️ Current position is None, using displacement as absolute position")
            raw_next_position = displacement_normalized
            next_position = torch.clamp(raw_next_position, -1.0, 1.0)
        
        # Exploration noise для разнообразия путей (в нормализованном пространстве)
        if self.config.use_exploration_noise:
            # Exploration noise тоже должен быть в нормализованном пространстве
            noise = torch.randn_like(next_position) * self.config.exploration_noise
            # Применяем шум с немедленным clamp для гарантии границ
            raw_next_position = raw_next_position + noise
            next_position = torch.clamp(next_position + noise, -1.0, 1.0)
            logger.debug(f"🎲 Added normalized exploration noise: std={self.config.exploration_noise}")
        
        # Применяем логику завершения потоков для новой трехплоскостной архитектуры
        next_position, is_terminated, termination_reasons = self._compute_next_position_relative(next_position, global_training_step, raw_next_position=raw_next_position)
        
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
                                   next_position: torch.Tensor,
                                   global_training_step: Optional[int] = None,
                                   raw_next_position: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Вычисляет следующую позицию для трехплоскостной архитектуры относительных координат
        
        DUAL OUTPUT PLANES архитектура:
        1. Входная плоскость: Z = depth/2 (центр куба) - normalized Z = 0.0
        2. Выходные плоскости: Z = 0 (normalized Z = -1.0) И Z = depth (normalized Z = +1.0)
        3. X/Y границы: отражение обрабатывается в FlowProcessor
        4. Потоки завершаются при достижении любой из двух выходных плоскостей
        
        ВАЖНО: Движение в любом Z-направлении валидно! Модель сама выбирает оптимальную
        выходную плоскость для каждого потока на основе обучающих данных.
        
        Args:
            next_position: [batch, 3] - позиция после применения смещения
            global_training_step: Шаг обучения для диагностики
            
        Returns:
            next_position: [batch, 3] - следующая позиция (целые координаты)
            is_terminated: [batch] - маска завершенных потоков  
            termination_reasons: List[str] - причины завершения для каждого потока
        """
        batch_size = next_position.shape[0]
        is_terminated = torch.zeros(batch_size, dtype=torch.bool, device=next_position.device)
        termination_reasons = []
        
        # ИСПРАВЛЕНО: проверяем завершение по Z координате в НОРМАЛИЗОВАННОМ пространстве
        # Z ≤ -1.0: достижение левой выходной плоскости (raw Z=0)
        # Z ≥ +1.0: достижение правой выходной плоскости (raw Z=depth)
        reached_z0_plane = next_position[:, 2] <= -1.0
        reached_zdepth_plane = next_position[:, 2] >= 1.0
        reached_output_plane = reached_z0_plane | reached_zdepth_plane
        
        # ДИАГНОСТИКА Z: логируем количество завершенных потоков
        if reached_output_plane.any():
            num_z0 = reached_z0_plane.sum().item()
            num_zdepth = reached_zdepth_plane.sum().item()
            logger.debug_forward(f"🔍 Z TERMINATION: z0_plane={num_z0}, zdepth_plane={num_zdepth}, total={reached_output_plane.sum().item()}")
        
        # ПРОБЛЕМА НАЙДЕНА: проверка границ должна быть в нормализованном пространстве [-1, 1]
        # НЕ в raw координатах решетки!
        
        # ДИАГНОСТИКА: логируем диапазоны координат
        if batch_size <= 10000:  # Избегаем логирования для больших батчей
            x_min, x_max = next_position[:, 0].min().item(), next_position[:, 0].max().item()
            y_min, y_max = next_position[:, 1].min().item(), next_position[:, 1].max().item()
            z_min, z_max = next_position[:, 2].min().item(), next_position[:, 2].max().item()
            logger.debug_forward(f"🔍 BOUNDS CHECK: positions range X[{x_min:.3f}, {x_max:.3f}], "
                               f"Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
        
        # Проверяем выход за границы X/Y по СЫРЫМ координатам до clamp, если они предоставлены
        pos_for_bounds = raw_next_position if raw_next_position is not None else next_position
        out_of_bounds_x = (pos_for_bounds[:, 0] < -1.0) | (pos_for_bounds[:, 0] > 1.0)
        out_of_bounds_y = (pos_for_bounds[:, 1] < -1.0) | (pos_for_bounds[:, 1] > 1.0)
        out_of_bounds_xy = out_of_bounds_x | out_of_bounds_y
        
        # ДИАГНОСТИКА: логируем количество потоков, требующих отражения (по "сырым" координатам до clamp)
        if out_of_bounds_xy.any():
            num_x_left = (pos_for_bounds[:, 0] < -1.0).sum().item()
            num_x_right = (pos_for_bounds[:, 0] > 1.0).sum().item()
            num_y_left = (pos_for_bounds[:, 1] < -1.0).sum().item()
            num_y_right = (pos_for_bounds[:, 1] > 1.0).sum().item()
            logger.debug_forward(f"🔍 OUT OF BOUNDS: X_left={num_x_left}, X_right={num_x_right}, "
                               f"Y_left={num_y_left}, Y_right={num_y_right}, total={out_of_bounds_xy.sum().item()}")
        
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
        
        # ИСПРАВЛЕНО: Проецирование на нормализованные выходные плоскости
        # Проецирование на Z=0 плоскость (norm Z = -1.0)
        if reached_z0_plane.any():
            final_position[reached_z0_plane, 2] = -1.0
        
        # Проецирование на Z=depth плоскость (norm Z = +1.0)
        if reached_zdepth_plane.any():
            final_position[reached_zdepth_plane, 2] = 1.0
        
        # ВАЖНО: НЕ выполняем округление координат в нормализованном пространстве — сохраняем непрерывные значения
        # Квантование выполняется только при индексации дискретных поверхностей/агрегации
        
        return final_position, is_terminated, termination_reasons
    
    def _calculate_displacement_scale(self, global_training_step: Optional[int]) -> float:
        """
        Вычисляет текущий масштаб смещений на основе системы разогрева
        
        Логика:
        - Первые warmup_steps: полный scale
        - Далее: постепенное убывание scale *= decay каждые update_interval шагов
        - Минимум: scale_min (натуральные смещения модели)
        
        Args:
            global_training_step: Текущий шаг обучения
            
        Returns:
            current_scale: Текущий масштаб смещений
        """
        if global_training_step is None or global_training_step < self.config.displacement_warmup_steps:
            # Фаза разогрева: полный scale
            return self.config.displacement_scale
        
        # Фаза убывания: считаем количество интервалов после warmup
        steps_after_warmup = global_training_step - self.config.displacement_warmup_steps
        decay_intervals = steps_after_warmup // self.config.displacement_scale_update_interval
        
        # Применяем экспоненциальное убывание
        current_scale = self.config.displacement_scale * (self.config.displacement_scale_decay ** decay_intervals)
        
        # Ограничиваем минимумом
        current_scale = max(current_scale, self.config.displacement_scale_min)
        
        return current_scale
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния GRU с небольшим шумом для лучшего обучения"""
        hidden = torch.randn(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=torch.float32
        ) * 0.01  # Маленькая дисперсия для стабильности
        logger.debug_init(f"🎲 Initialized GRU hidden state with noise: std=0.01, shape={hidden.shape}")
        return hidden
    
    # УДАЛЕН: check_energy_level() - в архитектуре относительных координат 
    # потоки не умирают от "недостатка энергии". Эмбеддинги - это данные, а не энергия.


def create_energy_carrier(config=None) -> EnergyCarrier:
    """Фабричная функция для создания EnergyCarrier"""
    return EnergyCarrier(config)