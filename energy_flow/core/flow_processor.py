"""
Flow Processor - механизм распространения энергии
=================================================

Управляет параллельной обработкой всех энергетических потоков.
Координирует взаимодействие между SimpleNeuron и EnergyCarrier.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import time

from ..utils.logging import get_logger, log_memory_state, gated_log, summarize_step, format_first_n, DEBUG_PERFORMANCE
from ..config import get_energy_config, create_debug_config, set_energy_config
from ..utils.device_manager import get_device_manager
from .simple_neuron import SimpleNeuron, create_simple_neuron
from .energy_carrier import EnergyCarrier, create_energy_carrier
from .energy_lattice import EnergyLattice, create_energy_lattice

logger = get_logger(__name__)


class FlowProcessor(nn.Module):
    """
    Механизм распространения энергии через решетку
    
    Координирует:
    - Параллельную обработку всех активных потоков
    - Взаимодействие SimpleNeuron и EnergyCarrier
    - Создание новых потоков
    - Управление жизненным циклом потоков
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: EnergyConfig с настройками
        """
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        
        # Основные компоненты
        self.lattice = create_energy_lattice(config)
        self.neuron = create_simple_neuron(config)
        self.carrier = create_energy_carrier(config)
        
        # Embedding mapper - ОБЯЗАТЕЛЬНО для архитектуры
        from .embedding_mapper import EnergyFlowMapper
        self.mapper = EnergyFlowMapper(config)
        # Переносим mapper компоненты на устройство
        self.mapper.input_mapper = self.mapper.input_mapper.to(self.device)
        self.mapper.output_collector = self.mapper.output_collector.to(self.device)
        
        # Переносим на устройство
        self.lattice = self.lattice.to(self.device)
        self.neuron = self.neuron.to(self.device)
        self.carrier = self.carrier.to(self.device)
        # Mapper уже инициализируется на правильном устройстве
        
        # Статистика производительности
        self.perf_stats = {
            'step_times': [],
            'flows_per_step': [],
            'gpu_memory_usage': []
        }
        
        # Статистика убийства потоков
        self.stats = {
            'flows_killed_backward': 0,
            'flows_killed_bounds': 0,
            'flows_killed_energy': 0
        }
        
        # Улучшенная статистика конвергенции со скользящим окном
        self.convergence_stats = {
            'completed_count_history': [],
            'no_improvement_steps': 0,
            'best_completed_count': 0,
            'moving_average_window': 5,  # Размер окна для скользящего среднего
            'last_moving_avg': 0.0,
            'improvement_threshold': 0.01  # Минимальное улучшение для считывания прогресса
        }
        
        # Счетчик всех созданных потоков для точной статистики конвергенции
        self.total_flows_created = 0
        
        # Параметры очистки памяти
        self.memory_cleanup_interval = 10  # Каждые 10 шагов
        self.memory_threshold_gb = 20.0    # Порог для очистки GPU кэша (20GB для RTX 5090)
        self.step_counter = 0
        
        logger.info(f"FlowProcessor initialized on {self.device}")
        logger.info(f"Components: Lattice {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}, "
                   f"SimpleNeuron, EnergyCarrier")
        
        if config.convergence_enabled:
            logger.info(f"Adaptive convergence enabled: threshold={config.convergence_threshold:.2f}, "
                       f"min_steps={config.convergence_min_steps}, patience={config.convergence_patience}")
        
        logger.info(f"Memory management: cleanup every {self.memory_cleanup_interval} steps, threshold={self.memory_threshold_gb}GB")
    
    def forward(self, input_embeddings: torch.Tensor, max_steps: Optional[int] = None, 
                global_training_step: Optional[int] = None) -> torch.Tensor:
        """
        Полный проход энергии через решетку
        
        Args:
            input_embeddings: [batch, 768] - входные эмбеддинги
            max_steps: Максимальное количество шагов (если None - depth решетки)
            global_training_step: Глобальный шаг обучения для curriculum learning
            
        Returns:
            output_embeddings: [batch, 768] - выходные эмбеддинги
        """
        batch_size = input_embeddings.shape[0]
        
        # Размещаем входную энергию с использованием маппера
        self.lattice.reset()
        flow_ids = self.lattice.place_initial_energy(input_embeddings, self.mapper)
        
        # Инициализируем счетчик потоков
        self.total_flows_created = len(flow_ids)
        
        # Определяем количество шагов
        if max_steps is None:
            max_steps = self.config.lattice_depth // 2  # Половина глубины решетки

        logger.info(f"Starting energy propagation: {len(flow_ids)} initial flows, max {max_steps} steps")
        
        # ДИАГНОСТИКА: проверяем начальные позиции потоков
        initial_flows = self.lattice.get_active_flows()
        if initial_flows:
            initial_z_positions = torch.stack([flow.position[2] for flow in initial_flows])
            gated_log(
                logger,
                'DEBUG_ENERGY',
                step=0,
                key='initial_positions',
                msg_or_factory=lambda: f"🏁 INITIAL positions: Z={format_first_n(initial_z_positions, n=10)}",
                first_n_steps=1,
                every=0,
            )
            if torch.any(initial_z_positions != 0):
                logger.debug_energy(
                    f"теперь это не ошибка: Initial flows do NOT start at Z=0! Found Z = {initial_z_positions.unique().tolist()}"
                )
        
        # Сбрасываем статистику конвергенции, сохраняя дефолтные параметры окна/порогов
        initial_flows_count = len(flow_ids)
        # Не пересоздаем словарь, чтобы не потерять ключи как moving_average_window
        self.convergence_stats.update({
            'completed_count_history': [],
            'no_improvement_steps': 0,
            'best_completed_count': 0,
            'last_moving_avg': 0.0,
        })
        # Гарантируем наличие обязательных ключей (на случай старых чекпоинтов/сериализаций)
        if 'moving_average_window' not in self.convergence_stats:
            self.convergence_stats['moving_average_window'] = 5
        if 'improvement_threshold' not in self.convergence_stats:
            self.convergence_stats['improvement_threshold'] = 0.01
        
        # Основной цикл распространения с adaptive convergence
        actual_steps = 0
        for step in range(max_steps):
            actual_steps = step + 1
            active_flows = self.lattice.get_active_flows()
            # Проверяем условие завершения: нет активных потоков
            if not active_flows:
                logger.info(f"No active flows at step {step}, stopping")
                break
            
            # Проверяем конвергенцию (adaptive max_steps)
            if self._check_convergence(step, initial_flows_count):
                logger.log(20, f"Early convergence at step {step}/{max_steps}")
                break
            
            # Если есть активные потоки - обрабатываем их
            if active_flows:
                self.step(active_flows, global_training_step=global_training_step)
            
            # Периодическая очистка памяти
            self.cleanup_memory_safe()
            
            # Логирование прогресса: частотный гейт и ленивое формирование
            stats = self.lattice.get_statistics()
            completion_rate = stats['total_completed'] / initial_flows_count if initial_flows_count > 0 else 0
            gated_log(
                logger,
                'INFO',
                step=step,
                key='flow_step_progress',
                msg_or_factory=lambda: summarize_step({
                    'active': stats['current_active'],
                    'completed': stats['total_completed'],
                    'rate': completion_rate,
                }, step=step, prefix='FLOW'),
                first_n_steps=5,
                every=getattr(self.config, 'log_interval', 10),
            )
            
            # ДЕТАЛИ (только для первых шагов) через частотный гейт
            if active_flows:
                z_positions = torch.stack([flow.position[2] for flow in active_flows])
                def _detail_stats():
                    boundary_stats = {
                        'z_min_boundary': (z_positions <= -0.95).sum().item(),
                        'z_max_boundary': (z_positions >= 0.95).sum().item(),
                        'z_center': ((z_positions > -0.2) & (z_positions < 0.2)).sum().item(),
                        'total': len(active_flows)
                    }
                    return (
                        f"📊 Z: min={z_positions.min():.2f}, max={z_positions.max():.2f}, "
                        f"mean={z_positions.mean():.2f}, std={z_positions.std():.2f}; "
                        f"bounds: z0={boundary_stats['z_min_boundary']}, "
                        f"zdepth={boundary_stats['z_max_boundary']}, center={boundary_stats['z_center']}, "
                        f"total={boundary_stats['total']}"
                    )
                gated_log(
                    logger,
                    'DEBUG_CONVERGENCE',
                    step=step,
                    key='flow_step_details',
                    msg_or_factory=_detail_stats,
                    first_n_steps=5,
                    every=0,
                )
                # Критическая проверка
                max_valid_z = self.config.lattice_depth - 1
                out_of_bounds_flows = (z_positions > max_valid_z * 2).sum().item()
                if out_of_bounds_flows > 0:
                    logger.error(
                        f"🚫 CRITICAL BOUNDS ERROR: {out_of_bounds_flows}/{len(active_flows)} flows have Z > {max_valid_z * 2} (expected max ≈ {max_valid_z})"
                    )
                    logger.error(f"🔍 Z-range in normalization: {self.config.normalization_manager.ranges.z_range}")
                # Гистограмма только в первые шаги по гейту
                def _hist_msg():
                    bins = 20
                    hist = torch.histc(z_positions, bins=bins, min=-1.0, max=1.0)
                    return f"📊 Z histogram (norm, bins={bins}): {format_first_n(hist, n=20)}"
                gated_log(
                    logger,
                    'INFO',
                    step=step,
                    key='flow_step_hist',
                    msg_or_factory=_hist_msg,
                    first_n_steps=3,
                    every=0,
                )
        
        # Собираем выходную энергию из буфера (БЕЗ преобразования в 768D!)
        output_surface_embeddings, completed_flows = self._collect_final_surface_output()
        
        # Если нет выходов, возвращаем нули (surface размерность!)
        if output_surface_embeddings.shape[0] == 0:
            logger.warning("No flows reached output, returning zero surface embeddings")
            surface_dim = self.config.lattice_width * self.config.lattice_height
            output_surface_embeddings = torch.zeros(batch_size, surface_dim, device=self.device)
        
        # Приводим к размеру батча
        if output_surface_embeddings.shape[0] < batch_size:
            # Дополняем нулями
            padding = torch.zeros(batch_size - output_surface_embeddings.shape[0], 
                                output_surface_embeddings.shape[1], device=self.device)
            output_surface_embeddings = torch.cat([output_surface_embeddings, padding], dim=0)
        elif output_surface_embeddings.shape[0] > batch_size:
            # Обрезаем
            output_surface_embeddings = output_surface_embeddings[:batch_size]
        
        # Финальная статистика
        final_stats = self.lattice.get_statistics()
        killed_backward = self.stats['flows_killed_backward']
        killed_bounds = self.stats['flows_killed_bounds']
        killed_energy = self.stats['flows_killed_energy']
        
        # Статистика adaptive max_steps
        steps_saved = max_steps - actual_steps
        if self.config.convergence_enabled and steps_saved > 0:
            speedup = max_steps / actual_steps if actual_steps > 0 else 1.0
            logger.log(20, f"Adaptive convergence saved {steps_saved} steps ({speedup:.2f}x speedup)")
        
        # Компактный свод по завершении шага
        summary = summarize_step({
            'completed': final_stats['total_completed'],
            'died': final_stats['total_died'],
            'killed_energy': killed_energy,
            'killed_backward': killed_backward,
            'killed_bounds': killed_bounds,
        }, step=actual_steps, prefix='FLOW-END')
        logger.info(f"Energy propagation complete ({actual_steps}/{max_steps} steps). {summary}")
        
        # ДОПОЛНИТЕЛЬНАЯ ДИАГНОСТИКА при проблемах
        if killed_backward > initial_flows_count * 0.8:  # Более 80% потоков умерли из-за backward движения
            logger.error(f"🚫 CRITICAL: {killed_backward}/{initial_flows_count} flows died from backward movement!")
            logger.error("🔍 Possible causes: bias not applied, wrong normalization, or curriculum disabled")
            if global_training_step is not None:
                logger.error(f"🔍 Current global_training_step: {global_training_step}")
            else:
                logger.error("🔍 global_training_step is None - curriculum learning disabled!")
        
        return output_surface_embeddings
    
    def _collect_final_output(self, batch_size: int = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Гибридный сбор финальной энергии
        
        Проверяет активные потоки и буфер, собирает энергию когда все готово.
        
        Args:
            batch_size: Размер батча для восстановления выходных эмбеддингов (опционально)
        
        Returns:
            output_embeddings: [batch, embedding_dim] - собранные эмбеддинги
            completed_flows: ID завершенных потоков
        """
        active_flows = self.lattice.get_active_flows()
        
        logger.debug(f"Final collection: {len(active_flows)} active flows")
        
        # НОВАЯ АРХИТЕКТУРА: Если есть активные потоки на выходной стороне - помечаем их как завершенные
        active_at_output = 0
        for flow in active_flows:
            z_pos = flow.position[2].item()
            if z_pos >= self.config.lattice_depth - 1:
                # Поток достиг выхода - помечаем как завершенный
                self.lattice._mark_flow_completed_zdepth_plane(flow.id)
                active_at_output += 1
            elif z_pos <= 0:
                # Поток достиг начала - помечаем как завершенный на Z=0
                self.lattice._mark_flow_completed_z0_plane(flow.id)  
                active_at_output += 1
        
        if active_at_output > 0:
            logger.debug(f"Marked {active_at_output} remaining flows as completed")
        
        # Определяем batch_size если не передан
        if batch_size is None:
            # Попробуем получить из config или по количеству активных потоков
            batch_size = self.config.batch_size
            if batch_size is None:
                logger.error(f"Cannot determine batch_size")

        # НОВАЯ АРХИТЕКТУРА: Собираем энергию напрямую из завершенных потоков (без буферизации)
        # ВОССТАНОВЛЕНИЕ 768D ЧЕРЕЗ MAPPER: используем surface-агрегацию и обратный маппер
        output_embeddings, completed_flows = self.lattice.collect_completed_flows_direct(self.mapper, expected_batch_size=batch_size)
        
        # Очищаем завершенные потоки после сбора
        if completed_flows:
            # Удаляем завершенные потоки из активного списка
            for flow_id in completed_flows:
                if flow_id in self.lattice.active_flows:
                    del self.lattice.active_flows[flow_id]
            logger.info(f"Collected and removed {len(completed_flows)} completed flows")
        
        return output_embeddings, completed_flows
    
    def _collect_final_surface_output(self) -> Tuple[torch.Tensor, List[int]]:
        """
        Собирает surface embeddings БЕЗ преобразования в 768D
        
        Returns:
            output_surface_embeddings: [batch, surface_dim] - surface embeddings
            completed_flows: ID завершенных потоков
        """
        active_flows = self.lattice.get_active_flows()
        
        logger.debug(f"Surface collection: {len(active_flows)} active flows")
        
        # НОВАЯ АРХИТЕКТУРА: Если есть активные потоки на выходной стороне - помечаем их как завершенные
        active_at_output = 0
        for flow in active_flows:
            z_pos = flow.position[2].item()
            if z_pos >= self.config.lattice_depth - 1:
                # Поток достиг выхода - помечаем как завершенный
                self.lattice._mark_flow_completed_zdepth_plane(flow.id)
                active_at_output += 1
            elif z_pos <= 0:
                # Поток достиг начала - помечаем как завершенный на Z=0
                self.lattice._mark_flow_completed_z0_plane(flow.id)
                active_at_output += 1
        
        if active_at_output > 0:
            logger.debug(f"Marked {active_at_output} remaining flows as completed")
        
        # НОВАЯ АРХИТЕКТУРА: Собираем surface embeddings напрямую из завершенных потоков (без буферизации)
        if self.lattice.tensor_storage is not None:
            output_surface_embeddings, completed_flows = self.lattice.collect_completed_flows_surface_direct_tensorized()
        else:
            output_surface_embeddings, completed_flows = self.lattice.collect_completed_flows_surface_direct()
        
        # Очищаем завершенные потоки после сбора
        if completed_flows:
            # Удаляем завершенные потоки из активного списка
            for flow_id in completed_flows:
                if flow_id in self.lattice.active_flows:
                    del self.lattice.active_flows[flow_id]
            logger.info(f"Collected and removed {len(completed_flows)} completed flows")
        
        return output_surface_embeddings, completed_flows
    
    def step(self, active_flows: Optional[List] = None, global_training_step: Optional[int] = None):
        """
        Один шаг распространения энергии для всех активных потоков
        ПОЛНАЯ ПАРАЛЛЕЛИЗАЦИЯ: все потоки обрабатываются одновременно вместо sequential batches
        
        Args:
            active_flows: Список активных потоков (если None - получаем из lattice)
        """
        start_time = time.time()
        phase = 'tensorized' if self.lattice.tensor_storage is not None else 'list'
        
        if self.lattice.tensor_storage is not None:
            # Tensorized fast path
            positions, energies, hidden_states, flow_ids, ages, steps_taken = self.lattice.tensor_storage.get_active_data()
            if positions.numel() == 0:
                return
            flows_count = positions.shape[0]
            t0 = time.time()
            self._process_flow_batch_tensorized(
                positions,
                energies,
                hidden_states,  # [batch, layers, hidden]
                flow_ids,
                ages,
                steps_taken,
                global_training_step=global_training_step
            )
            proc_time = (time.time() - t0) * 1000.0
            if logger.isEnabledFor(10):
                logger.log(DEBUG_PERFORMANCE, f"STEP[{phase}] flows={flows_count}: process_batch_tensorized={proc_time:.1f}ms")
            # Lazy consistency logs for first few flows
            try:
                if logger.isEnabledFor(10):  # DEBUG
                    # 1) Первые K потоков: сравнение позиций/энергий
                    k = min(3, flow_ids.shape[0])
                    for i in range(k):
                        fid = flow_ids[i].item()
                        pos_s = positions[i].tolist()
                        eng_s = energies[i].view(-1).item() if energies[i].numel()==1 else energies[i][0].item()
                        if fid in self.lattice.active_flows:
                            f = self.lattice.active_flows[fid]
                            pos_l = [round(v.item(), 6) for v in f.position]
                            eng_l = f.energy.item() if f.energy.numel()==1 else f.energy[0].item()
                            logger.debug(f"CONSISTENCY flow_id={fid}: pos_storage={list(map(lambda x: round(x,6), pos_s))} pos_lattice={pos_l}; energy_storage={eng_s:.6f} energy_lattice={eng_l:.6f}")
                    # 2) Сводка по количеству активных и первые ID
                    active_lattice_ids = [fid for fid, fl in self.lattice.active_flows.items() if fl.is_active]
                    count_lattice = len(active_lattice_ids)
                    count_storage = flow_ids.shape[0]
                    sample_ids_l = active_lattice_ids[:3]
                    sample_ids_s = flow_ids[:3].detach().cpu().tolist()
                    logger.debug(f"CONSISTENCY counts: lattice_active={count_lattice}, storage_active={count_storage}; sample_lattice_ids={sample_ids_l}, sample_storage_ids={sample_ids_s}")
            except Exception as e:
                logger.debug(f"Consistency log skipped due to error: {e}")
        else:
            if active_flows is None:
                active_flows = self.lattice.get_active_flows()
            
            if not active_flows:
                return
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: убираем Sequential Processing Bottleneck!
            # Вместо цикла с маленькими batch'ами - обрабатываем ВСЕ потоки сразу
            # Это позволяет GPU cores работать параллельно с 1000+ потоками одновременно
            
            flows_count = len(active_flows)
            max_flows_per_step = self.config.max_active_flows  # RTX 5090 может обработать все сразу
            if flows_count <= max_flows_per_step:
                # Оптимальный случай: обрабатываем ВСЕ потоки одним большим batch'ем
                t0 = time.time()
                self._process_flow_batch(active_flows, global_training_step=global_training_step)
                proc_time = (time.time() - t0) * 1000.0
                if logger.isEnabledFor(10):
                    logger.log(DEBUG_PERFORMANCE, f"STEP[{phase}] flows={flows_count}: process_flow_batch={proc_time:.1f}ms")
            else:
                # Fallback: если слишком много потоков (>200K), делим на крупные chunk'и
                optimal_chunk_size = max_flows_per_step // 2  # 100K потоков за раз
                
                for i in range(0, flows_count, optimal_chunk_size):
                    chunk_flows = active_flows[i:i + optimal_chunk_size]
                    t0 = time.time()
                    self._process_flow_batch(chunk_flows, global_training_step=global_training_step)
                    proc_time = (time.time() - t0) * 1000.0
                    if logger.isEnabledFor(10):
                        logger.log(DEBUG_PERFORMANCE, f"STEP[{phase}] chunk[{i}:{i+len(chunk_flows)}]: process_flow_batch={proc_time:.1f}ms")
                        logger.log(DEBUG_PERFORMANCE, f"STEP[{phase}] chunk[{i}:{i+len(chunk_flows)}]: process_flow_batch={proc_time:.1f}ms")
        
        # Статистика
        step_time = time.time() - start_time
        self.perf_stats['step_times'].append(step_time)
        self.perf_stats['flows_per_step'].append(flows_count)
        
        if self.device.type == 'cuda':
            memory_info = self.device_manager.get_memory_info()
            self.perf_stats['gpu_memory_usage'].append(memory_info['gpu_allocated_gb'])
    
    def _sanitize_tensor(self, t: torch.Tensor, clip_value: float = 10.0, clamp_only: bool = False) -> torch.Tensor:
        """Заменяет NaN/Inf и ограничивает экстремальные значения для стабильности RNN."""
        if t is None:
            return t
        t = torch.nan_to_num(t, nan=0.0, posinf=clip_value, neginf=-clip_value)
        if not clamp_only:
            return t.clamp_(-clip_value, clip_value)
        return t

    def _process_flow_batch(self, flows, global_training_step: Optional[int] = None):
        """Обрабатывает батч потоков с векторизованными операциями"""
        batch_size = len(flows)
        t_collect = time.time()
        
        # Собираем данные потоков и ID
        positions = torch.stack([f.position for f in flows])  # [batch, 3]
        energies = torch.stack([f.energy for f in flows])     # [batch, embedding_dim]
        hidden_states = torch.stack([f.hidden_state for f in flows])  # [batch, layers, hidden]

        # Санитизация входов (нормализованные позиции и энергии)
        positions = self._sanitize_tensor(positions, clip_value=1.0)  # clamp к [-1,1]
        energies = self._sanitize_tensor(energies, clip_value=10.0)
        flow_ids = torch.tensor([f.id for f in flows], dtype=torch.long, device=positions.device)
        
        # Транспонируем hidden states для GRU и делаем непрерывными
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # [layers, batch, hidden]
        
        collect_ms = (time.time() - t_collect) * 1000.0
        # 1. SimpleNeuron обрабатывает позиции и энергии
        t_neuron = time.time()
        neuron_output = self.neuron(positions, energies)  # [batch, neuron_output_dim]
        # Санитизация выхода нейрона
        neuron_output = self._sanitize_tensor(neuron_output, clip_value=10.0)
        neuron_ms = (time.time() - t_neuron) * 1000.0
        
        # Собираем возраста потоков для progressive bias
        ages = torch.tensor([flow.age for flow in flows], dtype=torch.float32, device=positions.device)
        
        # 2. EnergyCarrier генерирует следующее состояние с curriculum learning
        t_carrier = time.time()
        carrier_output, new_hidden = self.carrier(
            neuron_output, 
            energies,
            hidden_states,
            positions,
            flow_age=ages,
            global_training_step=global_training_step  # Передаем глобальный шаг
        )
        # Санитизация выходов carrier
        try:
            carrier_output.next_position = self._sanitize_tensor(getattr(carrier_output, 'next_position', None), clip_value=1.0)
            carrier_output.energy_value = self._sanitize_tensor(getattr(carrier_output, 'energy_value', None), clip_value=10.0)
            if hasattr(carrier_output, 'raw_next_position') and carrier_output.raw_next_position is not None:
                carrier_output.raw_next_position = self._sanitize_tensor(carrier_output.raw_next_position, clip_value=1.0)
        except Exception:
            pass
        
        # Транспонируем обратно и делаем непрерывными
        new_hidden = new_hidden.transpose(0, 1).contiguous()  # [batch, layers, hidden]
        carrier_ms = (time.time() - t_carrier) * 1000.0
        
        # 3. ВЕКТОРИЗОВАННАЯ обработка результатов
        t_results = time.time()
        self._process_results_vectorized(flows, flow_ids, positions, carrier_output, new_hidden)
        results_ms = (time.time() - t_results) * 1000.0
        if logger.isEnabledFor(10):
            logger.log(DEBUG_PERFORMANCE, f"BATCH timings: collect={collect_ms:.1f}ms, neuron={neuron_ms:.1f}ms, carrier={carrier_ms:.1f}ms, results={results_ms:.1f}ms (batch={batch_size})")
        # Throughput metric (flows per second) for this batch
        try:
            fps = (batch_size / ((collect_ms + neuron_ms + carrier_ms + results_ms) / 1000.0)) if (collect_ms + neuron_ms + carrier_ms + results_ms)  0 else 0.0
            logger.log(DEBUG_PERFORMANCE, f"Throughput[step_batch]: {fps:.1f} flows/s (batch={batch_size})")
        except Exception:
            pass
    
    def _process_results_vectorized(self, flows, flow_ids, current_positions, carrier_output, new_hidden):
        """Векторизованная обработка результатов carrier_output с поддержкой относительных координат"""
        batch_size = len(flows)
        device = current_positions.device
        
        # УДАЛЕНО: energy_alive_mask - в новой архитектуре потоки не умирают от недостатка энергии
        # Все потоки считаются живыми, завершение только по termination_reasons
        
        # Обрабатываем termination_reasons из EnergyCarrier
        termination_reasons = carrier_output.termination_reason
        is_terminated = carrier_output.is_terminated  # [batch]
        
        # Разбираем причины завершения для статистики
        reached_z0_count = sum(1 for reason in termination_reasons if reason == "reached_z0_plane")
        reached_zdepth_count = sum(1 for reason in termination_reasons if reason == "reached_zdepth_plane")
        reflection_needed_count = sum(1 for reason in termination_reasons if reason == "xy_reflection_needed")
        active_count = sum(1 for reason in termination_reasons if reason == "active")
        
        # УДАЛЕНО: energy_dead_count - потоки больше не умирают от энергии
        # Статистика теперь основана только на termination_reasons
        
        logger.debug_energy(f"🎯 Termination breakdown: z0={reached_z0_count}, zdepth={reached_zdepth_count}, "
                           f"reflection={reflection_needed_count}, active={active_count}")
        
        # В новой архитектуре логика намного проще - только 3 типа потоков:
        # 1. Достигшие выходных плоскостей (буферизуем)
        # 2. Требующие отражения (применяем отражение) 
        # 3. Активные (обновляем позицию)
        
        # Маска потоков, достигших выходных плоскостей
        output_reached_mask = is_terminated
        
        # Маска потоков, требующих отражения (по raw_next_position до clamp)
        reflection_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        raw_next = getattr(carrier_output, 'raw_next_position', None)
        if raw_next is not None:
            reflection_mask = (raw_next[:, 0] < -1.0) | (raw_next[:, 0] > 1.0) | (raw_next[:, 1] < -1.0) | (raw_next[:, 1] > 1.0)
        else:
            for i, reason in enumerate(termination_reasons):
                if reason == "xy_reflection_needed":
                    reflection_mask[i] = True
        
        # Маска активных потоков
        active_mask = ~is_terminated
        
        # ДВУХУРОВНЕВАЯ ПРОЕКЦИОННАЯ СИСТЕМА
        # Проверяем потоки, которые сделали depth/2 шагов но не достигли выходных плоскостей
        t_projmask = time.time()
        depth_half = self.config.lattice_depth // 2
        # Векторизованное вычисление projection_mask (вместо цикла)
        try:
            steps_taken_tensor = torch.tensor([f.steps_taken for f in flows], device=device)
            projection_mask = active_mask & (steps_taken_tensor >= depth_half)
        except Exception:
            # Fallback: пустая маска
            projection_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        projmask_ms = (time.time() - t_projmask) * 1000.0
        if logger.isEnabledFor(10):
            logger.log(DEBUG_PERFORMANCE, f"RESULTS: build_masks: projection_mask={projmask_ms:.1f}ms (count={int(projection_mask.sum().item())})")
        
        # Обрабатываем потоки для проекции
        t_project = time.time()
        if projection_mask.any():
            projected_count = projection_mask.sum().item()
            logger.info(f"📊 Projecting {projected_count} flows to nearest output surface (completed ≥{depth_half} steps)")
            
            # Извлекаем потоки для проекции
            projection_flow_ids = flow_ids[projection_mask]
            projection_positions = carrier_output.next_position[projection_mask]
            projection_energies = carrier_output.energy_value[projection_mask]
            projection_hidden = new_hidden[projection_mask]
            
            # Векторизованно определяем ближайшую поверхность (в НОРМАЛИЗОВАННОМ Z)
            proj_z = projection_positions[:, 2]
            dist_z0 = (proj_z - self.lattice.norm_z0).abs()
            dist_zd = (proj_z - self.lattice.norm_zdepth).abs()
            to_z0_mask = dist_z0 <= dist_zd
            to_zd_mask = ~to_z0_mask
            
            # Формируем проецированные позиции
            projected_positions = projection_positions.clone()
            projected_positions[to_z0_mask, 2] = self.lattice.norm_z0
            projected_positions[to_zd_mask, 2] = self.lattice.norm_zdepth
            
            # Обновляем состояния потоков батчем
            self.lattice.batch_update_flows(
                projection_flow_ids,
                projected_positions,
                projection_energies,
                projection_hidden,
            )
            
            # Помечаем завершенными батчем
            if to_z0_mask.any():
                self.lattice.mark_flows_completed_z0_batch(projection_flow_ids[to_z0_mask])
            if to_zd_mask.any():
                self.lattice.mark_flows_completed_zdepth_batch(projection_flow_ids[to_zd_mask])
            
            # Убираем проецированные потоки из активных
            active_mask = active_mask & ~projection_mask
        
        # Дополнительная фильтрация по длине смещения (переосмысленный carrier_dropout)
        if self.config.enable_displacement_filtering:
            # Вычисляем смещения из текущих и следующих позиций
            displacements = carrier_output.next_position - current_positions  # [batch, 3]
            displacement_lengths = torch.norm(displacements, dim=1)  # [batch]
            
            # Маска потоков с маленькими смещениями ("топчущиеся")
            small_displacement_mask = displacement_lengths < self.config.min_displacement_threshold
            
            # Логируем статистику фильтрации
            if small_displacement_mask.any():
                filtered_count = small_displacement_mask.sum().item()
                logger.debug_relative(f"🔍 Filtered {filtered_count}/{batch_size} flows with small displacements "
                                     f"(< {self.config.min_displacement_threshold:.2f})")
            
            # Исключаем потоки с маленькими смещениями из активных
            active_mask = active_mask & ~small_displacement_mask
        
        # 2. Буферизуем потоки, достигшие выходных плоскостей
        if output_reached_mask.any():
            output_flow_ids = flow_ids[output_reached_mask]
            output_positions = carrier_output.next_position[output_reached_mask]
            output_energies = carrier_output.energy_value[output_reached_mask]
            output_hidden = new_hidden[output_reached_mask]
            
            # Обновляем позиции и буферизуем
            for i, flow_id in enumerate(output_flow_ids):
                flow_id_item = flow_id.item()
                new_position = output_positions[i]
                new_energy = output_energies[i]
                new_hidden_state = output_hidden[i]
                
                # НОВАЯ АРХИТЕКТУРА: Помечаем поток как завершенный без буферизации
                z_pos = new_position[2].item()
                if z_pos <= 0:
                    self.lattice._mark_flow_completed_z0_plane(flow_id_item)
                elif z_pos >= self.config.lattice_depth:
                    self.lattice._mark_flow_completed_zdepth_plane(flow_id_item)
                
                # Обновляем поток перед буферизацией
                if flow_id_item in self.lattice.active_flows:
                    self.lattice.active_flows[flow_id_item].position = new_position
                    self.lattice.active_flows[flow_id_item].energy = new_energy
                    self.lattice.active_flows[flow_id_item].hidden_state = new_hidden_state
                    self.lattice.active_flows[flow_id_item].age += 1
        
        # 3. Применяем отражение границ (если включено)
        if reflection_mask.any() and self.config.boundary_reflection_enabled:
            reflection_flow_ids = flow_ids[reflection_mask]
            reflection_count = reflection_mask.sum().item()
            
            # ДИАГНОСТИКА: логируем агрегированные счетчики и примеры до отражения
            reflection_positions_before = (raw_next if raw_next is not None else carrier_output.next_position)[reflection_mask]
            x_left = (reflection_positions_before[:, 0] < -1.0).sum().item()
            x_right = (reflection_positions_before[:, 0] > 1.0).sum().item()
            y_left = (reflection_positions_before[:, 1] < -1.0).sum().item()
            y_right = (reflection_positions_before[:, 1] > 1.0).sum().item()
            logger.debug_reflection(f"🔄 BEFORE reflection: {reflection_count} flows need reflection | X_left={x_left}, X_right={x_right}, Y_left={y_left}, Y_right={y_right}")
            # Показываем первые 3 примера потоков до отражения
            for i in range(min(3, reflection_count)):
                flow_id = reflection_flow_ids[i].item()
                pos = reflection_positions_before[i]
                logger.debug_reflection(f"🔄 Flow {flow_id} before: X={pos[0].item():.6f}, Y={pos[1].item():.6f}, Z={pos[2].item():.6f}")
            
            reflection_positions = self.reflect_boundaries(reflection_positions_before, reflection_flow_ids)
            reflection_energies = carrier_output.energy_value[reflection_mask]
            reflection_hidden = new_hidden[reflection_mask]
            
            # Обновляем потоки с отраженными позициями
            self.lattice.batch_update_flows(
                reflection_flow_ids,
                reflection_positions,
                reflection_energies,
                reflection_hidden
            )
        
        # 4. Обновляем активные потоки
        final_active_mask = active_mask
        if reflection_mask.any() and not self.config.boundary_reflection_enabled:
            # Если отражение отключено, потоки с xy_reflection_needed остаются активными
            final_active_mask = active_mask | reflection_mask
        
        if final_active_mask.any():
            active_flow_ids = flow_ids[final_active_mask]
            active_positions = carrier_output.next_position[final_active_mask]
            active_energies = carrier_output.energy_value[final_active_mask]
            active_hidden = new_hidden[final_active_mask]
            
            # Batch update all active flows
            self.lattice.batch_update_flows(
                active_flow_ids,
                active_positions,
                active_energies,
                active_hidden
            )
        
        # 5. Обработка spawn потоков (оптимизированная)
        # Проверяем spawn на основе длины смещения, если включено
        if self.config.movement_based_spawn:
            movement_spawn_info = self._check_movement_spawn(current_positions, carrier_output.next_position, flow_ids)
            if movement_spawn_info:
                carrier_output.spawn_info.extend(movement_spawn_info)
                logger.debug_spawn_movement(f"🎆 Added {len(movement_spawn_info)} movement-based spawns")
        
        self._process_spawns_optimized(flows, carrier_output, final_active_mask, current_positions)
    
    def _process_flow_batch_tensorized(self,
                                      positions: torch.Tensor,
                                      energies: torch.Tensor,
                                      hidden_states: torch.Tensor,
                                      flow_ids: torch.Tensor,
                                      ages: torch.Tensor,
                                      steps_taken: torch.Tensor,
                                      global_training_step: Optional[int] = None):
        """Полностью тензоризованный путь обработки батча потоков"""
        batch_size = positions.shape[0]
        device = positions.device
        
        # Приводим hidden_states к [layers, batch, hidden]
        hidden_for_gru = hidden_states.transpose(0, 1).contiguous()
        
        # 1. SimpleNeuron
        neuron_output = self.neuron(positions, energies)
        
        # 2. EnergyCarrier
        carrier_output, new_hidden = self.carrier(
            neuron_output,
            energies,
            hidden_for_gru,
            positions,
            flow_age=ages.to(device),
            global_training_step=global_training_step
        )
        new_hidden_bt = new_hidden.transpose(0, 1).contiguous()  # [batch, layers, hidden]
        
        # Termination masks
        is_terminated = carrier_output.is_terminated
        termination_reasons = carrier_output.termination_reason
        
        # Reflection mask by raw positions if provided; fallback to reasons list
        reflection_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        raw_next = getattr(carrier_output, 'raw_next_position', None)
        if raw_next is not None:
            reflection_mask = (raw_next[:, 0] < -1.0) | (raw_next[:, 0] > 1.0) | (raw_next[:, 1] < -1.0) | (raw_next[:, 1] > 1.0)
        else:
            for i, reason in enumerate(termination_reasons):
                if reason == "xy_reflection_needed":
                    reflection_mask[i] = True
        
        active_mask = ~is_terminated
        
        # Projection mask using steps_taken
        projection_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        depth_half = self.config.lattice_depth // 2
        projection_mask = active_mask & (steps_taken.to(device) >= depth_half)
        
        # Handle projections
        if projection_mask.any():
            proj_ids = flow_ids[projection_mask]
            proj_pos = carrier_output.next_position[projection_mask]
            proj_energy = carrier_output.energy_value[projection_mask]
            proj_hidden = new_hidden_bt[projection_mask]
            # Decide nearest surface by Z distance in raw space assumption consistent with existing logic
            z_curr = proj_pos[:, 2]
            dist_z0 = (z_curr - 0).abs()
            dist_zd = (z_curr - self.config.lattice_depth).abs()
            to_z0 = dist_z0 <= dist_zd
            # For logging and marking
            ids_list = proj_ids.detach().cpu().tolist()
            z0_ids = [ids_list[i] for i, flag in enumerate(to_z0.detach().cpu().tolist()) if flag]
            zd_ids = [ids_list[i] for i, flag in enumerate((~to_z0).detach().cpu().tolist()) if flag]
            if z0_ids:
                self.lattice._mark_flow_completed_z0_plane_batch = getattr(self.lattice, "_mark_flow_completed_z0_plane", None)
                for fid in z0_ids:
                    self.lattice._mark_flow_completed_z0_plane(fid)
            if zd_ids:
                for fid in zd_ids:
                    self.lattice._mark_flow_completed_zdepth_plane(fid)
            active_mask = active_mask & (~projection_mask)
        
        # Output reached processing
        if is_terminated.any():
            out_ids = flow_ids[is_terminated]
            out_pos = carrier_output.next_position[is_terminated]
            out_energy = carrier_output.energy_value[is_terminated]
            out_hidden = new_hidden_bt[is_terminated]
            # Mark completed depending on normalized Z
            z_vals = out_pos[:, 2]
            to_z0_mask = (z_vals - self.lattice.norm_z0).abs() <= (z_vals - self.lattice.norm_zdepth).abs()
            to_zd_mask = ~to_z0_mask
            # Проецируем Z к плоскостям для консистентности
            out_pos[to_z0_mask, 2] = self.lattice.norm_z0
            out_pos[to_zd_mask, 2] = self.lattice.norm_zdepth
            # Обновляем состояния потоков сразу
            self.lattice.batch_update_flows(out_ids, out_pos, out_energy, out_hidden)
            # Отмечаем завершенными батчем
            if to_z0_mask.any():
                self.lattice.mark_flows_completed_z0_batch(out_ids[to_z0_mask])
            if to_zd_mask.any():
                self.lattice.mark_flows_completed_zdepth_batch(out_ids[to_zd_mask])
        
        # Reflection
        if reflection_mask.any() and self.config.boundary_reflection_enabled:
            refl_ids = flow_ids[reflection_mask]
            refl_pos_before = (raw_next if raw_next is not None else carrier_output.next_position)[reflection_mask]
            refl_pos = self.reflect_boundaries(refl_pos_before, refl_ids)
            refl_energy = carrier_output.energy_value[reflection_mask]
            refl_hidden = new_hidden_bt[reflection_mask]
            self.lattice.batch_update_flows(refl_ids, refl_pos, refl_energy, refl_hidden)
        
        # Active updates
        final_active = active_mask | (reflection_mask & ~self.config.boundary_reflection_enabled)
        if final_active.any():
            act_ids = flow_ids[final_active]
            act_pos = carrier_output.next_position[final_active]
            act_energy = carrier_output.energy_value[final_active]
            act_hidden = new_hidden_bt[final_active]
            self.lattice.batch_update_flows(act_ids, act_pos, act_energy, act_hidden)
        
        # Movement-based spawns
        if self.config.movement_based_spawn:
            mv_info = self._check_movement_spawn(positions, carrier_output.next_position, flow_ids)
            if mv_info:
                # Reuse existing spawn handler that expects flows and indices; build minimal flows proxy
                class _F: pass
                flows_proxy = []
                id_to_idx = {fid.item(): i for i, fid in enumerate(flow_ids)}
                for fid in flow_ids.detach().cpu().tolist():
                    f = _F()
                    f.id = fid
                    flows_proxy.append(f)
                alive_mask = final_active
                self._process_spawns_optimized(flows_proxy, type("O", (), {"spawn_info": mv_info})(), alive_mask, positions)
        
        return
    
    def reflect_boundaries(self, position: torch.Tensor, flow_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Отражение границ для нормализованного пространства [-1, 1]
        
        Args:
            position: [batch, 3] - позиции для отражения в нормализованном пространстве
            flow_ids: [batch] - ID потоков (для логирования)
            
        Returns:
            reflected_position: [batch, 3] - позиции с отраженными X/Y в [-1, 1]
        """
        reflected_pos = position.clone()
        x, y, z = reflected_pos[:, 0], reflected_pos[:, 1], reflected_pos[:, 2]
        
        # Отражение X координаты в нормализованном пространстве [-1, 1]
        x = torch.where(x < -1.0, -2.0 - x, x)  # Отражение от левой границы -1
        x = torch.where(x > 1.0, 2.0 - x, x)    # Отражение от правой границы 1
        
        # Отражение Y координаты в нормализованном пространстве [-1, 1]
        y = torch.where(y < -1.0, -2.0 - y, y)
        y = torch.where(y > 1.0, 2.0 - y, y)
        
        # Z остается без изменений (движение только к выходным плоскостям)
        
        reflected_pos[:, 0] = x
        reflected_pos[:, 1] = y

        # Детальное логирование первых примеров отражения
        num_reflected = position.shape[0]

        # Считаем количество отражений по осям ПО СЫРЫМ ВХОДНЫМ позициям
        x_reflected_left = (position[:, 0] < -1.0).sum().item()
        x_reflected_right = (position[:, 0] > 1.0).sum().item()
        y_reflected_left = (position[:, 1] < -1.0).sum().item()
        y_reflected_right = (position[:, 1] > 1.0).sum().item()

        # Логируем детальные примеры первых 3-х отражений
        reflection_examples = []
        for i in range(min(3, num_reflected)):
            orig_x, orig_y, orig_z = position[i, 0].item(), position[i, 1].item(), position[i, 2].item()
            new_x, new_y = x[i].item(), y[i].item()

            # Определяем тип отражения
            reflection_type = []
            if orig_x < -1.0:
                reflection_type.append(f"X<-1({orig_x:.3f}→{new_x:.3f})")
            elif orig_x > 1.0:
                reflection_type.append(f"X>1({orig_x:.3f}→{new_x:.3f})")
            if orig_y < -1.0:
                reflection_type.append(f"Y<-1({orig_y:.3f}→{new_y:.3f})")
            elif orig_y > 1.0:
                reflection_type.append(f"Y>1({orig_y:.3f}→{new_y:.3f})")

            if reflection_type:
                reflection_examples.append(f"flow_{i}[{','.join(reflection_type)}]")

        # Агрегированная статистика
        logger.debug_reflection(
            f"🔄 Reflected candidates: {num_reflected} | X_left={x_reflected_left}, X_right={x_reflected_right}, "
            f"Y_left={y_reflected_left}, Y_right={y_reflected_right}"
        )

        # Детальные примеры
        if reflection_examples:
            logger.debug_reflection(f"🔄 Examples: {', '.join(reflection_examples)}")

        # Финальные диапазоны
        logger.debug_reflection(
            f"🔄 Post-reflection ranges: X[{x.min().item():.3f}, {x.max().item():.3f}], "
            f"Y[{y.min().item():.3f}, {y.max().item():.3f}]"
        )

        # ДИАГНОСТИКА: показываем результат отражения для первых 3 потоков
        if flow_ids is not None:
            logger.debug_reflection(f"🔄 AFTER reflection examples:")
            for i in range(min(3, len(position))):
                flow_id = flow_ids[i].item()
                orig_pos = position[i]
                new_pos = reflected_pos[i]
                logger.debug_reflection(
                    f"🔄 Flow {flow_id} after: X={new_pos[0].item():.6f}, Y={new_pos[1].item():.6f}, Z={new_pos[2].item():.6f} "
                    f"(ΔX={new_pos[0].item() - orig_pos[0].item():.6f}, ΔY={new_pos[1].item() - orig_pos[1].item():.6f})"
                )

        return reflected_pos
    
    def _check_movement_spawn(self, current_positions: torch.Tensor, 
                             next_positions: torch.Tensor, 
                             flow_ids: torch.Tensor) -> List:
        """
        Проверяет spawn на основе длины смещения
        
        Args:
            current_positions: [batch, 3] - текущие позиции
            next_positions: [batch, 3] - следующие позиции
            flow_ids: [batch] - ID потоков
            
        Returns:
            spawn_info: Список SpawnInfo для новых потоков
        """
        # Вычисляем смещения
        displacement = next_positions - current_positions  # [batch, 3]
        displacement_lengths = torch.norm(displacement, dim=1)  # [batch]
        
        # Порог для spawn в нормализованном пространстве [-1, 1]
        threshold = self.config.spawn_movement_threshold_ratio  # Прямо в нормализованном пространстве
        
        # Маска для spawn
        spawn_mask = displacement_lengths > threshold
        
        if not spawn_mask.any():
            return []
        
        spawn_info_list = []
        total_candidates = spawn_mask.sum().item()
        total_potential_spawns = 0
        total_actual_spawns = 0
        total_limited_spawns = 0
        spawn_examples = []
        
        # Обрабатываем каждый поток, который должен создать spawn
        spawn_indices = torch.where(spawn_mask)[0]
        for idx in spawn_indices:
            idx_val = idx.item()
            delta_length = displacement_lengths[idx].item()
            flow_id = flow_ids[idx].item()
            
            # Количество дополнительных потоков (исправленная формула)
            potential_spawns = int(delta_length / threshold) - 1
            actual_spawns = min(potential_spawns, self.config.max_spawn_per_step)
            
            total_potential_spawns += potential_spawns
            
            if actual_spawns > 0:
                # Получаем энергию родительского потока
                if flow_id in self.lattice.active_flows:
                    parent_energy = self.lattice.active_flows[flow_id].energy
                    spawn_energies = [parent_energy.clone() for _ in range(actual_spawns)]
                    
                    # Создаем SpawnInfo структуру (должна быть импортирована)
                    from .energy_carrier import SpawnInfo
                    spawn_info = SpawnInfo(
                        energies=spawn_energies,
                        parent_batch_idx=idx_val  # Индекс в батче
                    )
                    spawn_info_list.append(spawn_info)
                    
                    total_actual_spawns += actual_spawns
                    if potential_spawns > actual_spawns:
                        total_limited_spawns += (potential_spawns - actual_spawns)
                    
                    # Детальные примеры для первых 3-х spawn'ов
                    if len(spawn_examples) < 3:
                        spawn_examples.append(f"flow_{flow_id}[disp={delta_length:.3f}→{actual_spawns}spawns]")
        
        # Агрегированное логирование
        if total_candidates > 0:
            logger.debug_spawn_movement(f"🎆 Movement spawn summary: {total_candidates} candidates, "
                                       f"{total_potential_spawns} potential → {total_actual_spawns} actual spawns")
            if total_limited_spawns > 0:
                logger.debug_spawn_movement(f"🎆 Limited by config: {total_limited_spawns} spawns restricted by max_spawn_per_step={self.config.max_spawn_per_step}")
            
            # Детальные примеры
            if spawn_examples:
                logger.debug_spawn_movement(f"🎆 Examples: {', '.join(spawn_examples)}")
        
        return spawn_info_list
    
    def _process_spawns_optimized(self, flows, carrier_output, alive_mask, current_positions):
        """Оптимизированная обработка spawn потоков"""
        if not carrier_output.spawn_info:
            return
        
        # Создаем индекс spawn_info по parent_batch_idx для O(1) поиска
        spawn_by_idx = {}
        for spawn_info in carrier_output.spawn_info:
            spawn_by_idx[spawn_info.parent_batch_idx] = spawn_info
        
        # Статистика spawn'ов
        total_spawn_requests = len(spawn_by_idx)
        total_spawned = 0
        spawn_examples = []
        parent_flows = []
        
        # Обрабатываем spawn'ы только для живых потоков
        alive_indices = torch.where(alive_mask)[0] if alive_mask.any() else torch.tensor([], dtype=torch.long)
        for idx in alive_indices:
            idx_val = idx.item()
            if idx_val in spawn_by_idx:
                spawn_info = spawn_by_idx[idx_val]
                if spawn_info.energies:
                    # Ограничиваем количество spawn'ов конфигом
                    spawn_energies = spawn_info.energies[:self.config.max_spawn_per_step]
                    flow_id = flows[idx_val].id
                    # Спавним новые потоки В ТОЙ ЖЕ ТОЧКЕ, где находился родитель ДО перемещения
                    start_pos = current_positions[idx_val]
                    new_flow_ids = self.lattice.spawn_flows(flow_id, spawn_energies, start_position=start_pos)
                    
                    # Обновляем счетчик всех созданных потоков
                    self.total_flows_created += len(spawn_energies)
                    total_spawned += len(spawn_energies)
                    
                    # Детальные примеры для первых 3-х spawn'ов
                    if len(spawn_examples) < 3:
                        spawn_examples.append(f"parent_{flow_id}→{len(spawn_energies)}flows")
                    parent_flows.append(flow_id)
        
        # Агрегированное логирование
        if total_spawn_requests > 0:
            logger.debug_spawn(f"🎆 Spawn summary: {total_spawn_requests} requests → {total_spawned} new flows created")
            
            # Детальные примеры
            if spawn_examples:
                logger.debug_spawn(f"🎆 Examples: {', '.join(spawn_examples)}")
            
            # Остальные spawn'ы в агрегированном виде
            if len(parent_flows) > 3:
                other_parents = parent_flows[3:]
                logger.debug_spawn(f"🎆 Additional parents: {len(other_parents)} flows (ids: {other_parents[:5]}{'...' if len(other_parents) > 5 else ''})")
    
    def cleanup_memory_safe(self):
        """
        Безопасная очистка памяти без удаления активных данных
        
        Очищает только завершенные потоки и GPU кэш при необходимости.
        Оптимизировано: реже, батчево, условно.
        """
        self.step_counter += 1
        
        # Менее частая очистка
        if self.step_counter % (self.memory_cleanup_interval * 2) != 0:
            return
        
        # 1. Батч-удаление завершенных потоков
        completed_ids = [fid for fid, flow in self.lattice.active_flows.items() if not flow.is_active]
        if completed_ids:
            # Используем pop с default, чтобы избежать KeyError при гонках
            for fid in completed_ids:
                self.lattice.active_flows.pop(fid, None)
            logger.debug(f"🧹 Cleaned {len(completed_ids)} completed flows")
        
        # 2. Условные операции с памятью GPU реже и при более высоком пороге
        if self.device.type == 'cuda':
            # Проверяем память не на каждом вызове, а еще реже
            if self.step_counter % (self.memory_cleanup_interval * 4) == 0:
                mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
                
                if mem_allocated > self.memory_threshold_gb * 1.5:
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                    mem_allocated_after = torch.cuda.memory_allocated() / 1e9
                    mem_freed = mem_allocated - mem_allocated_after
                    if mem_freed > 0.1:
                        logger.info(f"🧹 GPU memory cleanup: freed {mem_freed:.2f}GB (alloc {mem_allocated:.2f}→{mem_allocated_after:.2f}GB, reserved {mem_reserved:.2f}GB)")
                else:
                    logger.debug(f"💾 Memory OK: allocated={mem_allocated:.2f}GB < threshold={self.memory_threshold_gb*1.5:.2f}GB")
        
    def _check_convergence(self, step: int, initial_flows_count: int) -> bool:
        """
        Улучшенная проверка конвергенции со скользящим окном
        
        Args:
            step: Текущий шаг
            initial_flows_count: Количество начальных потоков
            
        Returns:
            True если следует остановить обучение
        """
        if not self.config.convergence_enabled:
            return False
        
        # Минимальное количество шагов
        if step < self.config.convergence_min_steps:
            return False
        
        # Получаем текущую статистику
        stats = self.lattice.get_statistics()
        completed_count = stats['total_completed']
        active_count = stats['current_active']
        
        # Добавляем в историю
        self.convergence_stats['completed_count_history'].append(completed_count)
        
        # Проверяем порог конвергенции (учитывая все созданные потоки, включая spawn'ы)
        completion_rate = completed_count / self.total_flows_created if self.total_flows_created > 0 else 0
        
        # УЛУЧШЕНИЕ 1: Вычисляем скользящее среднее для стабильности
        window_size = self.convergence_stats.get('moving_average_window', 5)
        if len(self.convergence_stats['completed_count_history']) >= window_size:
            recent_counts = self.convergence_stats['completed_count_history'][-window_size:]
            moving_avg = sum(recent_counts) / window_size
            
            # Проверяем улучшение относительно предыдущего скользящего среднего
            last_moving_avg = self.convergence_stats.get('last_moving_avg', 0.0)
            improvement_threshold = self.convergence_stats.get('improvement_threshold', 0.01)
            if last_moving_avg > 0:
                improvement = (moving_avg - last_moving_avg) / last_moving_avg
                if improvement < improvement_threshold:
                    self.convergence_stats['no_improvement_steps'] += 1
                else:
                    self.convergence_stats['no_improvement_steps'] = 0
            
            self.convergence_stats['last_moving_avg'] = moving_avg
            
            logger.log(20, f"Convergence check step {step}: {completed_count}/{self.total_flows_created} completed "
                          f"(rate={completion_rate:.2f}, moving_avg={moving_avg:.1f}, active={active_count})")
        else:
            logger.log(20, f"Convergence check step {step}: {completed_count}/{self.total_flows_created} completed "
                          f"(rate={completion_rate:.2f}, active={active_count}) - building history")
        
        # Условие 1: Достигнут порог конвергенции
        if completion_rate >= self.config.convergence_threshold:
            logger.log(20, f"✅ Convergence threshold reached: {completion_rate:.2f} >= {self.config.convergence_threshold:.2f}")
            return True
        
        # УЛУЧШЕНИЕ 2: Проверяем стагнацию - нет активных потоков и мало завершенных
        if active_count == 0 and completion_rate < 0.5:
            logger.log(20, f"⚠️ Stagnation detected: no active flows, only {completion_rate:.2f} completion rate")
            return True
        
        # Условие 2: Обновленный Patience на основе скользящего среднего
        if completed_count > self.convergence_stats['best_completed_count']:
            self.convergence_stats['best_completed_count'] = completed_count
            # Сбрасываем счетчик только при значительном улучшении
            if self.convergence_stats['best_completed_count'] - completed_count > 1:
                self.convergence_stats['no_improvement_steps'] = 0
        
        if self.convergence_stats['no_improvement_steps'] >= self.config.convergence_patience:
            logger.log(20, f"⏸️ Convergence patience exceeded: {self.convergence_stats['no_improvement_steps']} "
                          f">= {self.config.convergence_patience} steps without improvement")
            return True
        
        # УЛУЧШЕНИЕ 3: Адаптивная остановка при высокой эффективности
        if step > 10 and completion_rate > 0.8:
            # Если уже достигли 80% и прогресс замедлился
            if self.convergence_stats['no_improvement_steps'] > self.config.convergence_patience // 2:
                logger.log(20, f"🎯 High efficiency early stop: {completion_rate:.2f} completion with slowing progress")
                return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Возвращает статистику производительности"""
        if not self.perf_stats['step_times']:
            return {}
        
        import numpy as np
        
        stats = {
            'avg_step_time': np.mean(self.perf_stats['step_times']),
            'max_step_time': np.max(self.perf_stats['step_times']),
            'avg_flows_per_step': np.mean(self.perf_stats['flows_per_step']),
            'max_flows_per_step': np.max(self.perf_stats['flows_per_step']),
            'avg_gpu_memory_gb': np.mean(self.perf_stats['gpu_memory_usage']) if self.perf_stats['gpu_memory_usage'] else 0,
            'lattice_stats': self.lattice.get_statistics()
        }
        
        # Добавляем статистику конвергенции
        if self.config.convergence_enabled and self.convergence_stats['completed_count_history']:
            stats['convergence_stats'] = {
                'best_completion_count': self.convergence_stats['best_completed_count'],
                'final_completion_count': self.convergence_stats['completed_count_history'][-1] if self.convergence_stats['completed_count_history'] else 0,
                'completion_trend': len(self.convergence_stats['completed_count_history'])
            }
        
        return stats
    
    def visualize_flow_state(self) -> Dict:
        """Возвращает данные для визуализации текущего состояния потоков"""
        active_flows = self.lattice.get_active_flows()
        
        # Собираем позиции и энергии
        positions = []
        energies = []
        ages = []
        
        for flow in active_flows:
            positions.append(flow.position.cpu().numpy())
            embedding_magnitude = torch.norm(flow.energy).item()
            energies.append(embedding_magnitude)
            ages.append(flow.age)
        
        return {
            'positions': positions,
            'energies': energies,
            'ages': ages,
            'total_flows': len(active_flows),
            'lattice_dims': (self.config.lattice_width, 
                           self.config.lattice_height,
                           self.config.lattice_depth)
        }


def create_flow_processor(config=None) -> FlowProcessor:
    """Фабричная функция для создания FlowProcessor"""
    return FlowProcessor(config)