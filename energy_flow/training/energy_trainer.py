"""
EnergyTrainer - основной тренировочный модуль для energy_flow архитектуры
=========================================================================

Полноценный тренировочный пайплайн с интеграцией text_bridge модуля:
- Комбинированное обучение energy flow + text decoders
- GPU оптимизация с CUDA по умолчанию  
- Централизованное логирование и метрики
- Чекпоинтинг и восстановление состояния

Архитектура тренировки:
input_text → TextToCubeEncoder → surface_embedding → FlowProcessor → 
output_surface_embedding → CubeToTextDecoder → predicted_text

Loss = energy_loss + text_loss_weight × text_loss
"""

# import torch
import torch as torch_module  # Алиас для избежания scoping конфликтов
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime
import json
import os
import psutil

from ..utils.logging import (
    get_logger,
    DEBUG_TRAINING,
    DEBUG_ENERGY,
    DEBUG_MEMORY,
    DEBUG_CONVERGENCE,
    DEBUG_PERFORMANCE,
    DEBUG_PROFILING,
    summarize_step,
    gated_log,
    debug_every,
    info_every,
    gated_scalar,
    gated_metrics,
    item_str,
    format_first_n,
)
from ..utils.device_manager import get_device_manager
from ..utils.checkpoint_utils import generate_checkpoint_path, create_checkpoint_summary
from ..config import EnergyConfig, get_energy_config, create_debug_config, set_energy_config
from ..core import FlowProcessor, EnergyLattice, SimpleNeuron, EnergyCarrier
from ..text_bridge import (
    TextToCubeEncoder, CubeToTextDecoder, TextCache,
    create_text_to_cube_encoder, create_cube_to_text_decoder, create_text_cache,
    CachedTextToCubeEncoder, CachedCubeToTextDecoder
)
from ..utils.memory_cleanup import collect_garbage  # CPU memory cleanup
from .checkpoint_loader import SimpleCheckpointLoader

logger = get_logger(__name__)


class EnergyTrainer:
    """
    Основной тренировочный модуль для energy_flow архитектуры
    
    Реализует полный цикл обучения с интеграцией text_bridge:
    - Энергетические потоки через 3D решетку
    - Параллельное обучение text decoders
    - Комбинированный loss (energy + text)
    - GPU оптимизация и мониторинг производительности
    """
    
    def __init__(self, config: Optional[EnergyConfig] = None):
        """
        Args:
            config: EnergyConfig с настройками обучения
        """
        # Конфигурация
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Device management  
        self.device_manager = get_device_manager() 
        self.device = self.device_manager.device
        
        logger.log(DEBUG_TRAINING, f"🚀 EnergyTrainer initialization on {self.device}")
        logger.log(DEBUG_TRAINING, f"Config: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}, "
                                  f"text_bridge={config.text_bridge_enabled}")
        
        # Инициализация компонентов
        self._init_core_components()
        self._init_text_bridge()
        self._init_optimizer()
        self._init_mixed_precision()
        
        # Стабилизационные твики обучения (временные): снизим LR и усилим клиппинг
        try:
            original_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(original_lr * 0.25, 1e-6)
            for pg in self.optimizer.param_groups:
                pg['lr'] = new_lr
            # Усилим gradient clip, если включен
            if getattr(self.config, 'gradient_clip', 0.0) <= 0.0:
                self.config.gradient_clip = 0.1
            else:
                self.config.gradient_clip = min(self.config.gradient_clip, 0.1)
            logger.info(f"Stability tweaks: lr {original_lr:.2e} -> {new_lr:.2e}, gradient_clip={self.config.gradient_clip}")
        except Exception:
            pass
        
        # Включим аномалии на первых шагах (потом отключим)
        self._anomaly_steps_remaining = 2
        
        # Метрики обучения
        self.training_history = {
            "total_losses": [],
            "energy_losses": [],
            "text_losses": [],
            "learning_rates": [],
            "flow_statistics": [],
            "performance_metrics": []
        }
        
        # Счетчики и статистика
        self.global_step = 0  # Глобальный счетчик шагов через все эпохи (для curriculum learning)
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Gradient accumulation состояние
        self.current_accumulation_step = 0
        self.accumulation_loss = 0.0
        self.accumulation_metrics = {}
        
        # Smart memory management для устранения empty_cache() overhead
        self.step_counter = 0
        self.memory_cleanup_interval = 10  # Cleanup только каждые 10 шагов вместо каждого шага
        self.memory_threshold_gb = 16.0    # Cleanup при превышении 16GB для RTX 5090
        
        # Checkpoint управление
        self.checkpoint_loader = SimpleCheckpointLoader()
        self.checkpoint_base_dir = Path("checkpoints/energy_flow")
        
        logger.log(DEBUG_TRAINING, "✅ EnergyTrainer successfully initialized")
        
        # Инициализация пиковых метрик GPU памяти для корректной телеметрии
        if torch_module.cuda.is_available():
            try:
                torch_module.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    
    def _init_core_components(self):
        """Инициализация основных компонентов energy_flow"""
        logger.log(DEBUG_TRAINING, "Initializing core energy_flow components...")
        
        # FlowProcessor объединяет все core компоненты
        self.flow_processor = FlowProcessor(self.config).to(self.device)
        
        # Извлекаем компоненты для прямого доступа
        self.energy_lattice = self.flow_processor.lattice
        self.simple_neuron = self.flow_processor.neuron
        self.energy_carrier = self.flow_processor.carrier
        
        logger.log(DEBUG_TRAINING, f"Core components initialized: "
                                  f"FlowProcessor, EnergyLattice({self.config.lattice_width}x{self.config.lattice_height}x{self.config.lattice_depth})")
    
    def _init_text_bridge(self):
        """Инициализация text_bridge компонентов"""
        if not self.config.text_bridge_enabled:
            logger.log(DEBUG_TRAINING, "Text bridge disabled, skipping initialization")
            self.text_encoder = None
            self.text_decoder = None
            self.text_cache = None
            return
            
        logger.log(DEBUG_TRAINING, "Initializing text_bridge components...")
        
        # Text cache (если включен)
        if self.config.text_cache_enabled:
            self.text_cache = create_text_cache(
                max_size=self.config.text_cache_size,
                cache_file=self.config.text_cache_file
            )
            logger.log(DEBUG_TRAINING, f"TextCache initialized with size {self.config.text_cache_size}")
        else:
            self.text_cache = None
        
        # Text encoder (text → surface embeddings)
        base_encoder = create_text_to_cube_encoder(self.config).to(self.device)
        if self.text_cache:
            self.text_encoder = CachedTextToCubeEncoder(base_encoder, self.text_cache)
        else:
            self.text_encoder = base_encoder
            
        # Text decoder (surface embeddings → text)
        base_decoder = create_cube_to_text_decoder(self.config).to(self.device)
        if self.text_cache:
            self.text_decoder = CachedCubeToTextDecoder(base_decoder, self.text_cache)
        else:
            self.text_decoder = base_decoder
        
        # Подсчет параметров для логирования
        encoder_params = sum(p.numel() for p in base_encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in base_decoder.parameters() if p.requires_grad)
        
        logger.log(DEBUG_TRAINING, f"Text bridge initialized: encoder({encoder_params:,} params), "
                                  f"decoder({decoder_params:,} params)")
    
    def _init_optimizer(self):
        """Инициализация оптимизатора и планировщика"""
        # Собираем все обучаемые параметры
        params = list(self.flow_processor.parameters())
        
        if self.config.text_bridge_enabled:
            # Добавляем параметры text_bridge компонентов
            # Для cached версий используем базовые модели
            if hasattr(self.text_encoder, 'encoder'):  # CachedTextToCubeEncoder
                params.extend(self.text_encoder.encoder.parameters())
            elif hasattr(self.text_encoder, 'parameters'):  # Direct TextToCubeEncoder
                params.extend(self.text_encoder.parameters())
                
            if hasattr(self.text_decoder, 'decoder'):  # CachedCubeToTextDecoder
                params.extend(self.text_decoder.decoder.parameters())
            elif hasattr(self.text_decoder, 'parameters'):  # Direct CubeToTextDecoder
                params.extend(self.text_decoder.parameters())
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Планировщик learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        total_params = sum(p.numel() for p in params if p.requires_grad)
        logger.log(DEBUG_TRAINING, f"Optimizer initialized: AdamW, lr={self.config.learning_rate}, "
                                  f"total_params={total_params:,}")
    
    def _init_mixed_precision(self):
        """Инициализация Mixed Precision Training"""
        if self.config.use_mixed_precision and torch_module.cuda.is_available():
            # GradScaler для автоматического scaling градиентов
            if self.config.use_gradient_scaling:
                self.scaler = torch_module.cuda.amp.GradScaler(
                    init_scale=self.config.gradient_scale_init,
                    enabled=True
                )
                logger.log(DEBUG_TRAINING, f"🔧 Mixed Precision: GradScaler initialized with scale={self.config.gradient_scale_init}")
            else:
                self.scaler = None
            
            # Настройка autocast параметров
            self.autocast_kwargs = {
                'device_type': 'cuda',
                'dtype': self.config.mixed_precision_dtype,
                'enabled': True
            }
            
            logger.log(DEBUG_TRAINING, f"🚀 Mixed Precision Training enabled: {self.config.mixed_precision_dtype}")
            logger.log(DEBUG_TRAINING, f"   Expected benefits: 1.5x speedup, 50% memory saving")
        else:
            self.scaler = None
            self.autocast_kwargs = {'enabled': False}
            
            if not self.config.use_mixed_precision:
                logger.log(DEBUG_TRAINING, "Mixed Precision Training disabled by config")
            else:
                logger.log(DEBUG_TRAINING, "Mixed Precision Training disabled: CUDA not available")
    
    def _snapshot_memory(self, stage: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Снимок состояния памяти (GPU/CPU) и важных флагов.
        Возвращает словарь метрик и логирует на уровне DEBUG.
        """
        metrics: Dict[str, Any] = {
            'stage': stage,
            'global_step': self.global_step,
        }
        # CPU RSS
        try:
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / 1e9
            metrics['cpu_rss_gb'] = round(rss_gb, 3)
        except Exception:
            pass
        # GPU metrics
        if torch_module.cuda.is_available():
            if logger.isEnabledFor(DEBUG_MEMORY):
                try:
                    torch_module.cuda.synchronize()
                except Exception:
                    pass
            try:
                metrics.update({
                    'gpu_alloc_gb': round(torch_module.cuda.memory_allocated() / 1e9, 3),
                    'gpu_reserved_gb': round(torch_module.cuda.memory_reserved() / 1e9, 3),
                    'gpu_max_alloc_gb': round(torch_module.cuda.max_memory_allocated() / 1e9, 3),
                    'gpu_max_reserved_gb': round(getattr(torch_module.cuda, 'max_memory_reserved', lambda: 0.0)() / 1e9, 3) if hasattr(torch_module.cuda, 'max_memory_reserved') else None,
                })
            except Exception:
                pass
        # AMP / scaler info
        try:
            metrics['autocast_enabled'] = bool(self.autocast_kwargs.get('enabled', False))
        except Exception:
            pass
        try:
            if self.scaler is not None and hasattr(self.scaler, 'get_scale'):
                metrics['grad_scale'] = float(self.scaler.get_scale())
        except Exception:
            pass
        # Merge extras
        if extra:
            metrics.update(extra)
        # Log compact line
        if logger.isEnabledFor(10):
            msg = (
                f"🧠 Mem[{stage}]: CPU {metrics.get('cpu_rss_gb','?')}GB; "
                f"GPU alloc {metrics.get('gpu_alloc_gb','?')}GB, res {metrics.get('gpu_reserved_gb','?')}GB, "
                f"max_alloc {metrics.get('gpu_max_alloc_gb','?')}GB"
            )
            logger.log(DEBUG_MEMORY, msg)
        return metrics
    
    def _compute_losses(self, input_texts: List[str], target_texts: List[str], 
                       teacher_input_embeddings: torch_module.Tensor, teacher_target_embeddings: torch_module.Tensor) -> Dict[str, Any]:
        """
        Вычисляет losses без обратного распространения (для validation)
        
        Args:
            input_texts: Список входных текстов (для text_bridge)
            target_texts: Список целевых текстов (для text_bridge)
            teacher_input_embeddings: Входные эмбеддинги от модели-учителя [batch, 768]
            teacher_target_embeddings: Целевые эмбеддинги от модели-учителя [batch, 768]
            
        Returns:
            Словарь с метриками (без обратного распространения)
        """
        batch_size = len(input_texts)
        step_start_time = time.time()
        
        try:
            # Включаем anomaly detection также на forward для первых шагов
            try:
                if self._anomaly_steps_remaining and self._anomaly_steps_remaining > 0:
                    torch_module.autograd.set_detect_anomaly(True)
            except Exception:
                pass
            # Ensure tensors are on the correct device
            try:
                teacher_input_embeddings = teacher_input_embeddings.to(self.device, non_blocking=True)
                teacher_target_embeddings = teacher_target_embeddings.to(self.device, non_blocking=True)
            except Exception:
                teacher_input_embeddings = teacher_input_embeddings.to(self.device)
                teacher_target_embeddings = teacher_target_embeddings.to(self.device)

            # 1. Основной forward pass куба с teacher embeddings
            flow_start_time = time.time()
            cube_output_surface = self.flow_processor.forward(teacher_input_embeddings)
            flow_time = time.time() - flow_start_time
            
            # 2. Маппим teacher target в surface для сравнения
            target_surface_output = self.flow_processor.mapper.input_mapper.forward(teacher_target_embeddings)
            target_surface_input = self.flow_processor.mapper.input_mapper.forward(teacher_input_embeddings)
            
            # Приводим к правильной форме
            if target_surface_output.dim() == 3:
                target_surface_output = target_surface_output.view(batch_size, -1)
            if target_surface_input.dim() == 3:
                target_surface_input = target_surface_input.view(batch_size, -1)
            
            # 3. Energy loss - сравниваем на уровне surface
            energy_loss = nn.functional.mse_loss(cube_output_surface, target_surface_output)
            
            # 4. Text Bridge без градиентов для validation
            text_loss = torch_module.tensor(0.0, device=self.device)
            if self.config.text_bridge_enabled and self.config.text_loss_weight > 0:
                try:
                    encoder_outputs = self.text_encoder.encode_text(input_texts)
                    # В validation режиме не требуем градиенты
                    target_surface_input_grad = target_surface_input.clone().detach()
                    encoder_loss = nn.functional.mse_loss(encoder_outputs, target_surface_input_grad)
                    text_loss = encoder_loss
                except Exception as e:
                    logger.warning(f"❌ Text bridge computation failed: {e}")
                    text_loss = torch_module.tensor(0.1, device=self.device)
            
            # 5. Комбинированный loss
            total_loss = energy_loss + self.config.text_loss_weight * text_loss
            
            # Статистика шага
            step_time = time.time() - step_start_time
            flow_stats = {'flows_reached_output': batch_size}
            
            return {
                'total_loss': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss),
                'energy_loss': energy_loss.item() if hasattr(energy_loss, 'item') else float(energy_loss),
                'text_loss': text_loss.item() if hasattr(text_loss, 'item') else float(text_loss),
                'flow_time': flow_time,
                'step_time': step_time,
                'flow_stats': flow_stats,
                'gradients_computed': False,
                'total_params_with_grads': 0,
            }
            
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            return {
                'total_loss': float('inf'),
                'energy_loss': float('inf'),
                'text_loss': float('inf'),
                'flow_time': 0,
                'step_time': 0,
                'flow_stats': {'error': str(e)},
                'gradients_computed': False,
                'total_params_with_grads': 0,
            }
    
    def train_step(self, input_texts: List[str], target_texts: List[str],
                   teacher_input_embeddings: torch_module.Tensor, teacher_target_embeddings: torch_module.Tensor,
                   global_training_step: Optional[int] = None) -> Dict[str, float]:
        """
        Один шаг обучения с расширенным логированием
        
        Args:
            input_texts: Список входных текстов (для text_bridge)
            target_texts: Список целевых текстов (для text_bridge)
            teacher_input_embeddings: Входные эмбеддинги от модели-учителя [batch, 768]
            teacher_target_embeddings: Целевые эмбеддинги от модели-учителя [batch, 768]
            
        Returns:
            Словарь с метриками шага
        """
        # Gradient accumulation: очищаем градиенты только в начале accumulation
        if torch_module.cuda.is_available() and logger.isEnabledFor(DEBUG_MEMORY):
            try:
                torch_module.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._snapshot_memory('begin', {'acc_step': self.current_accumulation_step + 1})
        if self.current_accumulation_step == 0:
            # set_to_none=True освобождает память под градиенты быстрее, снижая пиковое потребление между батчами
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulation_loss = 0.0
            self.accumulation_metrics = {}
        
        batch_size = len(input_texts)
        step_start_time = time.time()
        
        # Диагностическое логирование
        logger.log(DEBUG_TRAINING, f"🔄 Starting train_step: batch_size={batch_size}, "
                                  f"accumulation_step={self.current_accumulation_step+1}/{self.config.gradient_accumulation_steps}")
        logger.log(DEBUG_TRAINING, f"📊 Input texts: {len(input_texts)} samples")
        logger.log(DEBUG_TRAINING, f"📊 Teacher embeddings: {teacher_input_embeddings.shape} -> {teacher_target_embeddings.shape}")

        if not torch_module.isfinite(teacher_input_embeddings).all() or not torch_module.isfinite(teacher_target_embeddings).all():
            logger.warning("Non-finite values detected in teacher embeddings; sanitizing")
            teacher_input_embeddings = torch_module.nan_to_num(teacher_input_embeddings)
            teacher_target_embeddings = torch_module.nan_to_num(teacher_target_embeddings)

        try:
            # Ensure tensors are on the correct device
            try:
                teacher_input_embeddings = teacher_input_embeddings.to(self.device, non_blocking=True)
                teacher_target_embeddings = teacher_target_embeddings.to(self.device, non_blocking=True)
            except Exception:
                teacher_input_embeddings = teacher_input_embeddings.to(self.device)
                teacher_target_embeddings = teacher_target_embeddings.to(self.device)

            # 1. Основное обучение куба с teacher embeddings С MIXED PRECISION
            flow_start_time = time.time()
            
            # Диагностика: проверяем градиенты входных данных
            logger.log(DEBUG_TRAINING, f"📈 Teacher input requires_grad: {teacher_input_embeddings.requires_grad}")
            logger.log(DEBUG_TRAINING, f"📈 Teacher target requires_grad: {teacher_target_embeddings.requires_grad}")
            
            # ПРИМЕНЯЕМ AUTOCAST ДЛЯ MIXED PRECISION (1.5x speedup, 50% memory)
            with torch_module.autocast(**self.autocast_kwargs):
                # Передаем глобальный шаг для curriculum learning в FlowProcessor
                cube_output_surface = self.flow_processor.forward(
                    teacher_input_embeddings, 
                    global_training_step=global_training_step or self.global_step
                )
            flow_time = time.time() - flow_start_time
            self._snapshot_memory('after_forward', {'flow_time_ms': flow_time * 1000.0})
            
            # Диагностика: редкое логирование формы и статистик выхода куба
            debug_every(
                logger,
                step=self.global_step,
                key="cube/output",
                msg_factory=lambda: (
                    f"📊 Cube surface: shape={tuple(cube_output_surface.shape)}, "
                    f"mean={item_str(cube_output_surface.mean())}, std={item_str(cube_output_surface.std())}"
                ),
                level=DEBUG_TRAINING,
                first_n_steps=3,
            )
            
            # Получаем статистику потоков
            flow_stats = {
                'active_flows': 0,
                'spawned_flows': 0,
                'flows_reached_output': batch_size
            }
            
            # 2. Маппим teacher target в surface для сравнения БЕЗ построения графа
            # Эти тензоры используются как цели, поэтому вычисляем их под no_grad(),
            # чтобы не удерживать граф и экономить память между микро-шагами
            with torch_module.no_grad():
                with torch_module.autocast(**self.autocast_kwargs):
                    target_surface_output = self.flow_processor.mapper.input_mapper.forward(teacher_target_embeddings)
                    target_surface_input = self.flow_processor.mapper.input_mapper.forward(teacher_input_embeddings)
            
            # Быстрые NaN-гварды на surface тензорах
            if not torch_module.isfinite(cube_output_surface).all() or not torch_module.isfinite(target_surface_output).all():
                logger.warning("⚠️ Non-finite tensors in cube/target surface; skipping micro-step safely")
                # Заполним безопасными значениями, чтобы метрики не ломались
                safe_metrics = {
                    'total_loss': float('inf'),
                    'energy_loss': float('inf'),
                    'text_loss': 0.0,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step_time': time.time() - step_start_time,
                    'flow_time': flow_time,
                    'active_flows': 0,
                    'spawned_flows': 0,
                    'flows_reached_output': 0,
                    'batch_size': batch_size,
                    'accumulation_complete': False,
                    'accumulation_step': self.current_accumulation_step
                }
                return safe_metrics
            
            # Диагностика: редкое логирование форм target surface
            debug_every(
                logger,
                step=self.global_step,
                key="cube/targets",
                msg_factory=lambda: (
                    f"📊 Target surfaces: output_shape={tuple(target_surface_output.shape)}, "
                    f"input_shape={tuple(target_surface_input.shape)}"
                ),
                level=DEBUG_TRAINING,
                first_n_steps=3,
            )
            
            # Приводим к правильной форме
            if target_surface_output.dim() == 3:
                target_surface_output = target_surface_output.view(batch_size, -1)
            if target_surface_input.dim() == 3:
                target_surface_input = target_surface_input.view(batch_size, -1)
            
            # Диагностика: проверяем градиенты после reshape
            logger.log(DEBUG_TRAINING, f"📈 Target surface output requires_grad: {target_surface_output.requires_grad}")
            logger.log(DEBUG_TRAINING, f"📈 Target surface input requires_grad: {target_surface_input.requires_grad}")
            
            # 3. Energy loss - сравниваем на уровне surface С MIXED PRECISION
            with torch_module.autocast(**self.autocast_kwargs):
                energy_loss = nn.functional.mse_loss(cube_output_surface, target_surface_output)
            self._snapshot_memory('after_energy_loss')
            
            # 4. Text Bridge обучение - ОПТИМИЗИРОВАННАЯ БАТЧЕВАЯ ВЕРСИЯ
            text_loss = torch_module.tensor(0.0, device=self.device)
            if self.config.text_bridge_enabled and self.config.text_loss_weight > 0:
                text_bridge_start_time = time.time()
                try:
                    # ОПТИМИЗАЦИЯ 1: Батчевое кодирование текста → surface
                    if logger.isEnabledFor(DEBUG_TRAINING):
                        logger.log(DEBUG_TRAINING, f"📝 Processing text bridge (batch={len(input_texts)})")
                    
                    encoder_outputs = self.text_encoder.encode_text(input_texts)
                    if logger.isEnabledFor(DEBUG_TRAINING):
                        logger.log(DEBUG_TRAINING, f"📊 Encoder outputs: {encoder_outputs.shape}")
                    
                    # Encoder loss: text → surface mapping
                    target_surface_input_grad = target_surface_input.clone().detach().requires_grad_(True)
                    encoder_loss = nn.functional.mse_loss(encoder_outputs, target_surface_input_grad)
                    
                    # ОПТИМИЗАЦИЯ 2: Используем УЖЕ ВЫЧИСЛЕННЫЙ cube_output_surface!
                    # Вместо повторных forward passes для каждого образца
                    decoder_loss = torch_module.tensor(0.0, device=self.device)
                    
                    try:
                        # Батчевое декодирование surface → text (градиенты не нужны)
                        with torch_module.no_grad():
                            predicted_texts = self.text_decoder.decode_surface(cube_output_surface)
                        
                        # Батчевое вычисление text similarity loss
                        if predicted_texts and len(predicted_texts) == len(target_texts):
                            similarities = []
                            for pred_text, target_text in zip(predicted_texts, target_texts):
                                if pred_text and target_text:
                                    pred_words = set(pred_text.lower().split())
                                    target_words = set(target_text.lower().split())
                                    intersection = len(pred_words & target_words)
                                    union = len(pred_words | target_words)
                                    similarity = intersection / max(union, 1)
                                    similarities.append(similarity)
                                else:
                                    similarities.append(0.0)
                            
                            # Vectorized similarity loss
                            similarities_tensor = torch_module.tensor(similarities, device=self.device)
                            decoder_loss = (1.0 - similarities_tensor).mean()
                            
                            if logger.isEnabledFor(DEBUG_TRAINING):
                                avg_similarity = similarities_tensor.mean().item()
                                logger.log(DEBUG_TRAINING, f"📝 Avg text similarity: {avg_similarity:.3f}")
                        else:
                            decoder_loss = torch_module.tensor(1.0, device=self.device)
                            if logger.isEnabledFor(DEBUG_TRAINING):
                                logger.log(DEBUG_TRAINING, f"⚠️ Text decoding mismatch: {len(predicted_texts)} vs {len(target_texts)}")
                    
                    except Exception as decode_error:
                        if logger.isEnabledFor(DEBUG_TRAINING):
                            logger.log(DEBUG_TRAINING, f"❌ Batch text decoding failed: {decode_error}")
                        decoder_loss = torch_module.tensor(1.0, device=self.device)
                    
                    # Combined text loss
                    text_loss = encoder_loss + 0.1 * decoder_loss
                    
                    # Performance logging
                    text_bridge_time = time.time() - text_bridge_start_time
                    self._snapshot_memory('after_text_bridge', {'text_bridge_time_ms': text_bridge_time * 1000.0})
                    if logger.isEnabledFor(DEBUG_TRAINING):
                        logger.log(DEBUG_TRAINING, 
                                 f"📊 Text bridge: encoder={encoder_loss:.4f}, decoder={decoder_loss:.4f}, "
                                 f"total={text_loss:.4f}, time={text_bridge_time*1000:.1f}ms")
                    
                except Exception as e:
                    logger.warning(f"❌ Text bridge computation failed: {e}")
                    text_loss = torch_module.tensor(0.1, device=self.device)
            
            # 5. Комбинированный loss (без forward_movement_reward - модель учится сама)
            total_loss = energy_loss + self.config.text_loss_weight * text_loss
            
            # Временная заглушка для forward_reward (для совместимости метрик)
            forward_reward = torch_module.tensor(0.0, device=self.device)
            
            # 6. Gradient accumulation: нормализуем loss 
            normalized_loss = total_loss / self.config.gradient_accumulation_steps

            # Guard against non-finite loss
            if not torch_module.isfinite(normalized_loss):
                logger.warning("⚠️ Non-finite loss detected (NaN/Inf). Skipping backward for this micro-step.")
                # Replace with zero to avoid breaking accumulation
                normalized_loss = torch_module.zeros((), device=self.device, dtype=total_loss.dtype)
            
            # Диагностика перед обратным распространением (редко и лениво)
            debug_every(
                logger,
                step=self.global_step,
                key="losses/detail",
                msg_factory=lambda: (
                    f"📊 Losses: energy={item_str(energy_loss)}, text={item_str(text_loss)}, "
                    f"total={item_str(total_loss)}, normalized={item_str(normalized_loss)}; "
                    f"requires_grad={bool(getattr(total_loss, 'requires_grad', False))}"
                ),
                level=DEBUG_TRAINING,
                first_n_steps=5,
            )
            
            # 8. Обратное распространение с normalized loss И GRADIENT SCALING
            self._snapshot_memory('pre_backward')
            backward_start = time.time()

            # Включаем anomaly detection на первых шагах для точного источника NaN
            anomaly_prev = None
            try:
                if self._anomaly_steps_remaining and self._anomaly_steps_remaining > 0:
                    anomaly_prev = torch_module.autograd.set_detect_anomaly(True)
            except Exception:
                anomaly_prev = None

            if self.scaler is not None:
                # Снизим init scale для устойчивости на первых шагах
                try:
                    if self._anomaly_steps_remaining and self._anomaly_steps_remaining > 0 and hasattr(self.scaler, 'update'):
                        # уменьшать scale вручную не рекомендуется, но можно снижать через _get_scale if needed
                        pass
                except Exception:
                    pass
                # Mixed precision backward pass с gradient scaling
                self.scaler.scale(normalized_loss).backward()
            else:
                # Обычный backward pass
                normalized_loss.backward()
            backward_time = (time.time() - backward_start) * 1000.0
            self._snapshot_memory('post_backward', {'backward_time_ms': backward_time, 'threads': getattr(torch_module, 'get_num_threads', lambda: None)()})
            # Отключаем anomaly detection после нескольких шагов
            try:
                if self._anomaly_steps_remaining and self._anomaly_steps_remaining > 0:
                    self._anomaly_steps_remaining -= 1
                    if self._anomaly_steps_remaining == 0:
                        torch_module.autograd.set_detect_anomaly(False)
            except Exception:
                pass
            
            # Накапливаем метрики для финального возврата
            self.accumulation_loss += total_loss.item()
            if not self.accumulation_metrics:
                self.accumulation_metrics = {
                    'energy_loss': energy_loss.item(),
                    'text_loss': text_loss.item(), 
                    'forward_reward': forward_reward.item(),
                    'batch_size': batch_size
                }
            else:
                self.accumulation_metrics['energy_loss'] += energy_loss.item()
                self.accumulation_metrics['text_loss'] += text_loss.item()
                self.accumulation_metrics['forward_reward'] += forward_reward.item()
                self.accumulation_metrics['batch_size'] += batch_size
            
            # Диагностика градиентов — вычисляем редко, чтобы избежать overhead
            grad_diag_needed = (
                logger.isEnabledFor(DEBUG_TRAINING)
                and (
                    self.current_accumulation_step + 1 >= self.config.gradient_accumulation_steps
                    or (self.global_step % max(1, self.config.log_interval * 2) == 0)
                )
            )
            if grad_diag_needed:
                total_params = 0
                grad_norms = []
                bad_grads = False
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None:
                            total_params += 1
                            g = param.grad
                            if torch_module.isfinite(g).all():
                                try:
                                    grad_norms.append(g.norm().item())
                                except Exception:
                                    pass
                            else:
                                bad_grads = True
                if grad_norms:
                    avg_grad_norm = sum(grad_norms) / len(grad_norms)
                    max_grad_norm = max(grad_norms)
                    debug_every(
                        logger,
                        step=self.global_step,
                        key="grads/stats",
                        msg_factory=lambda: f"📊 Gradients: {total_params} params, avg_norm={avg_grad_norm:.6f}, max_norm={max_grad_norm:.6f}",
                        level=DEBUG_TRAINING,
                        first_n_steps=3,
                    )
                if bad_grads:
                    logger.warning("⚠️ Detected NaN/Inf in gradients. This micro-step will be skipped.")
                # Zero any non-finite grads to prevent optimizer corruption
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.grad is not None and not torch_module.isfinite(param.grad).all():
                            param.grad = None
            
            # Увеличиваем счетчик accumulation
            self.current_accumulation_step += 1
            
            # Gradient clipping и optimizer step только на финальном accumulation шаге
            is_accumulation_complete = self.current_accumulation_step >= self.config.gradient_accumulation_steps
            
            if is_accumulation_complete:
                if self.scaler is not None:
                    pre_step_scale = None
                    try:
                        pre_step_scale = float(self.scaler.get_scale())
                    except Exception:
                        pass
                    # Unscale перед манипуляциями
                    self.scaler.unscale_(self.optimizer)
                    # Очистка и мягкий per-param клиппинг
                    self._clean_and_clip_grads(max_norm=getattr(self.config, 'gru_max_gradient_norm', 1.0))
                    # Глобальный clipping
                    if self.config.gradient_clip > 0:
                        torch_module.nn.utils.clip_grad_norm_(
                            self.optimizer.param_groups[0]['params'],
                            self.config.gradient_clip
                        )
                    if getattr(self.config, 'enable_detailed_gradient_monitoring', False):
                        self._monitor_gradients(self.flow_processor)
                    # Optimizer step с scaling check
                    self.scaler.step(self.optimizer)
                    self.scaler.update()  # Обновляем scale factor
                    self._snapshot_memory('post_optimizer_step', {'grad_scale_before': pre_step_scale, 'grad_scale_after': float(self.scaler.get_scale()) if hasattr(self.scaler, 'get_scale') else None})
                else:
                    # ОБЫЧНЫЙ путь: очистка и клиппинг
                    self._clean_and_clip_grads(max_norm=getattr(self.config, 'gru_max_gradient_norm', 1.0))
                    if self.config.gradient_clip > 0:
                        torch_module.nn.utils.clip_grad_norm_(
                            self.optimizer.param_groups[0]['params'],
                            self.config.gradient_clip
                        )
                    if getattr(self.config, 'enable_detailed_gradient_monitoring', False):
                        self._monitor_gradients(self.flow_processor)
                    self.optimizer.step()
                    self._snapshot_memory('post_optimizer_step')
                
                self.global_step += 1  # Увеличиваем global_step только после полного accumulation
                
                # Сбрасываем accumulation счетчик
                self.current_accumulation_step = 0
            
            # Статистика шага
            step_time = time.time() - step_start_time
            
            # Возвращаем метрики в зависимости от состояния accumulation
            if is_accumulation_complete:
                # Финальные accumulated метрики
                step_metrics = {
                    'total_loss': self.accumulation_loss / self.config.gradient_accumulation_steps,
                    'energy_loss': self.accumulation_metrics['energy_loss'] / self.config.gradient_accumulation_steps,
                    'text_loss': self.accumulation_metrics['text_loss'] / self.config.gradient_accumulation_steps,
                    'forward_reward': self.accumulation_metrics['forward_reward'] / self.config.gradient_accumulation_steps,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step_time': step_time,
                    'flow_time': flow_time,
                    'active_flows': flow_stats.get('active_flows', 0),
                    'spawned_flows': flow_stats.get('spawned_flows', 0),
                    'flows_reached_output': flow_stats.get('flows_reached_output', 0),
                    'batch_size': self.accumulation_metrics['batch_size'],
                    'effective_batch_size': self.accumulation_metrics['batch_size'],  # Реальный accumulated размер
                    'accumulation_complete': True
                }
            else:
                # Промежуточные метрики (accumulating)
                step_metrics = {
                    'total_loss': total_loss.item(),
                    'energy_loss': energy_loss.item(),
                    'text_loss': text_loss.item(),
                    'forward_reward': forward_reward.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step_time': step_time,
                    'flow_time': flow_time,
                    'active_flows': flow_stats.get('active_flows', 0),
                    'spawned_flows': flow_stats.get('spawned_flows', 0),
                    'flows_reached_output': flow_stats.get('flows_reached_output', 0),
                    'batch_size': batch_size,
                    'effective_batch_size': batch_size,
                    'accumulation_complete': False,
                    'accumulation_step': self.current_accumulation_step
                }
            
            # УСЛОВНЫЕ МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ (только при нужном уровне логирования)
            if logger.isEnabledFor(DEBUG_PERFORMANCE):
                try:
                    # Throughput metrics
                    throughput_samples_per_sec = batch_size / step_time if step_time > 0 else 0
                    
                    # GPU utilization (если доступно)
                    gpu_utilization = 0
                    memory_used_gb = 0
                    memory_utilization_percent = 0
                    
                    if torch_module.cuda.is_available():
                        memory_used_gb = torch_module.cuda.memory_allocated() / 1e9
                        memory_reserved_gb = torch_module.cuda.memory_reserved() / 1e9
                        
                        # GPU utilization требует nvidia-ml-py3, может быть недоступно
                        try:
                            gpu_utilization = torch_module.cuda.utilization() if hasattr(torch_module.cuda, 'utilization') else 0
                        except:
                            gpu_utilization = 0
                        
                        # Memory utilization как процент от выделенной памяти
                        if memory_reserved_gb > 0:
                            memory_utilization_percent = (memory_used_gb / memory_reserved_gb) * 100
                    
                    # Добавляем performance метрики
                    step_metrics.update({
                        'throughput_samples_per_sec': throughput_samples_per_sec,
                        'gpu_utilization_percent': gpu_utilization,
                        'memory_used_gb': memory_used_gb,
                        'memory_utilization_percent': memory_utilization_percent,
                    })
                    
                    # Text bridge timing (если был активен)
                    if 'text_bridge_time' in locals():
                        step_metrics['text_bridge_time_ms'] = text_bridge_time * 1000
                        step_metrics['energy_computation_time_ms'] = flow_time * 1000
                    
                    logger.log(DEBUG_PERFORMANCE, 
                             f"⚡ Performance: {throughput_samples_per_sec:.1f} samples/s, "
                             f"GPU: {gpu_utilization:.0f}%, Memory: {memory_used_gb:.1f}GB ({memory_utilization_percent:.0f}%)")
                
                except Exception as perf_error:
                    # Не прерываем обучение из-за ошибок в метриках
                    if logger.isEnabledFor(DEBUG_TRAINING):
                        logger.log(DEBUG_TRAINING, f"Performance metrics error: {perf_error}")
            
            # ДЕТАЛЬНОЕ ПРОФИЛИРОВАНИЕ (только при DEBUG_PROFILING)
            if logger.isEnabledFor(DEBUG_PROFILING):
                try:
                    # Детальные тайминги компонентов
                    energy_percentage = (flow_time / step_time * 100) if step_time > 0 else 0
                    text_bridge_percentage = 0
                    
                    if 'text_bridge_time' in locals():
                        text_bridge_percentage = (text_bridge_time / step_time * 100) if step_time > 0 else 0
                    
                    logger.log(DEBUG_PROFILING,
                             f"🔬 Profiling: Energy {energy_percentage:.1f}%, "
                             f"TextBridge {text_bridge_percentage:.1f}%, "
                             f"Other {100 - energy_percentage - text_bridge_percentage:.1f}%")
                
                except Exception as prof_error:
                    pass  # Игнорируем ошибки профилирования
            
            # Логирование только при завершении accumulation или для промежуточных шагов в debug режиме
            if is_accumulation_complete:
                # Компактный свод по шагу с гейтом частоты
                avg_loss = self.accumulation_loss / self.config.gradient_accumulation_steps
                avg_energy = self.accumulation_metrics['energy_loss'] / self.config.gradient_accumulation_steps
                avg_text = self.accumulation_metrics['text_loss'] / self.config.gradient_accumulation_steps
                avg_forward_reward = self.accumulation_metrics['forward_reward'] / self.config.gradient_accumulation_steps
                gated_log(
                    logger,
                    DEBUG_TRAINING,
                    step=self.global_step,
                    key='train_step_summary',
                    msg_or_factory=lambda: summarize_step({
                        'loss': avg_loss,
                        'energy': avg_energy,
                        'text': avg_text,
                        'fwd': avg_forward_reward,
                        'lr': self.optimizer.param_groups[0]['lr'],
                    }, step=self.global_step, prefix='TRAIN'),
                    first_n_steps=5,
                    every=self.config.log_interval,
                )
            elif logger.isEnabledFor(DEBUG_TRAINING):
                gated_log(
                    logger,
                    DEBUG_TRAINING,
                    step=self.current_accumulation_step,
                    key='train_accumulating',
                    msg_or_factory=lambda: (
                        f"🔄 Accumulating {self.current_accumulation_step}/{self.config.gradient_accumulation_steps}: "
                        f"total_loss={item_str(total_loss)}, forward_reward={item_str(forward_reward)}"
                    ),
                    first_n_steps=2,
                    every=0,
                )
            
            # SMART MEMORY MANAGEMENT: Conditional cleanup вместо агрессивного empty_cache()
            # Устраняет 10-20% performance penalty от forced memory reallocation
            self.step_counter += 1

            # Periodic CPU garbage collection to cap host RAM growth
            try:
                if (self.step_counter % 5) == 0:
                    collect_garbage()
            except Exception:
                pass

            # Proactively drop large temporaries to let GC reclaim GPU memory sooner
            try:
                for _tmp in [
                    'cube_output_surface', 'target_surface_output', 'target_surface_input',
                    'encoder_outputs', 'similarities_tensor'
                ]:
                    if _tmp in locals():
                        obj = locals()[_tmp]
                        if isinstance(obj, torch_module.Tensor):
                            del obj
                del _tmp
            except Exception:
                pass

            if torch_module.cuda.is_available():
                if logger.isEnabledFor(DEBUG_MEMORY):
                    # Synchronize only when profiling memory usage
                    try:
                        torch_module.cuda.synchronize()
                    except Exception:
                        pass

                current_alloc_gb = torch_module.cuda.memory_allocated() / 1e9
                current_reserved_gb = torch_module.cuda.memory_reserved() / 1e9
                current_gb = max(current_alloc_gb, current_reserved_gb)
                
                # Cleanup только при необходимости (каждые N шагов ИЛИ при превышении threshold по alloc ИЛИ reserved)
                should_cleanup = (
                    self.step_counter % self.memory_cleanup_interval == 0 or  # Каждые 10 шагов
                    current_alloc_gb > self.memory_threshold_gb or
                    current_reserved_gb > max(self.memory_threshold_gb, current_alloc_gb * 1.5)
                )
                
                if should_cleanup:
                    # Снимем пиковые метрики перед очисткой
                    self._snapshot_memory('pre_cleanup', {'current_gb': current_gb})
                    # Use best-effort cleanup sequence (GPU + CPU)
                    try:
                        torch_module.cuda.empty_cache()
                        if hasattr(torch_module.cuda, 'ipc_collect'):
                            torch_module.cuda.ipc_collect()
                    except Exception:
                        pass
                    # Periodic CPU GC to reduce host RAM pressure
                    try:
                        collect_garbage()
                    except Exception:
                        pass
                    memory_after_alloc = torch_module.cuda.memory_allocated() / 1e9
                    memory_after_reserved = torch_module.cuda.memory_reserved() / 1e9
                    
                    if logger.isEnabledFor(DEBUG_PERFORMANCE):
                        logger.log(
                            DEBUG_PERFORMANCE,
                            f"🧹 Smart cleanup: alloc {current_alloc_gb:.1f}→{memory_after_alloc:.1f}GB, "
                            f"reserved {current_reserved_gb:.1f}→{memory_after_reserved:.1f}GB "
                            f"(step {self.step_counter}, interval={self.memory_cleanup_interval})"
                        )
                    # После очистки сбрасываем пик и логируем
                    try:
                        torch_module.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                    self._snapshot_memory('post_cleanup')
                elif logger.isEnabledFor(DEBUG_PERFORMANCE):
                    logger.log(
                        DEBUG_PERFORMANCE,
                        f"⚡ Skipped cleanup: alloc {current_alloc_gb:.1f}GB, reserved {current_reserved_gb:.1f}GB < {self.memory_threshold_gb:.1f}GB threshold"
                    )
            
            # Early release of large Python references to help GC
            try:
                del input_texts
                del target_texts
            except Exception:
                pass
            
            # Финальный снимок памяти для шага
            end_metrics = self._snapshot_memory('end', {'accumulation_complete': bool(is_accumulation_complete)})
            # Если заметен скачок CPU RSS за шаг, логируем предупреждение
            try:
                if 'cpu_rss_gb' in end_metrics:
                    # Сравнить с begin невозможно напрямую, но backward_time_ms подскажет
                    if end_metrics.get('backward_time_ms') and end_metrics['backward_time_ms'] > 2000:
                        logger.log(DEBUG_PERFORMANCE, f"⏳ Backward took {end_metrics['backward_time_ms']:.0f}ms; CPU_RSS={end_metrics['cpu_rss_gb']}GB")
            except Exception:
                pass
            return step_metrics
            
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Возвращаем dummy метрики для продолжения обучения
            return {
                'total_loss': float('inf'),
                'energy_loss': float('inf'),
                'text_loss': 0.0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'step_time': time.time() - step_start_time,
                'flow_time': 0.0,
                'active_flows': 0,
                'spawned_flows': 0,
                'flows_reached_output': 0,
                'batch_size': batch_size,
                'error': str(e)
            }

    def _clean_and_clip_grads(self, max_norm: float = 1.0) -> None:
        """Обнуляет NaN/Inf градиенты и мягко клиппит per-param для численной стабильности."""
        try:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    # Sanitize non-finite grads
                    if not torch_module.isfinite(g).all():
                        logger.warning("🧯 Non-finite gradient detected; zeroing it")
                        p.grad = torch_module.zeros_like(g)
                        continue
                    # Per-param gentle clip
                    if max_norm and max_norm > 0:
                        gn = g.norm()
                        if gn > max_norm:
                            scale = (max_norm / (gn + 1e-6)).to(g.dtype)
                            p.grad.mul_(scale)
        except Exception:
            pass

    def _monitor_gradients(self, model: torch_module.nn.Module) -> None:
        """Подробный мониторинг градиентов по параметрам (включаем по флагу)."""
        try:
            total = 0; nan_cnt = 0; inf_cnt = 0; large_cnt = 0
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                total += 1
                g = p.grad
                if torch_module.isnan(g).any():
                    nan_cnt += 1
                    logger.error(f"NaN gradient in {name}")
                    continue
                if torch_module.isinf(g).any():
                    inf_cnt += 1
                    logger.error(f"Inf gradient in {name}")
                    continue
                n = g.norm().item()
                if n > 100.0:
                    large_cnt += 1
                    logger.warning(f"Large gradient ({n:.2f}) in {name}")
            if nan_cnt or inf_cnt:
                logger.error(f"Gradient health: {nan_cnt} NaN, {inf_cnt} Inf, {large_cnt} large out of {total}")
            elif large_cnt:
                logger.warning(f"Gradient health: {large_cnt} large out of {total}")
        except Exception:
            pass
    
    def train_epoch(self, dataloader: DataLoader, teacher_embeddings_loader) -> Dict[str, float]:
        """
        Обучение на одной эпохе
        
        Args:
            dataloader: DataLoader с парами (input_texts, target_texts)
            teacher_embeddings_loader: Итератор с teacher embeddings парами
            
        Returns:
            Усредненные метрики по эпохе
        """
        self.flow_processor.train()
        if self.config.text_bridge_enabled:
            self.text_encoder.train() if hasattr(self.text_encoder, 'train') else None
            self.text_decoder.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'energy_loss': 0.0,
            'text_loss': 0.0,
            'step_time': 0.0,
            'flow_time': 0.0,
            'active_flows': 0.0,
            'spawned_flows': 0.0,
            'flows_reached_output': 0.0
        }
        
        total_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (batch_data, teacher_data) in enumerate(zip(dataloader, teacher_embeddings_loader)):
            # Распаковка данных текстов
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                input_texts, target_texts = batch_data[0], batch_data[1]
            else:
                logger.warning(f"Unexpected batch format: {type(batch_data)}")
                continue
            
            # Распаковка teacher embeddings
            if isinstance(teacher_data, (list, tuple)) and len(teacher_data) >= 2:
                teacher_input_emb, teacher_target_emb = teacher_data[0], teacher_data[1]
            else:
                logger.warning(f"Unexpected teacher embeddings format: {type(teacher_data)}")
                continue

            # Ensure teacher embeddings reside on the correct device (move batch-wise)
            try:
                teacher_input_emb = teacher_input_emb.to(self.device, non_blocking=True)
                teacher_target_emb = teacher_target_emb.to(self.device, non_blocking=True)
            except Exception:
                # Fallback in case non_blocking not supported
                teacher_input_emb = teacher_input_emb.to(self.device)
                teacher_target_emb = teacher_target_emb.to(self.device)
            
            # Один шаг обучения - передаем глобальный шаг для curriculum learning  
            step_metrics = self.train_step(input_texts, target_texts, teacher_input_emb, teacher_target_emb, 
                                         global_training_step=self.global_step)
            
            # Аккумулируем метрики
            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]
            
            total_batches += 1
            
            # Periodic logging внутри эпохи
            if batch_idx % (self.config.log_interval * 5) == 0:
                logger.log(DEBUG_TRAINING,
                          f"Epoch {self.epoch}, Batch {batch_idx}/{len(dataloader)}: "
                          f"loss={step_metrics.get('total_loss', 0):.4f}")
        
        # Усреднение метрик по эпохе
        if total_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= total_batches
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['total_batches'] = total_batches
        
        # Обновление планировщика
        self.scheduler.step(epoch_metrics['total_loss'])
        
        # Логирование эпохи
        epoch_summary = summarize_step({'loss': epoch_metrics['total_loss'], 'time_s': epoch_time, 'batches': total_batches}, prefix='EPOCH')
        logger.log(DEBUG_TRAINING, f"✅ {epoch_summary}")
        logger.log(DEBUG_CONVERGENCE,
                  f"Convergence stats: flows_reached_output={epoch_metrics['flows_reached_output']:.1f}, "
                  f"active_flows={epoch_metrics['active_flows']:.1f}")
        
        self.epoch += 1
        return epoch_metrics
    
    def train(self, dataloader: DataLoader, teacher_embeddings_loader, num_epochs: int = 10) -> Dict[str, List]:
        """
        Полный цикл обучения
        
        Args:
            dataloader: DataLoader с текстовыми данными
            teacher_embeddings_loader: DataLoader с teacher embeddings
            num_epochs: Количество эпох
            
        Returns:
            История обучения
        """
        logger.info(f"🚀 Starting training: {num_epochs} epochs, batch_size={self.config.batch_size}")
        
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(dataloader, teacher_embeddings_loader)
            
            # Сохранение метрик
            for key in epoch_metrics:
                if key in self.training_history:
                    self.training_history[key].append(epoch_metrics[key])
            
            # Умное чекпоинтинг лучшей модели
            if epoch_metrics['total_loss'] < self.best_loss:
                self.best_loss = epoch_metrics['total_loss']
                self.save_smart_checkpoint(
                    current_loss=epoch_metrics['total_loss'],
                    is_best=True,
                    custom_suffix=f"step_{self.global_step}"
                )
            
            # Периодические умные чекпоинты
            if epoch % self.config.checkpoint_interval == 0:
                self.save_smart_checkpoint(
                    current_loss=epoch_metrics['total_loss'],
                    is_best=False,
                    custom_suffix=f"periodic_step_{self.global_step}"
                )
        
        training_time = time.time() - training_start_time
        
        logger.info(f"✅ Training completed: {num_epochs} epochs, "
                   f"total_time={training_time:.1f}s, "
                   f"best_loss={self.best_loss:.4f}")
        
        return self.training_history
    
    def validate(self, input_texts: List[str], target_texts: List[str], 
                 teacher_input_embeddings: torch_module.Tensor, teacher_target_embeddings: torch_module.Tensor) -> Dict[str, Any]:
        """
        Валидация модели
        
        Args:
            input_texts: Входные тексты для валидации
            target_texts: Целевые тексты
            teacher_input_embeddings: Teacher input embeddings [batch, 768]
            teacher_target_embeddings: Teacher target embeddings [batch, 768]
            
        Returns:
            Метрики валидации и примеры предсказаний
        """
        self.flow_processor.eval()
        if self.config.text_bridge_enabled:
            if hasattr(self.text_encoder, 'eval'):
                self.text_encoder.eval()
            self.text_decoder.eval()

        with torch_module.no_grad():
            # Валидация БЕЗ обратного распространения - только forward pass
            val_metrics = self._compute_losses(input_texts, target_texts, teacher_input_embeddings, teacher_target_embeddings)
            
            # Генерируем примеры для анализа качества
            examples = []
            if self.config.text_bridge_enabled:
                num_examples = min(3, len(input_texts))
                for i in range(num_examples):
                    try:
                        # Используем teacher embeddings для демонстрации (правильная архитектура)
                        surface_input = teacher_input_embeddings[i:i+1]
                        try:
                            surface_input = surface_input.to(self.device, non_blocking=True)
                        except Exception:
                            surface_input = surface_input.to(self.device)
                        surface_output = self.flow_processor.forward(surface_input)  # [1, surface_dim]

                        # Декодируем surface embedding в текст (surface_output уже имеет правильный размер [1, surface_dim])
                        predicted_texts = self.text_decoder.decode_surface(surface_output)  # [1, surface_dim] -> List[str]
                        predicted_text = predicted_texts[0] if predicted_texts else ""  # Берем первый результат или пустую строку
                        
                        examples.append({
                            'input': input_texts[i],
                            'target': target_texts[i],
                            'predicted': predicted_text
                        })
                    except Exception as e:
                        logger.warning(f"Example generation failed for sample {i}: {e}")
        
        val_metrics['examples'] = examples
        
        logger.log(DEBUG_TRAINING, f"Validation: loss={val_metrics.get('total_loss', 0):.4f}")
        if examples:
            logger.log(DEBUG_TRAINING, f"Example - Input: '{examples[0]['input'][:50]}...'")
            logger.log(DEBUG_TRAINING, f"Example - Predicted: '{examples[0]['predicted'][:50]}...'")
        
        return val_metrics
    
    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Сохранение чекпоинта модели (legacy метод)"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'model_state_dict': self.flow_processor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }
        
        if self.config.text_bridge_enabled:
            if hasattr(self.text_encoder, 'state_dict'):
                checkpoint['text_encoder_state_dict'] = self.text_encoder.state_dict()
            checkpoint['text_decoder_state_dict'] = self.text_decoder.state_dict()

        torch_module.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def save_smart_checkpoint(
        self, 
        current_loss: float, 
        is_best: bool = False, 
        custom_suffix: Optional[str] = None,
        save_to_active: bool = True
    ) -> Path:
        """
        Сохранение чекпоинта с умным именованием
        
        Args:
            current_loss: Текущий loss для включения в имя
            is_best: Является ли чекпоинт лучшим
            custom_suffix: Дополнительный суффикс для имени
            save_to_active: Сохранять ли в active директорию
            
        Returns:
            Путь к сохраненному чекпоинту
        """
        # Определяем директорию для сохранения
        if save_to_active:
            base_dir = self.checkpoint_base_dir / "active"
        else:
            base_dir = self.checkpoint_base_dir
        
        # Генерируем путь с умным именованием
        checkpoint_path = generate_checkpoint_path(
            config=self.config,
            epoch=self.epoch,
            loss=current_loss,
            base_dir=base_dir,
            is_best=is_best,
            custom_suffix=custom_suffix
        )
        
        # Создаем папку, если не существует
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем данные чекпоинта
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'model_state_dict': self.flow_processor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            
            # Дополнительная информация для умного управления
            'current_loss': current_loss,
            'is_best_checkpoint': is_best,
            'save_timestamp': datetime.now().isoformat(),
            'custom_suffix': custom_suffix
        }
        
        # Добавляем text_bridge состояния
        if self.config.text_bridge_enabled:
            if hasattr(self.text_encoder, 'state_dict'):
                checkpoint['text_encoder_state_dict'] = self.text_encoder.state_dict()
            elif hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'state_dict'):
                checkpoint['text_encoder_state_dict'] = self.text_encoder.encoder.state_dict()
                
            if hasattr(self.text_decoder, 'state_dict'):
                checkpoint['text_decoder_state_dict'] = self.text_decoder.state_dict()
            elif hasattr(self.text_decoder, 'decoder') and hasattr(self.text_decoder.decoder, 'state_dict'):
                checkpoint['text_decoder_state_dict'] = self.text_decoder.decoder.state_dict()
        
        # Сохраняем чекпоинт
        torch_module.save(checkpoint, checkpoint_path)

        # Создаем summary для логирования
        summary = create_checkpoint_summary(checkpoint_path)
        
        # Логирование
        prefix = "🏆 BEST" if is_best else "💾"
        logger.info(f"{prefix} Smart checkpoint saved:")
        logger.info(f"   📁 Path: {checkpoint_path}")
        logger.info(f"   📊 Epoch: {self.epoch}, Loss: {current_loss:.4f}")
        logger.info(f"   📏 Size: {summary.get('size_mb', 0):.1f} MB")
        if custom_suffix:
            logger.info(f"   🏷️  Suffix: {custom_suffix}")
        
        return checkpoint_path
    
    def load_smart_checkpoint(
        self, 
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_latest: bool = False,
        load_best: bool = False,
        strict_validation: bool = False
    ) -> bool:
        """
        Загрузка чекпоинта с умным поиском и валидацией конфигурации
        
        Args:
            checkpoint_path: Конкретный путь к чекпоинту
            load_latest: Загрузить последний чекпоинт из active
            load_best: Загрузить лучший чекпоинт из active
            strict_validation: Если True, несовместимость конфигураций прерывает загрузку
            
        Returns:
            True если загрузка прошла успешно
        """
        checkpoint_data = None
        loaded_path = None
        
        if checkpoint_path:
            # Загружаем конкретный чекпоинт с валидацией
            checkpoint_data = self.checkpoint_loader.load_checkpoint(
                checkpoint_path, 
                current_config=self.config,
                strict_validation=strict_validation
            )
            loaded_path = Path(checkpoint_path)
        elif load_best:
            # Загружаем лучший чекпоинт с валидацией
            checkpoint_data = self.checkpoint_loader.load_best_checkpoint(
                current_config=self.config,
                strict_validation=strict_validation
            )
            if checkpoint_data:
                from ..utils.checkpoint_utils import find_best_checkpoint
                best_path = find_best_checkpoint(self.checkpoint_loader.active_dir)
                loaded_path = best_path
        elif load_latest:
            # Загружаем последний чекпоинт с валидацией
            checkpoint_data = self.checkpoint_loader.load_latest_checkpoint(
                current_config=self.config,
                strict_validation=strict_validation
            )
            if checkpoint_data:
                from ..utils.checkpoint_utils import find_latest_checkpoint
                latest_path = find_latest_checkpoint(self.checkpoint_loader.active_dir)
                loaded_path = latest_path
        
        if checkpoint_data is None:
            logger.warning("No checkpoint loaded")
            return False
        
        try:
            # Восстанавливаем состояние
            self.epoch = checkpoint_data.get('epoch', 0)
            self.global_step = checkpoint_data.get('global_step', 0)
            self.best_loss = checkpoint_data.get('best_loss', float('inf'))
            self.training_history = checkpoint_data.get('training_history', {
                "total_losses": [], "energy_losses": [], "text_losses": [],
                "learning_rates": [], "flow_statistics": [], "performance_metrics": []
            })
            
            # Загружаем веса моделей
            if 'model_state_dict' in checkpoint_data:
                self.flow_processor.load_state_dict(checkpoint_data['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                
            if 'scheduler_state_dict' in checkpoint_data:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Загружаем text_bridge состояния
            if self.config.text_bridge_enabled:
                if 'text_encoder_state_dict' in checkpoint_data:
                    if hasattr(self.text_encoder, 'load_state_dict'):
                        self.text_encoder.load_state_dict(checkpoint_data['text_encoder_state_dict'])
                    elif hasattr(self.text_encoder, 'encoder'):
                        self.text_encoder.encoder.load_state_dict(checkpoint_data['text_encoder_state_dict'])
                
                if 'text_decoder_state_dict' in checkpoint_data:
                    if hasattr(self.text_decoder, 'load_state_dict'):
                        self.text_decoder.load_state_dict(checkpoint_data['text_decoder_state_dict'])
                    elif hasattr(self.text_decoder, 'decoder'):
                        self.text_decoder.decoder.load_state_dict(checkpoint_data['text_decoder_state_dict'])
            
            logger.info(f"✅ Smart checkpoint loaded successfully:")
            logger.info(f"   📁 From: {loaded_path.name if loaded_path else 'Unknown'}")
            logger.info(f"   📊 Epoch: {self.epoch}, Step: {self.global_step}")
            logger.info(f"   🎯 Best loss: {self.best_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint state: {e}")
            return False
    
    def load_checkpoint(self, filepath: Union[str, Path], strict_validation: bool = False) -> None:
        """Загрузка чекпоинта модели с валидацией конфигурации
        
        Args:
            filepath: Путь к чекпоинту
            strict_validation: Если True, несовместимость конфигураций вызывает ошибку
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        # Используем checkpoint_loader для загрузки с валидацией
        checkpoint = self.checkpoint_loader.load_checkpoint(
            filepath,
            current_config=self.config,
            strict_validation=strict_validation
        )
        
        if checkpoint is None:
            raise RuntimeError(f"Failed to load checkpoint from {filepath}")

        # Восстановление состояния
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        # Загрузка весов
        self.flow_processor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.config.text_bridge_enabled:
            if 'text_encoder_state_dict' in checkpoint and hasattr(self.text_encoder, 'load_state_dict'):
                self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            if 'text_decoder_state_dict' in checkpoint:
                self.text_decoder.load_state_dict(checkpoint['text_decoder_state_dict'])
        
        logger.info(f"📁 Checkpoint loaded: {filepath}, epoch={self.epoch}, step={self.global_step}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели и её параметрах"""
        info = {
            'config': self.config.to_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'device': str(self.device)
        }
        
        # Подсчет параметров
        flow_params = sum(p.numel() for p in self.flow_processor.parameters() if p.requires_grad)
        info['flow_processor_parameters'] = flow_params
        
        if self.config.text_bridge_enabled:
            # Подсчет параметров text_bridge компонентов
            if hasattr(self.text_encoder, 'model'):  # Cached version
                encoder_params = sum(p.numel() for p in self.text_encoder.model.parameters() if p.requires_grad)
            else:  # Direct model
                encoder_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            info['text_encoder_parameters'] = encoder_params
            
            if hasattr(self.text_decoder, 'model'):  # Cached version
                decoder_params = sum(p.numel() for p in self.text_decoder.model.parameters() if p.requires_grad)
            else:  # Direct model
                decoder_params = sum(p.numel() for p in self.text_decoder.parameters() if p.requires_grad)
            info['text_decoder_parameters'] = decoder_params
        
        return info


def create_energy_trainer(config: Optional[EnergyConfig] = None) -> EnergyTrainer:
    """Фабричная функция для создания EnergyTrainer"""
    return EnergyTrainer(config)