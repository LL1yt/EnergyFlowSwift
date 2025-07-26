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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime
import json

from ..utils.logging import get_logger, DEBUG_TRAINING, DEBUG_ENERGY, DEBUG_CONVERGENCE
from ..utils.device_manager import get_device_manager
from ..config import EnergyConfig, get_energy_config, create_debug_config, set_energy_config
from ..core import FlowProcessor, EnergyLattice, SimpleNeuron, EnergyCarrier
from ..text_bridge import (
    TextToCubeEncoder, CubeToTextDecoder, TextCache,
    create_text_to_cube_encoder, create_cube_to_text_decoder, create_text_cache,
    CachedTextToCubeEncoder, CachedCubeToTextDecoder
)

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
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        logger.log(DEBUG_TRAINING, "✅ EnergyTrainer successfully initialized")
    
    def _init_core_components(self):
        """Инициализация основных компонентов energy_flow"""
        logger.log(DEBUG_TRAINING, "Initializing core energy_flow components...")
        
        # FlowProcessor объединяет все core компоненты
        self.flow_processor = FlowProcessor(self.config).to(self.device)
        
        # Извлекаем компоненты для прямого доступа
        self.energy_lattice = self.flow_processor.energy_lattice
        self.simple_neuron = self.flow_processor.simple_neuron
        self.energy_carrier = self.flow_processor.energy_carrier
        
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
        base_encoder = create_text_to_cube_encoder(self.config.surface_dimension).to(self.device)
        if self.text_cache:
            self.text_encoder = CachedTextToCubeEncoder(base_encoder, self.text_cache)
        else:
            self.text_encoder = base_encoder
            
        # Text decoder (surface embeddings → text)
        base_decoder = create_cube_to_text_decoder(self.config.surface_dimension).to(self.device)
        if self.text_cache:
            self.text_decoder = CachedCubeToTextDecoder(base_decoder, self.text_cache)
        else:
            self.text_decoder = base_decoder
        
        logger.log(DEBUG_TRAINING, f"Text bridge initialized: encoder({base_encoder.count_parameters()} params), "
                                  f"decoder({base_decoder.count_parameters()} params)")
    
    def _init_optimizer(self):
        """Инициализация оптимизатора и планировщика"""
        # Собираем все обучаемые параметры
        params = list(self.flow_processor.parameters())
        
        if self.config.text_bridge_enabled:
            # Добавляем параметры text_bridge компонентов
            if hasattr(self.text_encoder, 'model'):  # Cached version
                params.extend(self.text_encoder.model.parameters())
            else:  # Direct model
                params.extend(self.text_encoder.parameters())
                
            if hasattr(self.text_decoder, 'model'):  # Cached version  
                params.extend(self.text_decoder.model.parameters())
            else:  # Direct model
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
            patience=10,
            verbose=True
        )
        
        total_params = sum(p.numel() for p in params if p.requires_grad)
        logger.log(DEBUG_TRAINING, f"Optimizer initialized: AdamW, lr={self.config.learning_rate}, "
                                  f"total_params={total_params:,}")
    
    def train_step(self, input_texts: List[str], target_texts: List[str]) -> Dict[str, float]:
        """
        Один шаг обучения
        
        Args:
            input_texts: Список входных текстов
            target_texts: Список целевых текстов
            
        Returns:
            Словарь с метриками шага
        """
        self.optimizer.zero_grad()
        
        batch_size = len(input_texts)
        step_start_time = time.time()
        
        try:
            # 1. Энкодирование входных текстов в surface embeddings
            if self.config.text_bridge_enabled:
                with torch.no_grad():  # Encoder не обучается в основном цикле
                    surface_inputs = []
                    for text in input_texts:
                        surface_emb = self.text_encoder.encode_text(text)
                        surface_inputs.append(surface_emb)
                    surface_inputs = torch.stack(surface_inputs, dim=0)  # [batch_size, surface_dim]
            else:
                # Fallback: случайные surface embeddings для тестирования
                surface_inputs = torch.randn(batch_size, self.config.surface_dimension, 
                                           device=self.device, dtype=self.config.dtype)
            
            # 2. Прохождение через energy flow архитектуру
            flow_start_time = time.time()
            surface_outputs = self.flow_processor.forward(surface_inputs, max_steps=50)
            flow_time = time.time() - flow_start_time
            
            # Получаем статистику потоков
            flow_stats = self.flow_processor.get_flow_statistics()
            
            # 3. Вычисление energy loss (MSE между входом и выходом)
            energy_loss = nn.functional.mse_loss(surface_outputs, surface_inputs)
            
            # 4. Text loss (если text_bridge включен)
            text_loss = torch.tensor(0.0, device=self.device)
            if self.config.text_bridge_enabled and self.config.text_loss_weight > 0:
                try:
                    # Декодируем выходные surface embeddings в текст
                    predicted_texts = []
                    for i in range(batch_size):
                        pred_text = self.text_decoder.decode_surface(surface_outputs[i])
                        predicted_texts.append(pred_text)
                    
                    # Энкодируем целевые тексты для сравнения
                    target_surface_embeddings = []
                    for text in target_texts:
                        target_emb = self.text_encoder.encode_text(text)
                        target_surface_embeddings.append(target_emb)
                    target_surface_embeddings = torch.stack(target_surface_embeddings, dim=0)
                    
                    # Text loss как MSE между surface embeddings
                    text_loss = nn.functional.mse_loss(surface_outputs, target_surface_embeddings)
                    
                except Exception as e:
                    logger.warning(f"Text loss computation failed: {e}")
                    text_loss = torch.tensor(0.0, device=self.device)
            
            # 5. Комбинированный loss
            total_loss = energy_loss + self.config.text_loss_weight * text_loss
            
            # 6. Обратное распространение
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Статистика шага
            step_time = time.time() - step_start_time
            
            step_metrics = {
                'total_loss': total_loss.item(),
                'energy_loss': energy_loss.item(), 
                'text_loss': text_loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'step_time': step_time,
                'flow_time': flow_time,
                'active_flows': flow_stats.get('active_flows', 0),
                'spawned_flows': flow_stats.get('spawned_flows', 0),
                'flows_reached_output': flow_stats.get('flows_reached_output', 0),
                'batch_size': batch_size
            }
            
            self.global_step += 1
            
            # Логирование
            if self.global_step % self.config.log_interval == 0:
                logger.log(DEBUG_TRAINING, 
                          f"Step {self.global_step}: total_loss={total_loss.item():.4f}, "
                          f"energy_loss={energy_loss.item():.4f}, text_loss={text_loss.item():.4f}")
                logger.log(DEBUG_ENERGY,
                          f"Flow stats: active={flow_stats.get('active_flows', 0)}, "
                          f"spawned={flow_stats.get('spawned_flows', 0)}, "
                          f"reached_output={flow_stats.get('flows_reached_output', 0)}")
            
            return step_metrics
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
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
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Обучение на одной эпохе
        
        Args:
            dataloader: DataLoader с парами (input_texts, target_texts)
            
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
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Распаковка данных (формат зависит от DataLoader)
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                input_texts, target_texts = batch_data[0], batch_data[1]
            else:
                logger.warning(f"Unexpected batch format: {type(batch_data)}")
                continue
            
            # Один шаг обучения
            step_metrics = self.train_step(input_texts, target_texts)
            
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
        logger.log(DEBUG_TRAINING,
                  f"✅ Epoch {self.epoch} completed: "
                  f"avg_loss={epoch_metrics['total_loss']:.4f}, "
                  f"time={epoch_time:.1f}s, batches={total_batches}")
        logger.log(DEBUG_CONVERGENCE,
                  f"Convergence stats: flows_reached_output={epoch_metrics['flows_reached_output']:.1f}, "
                  f"active_flows={epoch_metrics['active_flows']:.1f}")
        
        self.epoch += 1
        return epoch_metrics
    
    def train(self, dataloader: DataLoader, num_epochs: int = 10) -> Dict[str, List]:
        """
        Полный цикл обучения
        
        Args:
            dataloader: DataLoader с обучающими данными
            num_epochs: Количество эпох
            
        Returns:
            История обучения
        """
        logger.info(f"🚀 Starting training: {num_epochs} epochs, batch_size={self.config.batch_size}")
        
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(dataloader)
            
            # Сохранение метрик
            for key in epoch_metrics:
                if key in self.training_history:
                    self.training_history[key].append(epoch_metrics[key])
            
            # Чекпоинтинг лучшей модели
            if epoch_metrics['total_loss'] < self.best_loss:
                self.best_loss = epoch_metrics['total_loss']
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            
            # Периодические чекпоинты
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        training_time = time.time() - training_start_time
        
        logger.info(f"✅ Training completed: {num_epochs} epochs, "
                   f"total_time={training_time:.1f}s, "
                   f"best_loss={self.best_loss:.4f}")
        
        return self.training_history
    
    def validate(self, input_texts: List[str], target_texts: List[str]) -> Dict[str, Any]:
        """
        Валидация модели
        
        Args:
            input_texts: Входные тексты для валидации
            target_texts: Целевые тексты
            
        Returns:
            Метрики валидации и примеры предсказаний
        """
        self.flow_processor.eval()
        if self.config.text_bridge_enabled:
            if hasattr(self.text_encoder, 'eval'):
                self.text_encoder.eval()
            self.text_decoder.eval()
        
        with torch.no_grad():
            val_metrics = self.train_step(input_texts, target_texts)
            
            # Генерируем примеры для анализа качества
            examples = []
            if self.config.text_bridge_enabled:
                num_examples = min(3, len(input_texts))
                for i in range(num_examples):
                    try:
                        # Прогоняем через полный pipeline
                        surface_input = self.text_encoder.encode_text(input_texts[i])
                        surface_output = self.flow_processor.forward(surface_input.unsqueeze(0), max_steps=50)
                        predicted_text = self.text_decoder.decode_surface(surface_output[0])
                        
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
        """Сохранение чекпоинта модели"""
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
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Загрузка чекпоинта модели"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
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
            if hasattr(self.text_encoder, 'count_parameters'):
                info['text_encoder_parameters'] = self.text_encoder.count_parameters()
            if hasattr(self.text_decoder, 'count_parameters'):
                info['text_decoder_parameters'] = self.text_decoder.count_parameters()
        
        return info


def create_energy_trainer(config: Optional[EnergyConfig] = None) -> EnergyTrainer:
    """Фабричная функция для создания EnergyTrainer"""
    return EnergyTrainer(config)