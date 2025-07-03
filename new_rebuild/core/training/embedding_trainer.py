#!/usr/bin/env python3
"""
Базовый тренер для обучения 3D куба на эмбедингах
===================================================

Главный компонент для training фазы, объединяющий:
1. EmbeddingTransformer - преобразование эмбедингов DistilBERT
2. MoE Connection Processor - обработка связей между клетками
3. TextDecoder - декодирование результатов обратно в текст
4. Loss функции и валидация

Архитектура:
Text → DistilBERT → EmbeddingTransformer → MoE Cube → EmbeddingTransformer → TextDecoder → Text

Принципы:
- Централизованная конфигурация из SimpleProjectConfig
- GPU оптимизации для RTX 5090
- Модульность и переиспользование компонентов
- Детальное логирование всех операций
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from torch.utils.data import DataLoader

from ...utils.logging import get_logger
from ...config.simple_config import SimpleProjectConfig

from ..common.interfaces import (
    EmbeddingProcessor,
    CubeInterface,
    TrainingInterface,
    create_embedding_processor,
)
from ..common.embedding_transformer import EmbeddingTransformer
from ..inference.text_decoder import SimpleTextDecoder, JointTextDecoder
from ..moe import create_moe_connection_processor
from ..lattice.lattice import Lattice3D
from .embedding_lattice_mapper import (
    create_embedding_lattice_mapper, 
    create_lattice_embedding_extractor,
    EmbeddingLatticeSettings
)

logger = get_logger(__name__)


class EmbeddingTrainer(TrainingInterface):
    """
    Основной тренер для обучения 3D куба на эмбедингах

    Реализует полный цикл: эмбединги → куб → эмбединги → текст
    с поддержкой различных loss функций и валидации.
    """

    def __init__(self, config: SimpleProjectConfig):
        self.config = config
        
        # Устанавливаем глобальную конфигурацию для компонентов, использующих get_project_config()
        from ...config import set_project_config
        set_project_config(config)
        
        # Используем device_manager из конфига вместо создания нового
        self.device_manager = config.device_manager
        self.device = self.device_manager.device

        logger.info(f"Инициализация EmbeddingTrainer на устройстве: {self.device}")

        # Компоненты архитектуры
        self._init_components()

        # Метрики обучения
        self.training_history = {
            "losses": [],
            "reconstruction_losses": [],
            "similarity_losses": [],
            "diversity_losses": [],
            "emergence_losses": [],
            "val_scores": [],
        }

        # Статистика производительности
        self.performance_stats = {
            "forward_times": [],
            "backward_times": [],
            "total_times": [],
        }

        logger.info("EmbeddingTrainer успешно инициализирован")

    def _init_components(self):
        """Инициализация всех компонентов архитектуры"""

        # 1. Embedding Transformer (Teacher ↔ Cube)
        self.embedding_transformer = EmbeddingTransformer(self.config).to(self.device)
        logger.info(
            f"EmbeddingTransformer: {self.embedding_transformer.get_parameter_count()} параметров"
        )

        # 2. Lattice Integration Components
        # Используем размеры из централизованной конфигурации
        lattice_dims = self.config.lattice.dimensions
        
        # Маппер эмбедингов в решетку
        self.lattice_mapper = create_embedding_lattice_mapper(self.config).to(self.device)
        
        # 3D решетка с MoE архитектурой (Lattice3D теперь использует get_project_config())
        self.lattice = Lattice3D().to(self.device)
        
        # Экстрактор эмбедингов из решетки
        self.lattice_extractor = create_lattice_embedding_extractor(self.config).to(self.device)
        
        logger.info(f"Lattice3D создана: {lattice_dims}, total_cells={np.prod(lattice_dims)}")

        # 3. Text Decoder (Cube → Text)
        if self.config.training_embedding.test_mode:
            self.text_decoder = SimpleTextDecoder(self.config).to(self.device)
        else:
            # Joint training для продуктивного режима
            self.text_decoder = JointTextDecoder(self.config).to(self.device)

        logger.info(f"Text Decoder: {type(self.text_decoder).__name__}")

        # 4. Настройки динамики решетки
        self.lattice_settings = EmbeddingLatticeSettings()
        
        # 5. Оптимизатор для всех trainable компонентов
        trainable_params = list(self.embedding_transformer.parameters())
        trainable_params.extend(list(self.lattice_mapper.parameters()))
        trainable_params.extend(list(self.lattice.parameters()))
        trainable_params.extend(list(self.lattice_extractor.parameters()))
        
        if hasattr(self.text_decoder, "parameters"):
            trainable_params.extend(list(self.text_decoder.parameters()))

        self.optimizer = optim.AdamW(
            trainable_params, lr=1e-4, weight_decay=1e-5  # Conservative learning rate
        )
        
        total_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Общее количество обучаемых параметров: {total_params:,}")


        # 6. Scheduler для learning rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def train_epoch(
        self, dataloader: DataLoader, optimizer=None, **kwargs
    ) -> Dict[str, float]:
        """Обучение одной эпохи"""
        if optimizer is None:
            optimizer = self.optimizer

        self.embedding_transformer.train()
        self.lattice_mapper.train()
        self.lattice.train()
        self.lattice_extractor.train()
        if hasattr(self.text_decoder, "train"):
            self.text_decoder.train()

        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "similarity": 0.0,
            "diversity": 0.0,
            "emergence": 0.0,
            "lattice": 0.0,
            "spatial": 0.0,
            "count": 0,
        }

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()

            # Извлечение данных из батча
            if isinstance(batch, dict):
                # Handle both 'embedding' (from dataloader) and 'embeddings' (legacy)
                key = "embedding" if "embedding" in batch else "embeddings"
                input_embeddings = batch[key].to(self.device)
                target_embeddings = batch.get("target_embeddings", input_embeddings).to(
                    self.device
                )
                texts = batch.get("texts", None)
            elif isinstance(batch, (list, tuple)):
                input_embeddings = batch[0].to(self.device)
                target_embeddings = (
                    batch[1].to(self.device) if len(batch) > 1 else input_embeddings
                )
                texts = batch[2] if len(batch) > 2 else None
            else:
                input_embeddings = batch.to(self.device)
                target_embeddings = input_embeddings
                texts = None

            # Forward pass
            forward_start = time.time()
            losses = self._forward_pass(input_embeddings, target_embeddings, texts)
            forward_time = time.time() - forward_start

            # Backward pass
            backward_start = time.time()
            total_loss = losses["total"]

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.embedding_transformer.parameters()]
                + [p for p in self.lattice_mapper.parameters()]
                + [p for p in self.lattice.parameters()]
                + [p for p in self.lattice_extractor.parameters()],
                max_norm=1.0,
            )

            optimizer.step()
            backward_time = time.time() - backward_start

            # Обновление метрик
            for key in epoch_losses:
                if key != "count":
                    epoch_losses[key] += losses.get(key, 0.0)
            epoch_losses["count"] += 1

            # Статистика производительности
            batch_time = time.time() - batch_start_time
            self.performance_stats["forward_times"].append(forward_time)
            self.performance_stats["backward_times"].append(backward_time)
            self.performance_stats["total_times"].append(batch_time)

            if batch_idx % 10 == 0:
                logger.debug_training(
                    f"Batch {batch_idx}: loss={total_loss.item():.4f}, "
                    f"forward={forward_time:.3f}s, backward={backward_time:.3f}s"
                )

        # Усреднение loss'ов
        for key in epoch_losses:
            if key != "count" and epoch_losses["count"] > 0:
                epoch_losses[key] /= epoch_losses["count"]

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Эпоха завершена: total_loss={epoch_losses['total']:.4f}, "
            f"время={epoch_time:.2f}s, батчей={epoch_losses['count']}"
        )

        # Обновление learning rate
        self.scheduler.step()

        return epoch_losses

    def _forward_pass(
        self,
        input_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        texts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Полный forward pass через всю архитектуру

        Поток: Teacher Embeddings → Surface → 3D Lattice → Emergent Dynamics → Surface → Teacher Embeddings
        """

        # 1. Teacher → Cube Surface (768D → 8×8 для 8×8×8 куба)
        surface_embeddings = self.embedding_transformer.transform_to_cube(input_embeddings)
        
        # Преобразуем 3D surface в плоский вектор для lattice_mapper
        batch_size = surface_embeddings.shape[0]
        surface_embeddings_flat = surface_embeddings.view(batch_size, -1)  # [batch, 64]

        # 2. Surface → 3D Lattice initialization
        lattice_states = self.lattice_mapper(surface_embeddings_flat)
        
        # 3. Сохраняем начальные состояния для loss'а согласованности
        initial_states = lattice_states.clone()

        # 4. Emergent dynamics (несколько шагов через MoE)
        # Устанавливаем начальные состояния в решетку
        logger.debug_training(f"🔧 Setting lattice states: {lattice_states.shape}")
        logger.debug_training(f"🔧 Lattice config dimensions: {self.config.lattice.dimensions}")
        logger.debug_training(f"🔧 Expected cells: {self.config.lattice.total_cells}")
        
        # ИСПРАВЛЕНО: Lattice ожидает [total_cells, state_size], убираем batch dimension
        # Для batch_size=1 просто берем первый элемент
        if lattice_states.shape[0] == 1:
            self.lattice.states = lattice_states[0]  # [total_cells, state_size]
        else:
            # Для batch_size > 1 нужна другая логика
            raise NotImplementedError("Batch processing not yet supported in lattice")
        
        for step in range(self.lattice_settings.lattice_steps):
            # Выполняем шаг решетки (обновляет внутренние состояния)
            self.lattice.forward()  # Updates internal states
            
            # Получаем обновленные состояния и добавляем batch dimension обратно
            current_lattice_states = self.lattice.states.unsqueeze(0)  # [1, total_cells, state_size]
            
            # Проверка сходимости (опционально)
            if step > 0 and self._check_convergence(current_lattice_states, initial_states):
                logger.debug_training(f"Сходимость достигнута на шаге {step}")
                break

        # 5. 3D Lattice → Surface extraction
        # Получаем финальные состояния с batch dimension
        final_lattice_states = self.lattice.states.unsqueeze(0)  # [1, total_cells, state_size]
        final_surface = self.lattice_extractor(final_lattice_states)  # [batch, 64]
        
        # Преобразуем плоский вектор обратно в 2D surface для transformer
        # Для куба 8×8×8, поверхность имеет размер 8×8 = 64
        surface_size_1d = final_surface.shape[1]  # 64
        surface_size_2d = int(surface_size_1d ** 0.5)  # 8
        final_surface_2d = final_surface.view(batch_size, surface_size_2d, surface_size_2d)  # [batch, 8, 8]

        # 6. Surface → Teacher embeddings (8×8 → 768D обратно)
        output_embeddings = self.embedding_transformer.transform_from_cube(final_surface_2d)

        # 7. Вычисление loss'ов (включая пространственную согласованность)
        losses = self._compute_losses(
            input_embeddings, output_embeddings, target_embeddings, texts,
            initial_states, final_lattice_states
        )

        return losses
    
    def _check_convergence(self, current_states: torch.Tensor, 
                          initial_states: torch.Tensor) -> bool:
        """Проверка сходимости динамики решетки"""
        diff = torch.norm(current_states - initial_states, dim=-1).mean()
        return diff < self.lattice_settings.convergence_threshold

    def _compute_losses(
        self,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        texts: Optional[List[str]] = None,
        initial_lattice_states: Optional[torch.Tensor] = None,
        final_lattice_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Вычисление всех компонентов loss функции"""

        losses = {}

        # 1. Reconstruction Loss (MSE между выходом и целевыми эмбедингами)
        reconstruction_loss = nn.functional.mse_loss(
            output_embeddings, target_embeddings
        )
        losses["reconstruction"] = (
            reconstruction_loss
            * self.config.training_embedding.reconstruction_weight
        )

        # 2. Similarity Loss (cosine similarity preservation)
        input_sim = torch.cosine_similarity(
            input_embeddings.unsqueeze(1), input_embeddings.unsqueeze(0), dim=2
        )
        output_sim = torch.cosine_similarity(
            output_embeddings.unsqueeze(1), output_embeddings.unsqueeze(0), dim=2
        )
        similarity_loss = nn.functional.mse_loss(output_sim, input_sim)
        losses["similarity"] = (
            similarity_loss * self.config.training_embedding.similarity_weight
        )

        # 3. Diversity Loss (поощрение разнообразия выходов)
        # Для batch_size=1 diversity loss не имеет смысла
        if output_embeddings.shape[0] > 1:
            output_mean = output_embeddings.mean(dim=0)
            diversity_loss = -torch.var(output_embeddings, dim=0).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=output_embeddings.device)
        losses["diversity"] = (
            diversity_loss * self.config.training_embedding.diversity_weight
        )

        # 4. Emergence Loss (поощрение эмерджентного поведения)
        # Измеряем, насколько выход отличается от простого копирования входа
        identity_loss = nn.functional.mse_loss(output_embeddings, input_embeddings)
        emergence_loss = -torch.log(
            identity_loss + 1e-8
        )  # Логарифмическое поощрение различий
        losses["emergence"] = (
            emergence_loss * self.config.training_embedding.emergence_weight
        )

        # 5. Lattice Dynamics Loss (если есть состояния решетки)
        if initial_lattice_states is not None and final_lattice_states is not None:
            # Поощряем контролируемые изменения в решетке
            lattice_change = torch.norm(final_lattice_states - initial_lattice_states, dim=-1)
            lattice_loss = lattice_change.mean()  # Не слишком большие изменения
            losses["lattice"] = lattice_loss * self.lattice_settings.lattice_loss_weight
            
            # Пространственная согласованность
            spatial_loss = self._compute_spatial_consistency_loss(final_lattice_states)
            losses["spatial"] = spatial_loss * self.lattice_settings.spatial_consistency_weight

        # 6. Общий loss
        losses["total"] = sum(losses.values())

        return losses
    
    def _compute_spatial_consistency_loss(self, lattice_states: torch.Tensor) -> torch.Tensor:
        """
        Вычисление loss'а пространственной согласованности
        
        Поощряет схожие состояния у соседних клеток.
        """
        batch_size, total_cells, state_size = lattice_states.shape
        
        # Простая аппроксимация: соседние клетки должны иметь похожие состояния
        # Для куба 8×8×8 берем ближайших соседей по индексам
        consistency_loss = 0.0
        num_comparisons = 0
        
        # Сравниваем каждую клетку с ее непосредственными соседями
        lattice_dim = round(total_cells ** (1/3))  # Предполагаем кубическую решетку
        
        for i in range(min(100, total_cells)):  # Ограничиваем для производительности
            for j in range(i+1, min(i+27, total_cells)):  # Проверяем соседей
                diff = torch.norm(lattice_states[:, i] - lattice_states[:, j], dim=-1)
                consistency_loss += diff.mean()
                num_comparisons += 1
        
        if num_comparisons > 0:
            consistency_loss /= num_comparisons
            
        return consistency_loss

    def validate_epoch(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """Валидация одной эпохи"""
        self.embedding_transformer.eval()
        self.lattice_mapper.eval()
        self.lattice.eval()
        self.lattice_extractor.eval()
        if hasattr(self.text_decoder, "eval"):
            self.text_decoder.eval()

        val_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "similarity": 0.0,
            "diversity": 0.0,
            "emergence": 0.0,
            "lattice": 0.0,
            "spatial": 0.0,
            "count": 0,
        }

        with torch.no_grad():
            for batch in dataloader:
                # Аналогично train_epoch, но без backward pass
                if isinstance(batch, dict):
                    # Handle both 'embedding' (from dataloader) and 'embeddings' (legacy)
                    key = "embedding" if "embedding" in batch else "embeddings"
                    input_embeddings = batch[key].to(self.device)
                    target_embeddings = batch.get(
                        "target_embeddings", input_embeddings
                    ).to(self.device)
                    texts = batch.get("texts", None)
                elif isinstance(batch, (list, tuple)):
                    input_embeddings = batch[0].to(self.device)
                    target_embeddings = (
                        batch[1].to(self.device) if len(batch) > 1 else input_embeddings
                    )
                    texts = batch[2] if len(batch) > 2 else None
                else:
                    input_embeddings = batch.to(self.device)
                    target_embeddings = input_embeddings
                    texts = None

                losses = self._forward_pass(input_embeddings, target_embeddings, texts)

                for key in val_losses:
                    if key != "count":
                        val_losses[key] += losses.get(key, torch.tensor(0.0)).item()
                val_losses["count"] += 1

        # Усреднение
        for key in val_losses:
            if key != "count" and val_losses["count"] > 0:
                val_losses[key] /= val_losses["count"]

        logger.info(f"Валидация: total_loss={val_losses['total']:.4f}")

        return val_losses

    def save_checkpoint(self, path: str, **metadata):
        """Сохранение checkpoint'а"""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "embedding_transformer": self.embedding_transformer.state_dict(),
            "lattice": self.lattice.state_dict(),
            "lattice_mapper": self.lattice_mapper.state_dict(),
            "lattice_extractor": self.lattice_extractor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "performance_stats": self.performance_stats,
            "config": self.config,
            **metadata,
        }

        if hasattr(self.text_decoder, "state_dict"):
            checkpoint["text_decoder"] = self.text_decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint сохранен: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Загрузка checkpoint'а"""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.embedding_transformer.load_state_dict(checkpoint["embedding_transformer"])
        self.lattice.load_state_dict(checkpoint["lattice"])
        self.lattice_mapper.load_state_dict(checkpoint["lattice_mapper"])
        self.lattice_extractor.load_state_dict(checkpoint["lattice_extractor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "text_decoder" in checkpoint and hasattr(
            self.text_decoder, "load_state_dict"
        ):
            self.text_decoder.load_state_dict(checkpoint["text_decoder"])

        self.training_history = checkpoint.get(
            "training_history", self.training_history
        )
        self.performance_stats = checkpoint.get(
            "performance_stats", self.performance_stats
        )

        logger.info(f"Checkpoint загружен: {checkpoint_path}")

        return checkpoint

    def get_training_summary(self) -> Dict[str, Any]:
        """Получение сводки по обучению"""
        return {
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.embedding_transformer.parameters())
            + sum(p.numel() for p in self.lattice_mapper.parameters())
            + sum(p.numel() for p in self.lattice.parameters())
            + sum(p.numel() for p in self.lattice_extractor.parameters()),
            "training_history": self.training_history,
            "performance_stats": {
                "avg_forward_time": sum(self.performance_stats["forward_times"])
                / max(len(self.performance_stats["forward_times"]), 1),
                "avg_backward_time": sum(self.performance_stats["backward_times"])
                / max(len(self.performance_stats["backward_times"]), 1),
                "avg_total_time": sum(self.performance_stats["total_times"])
                / max(len(self.performance_stats["total_times"]), 1),
            },
        }


# === ФАБРИЧНАЯ ФУНКЦИЯ ===


def create_embedding_trainer(config: SimpleProjectConfig) -> EmbeddingTrainer:
    """Фабричная функция для создания тренера"""
    return EmbeddingTrainer(config)
