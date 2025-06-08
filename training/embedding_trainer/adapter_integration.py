"""
🔗 Universal Adapter Integration
Интеграция универсального адаптера с системой обучения CubeTrainer
Поддержка любых teacher моделей и размеров куба
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import yaml

# Импорты нашей системы
from .cube_trainer import CubeTrainer, TrainingConfig, EmbeddingMetrics
from data.embedding_adapter.universal_adapter import (
    UniversalEmbeddingAdapter, 
    AdapterManager, 
    create_adapter_for_cube,
    KNOWN_MODELS
)

logger = logging.getLogger(__name__)


class SimpleWrapper:
    """
    Простой wrapper для EmbeddingProcessor чтобы имитировать CubeTrainer API
    Используется когда мы работаем напрямую с surface embeddings
    """
    
    def __init__(self, embedding_processor, device: str, learning_rate: float):
        self.embedding_processor = embedding_processor
        self.device = device
        
        # Создаем optimizer для embedding_processor
        self.optimizer = optim.AdamW(
            self.embedding_processor.parameters(), 
            lr=learning_rate
        )
    
    def forward(self, surface_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass через embedding processor"""
        return self.embedding_processor.forward(surface_embeddings)
    
    def get_info(self) -> Dict[str, Any]:
        """Информация о processor"""
        return {
            "type": "SimpleWrapper",
            "device": self.device,
            "total_parameters": sum(p.numel() for p in self.embedding_processor.parameters())
        }


@dataclass
class AdapterIntegrationConfig:
    """Конфигурация интеграции адаптера с системой обучения"""
    
    # Teacher model settings
    teacher_model: str = "Meta-Llama-3-8B"
    teacher_embedding_dim: Optional[int] = None  # Auto-detect from KNOWN_MODELS
    
    # Cube settings
    cube_dimensions: Tuple[int, int, int] = (15, 15, 11)
    surface_strategy: str = "single"  # single | triple | full
    
    # Adapter settings
    adapter_strategy: str = "learned_linear"  # learned_linear | hierarchical | attention_based | autoencoder
    use_reconstruction_loss: bool = True
    reconstruction_weight: float = 0.1
    
    # Training settings
    adapter_learning_rate: float = 0.001
    cube_learning_rate: float = 0.0005
    joint_training: bool = True  # Обучаем adapter + cube совместно
    
    # Integration settings
    gradient_balancing: bool = True
    adapter_warmup_epochs: int = 3
    
    def __post_init__(self):
        if self.teacher_embedding_dim is None:
            if self.teacher_model in KNOWN_MODELS:
                self.teacher_embedding_dim = KNOWN_MODELS[self.teacher_model]["embedding_dim"]
            else:
                raise ValueError(f"Unknown teacher model: {self.teacher_model}")


class AdapterCubeTrainer:
    """
    Интегрированный тренер с универсальным адаптером
    
    Комбинирует:
    - UniversalEmbeddingAdapter (любая модель → surface размер)
    - CubeTrainer (3D cubic processing)
    - Joint training pipeline
    """
    
    def __init__(self, 
                 config: Optional[Union[AdapterIntegrationConfig, str, Dict]] = None,
                 device: str = "cpu"):
        """
        Инициализация интегрированного тренера
        
        Args:
            config: Конфигурация интеграции
            device: Устройство для вычислений
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔗 Initializing AdapterCubeTrainer...")
        
        # Загрузка конфигурации
        self.config = self._load_config(config)
        self.device = torch.device(device)
        
        # Компоненты системы
        self.adapter = None
        self.cube_trainer = None
        self.adapter_optimizer = None
        self.joint_optimizer = None
        
        # Состояние обучения
        self.current_epoch = 0
        self.training_history = []
        self.adapter_warmup_complete = False
        
        self.logger.info(f"✅ AdapterCubeTrainer configured:")
        self.logger.info(f"   Teacher: {self.config.teacher_model} ({self.config.teacher_embedding_dim}D)")
        self.logger.info(f"   Cube: {self.config.cube_dimensions}")
        self.logger.info(f"   Surface strategy: {self.config.surface_strategy}")
        self.logger.info(f"   Adapter strategy: {self.config.adapter_strategy}")
    
    def _load_config(self, config: Optional[Union[AdapterIntegrationConfig, str, Dict]]) -> AdapterIntegrationConfig:
        """Загрузка конфигурации интеграции"""
        if config is None:
            return AdapterIntegrationConfig()
        
        elif isinstance(config, AdapterIntegrationConfig):
            return config
        
        elif isinstance(config, str):
            # Загрузка из YAML
            try:
                with open(config, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return AdapterIntegrationConfig(**config_data)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config}: {e}")
                return AdapterIntegrationConfig()
        
        elif isinstance(config, dict):
            try:
                return AdapterIntegrationConfig(**config)
            except Exception as e:
                self.logger.warning(f"Failed to create config from dict: {e}")
                return AdapterIntegrationConfig()
        
        else:
            return AdapterIntegrationConfig()
    
    def initialize_components(self):
        """Инициализация всех компонентов интегрированной системы"""
        self.logger.info("🔧 Initializing integrated training components...")
        
        # 1. Создание универсального адаптера
        self._setup_adapter()
        
        # 2. Настройка CubeTrainer
        self._setup_cube_trainer()
        
        # 3. Настройка оптимизаторов
        self._setup_optimizers()
        
        # 4. Проверка совместимости
        self._validate_integration()
        
        self.logger.info("🎯 All integrated components initialized successfully!")
    
    def _setup_adapter(self):
        """Создание и настройка универсального адаптера"""
        self.logger.info(f"🔧 Setting up adapter: {self.config.teacher_model} → surface...")
        
        # Вычисляем surface размер в зависимости от стратегии
        surface_size = self._calculate_surface_size()
        
        # Создаем адаптер
        self.adapter = UniversalEmbeddingAdapter(
            input_dim=self.config.teacher_embedding_dim,
            output_dim=surface_size,
            strategy=self.config.adapter_strategy
        )
        
        self.adapter.to(self.device)
        
        compression_ratio = self.adapter.get_compression_ratio()
        param_count = self.adapter.get_parameter_count()
        
        self.logger.info(f"✅ Adapter created:")
        self.logger.info(f"   {self.config.teacher_embedding_dim}D → {surface_size}D")
        self.logger.info(f"   Compression: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
        self.logger.info(f"   Parameters: {param_count:,}")
    
    def _setup_cube_trainer(self):
        """Настройка CubeTrainer с правильными размерами"""
        self.logger.info("🔧 Setting up CubeTrainer...")
        
        # Для универсального адаптера нам нужен direct EmbeddingProcessor
        # без EmbeddingReshaper, так как мы работаем с surface embeddings
        surface_size = self._calculate_surface_size()
        
        # Создаем EmbeddingProcessor напрямую
        from core.embedding_processor import EmbeddingProcessor, EmbeddingConfig
        
        processor_config = EmbeddingConfig(
            lattice_size=tuple(self.config.cube_dimensions),
            device=str(self.device),
            input_dim=surface_size,  # Surface размер
            output_dim=surface_size,  # Surface размер
            processing_mode="surface_direct"  # Новый режим для surface-only
        )
        
        self.embedding_processor = EmbeddingProcessor(config=processor_config)
        
        # Создаем простой wrapper для совместимости с CubeTrainer API
        self.cube_trainer = SimpleWrapper(
            embedding_processor=self.embedding_processor,
            device=self.device,
            learning_rate=self.config.cube_learning_rate
        )
        
        self.logger.info(f"✅ Direct EmbeddingProcessor initialized for {surface_size}D surface embeddings")
    
    def _setup_optimizers(self):
        """Настройка оптимизаторов для joint training"""
        self.logger.info("🔧 Setting up optimizers...")
        
        if self.config.joint_training:
            # Joint optimizer для обоих компонентов
            all_params = list(self.adapter.parameters()) + list(self.embedding_processor.parameters())
            self.joint_optimizer = optim.AdamW(all_params, lr=self.config.cube_learning_rate)
            self.logger.info("✅ Joint optimizer configured (adapter + embedding_processor)")
        else:
            # Separate optimizers
            self.adapter_optimizer = optim.Adam(self.adapter.parameters(), lr=self.config.adapter_learning_rate)
            # cube_trainer уже имеет свой optimizer
            self.logger.info("✅ Separate optimizers configured")
    
    def _calculate_surface_size(self) -> int:
        """Вычисление размера surface в зависимости от стратегии"""
        x, y, z = self.config.cube_dimensions
        face_size = x * y
        
        if self.config.surface_strategy == "single":
            return face_size
        elif self.config.surface_strategy == "triple":
            return 3 * face_size  # front, back, top
        elif self.config.surface_strategy == "full":
            return 6 * face_size  # all faces
        else:
            raise ValueError(f"Unknown surface strategy: {self.config.surface_strategy}")
    
    def _validate_integration(self):
        """Проверка совместимости всех компонентов"""
        self.logger.info("🔍 Validating component integration...")
        
        # Тестовый forward pass
        test_input = torch.randn(2, self.config.teacher_embedding_dim).to(self.device)
        
        # Adapter forward
        adapter_output = self.adapter(test_input)
        expected_surface_size = self._calculate_surface_size()
        
        assert adapter_output.shape == (2, expected_surface_size), \
            f"Adapter output shape mismatch: {adapter_output.shape} vs expected (2, {expected_surface_size})"
        
        # EmbeddingProcessor forward  
        processor_output = self.embedding_processor.forward(adapter_output)
        
        assert processor_output.shape == adapter_output.shape, \
            f"Processor output shape mismatch: {processor_output.shape} vs expected {adapter_output.shape}"
        
        self.logger.info("✅ Integration validation passed!")
        self.logger.info(f"   Pipeline: {test_input.shape} → {adapter_output.shape} → {processor_output.shape}")
    
    def forward(self, teacher_embeddings: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass через adapter + cube
        
        Args:
            teacher_embeddings: Эмбединги от teacher модели [batch, teacher_dim]
            return_intermediate: Возвращать промежуточные результаты
            
        Returns:
            output: Финальный результат [batch, surface_size]
            intermediate: Словарь промежуточных результатов (если запрошено)
        """
        # 1. Adapter: teacher_dim → surface_size
        if self.config.use_reconstruction_loss:
            surface_embeddings, reconstructed = self.adapter(teacher_embeddings, return_reconstruction=True)
        else:
            surface_embeddings = self.adapter(teacher_embeddings)
            reconstructed = None
        
        # 2. Cube: surface_size → surface_size
        processed_embeddings = self.embedding_processor.forward(surface_embeddings)
        
        if return_intermediate:
            return {
                "output": processed_embeddings,
                "surface_embeddings": surface_embeddings,
                "reconstructed": reconstructed,
                "teacher_embeddings": teacher_embeddings
            }
        else:
            return processed_embeddings
    
    def compute_loss(self, 
                    teacher_embeddings: torch.Tensor, 
                    target_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Вычисление комбинированной loss function
        
        Args:
            teacher_embeddings: Input эмбединги от teacher модели
            target_embeddings: Target эмбединги (также от teacher модели)
            
        Returns:
            Dict с различными компонентами loss
        """
        # Forward pass с промежуточными результатами
        results = self.forward(teacher_embeddings, return_intermediate=True)
        
        # Конвертируем target также через adapter
        target_surface = self.adapter(target_embeddings)
        
        # Main task loss: processed surface → target surface
        main_loss = nn.functional.cosine_embedding_loss(
            results["output"], 
            target_surface,
            torch.ones(target_surface.size(0)).to(self.device)
        )
        
        losses = {"main_loss": main_loss}
        
        # Reconstruction loss (если включен)
        if self.config.use_reconstruction_loss and results["reconstructed"] is not None:
            reconstruction_loss = self.adapter.compute_reconstruction_loss(
                teacher_embeddings, 
                results["reconstructed"]
            )
            losses["reconstruction_loss"] = reconstruction_loss
        
        # Total loss
        total_loss = losses["main_loss"]
        if "reconstruction_loss" in losses:
            total_loss = total_loss + self.config.reconstruction_weight * losses["reconstruction_loss"]
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def train_step(self, 
                  question_embeddings: torch.Tensor, 
                  answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Один шаг обучения
        
        Args:
            question_embeddings: Question эмбединги от teacher модели
            answer_embeddings: Answer эмбединги от teacher модели
            
        Returns:
            Метрики шага обучения
        """
        if self.config.joint_training:
            return self._joint_train_step(question_embeddings, answer_embeddings)
        else:
            return self._separate_train_step(question_embeddings, answer_embeddings)
    
    def _joint_train_step(self, 
                         question_embeddings: torch.Tensor, 
                         answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """Joint training step (adapter + cube одновременно)"""
        self.adapter.train()
        self.embedding_processor.train()
        
        # Forward pass
        losses = self.compute_loss(question_embeddings, answer_embeddings)
        
        # Backward pass
        self.joint_optimizer.zero_grad()
        losses["total_loss"].backward()
        
        # Gradient clipping
        if self.config.gradient_balancing:
            nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.embedding_processor.parameters(), max_norm=1.0)
        
        self.joint_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            results = self.forward(question_embeddings, return_intermediate=True)
            target_surface = self.adapter(answer_embeddings)
            
            qa_similarity = torch.cosine_similarity(
                results["output"], 
                target_surface, 
                dim=1
            ).mean().item()
        
        return {
            "total_loss": losses["total_loss"].item(),
            "main_loss": losses["main_loss"].item(),
            "reconstruction_loss": losses.get("reconstruction_loss", torch.tensor(0.0)).item(),
            "qa_similarity": qa_similarity
        }
    
    def _separate_train_step(self, 
                           question_embeddings: torch.Tensor, 
                           answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """Separate training (adapter first, then cube)"""
        if not self.adapter_warmup_complete and self.current_epoch < self.config.adapter_warmup_epochs:
            # Adapter warmup phase
            return self._adapter_warmup_step(question_embeddings, answer_embeddings)
        else:
            # Main training phase
            self.adapter_warmup_complete = True
            return self._cube_training_step(question_embeddings, answer_embeddings)
    
    def _adapter_warmup_step(self, 
                           question_embeddings: torch.Tensor, 
                           answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """Adapter warmup step (только adapter)"""
        self.adapter.train()
        self.embedding_processor.eval()  # Freeze cube
        
        # Adapter training только на reconstruction
        _, reconstructed = self.adapter(question_embeddings, return_reconstruction=True)
        reconstruction_loss = self.adapter.compute_reconstruction_loss(question_embeddings, reconstructed)
        
        self.adapter_optimizer.zero_grad()
        reconstruction_loss.backward()
        self.adapter_optimizer.step()
        
        return {
            "total_loss": reconstruction_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "qa_similarity": 0.0,  # Not applicable during warmup
            "phase": "adapter_warmup"
        }
    
    def _cube_training_step(self, 
                          question_embeddings: torch.Tensor, 
                          answer_embeddings: torch.Tensor) -> Dict[str, float]:
        """Cube training step (только cube, adapter frozen)"""
        self.adapter.eval()  # Freeze adapter
        self.embedding_processor.train()
        
        # Forward через frozen adapter
        with torch.no_grad():
            question_surface = self.adapter(question_embeddings)
            answer_surface = self.adapter(answer_embeddings)
        
        # Cube training
        processed_surface = self.embedding_processor.forward(question_surface)
        
        main_loss = nn.functional.cosine_embedding_loss(
            processed_surface,
            answer_surface,
            torch.ones(answer_surface.size(0)).to(self.device)
        )
        
        self.cube_trainer.optimizer.zero_grad()
        main_loss.backward()
        self.cube_trainer.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            qa_similarity = torch.cosine_similarity(
                processed_surface, 
                answer_surface, 
                dim=1
            ).mean().item()
        
        return {
            "total_loss": main_loss.item(),
            "main_loss": main_loss.item(),
            "qa_similarity": qa_similarity,
            "phase": "cube_training"
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Информация о интегрированной системе"""
        adapter_info = {
            "input_dim": self.adapter.input_dim if self.adapter else None,
            "output_dim": self.adapter.output_dim if self.adapter else None,
            "strategy": self.config.adapter_strategy,
            "compression_ratio": self.adapter.get_compression_ratio() if self.adapter else None,
            "parameters": self.adapter.get_parameter_count() if self.adapter else None
        }
        
        cube_info = self.cube_trainer.get_info() if self.cube_trainer else {}
        
        return {
            "teacher_model": self.config.teacher_model,
            "cube_dimensions": self.config.cube_dimensions,
            "surface_strategy": self.config.surface_strategy,
            "joint_training": self.config.joint_training,
            "current_epoch": self.current_epoch,
            "adapter_warmup_complete": self.adapter_warmup_complete,
            "adapter": adapter_info,
            "cube": cube_info,
            "total_parameters": (adapter_info.get("parameters", 0) + 
                               cube_info.get("total_parameters", 0) if cube_info else 0)
        }


# Удобные функции для создания интегрированных тренеров

def create_llama3_cube_trainer(cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                              adapter_strategy: str = "learned_linear",
                              device: str = "cpu") -> AdapterCubeTrainer:
    """Создание тренера для LLaMA-3-8B"""
    config = AdapterIntegrationConfig(
        teacher_model="Meta-Llama-3-8B",
        cube_dimensions=cube_dimensions,
        adapter_strategy=adapter_strategy,
        joint_training=True
    )
    
    trainer = AdapterCubeTrainer(config=config, device=device)
    trainer.initialize_components()
    
    return trainer


def create_distilbert_cube_trainer(cube_dimensions: Tuple[int, int, int] = (15, 15, 11),
                                  adapter_strategy: str = "learned_linear", 
                                  device: str = "cpu") -> AdapterCubeTrainer:
    """Создание тренера для DistilBERT (для сравнения)"""
    config = AdapterIntegrationConfig(
        teacher_model="DistilBERT",
        cube_dimensions=cube_dimensions,
        adapter_strategy=adapter_strategy,
        joint_training=True
    )
    
    trainer = AdapterCubeTrainer(config=config, device=device)
    trainer.initialize_components()
    
    return trainer


# Экспорт
__all__ = [
    "AdapterIntegrationConfig",
    "AdapterCubeTrainer", 
    "create_llama3_cube_trainer",
    "create_distilbert_cube_trainer"
] 