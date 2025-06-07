"""
CubeTrainer - Основной класс для обучения 3D Cubic Core

Этот модуль реализует центральный тренер для обучения 3D клеточной нейронной сети
на эмбединг→эмбединг трансформациях.

Ключевые возможности:
- Autoencoder режим (embedding → embedding)
- Dialogue режим (question_embedding → answer_embedding)
- Интеграция с готовыми компонентами (EmbeddingProcessor, EmbeddingReshaper)
- Система метрик и логирования
- Checkpoint управление

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0 (Phase 3.1 - Stage 1.1)
Дата: 6 июня 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import yaml

# Импорты готовых компонентов
try:
    from core.embedding_processor import EmbeddingProcessor
    from data.embedding_reshaper import EmbeddingReshaper
    from data.embedding_loader import EmbeddingLoader
    from utils.config_manager import ConfigManager, ConfigManagerSettings
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    mode: str = "autoencoder"  # autoencoder | dialogue | mixed
    device: str = "cpu"        # cpu | cuda
    random_seed: int = 42
    
    # Архитектура
    lattice_size: List[int] = None
    embedding_dim: int = 768
    batch_size: int = 32
    
    # Обучение
    learning_rate: float = 0.001
    epochs: int = 50
    optimizer: str = "adam"    # adam | sgd | adamw
    loss_function: str = "cosine"  # cosine | mse | combined
    
    # Качество и сходимость
    target_similarity: float = 0.90
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 10
    
    # Логирование
    log_interval: int = 10
    save_interval: int = 25
    checkpoint_dir: str = "checkpoints/embedding_trainer"
    
    def __post_init__(self):
        if self.lattice_size is None:
            self.lattice_size = [8, 8, 8]


class EmbeddingMetrics:
    """Система метрик для оценки качества обучения"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss()
        
    def calculate_cosine_similarity(self, input_emb: torch.Tensor, output_emb: torch.Tensor) -> float:
        """Вычисление cosine similarity между эмбедингами"""
        with torch.no_grad():
            similarity = self.cosine_similarity(input_emb, output_emb)
            return similarity.mean().item()
    
    def calculate_mse_loss(self, input_emb: torch.Tensor, output_emb: torch.Tensor) -> float:
        """Вычисление MSE loss между эмбедингами"""
        with torch.no_grad():
            loss = self.mse_loss(input_emb, output_emb)
            return loss.item()
    
    def calculate_semantic_preservation(self, input_emb: torch.Tensor, output_emb: torch.Tensor) -> float:
        """Оценка сохранения семантики (комбинированная метрика)"""
        cosine_sim = self.calculate_cosine_similarity(input_emb, output_emb)
        mse_loss = self.calculate_mse_loss(input_emb, output_emb)
        
        # Комбинированная метрика: высокая cosine similarity + низкий MSE
        semantic_score = cosine_sim * (1.0 / (1.0 + mse_loss))
        return semantic_score
    
    def compute_batch_metrics(self, input_batch: torch.Tensor, output_batch: torch.Tensor) -> Dict[str, float]:
        """Вычисление всех метрик для batch"""
        return {
            'cosine_similarity': self.calculate_cosine_similarity(input_batch, output_batch),
            'mse_loss': self.calculate_mse_loss(input_batch, output_batch),
            'semantic_preservation': self.calculate_semantic_preservation(input_batch, output_batch)
        }


class CubeTrainer:
    """
    Основной класс для обучения 3D Cubic Core
    
    Этот класс управляет процессом обучения центрального процессора системы,
    интегрируясь с готовыми компонентами модульной архитектуры.
    """
    
    def __init__(self, 
                 config: Optional[Union[TrainingConfig, str, Dict]] = None,
                 mode: str = "autoencoder",
                 device: str = "cpu"):
        """
        Инициализация CubeTrainer
        
        Args:
            config: Конфигурация обучения (TrainingConfig, путь к YAML или dict)
            mode: Режим обучения (autoencoder, dialogue, mixed)
            device: Устройство для вычислений (cpu, cuda)
        """
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.logger.info("🚀 Initializing CubeTrainer...")
        
        # Проверка доступности зависимостей
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies are not available. "
                            "Make sure core.embedding_processor, data.embedding_reshaper, "
                            "and data.embedding_loader are implemented.")
        
        # Загрузка конфигурации
        self.config = self._load_config(config, mode, device)
        
        # Установка устройства
        self.device = torch.device(self.config.device)
        
        # Установка random seed для воспроизводимости
        self._set_random_seed(self.config.random_seed)
        
        # Инициализация компонентов
        self.embedding_processor = None
        self.embedding_reshaper = None
        self.embedding_loader = None
        
        # Компоненты обучения
        self.optimizer = None
        self.loss_function = None
        self.metrics = EmbeddingMetrics(device=self.config.device)
        
        # Состояние обучения
        self.current_epoch = 0
        self.training_history = []
        self.best_metrics = {}
        
        # Создание директории для checkpoint'ов
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"✅ CubeTrainer initialized successfully")
        self.logger.info(f"   Mode: {self.config.mode}")
        self.logger.info(f"   Device: {self.config.device}")
        self.logger.info(f"   Lattice size: {self.config.lattice_size}")
        
    def _setup_logging(self):
        """Настройка логирования"""
        # Базовая настройка - можно будет расширить
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self, config: Optional[Union[TrainingConfig, str, Dict]], 
                    mode: str, device: str) -> TrainingConfig:
        """Загрузка и валидация конфигурации"""
        if config is None:
            # Создание конфигурации по умолчанию
            return TrainingConfig(mode=mode, device=device)
        
        elif isinstance(config, TrainingConfig):
            return config
        
        elif isinstance(config, str):
            # Загрузка из YAML файла
            try:
                with open(config, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Извлечение секции embedding_trainer если есть
                if 'embedding_trainer' in config_data:
                    config_data = config_data['embedding_trainer']
                
                return TrainingConfig(**config_data)
            
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config}: {e}")
                return TrainingConfig(mode=mode, device=device)
        
        elif isinstance(config, dict):
            # Создание из словаря
            try:
                return TrainingConfig(**config)
            except Exception as e:
                self.logger.warning(f"Failed to create config from dict: {e}")
                return TrainingConfig(mode=mode, device=device)
        
        else:
            self.logger.warning(f"Unknown config type: {type(config)}")
            return TrainingConfig(mode=mode, device=device)
    
    def _set_random_seed(self, seed: int):
        """Установка random seed для воспроизводимости"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def initialize_components(self):
        """
        Инициализация основных компонентов для обучения
        
        Создает и настраивает:
        - EmbeddingProcessor (3D Cubic Core)
        - EmbeddingReshaper (конвертация форматов)
        - EmbeddingLoader (источник данных)
        """
        self.logger.info("🔧 Initializing training components...")
        
        try:
            # 1. EmbeddingReshaper для конвертации форматов
            self.embedding_reshaper = EmbeddingReshaper(
                input_dim=self.config.embedding_dim,
                cube_shape=self.config.lattice_size
            )
            self.logger.info(f"✅ EmbeddingReshaper initialized: {self.config.embedding_dim}D ↔ {self.config.lattice_size}")
            
            # 2. EmbeddingProcessor (3D Cubic Core)
            self.embedding_processor = EmbeddingProcessor(
                lattice_size=self.config.lattice_size,
                device=self.config.device
            )
            self.logger.info(f"✅ EmbeddingProcessor initialized: {self.config.lattice_size}")
            
            # 3. EmbeddingLoader для данных
            self.embedding_loader = EmbeddingLoader()
            self.logger.info(f"✅ EmbeddingLoader initialized")
            
            # 4. Настройка loss function
            self._setup_loss_function()
            
            # 5. Настройка optimizer
            self._setup_optimizer()
            
            self.logger.info("🎯 All training components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def _setup_loss_function(self):
        """Настройка функции потерь"""
        if self.config.loss_function == "cosine":
            # Cosine similarity loss (1 - cosine_similarity)
            self.loss_function = lambda x, y: 1 - nn.functional.cosine_similarity(x, y, dim=1).mean()
        
        elif self.config.loss_function == "mse":
            self.loss_function = nn.MSELoss()
        
        elif self.config.loss_function == "combined":
            # Комбинированная loss function
            mse_loss = nn.MSELoss()
            cosine_weight = 0.7
            mse_weight = 0.3
            
            def combined_loss(x, y):
                cosine_loss = 1 - nn.functional.cosine_similarity(x, y, dim=1).mean()
                mse_loss_val = mse_loss(x, y)
                return cosine_weight * cosine_loss + mse_weight * mse_loss_val
            
            self.loss_function = combined_loss
        
        else:
            self.logger.warning(f"Unknown loss function: {self.config.loss_function}, using cosine")
            self.loss_function = lambda x, y: 1 - nn.functional.cosine_similarity(x, y, dim=1).mean()
        
        self.logger.info(f"✅ Loss function configured: {self.config.loss_function}")
    
    def _setup_optimizer(self):
        """Настройка оптимизатора"""
        # Получаем параметры для оптимизации (только параметры куба!)
        if self.embedding_processor is None:
            raise ValueError("EmbeddingProcessor must be initialized before setting up optimizer")
        
        # Важно: обучаем только параметры 3D Cubic Core
        trainable_params = list(self.embedding_processor.parameters())
        
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(trainable_params, lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(trainable_params, lr=self.config.learning_rate)
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(trainable_params, lr=self.config.learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {self.config.optimizer}, using Adam")
            self.optimizer = optim.Adam(trainable_params, lr=self.config.learning_rate)
        
        self.logger.info(f"✅ Optimizer configured: {self.config.optimizer}")
        self.logger.info(f"   Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через полный pipeline
        
        Args:
            input_embedding: Входной эмбединг (batch_size, embedding_dim)
            
        Returns:
            Обработанный эмбединг (batch_size, embedding_dim)
        """
        if self.embedding_reshaper is None or self.embedding_processor is None:
            raise ValueError("Components must be initialized before forward pass")
        
        # 1. Конвертация 1D → 3D
        matrix_3d = self.embedding_reshaper.vector_to_matrix(input_embedding)
        
        # 2. Обработка через 3D Cubic Core
        processed_matrix = self.embedding_processor.process(matrix_3d)
        
        # 3. Конвертация 3D → 1D
        output_embedding = self.embedding_reshaper.matrix_to_vector(processed_matrix)
        
        return output_embedding
    
    def get_info(self) -> Dict[str, Any]:
        """Получение информации о тренере"""
        return {
            'mode': self.config.mode,
            'device': self.config.device,
            'lattice_size': self.config.lattice_size,
            'embedding_dim': self.config.embedding_dim,
            'current_epoch': self.current_epoch,
            'optimizer': self.config.optimizer,
            'loss_function': self.config.loss_function,
            'target_similarity': self.config.target_similarity,
            'components_initialized': all([
                self.embedding_processor is not None,
                self.embedding_reshaper is not None,
                self.embedding_loader is not None
            ])
        }
    
    def set_mode(self, mode: str):
        """Изменение режима обучения"""
        if mode in ["autoencoder", "dialogue", "mixed"]:
            self.config.mode = mode
            self.logger.info(f"✅ Training mode changed to: {mode}")
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported: autoencoder, dialogue, mixed")
    
    def __repr__(self):
        return (f"CubeTrainer(mode={self.config.mode}, "
                f"device={self.config.device}, "
                f"lattice_size={self.config.lattice_size})")


# Экспорт основных классов
__all__ = ['CubeTrainer', 'TrainingConfig', 'EmbeddingMetrics'] 