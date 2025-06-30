#!/usr/bin/env python3
"""
Преобразователи эмбедингов между различными размерностями
========================================================

Модули для преобразования эмбедингов между Teacher моделью и 3D кубом.
Поддерживает различные стратегии преобразования с сохранением семантики.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .interfaces import EmbeddingProcessor
from ...utils.logging import get_logger, LogContext
from ...utils.device_manager import get_device_manager
from ...config.simple_config import SimpleProjectConfig

logger = get_logger(__name__)


class EmbeddingTransformer(nn.Module, EmbeddingProcessor):
    """
    Базовый преобразователь эмбедингов
    
    Выполняет преобразование между размерностями teacher модели (768D) 
    и поверхности куба (37×37 = 1369D) с сохранением семантической информации.
    """
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager()
        
        # Размерности
        self.teacher_dim = config.embedding.teacher_embedding_dim
        self.cube_dim = config.cube_embedding_dim
        self.surface_dim = config.cube_surface_dim
        
        # Проверяем соответствие размерностей
        expected_cube_dim = self.surface_dim ** 2
        if self.cube_dim != expected_cube_dim:
            raise ValueError(f"Cube dimension mismatch: {self.cube_dim} != {expected_cube_dim}")
        
        # Создаем трансформеры на основе типа преобразования
        transformation_type = config.embedding.transformation_type
        
        if transformation_type == "linear":
            self._build_linear_transformers()
        elif transformation_type == "hierarchical":
            self._build_hierarchical_transformers()
        elif transformation_type == "attention":
            self._build_attention_transformers()
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
        
        # Дополнительные компоненты
        if config.embedding.use_layer_norm:
            self.layer_norm_to = nn.LayerNorm(self.cube_dim)
            self.layer_norm_from = nn.LayerNorm(self.teacher_dim)
        
        self.dropout = nn.Dropout(config.embedding.dropout_rate)
        
        # Позиционное кодирование для поверхности куба
        self.positional_encoding = nn.Parameter(
            torch.randn(self.surface_dim, self.surface_dim) * 0.1
        )
        
        self.logger.info(f"[SYNC] EmbeddingTransformer initialized: {self.teacher_dim}D ↔ {self.cube_dim}D")
    
    def _build_linear_transformers(self):
        """Построение простых линейных преобразователей"""
        # Teacher → Cube
        self.to_cube = nn.Sequential(
            nn.Linear(self.teacher_dim, self.cube_dim),
            nn.GELU(),
            nn.Linear(self.cube_dim, self.cube_dim)
        )
        
        # Cube → Teacher  
        self.from_cube = nn.Sequential(
            nn.Linear(self.cube_dim, self.teacher_dim),
            nn.GELU(),
            nn.Linear(self.teacher_dim, self.teacher_dim)
        )
        
        # Residual connections если включены
        if self.config.embedding.use_residual_connections:
            self.residual_to = nn.Linear(self.teacher_dim, self.cube_dim)
            self.residual_from = nn.Linear(self.cube_dim, self.teacher_dim)
    
    def _build_hierarchical_transformers(self):
        """Построение иерархических преобразователей (улучшенный вариант)"""
        # Иерархическое расширение: 768 → 512 → 1024 → 1369
        self.to_cube = nn.Sequential(
            nn.Linear(self.teacher_dim, 512),
            nn.GELU(),
            nn.Dropout(self.config.embedding.dropout_rate),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(self.config.embedding.dropout_rate),
            nn.Linear(1024, self.cube_dim)
        )
        
        # Иерархическое сжатие: 1369 → 1024 → 512 → 768
        self.from_cube = nn.Sequential(
            nn.Linear(self.cube_dim, 1024),
            nn.GELU(),
            nn.Dropout(self.config.embedding.dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(self.config.embedding.dropout_rate),
            nn.Linear(512, self.teacher_dim)
        )
        
        # Residual connections через промежуточные размерности
        if self.config.embedding.use_residual_connections:
            self.residual_to_1 = nn.Linear(self.teacher_dim, 1024)
            self.residual_to_2 = nn.Linear(1024, self.cube_dim)
            self.residual_from_1 = nn.Linear(self.cube_dim, 1024)
            self.residual_from_2 = nn.Linear(1024, self.teacher_dim)
    
    def _build_attention_transformers(self):
        """Построение attention-based преобразователей"""
        # Пока реализуем как linear, attention добавим позже
        self._build_linear_transformers()
        
        # TODO: Реализовать attention механизм
        self.logger.info("[WARN]  Attention transformers not fully implemented yet, using linear")
    
    def transform_to_cube(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Преобразование Teacher embeddings → Cube surface embeddings
        
        Args:
            embeddings: Tensor размера [batch, teacher_dim]
        
        Returns:
            Tensor размера [batch, surface_dim, surface_dim]
        """
        with LogContext("embedding_transform", direction="to_cube"):
            batch_size = embeddings.size(0)
            
            # Основное преобразование
            cube_flat = self.to_cube(embeddings)
            
            # Residual connection если включена
            if (hasattr(self, 'residual_to') and 
                self.config.embedding.use_residual_connections):
                cube_flat = cube_flat + self.residual_to(embeddings)
            
            # Layer normalization
            if hasattr(self, 'layer_norm_to'):
                cube_flat = self.layer_norm_to(cube_flat)
            
            # Dropout
            cube_flat = self.dropout(cube_flat)
            
            # Преобразование в 2D поверхность
            cube_2d = cube_flat.view(batch_size, self.surface_dim, self.surface_dim)
            
            # Добавление позиционного кодирования
            cube_2d = cube_2d + self.positional_encoding.unsqueeze(0)
            
            return cube_2d
    
    def transform_from_cube(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Преобразование Cube surface embeddings → Teacher embeddings
        
        Args:
            embeddings: Tensor размера [batch, surface_dim, surface_dim]
        
        Returns:
            Tensor размера [batch, teacher_dim]
        """
        with LogContext("embedding_transform", direction="from_cube"):
            batch_size = embeddings.size(0)
            
            # Убираем позиционное кодирование
            cube_2d = embeddings - self.positional_encoding.unsqueeze(0)
            
            # Плоское представление
            cube_flat = cube_2d.view(batch_size, self.cube_dim)
            
            # Основное преобразование
            teacher_emb = self.from_cube(cube_flat)
            
            # Residual connection если включена
            if (hasattr(self, 'residual_from') and 
                self.config.embedding.use_residual_connections):
                teacher_emb = teacher_emb + self.residual_from(cube_flat)
            
            # Layer normalization
            if hasattr(self, 'layer_norm_from'):
                teacher_emb = self.layer_norm_from(teacher_emb)
            
            return teacher_emb
    
    def get_compression_ratio(self) -> float:
        """Вычисляет коэффициент сжатия/расширения"""
        return self.cube_dim / self.teacher_dim
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Возвращает количество параметров по компонентам"""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts


class HierarchicalEmbeddingTransformer(EmbeddingTransformer):
    """
    Расширенный иерархический преобразователь с улучшенными residual connections
    """
    
    def __init__(self, config: SimpleProjectConfig):
        # Принудительно устанавливаем иерархический тип
        config.embedding.transformation_type = "hierarchical"
        super().__init__(config)
    
    def transform_to_cube(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Иерархическое преобразование с multiple residual connections"""
        with LogContext("hierarchical_transform", direction="to_cube"):
            batch_size = embeddings.size(0)
            
            # Поэтапное преобразование с residual connections
            h1 = F.gelu(self.to_cube[0](embeddings))  # 768 → 512
            h1 = self.to_cube[1](h1)  # dropout
            
            h2 = F.gelu(self.to_cube[3](h1))  # 512 → 1024
            h2 = self.to_cube[4](h2)  # dropout
            
            # Residual на промежуточном уровне
            if hasattr(self, 'residual_to_1'):
                h2 = h2 + self.residual_to_1(embeddings)
            
            h3 = self.to_cube[6](h2)  # 1024 → 1369
            
            # Финальный residual
            if hasattr(self, 'residual_to_2'):
                h3 = h3 + self.residual_to_2(h2)
            
            # Нормализация и reshape
            if hasattr(self, 'layer_norm_to'):
                h3 = self.layer_norm_to(h3)
            
            h3 = self.dropout(h3)
            cube_2d = h3.view(batch_size, self.surface_dim, self.surface_dim)
            cube_2d = cube_2d + self.positional_encoding.unsqueeze(0)
            
            return cube_2d


def create_embedding_transformer(config: SimpleProjectConfig) -> EmbeddingTransformer:
    """Фабричная функция для создания преобразователя эмбедингов"""
    transformation_type = config.embedding.transformation_type
    
    if transformation_type == "hierarchical":
        return HierarchicalEmbeddingTransformer(config)
    else:
        return EmbeddingTransformer(config)


# === UTILITY FUNCTIONS ===

def test_embedding_transformer(config: SimpleProjectConfig, batch_size: int = 4):
    """Тестирование преобразователя эмбедингов"""
    logger.info("[TEST] Testing EmbeddingTransformer...")
    
    transformer = create_embedding_transformer(config)
    device_manager = get_device_manager()
    transformer = device_manager.transfer_module(transformer)
    
    # Создаем тестовые данные
    teacher_embeddings = torch.randn(batch_size, config.embedding.teacher_embedding_dim)
    teacher_embeddings = device_manager.ensure_device(teacher_embeddings)
    
    # Прямое преобразование
    cube_embeddings = transformer.transform_to_cube(teacher_embeddings)
    logger.info(f"  Teacher → Cube: {teacher_embeddings.shape} → {cube_embeddings.shape}")
    
    # Обратное преобразование
    reconstructed = transformer.transform_from_cube(cube_embeddings)
    logger.info(f"  Cube → Teacher: {cube_embeddings.shape} → {reconstructed.shape}")
    
    # Проверяем размерности
    assert reconstructed.shape == teacher_embeddings.shape
    
    # Вычисляем reconstruction loss
    mse_loss = F.mse_loss(reconstructed, teacher_embeddings)
    cosine_sim = F.cosine_similarity(reconstructed, teacher_embeddings).mean()
    
    logger.info(f"  [DATA] Reconstruction MSE: {mse_loss:.6f}")
    logger.info(f"  [DATA] Cosine Similarity: {cosine_sim:.6f}")
    
    # Информация о параметрах
    param_counts = transformer.get_parameter_count()
    logger.info(f"  [TOOL] Total parameters: {param_counts['total']:,}")
    
    logger.info("[OK] EmbeddingTransformer test completed!")
    
    return transformer, {
        'mse_loss': mse_loss.item(),
        'cosine_similarity': cosine_sim.item(),
        'parameter_count': param_counts['total']
    }