#!/usr/bin/env python3
"""
Маппинг эмбеддингов для энергетической архитектуры
==================================================

Адаптация из new_rebuild для energy_flow:
- Проекция 768D эмбеддингов на входную поверхность решетки
- Каждая клетка получает скалярную энергию
- Нормализация для стабильной работы
- Обратное преобразование для выходной поверхности
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np

from ..config.energy_config import get_energy_config
from ..utils.logging import get_logger, DEBUG_ENERGY

logger = get_logger(__name__)


class EnergyEmbeddingMapper(nn.Module):
    """
    Проецирует эмбеддинги обучающей модели на входную поверхность решетки
    
    768D -> surface_dim (width * height)
    Каждая клетка получает скалярное значение энергии
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Конфигурация
        self.config = config or get_energy_config()
        
        # Размеры
        self.input_dim = self.config.input_embedding_dim_from_teacher  # 768
        self.width = self.config.lattice_width
        self.height = self.config.lattice_height
        self.surface_dim = self.width * self.height  # 400 для 20x20
        
        # Проекция с адаптивной нормализацией
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.surface_dim * 4),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim * 4),
            nn.Dropout(0.1),
            nn.Linear(self.surface_dim * 4, self.surface_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim * 2),
            nn.Linear(self.surface_dim * 2, self.surface_dim),
            nn.Tanh()  # Выход в [-1, 1]
        )
        
        # Адаптивная нормализация энергии
        self.energy_scale = nn.Parameter(torch.ones(1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
        
        logger.info(
            f"EnergyEmbeddingMapper: {self.input_dim}D -> {self.surface_dim}D "
            f"(lattice {self.width}x{self.height})"
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Проецирует эмбеддинги на поверхность
        
        Args:
            embeddings: [batch, 768] - входные эмбеддинги
            
        Returns:
            surface_energy: [batch, width, height] - энергия для каждой клетки
        """
        batch_size = embeddings.shape[0]
        
        # Проекция
        projected = self.projection(embeddings)  # [batch, surface_dim]
        
        # Адаптивная нормализация
        normalized = projected * self.energy_scale + self.energy_bias
        
        # Reshape в 2D сетку
        surface_energy = normalized.view(batch_size, self.height, self.width)
        
        # Логирование статистики
        if logger.isEnabledFor(DEBUG_ENERGY):
            energy_stats = {
                'mean': float(normalized.mean()),
                'std': float(normalized.std()),
                'min': float(normalized.min()),
                'max': float(normalized.max()),
                'scale': float(self.energy_scale),
                'bias': float(self.energy_bias)
            }
            logger.log(DEBUG_ENERGY, f"Energy projection stats: {energy_stats}")
        
        return surface_energy
    
    def get_cell_energies(self, surface_energy: torch.Tensor) -> List[Tuple[Tuple[int, int], torch.Tensor]]:
        """
        Извлекает энергию для каждой клетки
        
        Args:
            surface_energy: [batch, height, width]
            
        Returns:
            cell_energies: [(position, energy)] для каждой клетки и батча
        """
        batch_size = surface_energy.shape[0]
        cell_energies = []
        
        for batch_idx in range(batch_size):
            for y in range(self.height):
                for x in range(self.width):
                    position = (x, y)
                    energy = surface_energy[batch_idx, y, x].unsqueeze(0)  # [1]
                    cell_energies.append((position, energy, batch_idx))
        
        return cell_energies


class EnergyOutputCollector(nn.Module):
    """
    Собирает энергию с выходной поверхности и восстанавливает эмбеддинги
    
    surface_dim -> 768D
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Конфигурация
        self.config = config or get_energy_config()
        
        # Размеры
        self.output_dim = self.config.output_embedding_dim_to_teacher  # 768
        self.width = self.config.lattice_width
        self.height = self.config.lattice_height
        self.surface_dim = self.width * self.height
        
        # Восстановление эмбеддингов
        self.reconstruction = nn.Sequential(
            nn.Linear(self.surface_dim, self.surface_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.surface_dim * 2, self.output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.output_dim * 2),
            nn.Linear(self.output_dim * 2, self.output_dim)
        )
        
        # Адаптивное взвешивание
        self.position_weights = nn.Parameter(torch.ones(self.height, self.width))
        
        logger.info(
            f"EnergyOutputCollector: {self.surface_dim}D -> {self.output_dim}D"
        )
    
    def forward(self, surface_energy: Dict[Tuple[int, int], torch.Tensor], 
                batch_size: int) -> torch.Tensor:
        """
        Восстанавливает эмбеддинги из энергии на выходной поверхности
        
        Args:
            surface_energy: {(x, y): energy_tensor} - энергия в клетках
            batch_size: размер батча
            
        Returns:
            embeddings: [batch, 768] - восстановленные эмбеддинги
        """
        device = next(iter(surface_energy.values())).device if surface_energy else torch.device('cpu')
        
        # Создаем пустую поверхность
        surface = torch.zeros(batch_size, self.height, self.width, device=device)
        
        # Заполняем энергией из потоков
        for (x, y), energy in surface_energy.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                # Применяем позиционные веса
                weighted_energy = energy * self.position_weights[y, x]
                surface[:, y, x] = weighted_energy.squeeze()
        
        # Flatten и восстановление
        flattened = surface.view(batch_size, -1)  # [batch, surface_dim]
        embeddings = self.reconstruction(flattened)  # [batch, 768]
        
        # Логирование
        if logger.isEnabledFor(DEBUG_ENERGY):
            coverage = len(surface_energy) / self.surface_dim * 100
            logger.log(DEBUG_ENERGY, f"Output collection: {len(surface_energy)} cells ({coverage:.1f}% coverage)")
        
        return embeddings


class EnergyFlowMapper:
    """
    Объединенный маппер для energy_flow архитектуры
    """
    
    def __init__(self, config=None):
        self.config = config or get_energy_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Компоненты
        self.input_mapper = EnergyEmbeddingMapper(config).to(self.device)
        self.output_collector = EnergyOutputCollector(config).to(self.device)
        
        logger.info("EnergyFlowMapper initialized")
    
    def map_to_surface(self, embeddings: torch.Tensor) -> List[Tuple[Tuple[int, int], torch.Tensor, int]]:
        """
        Маппинг эмбеддингов на входную поверхность
        
        Returns:
            [(position, energy, batch_idx)] для создания потоков
        """
        surface_energy = self.input_mapper(embeddings)
        return self.input_mapper.get_cell_energies(surface_energy)
    
    def collect_from_surface(self, surface_energy: Dict[Tuple[int, int], torch.Tensor], 
                           batch_size: int) -> torch.Tensor:
        """
        Сбор энергии с выходной поверхности
        """
        return self.output_collector(surface_energy, batch_size)
    
    def parameters(self):
        """Все обучаемые параметры"""
        for p in self.input_mapper.parameters():
            yield p
        for p in self.output_collector.parameters():
            yield p