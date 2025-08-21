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
from ..utils.logging import get_logger, DEBUG_ENERGY, gated_log, item_str

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
        
        # Иерархическая проекция (вдохновлено embedding_transformer)
        intermediate_dim = max(self.input_dim // 2, self.surface_dim * 2)
        
        self.projection = nn.Sequential(
            # Первый этап: понижение размерности с сохранением информации
            nn.Linear(self.input_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Dropout(0.1),
            
            # Второй этап: подготовка к распределению по поверхности
            nn.Linear(intermediate_dim, self.surface_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim * 2),
            nn.Dropout(0.05),
            
            # Финальная проекция на поверхность
            nn.Linear(self.surface_dim * 2, self.surface_dim),
            nn.Tanh()  # Выход в [-1, 1] для стабильной энергии
        )
        
        # Позиционное кодирование для лучшего распределения энергии
        self.positional_encoding = nn.Parameter(
            torch.randn(self.height, self.width) * 0.02
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
        
        # Добавляем позиционное кодирование для лучшего распределения
        surface_energy = surface_energy + self.positional_encoding.unsqueeze(0)
        
        # Логирование статистики (лениво и гейтированно)
        gated_log(
            logger,
            DEBUG_ENERGY,
            step=0,
            key='energy_projection_stats',
            msg_or_factory=lambda: (
                f"Energy projection stats: "
                f"mean={item_str(normalized.mean())}, std={item_str(normalized.std())}, "
                f"min={item_str(normalized.min())}, max={item_str(normalized.max())}, "
                f"scale={item_str(self.energy_scale.squeeze())}, bias={item_str(self.energy_bias.squeeze())}"
            ),
            first_n_steps=3,
            every=0,
        )
        
        return surface_energy
    
    def get_cell_energies(self, surface_energy: torch.Tensor) -> List[Tuple[Tuple[int, int], torch.Tensor]]:
        """
        Извлекает энергию для каждой клетки - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
        Вместо triple nested loop используем векторизацию и list comprehension
        
        Args:
            surface_energy: [batch, height, width]
            
        Returns:
            cell_energies: [(position, energy, batch_idx)] для каждой клетки и батча
        """
        batch_size = surface_energy.shape[0]
        
        # ОПТИМИЗАЦИЯ: Создаем все индексы заранее векторизованно
        # Это в 100x быстрее чем nested loops
        x_indices = torch.arange(self.width, device=surface_energy.device)
        y_indices = torch.arange(self.height, device=surface_energy.device)
        batch_indices = torch.arange(batch_size, device=surface_energy.device)
        
        # Meshgrid для всех комбинаций индексов
        batch_grid, y_grid, x_grid = torch.meshgrid(batch_indices, y_indices, x_indices, indexing='ij')
        
        # Flatten все grid'ы
        batch_flat = batch_grid.flatten()
        y_flat = y_grid.flatten()
        x_flat = x_grid.flatten()
        
        # Извлекаем все энергии одной векторной операцией
        energies_flat = surface_energy.flatten()
        
        # Создаем список с помощью zip (намного быстрее чем циклы)
        cell_energies = [
            ((int(x_flat[i]), int(y_flat[i])), energies_flat[i].unsqueeze(0), int(batch_flat[i]))
            for i in range(len(energies_flat))
        ]
        
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
        
        # Восстановление эмбеддингов (обратное преобразование к входному маппингу)
        intermediate_dim = max(self.output_dim // 2, self.surface_dim * 2)
        
        self.reconstruction = nn.Sequential(
            # Первый этап: расширение от поверхности
            nn.Linear(self.surface_dim, self.surface_dim * 2),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim * 2),
            nn.Dropout(0.05),
            
            # Второй этап: промежуточное расширение
            nn.Linear(self.surface_dim * 2, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Dropout(0.1),
            
            # Финальная проекция к teacher размерности
            nn.Linear(intermediate_dim, self.output_dim)
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
        # Создаем пустую поверхность на том же устройстве, что и веса
        surface = torch.zeros(batch_size, self.height, self.width, device=self.position_weights.device)
        
        # Заполняем энергией из потоков
        for (x, y), energy in surface_energy.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                # DEBUG: проверяем устройства (редко)
                gated_log(
                    logger,
                    DEBUG_ENERGY,
                    step=0,
                    key='devices_check',
                    msg_or_factory=lambda: (
                        f"Devices - energy: {energy.device}, "
                        f"position_weights: {self.position_weights.device}, surface: {surface.device}"
                    ),
                    first_n_steps=1,
                    every=0,
                )
                
                # energy уже должна быть на правильном устройстве (default CUDA)
                
                # Применяем позиционные веса
                weighted_energy = energy * self.position_weights[y, x]
                surface[:, y, x] = weighted_energy.squeeze()
        
        # Flatten и восстановление
        flattened = surface.view(batch_size, -1)  # [batch, surface_dim]
        embeddings = self.reconstruction(flattened)  # [batch, 768]
        
        # Логирование (ленивое)
        gated_log(
            logger,
            DEBUG_ENERGY,
            step=0,
            key='output_collection_coverage',
            msg_or_factory=lambda: (
                f"Output collection: {len(surface_energy)} cells ("
                f"{(len(surface_energy) / self.surface_dim * 100):.1f}% coverage)"
            ),
            first_n_steps=3,
            every=0,
        )
        
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
