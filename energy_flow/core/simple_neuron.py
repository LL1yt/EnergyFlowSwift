"""
Simple Neuron - базовый нейрон-автомат для энергетической архитектуры
====================================================================

Простой нейрон с ~1000 параметров, который:
1. Получает координаты клетки (x, y, z) 
2. Получает часть входного эмбеддинга
3. Преобразует их в выход для GRU

Общие веса для всех нейронов в решетке.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.logging import get_logger
from ..config import get_energy_config, create_debug_config, set_energy_config

logger = get_logger(__name__)


class SimpleNeuron(nn.Module):
    """
    Простой нейрон-автомат для обработки энергии в клетке
    
    Принимает:
    - Координаты клетки (3D)
    - Часть входного эмбеддинга
    
    Выдает:
    - Вектор признаков для GRU
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
        
        # Берем все параметры из конфига
        self.coord_dim = 3  # Всегда 3D координаты
        self.embedding_dim = config.embedding_per_cell
        self.hidden_dim = config.neuron_hidden_dim
        self.output_dim = config.neuron_output_dim
        dropout = config.carrier_dropout  # Используем общий dropout
        
        # Входной размер: координаты + эмбеддинг
        input_dim = self.coord_dim + self.embedding_dim
        
        # Сеть преобразования
        self.layers = nn.Sequential(
            # Первый слой с позиционным кодированием
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Второй слой для нелинейности
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Выходной слой
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Позиционное кодирование для координат
        self.coord_encoder = nn.Linear(self.coord_dim, self.coord_dim * 4)
        
        # Устанавливаем размеры решетки из конфига
        self._lattice_dims = (config.lattice_width, config.lattice_height, config.lattice_depth)
        
        # Инициализация весов
        self._init_weights()
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SimpleNeuron initialized with {total_params:,} parameters")
        logger.debug(f"Architecture: input={input_dim} → hidden={self.hidden_dim} → output={self.output_dim}")
    
    def _init_weights(self):
        """Инициализация весов Xavier/He"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization для GELU активации
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                position: torch.Tensor,
                embedding_part: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через нейрон
        
        Args:
            position: [batch, 3] - координаты клетки (x, y, z)
            embedding_part: [batch, embedding_dim] - часть эмбеддинга для этой клетки
            
        Returns:
            output: [batch, output_dim] - выход для передачи в GRU
        """
        batch_size = position.shape[0]
        
        # Проверка размерностей
        assert position.shape == (batch_size, self.coord_dim), \
            f"Expected position shape ({batch_size}, {self.coord_dim}), got {position.shape}"
        assert embedding_part.shape == (batch_size, self.embedding_dim), \
            f"Expected embedding shape ({batch_size}, {self.embedding_dim}), got {embedding_part.shape}"
        
        # Нормализуем координаты в [-1, 1]
        if hasattr(self, '_lattice_dims'):
            norm_position = self._normalize_coordinates(position)
        else:
            # Если размеры решетки неизвестны, используем стандартную нормализацию
            norm_position = position / 100.0  # Предполагаем макс размер 100
        
        # Кодируем позицию
        encoded_position = self.coord_encoder(norm_position)
        encoded_position = encoded_position[:, :self.coord_dim]  # Берем только нужную часть
        
        # Объединяем позицию и эмбеддинг
        combined_input = torch.cat([encoded_position, embedding_part], dim=-1)
        
        # Проход через сеть
        output = self.layers(combined_input)
        
        return output
    
    def _normalize_coordinates(self, position: torch.Tensor) -> torch.Tensor:
        """Нормализация координат в диапазон [-1, 1]"""
        # position: [batch, 3] с значениями (x, y, z)
        normalized = position.clone()
        
        # Нормализуем каждую координату
        if hasattr(self, '_lattice_dims'):
            dims = torch.tensor(self._lattice_dims, device=position.device, dtype=position.dtype)
            normalized = (position / (dims - 1)) * 2 - 1
        
        return normalized
    
    def set_lattice_dimensions(self, width: int, height: int, depth: int):
        """
        Установить размеры решетки для правильной нормализации координат
        
        Args:
            width: Ширина решетки
            height: Высота решетки  
            depth: Глубина решетки
        """
        self._lattice_dims = (width, height, depth)
        logger.debug(f"Lattice dimensions set to {self._lattice_dims}")
    
    def compute_activation_pattern(self, 
                                 position: torch.Tensor,
                                 embedding_part: torch.Tensor) -> torch.Tensor:
        """
        Вычислить паттерн активации нейрона (для визуализации)
        
        Returns:
            activation_pattern: [batch, hidden_dim] - активации скрытого слоя
        """
        with torch.no_grad():
            # Нормализуем координаты
            if hasattr(self, '_lattice_dims'):
                norm_position = self._normalize_coordinates(position)
            else:
                norm_position = position / 100.0
            
            # Кодируем позицию
            encoded_position = self.coord_encoder(norm_position)[:, :self.coord_dim]
            
            # Объединяем входы
            combined_input = torch.cat([encoded_position, embedding_part], dim=-1)
            
            # Получаем активации первого скрытого слоя
            x = self.layers[0](combined_input)  # Linear
            x = self.layers[1](x)  # LayerNorm
            activation = self.layers[2](x)  # GELU
            
            return activation


def create_simple_neuron(config=None) -> SimpleNeuron:
    """Фабричная функция для создания SimpleNeuron"""
    return SimpleNeuron(config)