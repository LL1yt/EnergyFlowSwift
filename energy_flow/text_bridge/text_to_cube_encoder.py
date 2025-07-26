#!/usr/bin/env python3
"""
TextToCubeEncoder для energy_flow архитектуры
=============================================

Преобразует текст в эмбеддинги поверхности куба с адаптивными размерностями.
Lightweight архитектура (~5M параметров) для эффективного обучения в основном цикле.

Архитектура:
Текст → Токенизация → TransformerEncoder (2 слоя) → Linear → Surface Embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Union, Dict, Optional
import math

from ..config import create_debug_config, set_energy_config
from ..utils.logging import get_logger, DEBUG_ENERGY

logger = get_logger(__name__)


class TextToCubeEncoder(nn.Module):
    """
    Преобразует текст в эмбеддинги поверхности куба
    
    Lightweight модель (~5M параметров) для обучения в основном цикле EnergyTrainer.
    Адаптивные размерности в зависимости от конфига куба.
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Device management
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Адаптивная размерность поверхности куба
        self.surface_dim = self.config.lattice_width * self.config.lattice_height
        
        # Токенизатор (используем DistilBERT как в документе)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Добавляем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Архитектура согласно документу
        self.hidden_dim = 256
        self.vocab_size = self.tokenizer.vocab_size
        
        # Embedding слой
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Positional encoding для лучшего понимания последовательности
        self.positional_encoding = nn.Parameter(
            torch.zeros(512, self.hidden_dim, device=self.device)  # Макс длина 512 токенов
        )
        self._init_positional_encoding()
        
        # Transformer encoder (2 слоя для lightweight архитектуры)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Проекция к размерности поверхности куба
        self.surface_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.surface_dim),
            nn.Tanh()  # Нормализация в [-1, 1] для совместимости с энергией
        )
        
        # Переносим на устройство
        self.to(self.device)
        
        # Инициализация весов
        self._init_weights()
        
        # Статистика параметров
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(
            f"TextToCubeEncoder инициализирован: "
            f"текст → {self.surface_dim}D поверхность куба "
            f"({self.config.lattice_width}×{self.config.lattice_height}), "
            f"{total_params:,} параметров"
        )
    
    def _init_positional_encoding(self):
        """Инициализация positional encoding (sinusoidal)"""
        position = torch.arange(512).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * 
                           (-math.log(10000.0) / self.hidden_dim))
        
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_text(self, texts: Union[str, List[str]],
                   max_length: int = 128) -> torch.Tensor:
        """
        Кодирует текст в эмбеддинги поверхности куба
        
        Args:
            texts: строка или список строк для кодирования
            max_length: максимальная длина токенизации
            
        Returns:
            surface_embeddings: [batch_size, surface_dim] - эмбеддинги поверхности куба
        """
        try:
            # Обеспечиваем список
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                logger.warning("Empty texts provided to encode_text")
                return torch.zeros(1, self.surface_dim, device=self.device)
            
            batch_size = len(texts)
            
            # Фильтруем пустые тексты
            valid_texts = [text if text and text.strip() else "empty" for text in texts]
            
            # Токенизация
            encoded = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Проверяем корректность токенизации
            if encoded['input_ids'].numel() == 0:
                logger.warning("Empty tokenization result")
                return torch.zeros(batch_size, self.surface_dim, device=self.device)
            
            # Используем default device (CUDA автоматически)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Token embeddings
            token_embeddings = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
            
            # Добавляем positional encoding
            seq_len = token_embeddings.shape[1]
            pos_encodings = self.positional_encoding[:seq_len].unsqueeze(0).to(self.device)
            embeddings = token_embeddings + pos_encodings
            
            # Создаем padding mask для transformer
            padding_mask = ~attention_mask.bool()
            
            # Transformer encoding
            encoded_embeddings = self.transformer_encoder(
                embeddings,
                src_key_padding_mask=padding_mask
            )  # [batch, seq_len, hidden_dim]
            
            # Агрегация последовательности
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoded_embeddings.size()).float()
            sum_embeddings = torch.sum(encoded_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            aggregated = sum_embeddings / sum_mask  # [batch, hidden_dim]
            
            # Проекция к поверхности куба
            surface_embeddings = self.surface_projection(aggregated)  # [batch, surface_dim]
            
            # Ensure requires_grad=True for training
            if not surface_embeddings.requires_grad:
                surface_embeddings.requires_grad_(True)
            
            # Логирование статистики
            if logger.isEnabledFor(DEBUG_ENERGY):
                stats = {
                    'batch_size': batch_size,
                    'surface_dim': self.surface_dim,
                    'mean': float(surface_embeddings.mean()),
                    'std': float(surface_embeddings.std()),
                    'min': float(surface_embeddings.min()),
                    'max': float(surface_embeddings.max()),
                    'seq_lengths': [int(mask.sum()) for mask in attention_mask],
                    'requires_grad': surface_embeddings.requires_grad
                }
                logger.log(DEBUG_ENERGY, f"TextToCubeEncoder статистика: {stats}")
                if len(valid_texts) <= 3:
                    logger.log(DEBUG_ENERGY, f"Примеры текстов: {valid_texts}")
            
            return surface_embeddings
            
        except Exception as e:
            logger.error(f"Error in encode_text: {e}")
            logger.error(f"Input texts: {texts}")
            if isinstance(texts, list):
                logger.error(f"Texts count: {len(texts)}")
                for i, text in enumerate(texts):
                    logger.error(f"Text {i}: '{text}' (type: {type(text)}, len: {len(str(text))})")
            
            # Возвращаем безопасный результат с градиентами
            batch_size = 1 if isinstance(texts, str) else max(len(texts), 1)
            return torch.zeros(batch_size, self.surface_dim, device=self.device, requires_grad=True)
    
    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Forward pass для совместимости с nn.Module
        """
        return self.encode_text(texts, **kwargs)
    
    def get_surface_shape(self) -> tuple:
        """
        Возвращает форму поверхности куба для reshape операций
        
        Returns:
            (width, height) - размеры поверхности
        """
        return (self.config.lattice_width, self.config.lattice_height)
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        Возвращает токенизатор для внешнего использования
        """
        return self.tokenizer
    
    def get_vocab_size(self) -> int:
        """
        Размер словаря токенизатора
        """
        return self.vocab_size
    
    def reshape_to_surface(self, surface_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Преобразует плоские эмбеддинги в 2D поверхность
        
        Args:
            surface_embeddings: [batch, surface_dim]
            
        Returns:
            surface_2d: [batch, height, width] - 2D представление поверхности
        """
        batch_size = surface_embeddings.shape[0]
        return surface_embeddings.view(
            batch_size, 
            self.config.lattice_height, 
            self.config.lattice_width
        )


class BatchTextToCubeEncoder:
    """
    Утилитарный класс для эффективной батчевой обработки текста
    """
    
    def __init__(self, encoder: TextToCubeEncoder, batch_size: int = 32):
        self.encoder = encoder
        self.batch_size = batch_size
        
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Эффективно кодирует большой список текстов батчами
        
        Args:
            texts: список текстов для кодирования
            
        Returns:
            surface_embeddings: [len(texts), surface_dim] - все эмбеддинги поверхности
        """
        if not texts:
            return torch.empty(0, self.encoder.surface_dim)
            
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encoder.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        return torch.cat(all_embeddings, dim=0)


# Утилитарные функции

def create_text_to_cube_encoder(config=None) -> TextToCubeEncoder:
    """
    Factory функция для создания TextToCubeEncoder
    
    Args:
        config: EnergyConfig или None для глобального
        
    Returns:
        TextToCubeEncoder instance
    """
    return TextToCubeEncoder(config)


def encode_text_list_to_surface(texts: List[str], config=None, 
                               batch_size: int = 32) -> torch.Tensor:
    """
    Удобная функция для кодирования списка текстов в поверхность куба
    
    Args:
        texts: список текстов
        config: EnergyConfig или None
        batch_size: размер батча
        
    Returns:
        surface_embeddings: [len(texts), surface_dim] - эмбеддинги поверхности
    """
    encoder = create_text_to_cube_encoder(config)
    batch_encoder = BatchTextToCubeEncoder(encoder, batch_size)
    
    return batch_encoder.encode_batch(texts)


def get_surface_dimensions(config=None) -> Dict[str, int]:
    """
    Возвращает размерности поверхности для данной конфигурации
    
    Returns:
        Dict с width, height, surface_dim
    """

    # Получаем конфигурацию
    if config is None:
        config = create_debug_config()
        set_energy_config(config)
    
    return {
        'width': config.lattice_width,
        'height': config.lattice_height,
        'surface_dim': config.lattice_width * config.lattice_height
    }