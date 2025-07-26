#!/usr/bin/env python3
"""
Text Encoder для energy_flow архитектуры
========================================

Преобразует текст в эмбеддинги куба (768D) с совместимостью с EnergyEmbeddingMapper.
Использует DistilBERT токенизатор и промежуточную нейросеть для адаптации.

Архитектура:
Текст → DistilBERT токенизация → Промежуточная нейросеть → 768D эмбеддинги
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Union, Dict, Optional
import numpy as np

from ..config.energy_config import get_energy_config
from ..utils.logging import get_logger, DEBUG_ENERGY

logger = get_logger(__name__)


class TextEncoder(nn.Module):
    """
    Преобразует текст в эмбеддинги куба
    
    Использует DistilBERT для токенизации и извлечения контекстных эмбеддингов,
    затем адаптирует их к формату energy_flow (768D).
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or get_energy_config()
        
        # Целевая размерность (совместимость с EnergyEmbeddingMapper)
        self.output_dim = self.config.input_embedding_dim_from_teacher  # 768
        
        # DistilBERT компоненты
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Замораживаем BERT для стабильности (можно разморозить позже)
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        # BERT выдает 768D, нам нужно 768D - прямое отображение с адаптацией
        self.adaptation_layers = nn.Sequential(
            # Адаптационный слой для улучшения представлений
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            
            # Проекция к целевой размерности
            nn.Linear(1024, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.Dropout(0.05)
        )
        
        # Специальные токены
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        
        logger.info(f"TextEncoder инициализирован: текст → {self.output_dim}D эмбеддинги")
        
    def encode_text(self, texts: Union[str, List[str]], 
                   max_length: int = 128) -> torch.Tensor:
        """
        Кодирует текст в эмбеддинги куба
        
        Args:
            texts: строка или список строк для кодирования
            max_length: максимальная длина токенизации
            
        Returns:
            embeddings: [batch_size, 768] - эмбеддинги для energy_flow
        """
        # Обеспечиваем список
        if isinstance(texts, str):
            texts = [texts]
            
        batch_size = len(texts)
        
        # Токенизация
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Перемещаем на правильное устройство
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        
        # BERT обработка
        with torch.no_grad():  # BERT заморожен
            bert_outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Используем [CLS] токен как представление предложения
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Адаптация к energy_flow формату
        adapted_embeddings = self.adaptation_layers(cls_embeddings)  # [batch_size, 768]
        
        # Логирование статистики
        if logger.isEnabledFor(DEBUG_ENERGY):
            stats = {
                'batch_size': batch_size,
                'mean': float(adapted_embeddings.mean()),
                'std': float(adapted_embeddings.std()),
                'min': float(adapted_embeddings.min()),
                'max': float(adapted_embeddings.max())
            }
            logger.log(DEBUG_ENERGY, f"TextEncoder статистика: {stats}")
            logger.log(DEBUG_ENERGY, f"Примеры текстов: {texts[:3]}...")  # Первые 3 для отладки
        
        return adapted_embeddings
    
    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Forward pass для совместимости с nn.Module
        """
        return self.encode_text(texts, **kwargs)
    
    def get_tokenizer(self) -> DistilBertTokenizer:
        """
        Возвращает токенизатор для внешнего использования
        """
        return self.tokenizer
    
    def get_vocab_size(self) -> int:
        """
        Размер словаря токенизатора
        """
        return self.tokenizer.vocab_size
    
    def unfreeze_bert(self):
        """
        Размораживает BERT для fine-tuning
        """
        for param in self.bert_model.parameters():
            param.requires_grad = True
        logger.info("BERT разморожен для fine-tuning")
    
    def freeze_bert(self):
        """
        Замораживает BERT
        """
        for param in self.bert_model.parameters():
            param.requires_grad = False
        logger.info("BERT заморожен")


class BatchTextEncoder:
    """
    Утилитарный класс для эффективной батчевой обработки текста
    """
    
    def __init__(self, encoder: TextEncoder, batch_size: int = 32):
        self.encoder = encoder
        self.batch_size = batch_size
        
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Эффективно кодирует большой список текстов батчами
        
        Args:
            texts: список текстов для кодирования
            
        Returns:
            embeddings: [len(texts), 768] - все эмбеддинги
        """
        if not texts:
            return torch.empty(0, self.encoder.output_dim)
            
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encoder.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        return torch.cat(all_embeddings, dim=0)


# Утилитарные функции

def create_text_encoder(config=None, freeze_bert: bool = True) -> TextEncoder:
    """
    Factory функция для создания TextEncoder
    
    Args:
        config: EnergyConfig или None для глобального
        freeze_bert: заморозить ли BERT веса
        
    Returns:
        TextEncoder instance
    """
    encoder = TextEncoder(config)
    
    if not freeze_bert:
        encoder.unfreeze_bert()
        
    return encoder


def encode_text_list(texts: List[str], config=None, 
                    batch_size: int = 32) -> torch.Tensor:
    """
    Удобная функция для кодирования списка текстов
    
    Args:
        texts: список текстов
        config: EnergyConfig или None
        batch_size: размер батча
        
    Returns:
        embeddings: [len(texts), 768] - эмбеддинги
    """
    encoder = create_text_encoder(config)
    batch_encoder = BatchTextEncoder(encoder, batch_size)
    
    return batch_encoder.encode_batch(texts)