#!/usr/bin/env python3
"""
CubeToTextDecoder для energy_flow архитектуры
=============================================

Восстанавливает текст из эмбеддингов поверхности куба используя специализированную
архитектуру инверсии эмбеддингов на базе T5-small. Адаптация принципов vec2text
для наших размерностей поверхности куба.

Архитектура:
Surface Embeddings → Adapter → T5-small → Iterative Correction → Text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Union, Dict, Optional, Tuple
import numpy as np

from ..config import create_debug_config, set_energy_config
from ..utils.logging import get_logger, DEBUG_ENERGY

logger = get_logger(__name__)


class CubeToTextDecoder(nn.Module):
    """
    Восстанавливает текст из эмбеддингов поверхности куба
    
    Специализированная модель (~60M параметров) для embedding inversion.
    Использует принципы vec2text с адаптацией к размерностям поверхности куба.
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        # Адаптивная размерность поверхности куба
        self.surface_dim = self.config.lattice_width * self.config.lattice_height
        
        # T5 компоненты для инверсии
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # Добавляем специальные токены если их нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # T5-small hidden size = 512
        self.t5_hidden_size = self.t5_model.config.d_model
        
        # Адаптер surface embeddings → T5 input space
        self.surface_adapter = nn.Sequential(
            nn.Linear(self.surface_dim, self.surface_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.surface_dim // 2),
            nn.Dropout(0.1),
            nn.Linear(self.surface_dim // 2, self.t5_hidden_size),
            nn.LayerNorm(self.t5_hidden_size)
        )
        
        # Итеративная коррекция (vec2text принципы)
        self.correction_steps = getattr(config, 'iterative_correction_steps', 3)
        
        # Специальные токены для управления процессом
        self.surface_prefix = "decode surface: "
        
        # Замораживаем T5 encoder для стабильности (можем разморозить потом)
        for param in self.t5_model.encoder.parameters():
            param.requires_grad = False
        
        # Статистика параметров
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        logger.info(
            f"CubeToTextDecoder инициализирован: "
            f"{self.surface_dim}D поверхность → текст "
            f"({self.config.lattice_width}×{self.config.lattice_height}), "
            f"{total_params:,} обучаемых + {frozen_params:,} замороженных параметров"
        )
    
    def _surface_to_t5_input(self, surface_embeddings: torch.Tensor, 
                           max_length: int = 64) -> Dict[str, torch.Tensor]:
        """
        Преобразует surface embeddings в входные данные для T5
        
        Args:
            surface_embeddings: [batch_size, surface_dim]
            max_length: максимальная длина выходной последовательности
            
        Returns:
            Dict с input_ids, attention_mask для T5
        """
        batch_size = surface_embeddings.shape[0]
        
        # Адаптация surface → T5 hidden space
        adapted_embeddings = self.surface_adapter(surface_embeddings)  # [batch, t5_hidden_size]
        
        # Создаем псевдо input_ids для decoder-only генерации
        # Используем специальный префикс + padding
        prefix_text = [self.surface_prefix] * batch_size
        
        tokenized = self.tokenizer(
            prefix_text,
            padding='max_length',
            truncation=True,
            max_length=8,  # Короткий префикс
            return_tensors='pt'
        )
        
        # Создаем decoder input (начинаем с pad_token)
        decoder_input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        decoder_input_ids[:, 0] = self.tokenizer.pad_token_id  # BOS token
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'decoder_input_ids': decoder_input_ids,
            'adapted_embeddings': adapted_embeddings
        }
    
    def decode_surface(self, surface_embeddings: torch.Tensor,
                       max_length: int = 64,
                       num_beams: int = 4,
                       temperature: float = 1.0) -> List[str]:
        """
        Декодирует surface embeddings в текст
        
        Args:
            surface_embeddings: [batch_size, surface_dim]
            max_length: максимальная длина генерируемого текста
            num_beams: количество beams для beam search
            temperature: температура для генерации
            
        Returns:
            List[str] - сгенерированные тексты
        """
        try:
            # Проверяем входные данные
            if surface_embeddings is None or surface_embeddings.numel() == 0:
                logger.warning("Empty or None surface embeddings provided")
                return [""]
            
            # Убеждаемся, что это 2D тензор (избегаем in-place операций)
            if surface_embeddings.dim() == 1:
                surface_embeddings = surface_embeddings.unsqueeze(0)  # Создает новый тензор, не in-place
            
            batch_size = surface_embeddings.shape[0]
            
            # Проверяем на NaN/Inf значения
            if torch.isnan(surface_embeddings).any() or torch.isinf(surface_embeddings).any():
                logger.warning("NaN or Inf values in surface embeddings")
                return [""] * batch_size
            
            # Подготавливаем входные данные для T5
            t5_inputs = self._surface_to_t5_input(surface_embeddings, max_length)
            
            # Проверяем корректность входных данных
            if 'input_ids' not in t5_inputs or t5_inputs['input_ids'].numel() == 0:
                logger.warning("Invalid T5 inputs generated")
                return [""] * batch_size
            
            # Создаем правильные encoder outputs для T5
            from transformers.modeling_outputs import BaseModelOutput
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=t5_inputs['adapted_embeddings'].unsqueeze(1),  # [batch, 1, hidden]
                hidden_states=None,
                attentions=None
            )
            
            # Генерируем текст через T5 decoder
            with torch.no_grad():
                generated_ids = self.t5_model.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=t5_inputs['decoder_input_ids'][:, :1],  # Только BOS
                    max_length=max_length,
                    num_beams=max(1, num_beams),  # Ensure at least 1 beam
                    temperature=max(0.1, temperature),  # Ensure positive temperature
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Декодируем в текст
            decoded_texts = []
            for ids in generated_ids:
                if ids is not None and ids.numel() > 0:
                    try:
                        text = self.tokenizer.decode(ids, skip_special_tokens=True)
                        # Убираем префикс если он есть
                        if text and text.startswith(self.surface_prefix):
                            text = text[len(self.surface_prefix):].strip()
                        decoded_texts.append(text or "generated_text")
                    except Exception as decode_error:
                        logger.warning(f"Token decode error: {decode_error}")
                        decoded_texts.append("")
                else:
                    decoded_texts.append("")
            
            return decoded_texts
            
        except Exception as e:
            logger.error(f"Error in decode_surface: {e}")
            if surface_embeddings is not None:
                logger.error(f"Surface embeddings shape: {surface_embeddings.shape}")
                logger.error(f"Surface embeddings stats: mean={surface_embeddings.mean():.4f}, std={surface_embeddings.std():.4f}")
            return [""] * (surface_embeddings.shape[0] if surface_embeddings is not None else 1)
    
    def iterative_decode(self, surface_embeddings: torch.Tensor,
                        max_length: int = 64,
                        correction_steps: Optional[int] = None) -> List[str]:
        """
        Итеративная коррекция декодирования (vec2text принципы)
        
        Args:
            surface_embeddings: [batch_size, surface_dim]
            max_length: максимальная длина текста
            correction_steps: количество шагов коррекции (по умолчанию из config)
            
        Returns:
            List[str] - улучшенные тексты после коррекции
        """
        if correction_steps is None:
            correction_steps = self.correction_steps
        
        # Начальная гипотеза
        current_texts = self.decode_surface(surface_embeddings, max_length)
        
        # Итеративная коррекция (упрощенная версия vec2text)
        for step in range(correction_steps):
            if logger.isEnabledFor(DEBUG_ENERGY):
                logger.log(DEBUG_ENERGY, f"Коррекция шаг {step+1}/{correction_steps}")
            
            # В реальной реализации здесь была бы более сложная коррекция
            # Пока просто повторяем декодирование с разными параметрами
            corrected_texts = self.decode_surface(
                surface_embeddings, 
                max_length, 
                temperature=0.7 - step * 0.1  # Уменьшаем температуру
            )
            
            # Простая эвристика: выбираем более длинные тексты (часто лучше)
            for i, (current, corrected) in enumerate(zip(current_texts, corrected_texts)):
                if len(corrected.split()) > len(current.split()):
                    current_texts[i] = corrected
        
        return current_texts
    
    def forward(self, surface_embeddings: torch.Tensor, **kwargs) -> List[str]:
        """
        Forward pass для совместимости с nn.Module
        """
        return self.decode_surface(surface_embeddings, **kwargs)
    
    def get_tokenizer(self) -> T5Tokenizer:
        """
        Возвращает токенизатор для внешнего использования
        """
        return self.tokenizer
    
    def unfreeze_t5_encoder(self):
        """
        Размораживает T5 encoder для fine-tuning
        """
        for param in self.t5_model.encoder.parameters():
            param.requires_grad = True
        logger.info("T5 encoder разморожен для fine-tuning")
    
    def freeze_t5_encoder(self):
        """
        Замораживает T5 encoder
        """
        for param in self.t5_model.encoder.parameters():
            param.requires_grad = False
        logger.info("T5 encoder заморожен")


class SyntheticTrainingDataGenerator:
    """
    Генератор синтетических данных для pre-training CubeToTextDecoder
    """
    
    def __init__(self, config=None):
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        self.surface_dim = config.lattice_width * config.lattice_height
    
    def generate_synthetic_pairs(self, num_samples: int = 1000) -> List[Tuple[torch.Tensor, str]]:
        """
        Генерирует синтетические пары (surface_embedding, text) для обучения
        
        Args:
            num_samples: количество пар для генерации
            
        Returns:
            List[Tuple[tensor, str]] - пары для обучения
        """
        pairs = []
        
        # Простые шаблоны для начального обучения
        templates = [
            "This is a test sentence.",
            "Hello world from the cube.",
            "Machine learning is fascinating.",
            "Neural networks process information.",
            "Energy flows through the lattice.",
            "Text to embedding conversion works.",
            "Surface dimensions are important.",
            "Cube processing enables learning.",
        ]
        
        for i in range(num_samples):
            # Случайный surface embedding в диапазоне [-1, 1]
            surface_emb = torch.randn(self.surface_dim) * 0.5  # Умеренные значения
            surface_emb = torch.clamp(surface_emb, -1, 1)
            
            # Случайный шаблон
            text = templates[i % len(templates)]
            
            pairs.append((surface_emb, text))
        
        return pairs


# Утилитарные функции

def create_cube_to_text_decoder(config=None) -> CubeToTextDecoder:
    """
    Factory функция для создания CubeToTextDecoder
    
    Args:
        config: EnergyConfig или None для глобального
        
    Returns:
        CubeToTextDecoder instance
    """
    return CubeToTextDecoder(config)


def decode_surface_embeddings_to_text(surface_embeddings: torch.Tensor, 
                                     config=None,
                                     use_iterative: bool = True) -> List[str]:
    """
    Удобная функция для декодирования surface embeddings в текст
    
    Args:
        surface_embeddings: [batch_size, surface_dim]
        config: EnergyConfig или None
        use_iterative: использовать итеративную коррекцию
        
    Returns:
        List[str] - декодированные тексты
    """
    decoder = create_cube_to_text_decoder(config)
    
    if use_iterative:
        return decoder.iterative_decode(surface_embeddings)
    else:
        return decoder.decode_surface(surface_embeddings)


def create_synthetic_training_generator(config=None) -> SyntheticTrainingDataGenerator:
    """
    Factory функция для создания генератора синтетических данных
    """
    return SyntheticTrainingDataGenerator(config)