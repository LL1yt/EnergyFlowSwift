#!/usr/bin/env python3
"""
Декодер эмбедингов в текст
=========================

Простой декодер для преобразования эмбедингов обратно в текст.
Включает базовое кэширование и возможность joint training с кубом.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import json
from pathlib import Path

from ...utils.logging import get_logger, LogContext
from ...utils.device_manager import get_device_manager
from ...config.simple_config import SimpleProjectConfig

logger = get_logger(__name__)


class EmbeddingTextCache:
    """
    Простой кэш для хранения пар эмбединг-текст
    
    Использует хэширование эмбедингов для быстрого поиска
    и LRU стратегию для управления размером.
    """
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        
        # Основные хранилища
        self.embedding_to_text = {}  # hash -> text
        self.text_to_embedding = {}  # text -> (hash, embedding)
        self.embeddings_store = {}   # hash -> embedding tensor
        
        # LRU tracking
        self.access_order = []  # Порядок доступа к элементам
        self.access_count = {}  # Счетчики использования
        
        self.logger = get_logger(__name__)
        
    def _hash_embedding(self, embedding: torch.Tensor) -> str:
        """Создает хэш для эмбединга"""
        # Округляем до 4 знаков для уменьшения вариативности
        rounded = torch.round(embedding * 10000) / 10000
        embedding_bytes = rounded.detach().cpu().numpy().tobytes()
        return hashlib.md5(embedding_bytes).hexdigest()[:16]
    
    def _hash_text(self, text: str) -> str:
        """Создает хэш для текста"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def get(self, embedding: torch.Tensor) -> Optional[str]:
        """Получение текста по эмбедингу"""
        emb_hash = self._hash_embedding(embedding)
        
        # Прямое попадание
        if emb_hash in self.embedding_to_text:
            self._update_access(emb_hash)
            return self.embedding_to_text[emb_hash]
        
        # Поиск похожих эмбедингов
        best_text = self._find_similar_embedding(embedding)
        return best_text
    
    def _find_similar_embedding(self, embedding: torch.Tensor) -> Optional[str]:
        """Поиск похожего эмбединга в кэше"""
        if not self.embeddings_store:
            return None
            
        best_similarity = -1
        best_text = None
        
        for cached_hash, cached_embedding in self.embeddings_store.items():
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0), 
                cached_embedding.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_text = self.embedding_to_text.get(cached_hash)
        
        if best_text:
            self.logger.debug(f"Found similar embedding with similarity {best_similarity:.3f}")
            
        return best_text
    
    def put(self, embedding: torch.Tensor, text: str):
        """Сохранение пары эмбединг-текст"""
        emb_hash = self._hash_embedding(embedding)
        text_hash = self._hash_text(text)
        
        # Проверяем лимит размера
        if len(self.embedding_to_text) >= self.max_size:
            self._evict_lru()
        
        # Сохраняем
        self.embedding_to_text[emb_hash] = text
        self.text_to_embedding[text] = (emb_hash, embedding.clone().detach())
        self.embeddings_store[emb_hash] = embedding.clone().detach()
        
        self._update_access(emb_hash)
    
    def _update_access(self, emb_hash: str):
        """Обновление порядка доступа для LRU"""
        if emb_hash in self.access_order:
            self.access_order.remove(emb_hash)
        self.access_order.append(emb_hash)
        self.access_count[emb_hash] = self.access_count.get(emb_hash, 0) + 1
    
    def _evict_lru(self):
        """Удаление наименее используемого элемента"""
        if not self.access_order:
            return
            
        lru_hash = self.access_order.pop(0)
        
        # Удаляем из всех хранилищ
        text = self.embedding_to_text.pop(lru_hash, None)
        if text:
            self.text_to_embedding.pop(text, None)
        self.embeddings_store.pop(lru_hash, None)
        self.access_count.pop(lru_hash, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        return {
            'size': len(self.embedding_to_text),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_count.values()) / max(len(self.access_count), 1),
            'similarity_threshold': self.similarity_threshold
        }
    
    def save(self, path: str):
        """Сохранение кэша на диск"""
        cache_data = {
            'embedding_to_text': self.embedding_to_text,
            'text_to_embedding': {k: (v[0], v[1].tolist()) for k, v in self.text_to_embedding.items()},
            'embeddings_store': {k: v.tolist() for k, v in self.embeddings_store.items()},
            'access_count': self.access_count
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(cache_data, f)
        
        self.logger.info(f"Cache saved to {path} ({len(self.embedding_to_text)} items)")
    
    def load(self, path: str):
        """Загрузка кэша с диска"""
        if not Path(path).exists():
            return
            
        with open(path, 'r') as f:
            cache_data = json.load(f)
        
        self.embedding_to_text = cache_data['embedding_to_text']
        self.text_to_embedding = {
            k: (v[0], torch.tensor(v[1])) 
            for k, v in cache_data['text_to_embedding'].items()
        }
        self.embeddings_store = {
            k: torch.tensor(v) 
            for k, v in cache_data['embeddings_store'].items()
        }
        self.access_count = cache_data['access_count']
        
        self.logger.info(f"Cache loaded from {path} ({len(self.embedding_to_text)} items)")


class SimpleTextDecoder(nn.Module):
    """
    Простой декодер эмбедингов в текст
    
    Использует предобученную модель (например, DistilBERT) для 
    приблизительного декодирования эмбедингов обратно в текст.
    """
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        self.device_manager = get_device_manager()
        
        # Настройки
        self.teacher_dim = config.embedding.teacher_embedding_dim
        self.decoder_model_name = config.embedding.decoder_model
        self.max_length = config.embedding.max_decode_length
        
        # Кэш
        self.cache_enabled = config.embedding.decoder_cache_enabled
        if self.cache_enabled:
            cache_path = Path(config.embedding.cache_dir) / "text_decoder_cache.json"
            self.cache = EmbeddingTextCache()
            self.cache.load(str(cache_path))
            self.cache_path = cache_path
        
        # Инициализация модели декодера будет lazy
        self._decoder_model = None
        self._tokenizer = None
        
        self.logger.info(f"🔤 SimpleTextDecoder initialized (cache: {self.cache_enabled})")
    
    def _init_decoder_model(self):
        """Lazy инициализация модели декодера"""
        if self._decoder_model is not None:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModel
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.decoder_model_name)
            self._decoder_model = AutoModel.from_pretrained(self.decoder_model_name)
            
            # Переносим на устройство
            self._decoder_model = self.device_manager.transfer_module(self._decoder_model)
            self._decoder_model.eval()  # Для inference
            
            self.logger.info(f"Decoder model {self.decoder_model_name} loaded")
            
        except ImportError:
            self.logger.warning("transformers not available, using dummy decoder")
            self._decoder_model = "dummy"
            self._tokenizer = "dummy"
    
    def decode_embeddings(self, embeddings: torch.Tensor, use_cache: bool = True) -> List[str]:
        """
        Декодирование эмбедингов в текст
        
        Args:
            embeddings: Tensor размера [batch, embedding_dim]
            use_cache: Использовать ли кэш
            
        Returns:
            Список декодированных текстов
        """
        with LogContext("text_decoding", batch_size=embeddings.size(0)):
            batch_size = embeddings.size(0)
            results = []
            cache_hits = 0
            
            for i in range(batch_size):
                embedding = embeddings[i]
                
                # Проверяем кэш
                if use_cache and self.cache_enabled:
                    cached_text = self.cache.get(embedding)
                    if cached_text:
                        results.append(cached_text)
                        cache_hits += 1
                        continue
                
                # Декодируем новый
                decoded_text = self._decode_single(embedding)
                results.append(decoded_text)
                
                # Сохраняем в кэш
                if use_cache and self.cache_enabled:
                    self.cache.put(embedding, decoded_text)
            
            self.logger.info(f"Decoded {batch_size} embeddings (cache hits: {cache_hits})")
            return results
    
    def _decode_single(self, embedding: torch.Tensor) -> str:
        """Декодирование одного эмбединга"""
        self._init_decoder_model()
        
        if self._decoder_model == "dummy":
            # Dummy декодер для тестирования
            emb_hash = hashlib.md5(embedding.detach().cpu().numpy().tobytes()).hexdigest()[:8]
            return f"[Generated from embedding {emb_hash}]"
        
        try:
            # Используем ближайший поиск по эмбедингам модели
            # Это приблизительный подход, позже можно улучшить
            
            # Генерируем случайные токены и ищем ближайший по эмбедингу
            with torch.no_grad():
                # Простая эвристика - генерируем несколько вариантов и выбираем лучший
                candidates = [
                    "This is a sample text.",
                    "Hello world example.",
                    "Generated response text.",
                    "Sample decoded message.",
                    "Embedding reconstruction."
                ]
                
                best_text = candidates[0]
                best_similarity = -1
                
                for candidate in candidates:
                    # Получаем эмбединг кандидата
                    inputs = self._tokenizer(candidate, return_tensors="pt", 
                                           padding=True, truncation=True)
                    inputs = {k: self.device_manager.ensure_device(v) for k, v in inputs.items()}
                    
                    outputs = self._decoder_model(**inputs)
                    candidate_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    # Вычисляем сходство
                    similarity = F.cosine_similarity(
                        embedding.unsqueeze(0), 
                        candidate_embedding.unsqueeze(0)
                    ).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_text = candidate
                
                return f"{best_text} (sim: {best_similarity:.3f})"
                
        except Exception as e:
            self.logger.warning(f"Decoding error: {e}")
            return f"[Decoding error: {str(e)[:50]}]"
    
    def update_training_pairs(self, embeddings: torch.Tensor, texts: List[str]):
        """Обновление кэша парами из обучающих данных"""
        if not self.cache_enabled:
            return
            
        for i, text in enumerate(texts):
            if i < embeddings.size(0):
                self.cache.put(embeddings[i], text)
        
        self.logger.info(f"Updated cache with {len(texts)} training pairs")
    
    def save_cache(self):
        """Сохранение кэша"""
        if self.cache_enabled and hasattr(self, 'cache_path'):
            self.cache.save(str(self.cache_path))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        if self.cache_enabled:
            return self.cache.get_stats()
        return {'cache_enabled': False}


class JointTextDecoder(nn.Module):
    """
    Декодер, который может обучаться вместе с кубом
    
    Расширенная версия для joint training с основной моделью.
    """
    
    def __init__(self, config: SimpleProjectConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Базовый декодер
        self.base_decoder = SimpleTextDecoder(config)
        
        # Дополнительная обучаемая часть
        self.embedding_dim = config.embedding.teacher_embedding_dim
        self.vocab_size = 50000  # Примерный размер словаря
        
        # Простая архитектура для joint learning
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.vocab_size)
        )
        
        self.training_mode = True  # Включено ли обучение декодера
        
    def forward(self, embeddings: torch.Tensor, target_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Forward pass для joint training
        
        Args:
            embeddings: Входные эмбединги
            target_texts: Целевые тексты (для обучения)
            
        Returns:
            Dict с результатами декодирования и loss
        """
        results = {
            'decoded_texts': [],
            'logits': None,
            'loss': None
        }
        
        # Основное декодирование
        if not self.training:
            # В режиме inference используем базовый декодер
            results['decoded_texts'] = self.base_decoder.decode_embeddings(embeddings)
        else:
            # В режиме обучения используем обучаемую часть
            logits = self.projection(embeddings)
            results['logits'] = logits
            
            # Если есть целевые тексты, вычисляем loss
            if target_texts and self.training_mode:
                # Здесь нужна токенизация целевых текстов
                # Упрощенная версия пока
                target_tokens = self._tokenize_texts(target_texts)
                if target_tokens is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, self.vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100
                    )
                    results['loss'] = loss
            
            # Декодируем для отображения
            results['decoded_texts'] = self._decode_from_logits(logits)
        
        return results
    
    def _tokenize_texts(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Простая токенизация текстов"""
        # Заглушка - в реальной реализации нужен токенизатор
        return None
    
    def _decode_from_logits(self, logits: torch.Tensor) -> List[str]:
        """Декодирование из logits"""
        # Простая эвристика - берем топ токены
        top_tokens = torch.argmax(logits, dim=-1)
        
        results = []
        for i in range(logits.size(0)):
            # Заглушка декодирования
            results.append(f"[Joint decoded: tokens {top_tokens[i][:5].tolist()}]")
        
        return results
    
    def set_training_mode(self, enabled: bool):
        """Включение/выключение обучения декодера"""
        self.training_mode = enabled
        self.logger.info(f"Decoder training mode: {enabled}")


def create_text_decoder(config: SimpleProjectConfig, joint_training: bool = False) -> nn.Module:
    """Фабричная функция для создания декодера"""
    if joint_training:
        return JointTextDecoder(config)
    else:
        return SimpleTextDecoder(config)