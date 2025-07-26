#!/usr/bin/env python3
"""
TextCache для energy_flow архитектуры
====================================

LRU кэш для кэширования известных пар текст ↔ surface_embeddings.
Поддерживает персистентное хранение и активацию через конфигурацию.

Функциональность:
- LRU кэширование с настраиваемым размером
- Персистентное хранение в файле
- Hash-based поиск для быстрого доступа
- Статистика использования кэша
"""

import torch
import hashlib
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import OrderedDict
import time
import threading

from ..config import create_debug_config, set_energy_config
from ..utils.logging import get_logger, DEBUG_ENERGY

logger = get_logger(__name__)


class TextCache:
    """
    LRU кэш для пар текст ↔ surface_embeddings с персистентным хранением
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 cache_file: Optional[str] = None,
                 config=None,
                 enabled: bool = True):
        """
        Args:
            max_size: максимальный размер кэша
            cache_file: файл для персистентного хранения (опционально)
            config: EnergyConfig для определения размерностей
            enabled: включен ли кэш
        """
        # Получаем конфигурацию
        if config is None:
            config = create_debug_config()
            set_energy_config(config)
        self.config = config
        
        self.max_size = max_size
        self.enabled = enabled
        self.surface_dim = config.lattice_width * config.lattice_height
        
        # LRU кэш (OrderedDict для эффективного LRU)
        self._text_to_surface_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._surface_to_text_cache: OrderedDict[str, str] = OrderedDict()
        
        # Персистентное хранение
        if cache_file is None:
            cache_file = f"text_cache_{self.surface_dim}d.pt"  # Изменено на .pt для torch.save
        self.cache_file = Path(cache_file)
        
        # Статистика
        self.stats = {
            'text_to_surface_hits': 0,
            'text_to_surface_misses': 0,
            'surface_to_text_hits': 0,
            'surface_to_text_misses': 0,
            'total_entries': 0,
            'evictions': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Загружаем существующий кэш
        self.load_cache()
        
        logger.info(
            f"TextCache инициализирован: max_size={max_size}, "
            f"surface_dim={self.surface_dim}, enabled={enabled}, "
            f"cache_file={self.cache_file}"
        )
    
    def _hash_text(self, text: str) -> str:
        """Создает hash для текста"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _hash_surface_embedding(self, surface_embedding: torch.Tensor) -> str:
        """Создает hash для surface embedding"""
        # Используем детерминированный hash на основе значений тензора
        # Округляем до 4 знаков для устойчивости к небольшим различиям
        rounded = torch.round(surface_embedding * 10000) / 10000
        
        # Используем встроенный hash PyTorch - работает на любом устройстве
        try:
            # Попробуем встроенный hash от PyTorch
            tensor_hash = hash(rounded.detach().flatten().sum().item())
            return hashlib.md5(str(tensor_hash).encode('utf-8')).hexdigest()
        except:
            # Fallback: используем строковое представление первых нескольких элементов
            first_elements = rounded.detach().flatten()[:10].tolist()
            tensor_str = str(first_elements)
            return hashlib.md5(tensor_str.encode('utf-8')).hexdigest()
    
    def put_text_to_surface(self, text: str, surface_embedding: torch.Tensor):
        """
        Кэширует пару текст → surface_embedding
        
        Args:
            text: исходный текст
            surface_embedding: соответствующий surface embedding [surface_dim]
        """
        if not self.enabled:
            return
            
        with self._lock:
            text_hash = self._hash_text(text)
            
            # Перемещаем в конец (most recently used)
            if text_hash in self._text_to_surface_cache:
                self._text_to_surface_cache.move_to_end(text_hash)
            else:
                # Добавляем новую запись, обеспечивая правильное устройство
                cloned_embedding = surface_embedding.clone().detach()
                # Сохраняем на том же устройстве что и оригинал
                self._text_to_surface_cache[text_hash] = cloned_embedding
                
                # Проверяем размер кэша
                if len(self._text_to_surface_cache) > self.max_size:
                    # Удаляем самую старую запись (LRU)
                    oldest_key = next(iter(self._text_to_surface_cache))
                    del self._text_to_surface_cache[oldest_key]
                    self.stats['evictions'] += 1
                
                self.stats['total_entries'] += 1
            
            # Дублируем в обратный кэш
            surface_hash = self._hash_surface_embedding(surface_embedding)
            self._surface_to_text_cache[surface_hash] = text
            
            # Логирование
            if logger.isEnabledFor(DEBUG_ENERGY):
                logger.log(DEBUG_ENERGY, 
                          f"Кэшировали text→surface: '{text[:50]}...' → {surface_embedding.shape}")
    
    def get_surface_from_text(self, text: str) -> Optional[torch.Tensor]:
        """
        Получает surface embedding по тексту из кэша
        
        Args:
            text: исходный текст
            
        Returns:
            surface_embedding или None если не найден
        """
        if not self.enabled:
            return None
            
        with self._lock:
            text_hash = self._hash_text(text)
            
            if text_hash in self._text_to_surface_cache:
                # Перемещаем в конец (most recently used)
                surface_embedding = self._text_to_surface_cache[text_hash]
                self._text_to_surface_cache.move_to_end(text_hash)
                self.stats['text_to_surface_hits'] += 1
                
                if logger.isEnabledFor(DEBUG_ENERGY):
                    logger.log(DEBUG_ENERGY, f"Cache HIT text→surface: '{text[:50]}...'")
                
                # Возвращаем clone на том же устройстве
                return surface_embedding.clone()
            else:
                self.stats['text_to_surface_misses'] += 1
                return None
    
    def put_surface_to_text(self, surface_embedding: torch.Tensor, text: str):
        """
        Кэширует пару surface_embedding → текст
        
        Args:
            surface_embedding: surface embedding [surface_dim]
            text: соответствующий текст
        """
        if not self.enabled:
            return
            
        with self._lock:
            surface_hash = self._hash_surface_embedding(surface_embedding)
            
            # Перемещаем в конец (most recently used)
            if surface_hash in self._surface_to_text_cache:
                self._surface_to_text_cache.move_to_end(surface_hash)
            else:
                # Добавляем новую запись
                self._surface_to_text_cache[surface_hash] = text
                
                # Проверяем размер кэша
                if len(self._surface_to_text_cache) > self.max_size:
                    # Удаляем самую старую запись (LRU)
                    oldest_key = next(iter(self._surface_to_text_cache))
                    del self._surface_to_text_cache[oldest_key]
                    self.stats['evictions'] += 1
            
            # Дублируем в прямой кэш
            text_hash = self._hash_text(text)
            cloned_embedding = surface_embedding.clone().detach()
            self._text_to_surface_cache[text_hash] = cloned_embedding
            
            # Логирование
            if logger.isEnabledFor(DEBUG_ENERGY):
                logger.log(DEBUG_ENERGY, 
                          f"Кэшировали surface→text: {surface_embedding.shape} → '{text[:50]}...'")
    
    def get_text_from_surface(self, surface_embedding: torch.Tensor) -> Optional[str]:
        """
        Получает текст по surface embedding из кэша
        
        Args:
            surface_embedding: surface embedding [surface_dim]
            
        Returns:
            text или None если не найден
        """
        if not self.enabled:
            return None
            
        with self._lock:
            surface_hash = self._hash_surface_embedding(surface_embedding)
            
            if surface_hash in self._surface_to_text_cache:
                # Перемещаем в конец (most recently used)
                text = self._surface_to_text_cache[surface_hash]
                self._surface_to_text_cache.move_to_end(surface_hash)
                self.stats['surface_to_text_hits'] += 1
                
                if logger.isEnabledFor(DEBUG_ENERGY):
                    logger.log(DEBUG_ENERGY, f"Cache HIT surface→text: '{text[:50]}...'")
                
                return text
            else:
                self.stats['surface_to_text_misses'] += 1
                return None
    
    def save_cache(self):
        """Сохраняет кэш в файл"""
        if not self.enabled:
            return
            
        try:
            with self._lock:
                # Используем torch.save для точного сохранения тензоров
                cache_data = {
                    'text_to_surface': dict(self._text_to_surface_cache),  # Сохраняем как есть
                    'surface_to_text': dict(self._surface_to_text_cache),
                    'stats': self.stats.copy(),
                    'surface_dim': self.surface_dim,
                    'config_info': {
                        'lattice_width': self.config.lattice_width,
                        'lattice_height': self.config.lattice_height,
                        'lattice_depth': self.config.lattice_depth
                    }
                }
                
                # Используем torch.save вместо pickle для точности
                torch.save(cache_data, self.cache_file)
                
                logger.info(f"Кэш сохранен в {self.cache_file} ({len(self._text_to_surface_cache)} записей)")
                
        except Exception as e:
            logger.warning(f"Ошибка сохранения кэша: {e}")
    
    def load_cache(self):
        """Загружает кэш из файла"""
        if not self.enabled or not self.cache_file.exists():
            return
            
        try:
            with self._lock:
                # Используем torch.load для точного восстановления тензоров
                cache_data = torch.load(self.cache_file, map_location='cpu')
                
                # Проверяем совместимость размерностей
                if cache_data.get('surface_dim') != self.surface_dim:
                    logger.warning(
                        f"Несовместимые размерности кэша: "
                        f"файл={cache_data.get('surface_dim')}, "
                        f"текущий={self.surface_dim}. Пропускаем загрузку."
                    )
                    return
                
                # Восстанавливаем тензоры на правильном устройстве  
                default_device = torch.get_default_device()
                for k, v in cache_data.get('text_to_surface', {}).items():
                    # Тензоры уже загружены torch.load, просто перемещаем на нужное устройство
                    if default_device.type == 'cuda':
                        v = v.to(default_device)
                    self._text_to_surface_cache[k] = v
                
                self._surface_to_text_cache.update(cache_data.get('surface_to_text', {}))
                
                # Восстанавливаем статистику
                if 'stats' in cache_data:
                    self.stats.update(cache_data['stats'])
                
                logger.info(
                    f"Кэш загружен из {self.cache_file} "
                    f"({len(self._text_to_surface_cache)} записей), "
                    f"device={default_device}"
                )
                
        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}")
    
    def clear(self):
        """Очищает весь кэш"""
        with self._lock:
            self._text_to_surface_cache.clear()
            self._surface_to_text_cache.clear()
            self.stats = {key: 0 for key in self.stats}
            logger.info("Кэш очищен")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        with self._lock:
            total_hits = self.stats['text_to_surface_hits'] + self.stats['surface_to_text_hits']
            total_requests = total_hits + self.stats['text_to_surface_misses'] + self.stats['surface_to_text_misses']
            
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'cache_size': len(self._text_to_surface_cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'enabled': self.enabled,
                'surface_dim': self.surface_dim
            }
    
    def __len__(self) -> int:
        """Размер кэша"""
        return len(self._text_to_surface_cache)
    
    def __del__(self):
        """Автоматически сохраняем кэш при удалении объекта"""
        try:
            self.save_cache()
        except:
            pass  # Игнорируем ошибки при деструкции


class CachedTextToCubeEncoder:
    """
    Wrapper для TextToCubeEncoder с кэшированием
    """
    
    def __init__(self, encoder, cache: Optional[TextCache] = None):
        """
        Args:
            encoder: TextToCubeEncoder instance
            cache: TextCache instance (создается автоматически если None)
        """
        self.encoder = encoder
        self.cache = cache or TextCache(config=encoder.config)
    
    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Кодирует текст с кэшированием
        """
        # Обеспечиваем список
        if isinstance(texts, str):
            texts = [texts]
            was_single = True
        else:
            was_single = False
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Проверяем кэш для каждого текста
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get_surface_from_text(text)
            if cached_embedding is not None:
                results.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Кодируем не кэшированные тексты
        if uncached_texts:
            uncached_embeddings = self.encoder.encode_text(uncached_texts, **kwargs)
            
            # Кэшируем новые результаты
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache.put_text_to_surface(text, embedding)
            
            # Добавляем к результатам
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                results.append((i, embedding))
        
        # Сортируем по исходному порядку
        results.sort(key=lambda x: x[0])
        final_embeddings = torch.stack([emb for _, emb in results])
        
        return final_embeddings[0] if was_single else final_embeddings


class CachedCubeToTextDecoder:
    """
    Wrapper для CubeToTextDecoder с кэшированием
    """
    
    def __init__(self, decoder, cache: Optional[TextCache] = None):
        """
        Args:
            decoder: CubeToTextDecoder instance  
            cache: TextCache instance (создается автоматически если None)
        """
        self.decoder = decoder
        self.cache = cache or TextCache(config=decoder.config)
    
    def decode_surface(self, surface_embeddings: torch.Tensor, **kwargs) -> List[str]:
        """
        Декодирует surface embeddings с кэшированием
        """
        batch_size = surface_embeddings.shape[0]
        results = []
        uncached_embeddings = []
        uncached_indices = []
        
        # Проверяем кэш для каждого embedding
        for i in range(batch_size):
            embedding = surface_embeddings[i]
            cached_text = self.cache.get_text_from_surface(embedding)
            if cached_text is not None:
                results.append((i, cached_text))
            else:
                uncached_embeddings.append(embedding)
                uncached_indices.append(i)
        
        # Декодируем не кэшированные embeddings
        if uncached_embeddings:
            uncached_batch = torch.stack(uncached_embeddings)
            uncached_texts = self.decoder.decode_surface(uncached_batch, **kwargs)
            
            # Кэшируем новые результаты
            for embedding, text in zip(uncached_embeddings, uncached_texts):
                self.cache.put_surface_to_text(embedding, text)
            
            # Добавляем к результатам
            for i, text in zip(uncached_indices, uncached_texts):
                results.append((i, text))
        
        # Сортируем по исходному порядку
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]


# Утилитарные функции

def create_text_cache(max_size: int = 10000, 
                     cache_file: Optional[str] = None,
                     config=None) -> TextCache:
    """Factory функция для создания TextCache"""
    return TextCache(max_size=max_size, cache_file=cache_file, config=config)


def create_cached_encoder(encoder, cache: Optional[TextCache] = None) -> CachedTextToCubeEncoder:
    """Factory функция для создания кэшированного encoder"""
    return CachedTextToCubeEncoder(encoder, cache)


def create_cached_decoder(decoder, cache: Optional[TextCache] = None) -> CachedCubeToTextDecoder:
    """Factory функция для создания кэшированного decoder"""
    return CachedCubeToTextDecoder(decoder, cache)