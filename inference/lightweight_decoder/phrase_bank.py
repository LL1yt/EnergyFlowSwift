"""
🏦 PHRASE BANK - Управление фразовой базой для декодирования

Этот модуль реализует систему хранения и поиска фраз на основе семантического сходства.
Основа для PhraseBankDecoder в Phase 2.7.1.

Компоненты:
- PhraseBank: основной класс для управления фразами
- PhraseIndex: быстрый поиск по similarity
- PhraseLoader: загрузка и подготовка данных
"""

import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import time

# Попытка импорта FAISS (может быть недоступен)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, falling back to linear search")

@dataclass
class PhraseEntry:
    """Запись фразы в банке"""
    text: str                    # Текст фразы
    embedding: torch.Tensor      # Эмбединг фразы (768D)
    frequency: int = 1           # Частота использования
    category: str = "general"    # Категория фразы
    length: int = 0              # Длина в словах
    
    def __post_init__(self):
        if self.length == 0:
            self.length = len(self.text.split())

class PhraseIndex:
    """Индекс для быстрого поиска фраз по similarity"""
    
    def __init__(self, embedding_dim: int = 768, index_type: str = "faiss"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.phrases = []  # Список PhraseEntry
        self.built = False
        
        # Инициализация индекса
        if index_type == "faiss" and FAISS_AVAILABLE:
            # FAISS index для быстрого поиска
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product для cosine similarity
            self.use_faiss = True
        else:
            # Fallback к linear search
            self.use_faiss = False
            self.embeddings_matrix = None
            logging.info(f"Using linear search index (FAISS not available or type={index_type})")
    
    def add_phrases(self, phrases: List[PhraseEntry]):
        """Добавление фраз в индекс"""
        self.phrases.extend(phrases)
        
        if self.use_faiss:
            # Подготовка эмбедингов для FAISS
            embeddings = []
            for phrase in phrases:
                # Нормализация для cosine similarity
                embedding = phrase.embedding.numpy()
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            embeddings_matrix = np.vstack(embeddings).astype(np.float32)
            self.index.add(embeddings_matrix)
        else:
            # Linear search подготовка
            all_embeddings = []
            for phrase in self.phrases:
                embedding = phrase.embedding.numpy()
                embedding = embedding / np.linalg.norm(embedding)  # Нормализация
                all_embeddings.append(embedding)
            
            self.embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
        
        self.built = True
        logging.info(f"Added {len(phrases)} phrases to index. Total: {len(self.phrases)}")
    
    def search(self, query_embedding: torch.Tensor, k: int = 10) -> List[Tuple[PhraseEntry, float]]:
        """Поиск k наиболее похожих фраз"""
        if not self.built:
            raise ValueError("Index not built. Call add_phrases() first.")
        
        # Нормализация query для cosine similarity
        query = query_embedding.numpy()
        query = query / np.linalg.norm(query)
        query = query.reshape(1, -1).astype(np.float32)
        
        if self.use_faiss:
            # FAISS поиск
            similarities, indices = self.index.search(query, k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.phrases):  # Валидный индекс
                    results.append((self.phrases[idx], float(similarity)))
            
            return results
        else:
            # Linear search
            similarities = np.dot(self.embeddings_matrix, query.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                results.append((self.phrases[idx], float(similarity)))
            
            return results

class PhraseLoader:
    """Загрузчик фраз из различных источников"""
    
    @staticmethod
    def create_sample_phrases(embedding_loader) -> List[PhraseEntry]:
        """Создание sample phrase bank для тестирования"""
        
        # Базовые фразы для тестирования
        sample_phrases = [
            # Общие фразы
            "Hello, how are you?",
            "Thank you very much.",
            "I don't understand.",
            "Could you please help me?",
            "That's a great idea!",
            
            # Технические фразы
            "Machine learning is fascinating.",
            "Neural networks process information.",
            "Artificial intelligence advances rapidly.",
            "Data science requires careful analysis.",
            "Computer vision enables automation.",
            
            # Разговорные фразы
            "What's the weather like today?",
            "I'm looking forward to it.",
            "Let me think about that.",
            "This is really interesting.",
            "I completely agree with you.",
            
            # Фразы для диалога
            "Can you explain that further?",
            "I see what you mean.",
            "That makes perfect sense.",
            "I have a question about this.",
            "Thank you for your patience.",
            
            # Креативные фразы  
            "The sun shines brightly today.",
            "Music fills the air with joy.",
            "Books open doors to knowledge.",
            "Technology shapes our future.",
            "Learning never truly ends."
        ]
        
        phrase_entries = []
        
        # Генерация эмбедингов для фраз батчами для эффективности
        try:
            # Получение всех эмбедингов сразу через правильный API
            embeddings = embedding_loader.load_from_llm(
                texts=sample_phrases,
                model_key="distilbert",
                use_cache=True
            )
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            return []
        
        # Создание phrase entries
        for i, phrase_text in enumerate(sample_phrases):
            try:
                embedding = embeddings[i]
                
                # Определение категории
                if "machine learning" in phrase_text.lower() or "neural" in phrase_text.lower():
                    category = "technical"
                elif "hello" in phrase_text.lower() or "thank you" in phrase_text.lower():
                    category = "greeting"
                elif "sun" in phrase_text.lower() or "music" in phrase_text.lower():
                    category = "creative"
                else:
                    category = "general"
                
                phrase_entry = PhraseEntry(
                    text=phrase_text,
                    embedding=embedding,
                    frequency=1,
                    category=category
                )
                phrase_entries.append(phrase_entry)
                
            except Exception as e:
                logging.warning(f"Failed to create embedding for phrase '{phrase_text}': {e}")
                continue
        
        logging.info(f"Created {len(phrase_entries)} sample phrases")
        return phrase_entries
    
    @staticmethod
    def load_from_file(file_path: str) -> List[PhraseEntry]:
        """Загрузка фраз из файла"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Phrase file not found: {file_path}")
        
        try:
            with open(path, 'rb') as f:
                phrase_data = pickle.load(f)
            
            # Конвертация в PhraseEntry если нужно
            if isinstance(phrase_data, list) and len(phrase_data) > 0:
                if isinstance(phrase_data[0], PhraseEntry):
                    return phrase_data
                else:
                    # Конвертация из других форматов
                    logging.warning("Converting phrase data format")
                    return []
            
        except Exception as e:
            logging.error(f"Failed to load phrases from {file_path}: {e}")
            return []
    
    @staticmethod
    def save_to_file(phrases: List[PhraseEntry], file_path: str):
        """Сохранение фраз в файл"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(phrases, f)
            
            logging.info(f"Saved {len(phrases)} phrases to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save phrases to {file_path}: {e}")

class PhraseBank:
    """Основной класс для управления phrase bank"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 similarity_threshold: float = 0.8,
                 max_phrases: int = 50000,
                 index_type: str = "faiss",
                 cache_size: int = 1000):
        
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_phrases = max_phrases
        self.cache_size = cache_size
        
        # Компоненты
        self.index = PhraseIndex(embedding_dim, index_type)
        self.cache = {}  # Простой cache для частых запросов
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0
        }
        
        logging.info(f"PhraseBank initialized: dim={embedding_dim}, threshold={similarity_threshold}")
    
    def add_phrases(self, phrases: List[PhraseEntry]):
        """Добавление фраз в банк"""
        if len(self.index.phrases) + len(phrases) > self.max_phrases:
            logging.warning(f"Phrase limit reached ({self.max_phrases}), truncating")
            phrases = phrases[:self.max_phrases - len(self.index.phrases)]
        
        self.index.add_phrases(phrases)
        
        # Очистка cache при обновлении
        self.cache.clear()
    
    def search_phrases(self, 
                      query_embedding: torch.Tensor, 
                      k: int = 10,
                      min_similarity: Optional[float] = None) -> List[Tuple[PhraseEntry, float]]:
        """Поиск похожих фраз"""
        
        start_time = time.time()
        
        # Использование threshold по умолчанию
        if min_similarity is None:
            min_similarity = self.similarity_threshold
        
        # Простой cache key (hash от embedding)
        cache_key = hash(query_embedding.numpy().tobytes())
        
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # Поиск через индекс
        results = self.index.search(query_embedding, k * 2)  # Больше результатов для фильтрации
        
        # Фильтрация по threshold
        filtered_results = [
            (phrase, similarity) for phrase, similarity in results
            if similarity >= min_similarity
        ]
        
        # Ограничение результатов
        filtered_results = filtered_results[:k]
        
        # Кэширование результата
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = filtered_results
        
        # Статистика
        search_time = time.time() - start_time
        self.stats['total_searches'] += 1
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) /
            self.stats['total_searches']
        )
        
        return filtered_results
    
    def get_best_phrase(self, query_embedding: torch.Tensor) -> Optional[Tuple[PhraseEntry, float]]:
        """Получение лучшей фразы для query"""
        results = self.search_phrases(query_embedding, k=1)
        
        if results:
            return results[0]
        return None
    
    def get_statistics(self) -> Dict:
        """Статистика использования"""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['total_searches'], 1) * 100
        )
        
        return {
            'total_phrases': len(self.index.phrases),
            'total_searches': self.stats['total_searches'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'avg_search_time_ms': f"{self.stats['avg_search_time'] * 1000:.2f}",
            'index_type': self.index.index_type,
            'faiss_available': FAISS_AVAILABLE
        }
    
    def load_sample_bank(self, embedding_loader):
        """Загрузка sample phrase bank для тестирования"""
        sample_phrases = PhraseLoader.create_sample_phrases(embedding_loader)
        self.add_phrases(sample_phrases)
        logging.info(f"Loaded sample bank with {len(sample_phrases)} phrases")
    
    def save_bank(self, file_path: str):
        """Сохранение phrase bank"""
        PhraseLoader.save_to_file(self.index.phrases, file_path)
    
    def load_bank(self, file_path: str):
        """Загрузка phrase bank из файла"""
        phrases = PhraseLoader.load_from_file(file_path)
        if phrases:
            self.add_phrases(phrases)
            logging.info(f"Loaded {len(phrases)} phrases from {file_path}")
        else:
            logging.warning(f"No phrases loaded from {file_path}")

# Логирование для модуля
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 