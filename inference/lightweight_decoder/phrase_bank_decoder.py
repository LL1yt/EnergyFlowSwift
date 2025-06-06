"""
🔤 PHRASE BANK DECODER - Декодирование через поиск фраз

Реализует декодирование эмбедингов в текст через поиск наиболее семантически
близких фраз в предобученном phrase bank.

Phase 2.7.1 - PhraseBankDecoder Implementation
"""

import torch
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .phrase_bank import PhraseBank, PhraseEntry

@dataclass
class DecodingConfig:
    """Конфигурация для декодирования"""
    max_candidates: int = 10          # Максимум кандидатов для поиска
    similarity_threshold: float = 0.8  # Минимальный threshold similarity
    assembly_method: str = "weighted"  # Метод сборки: weighted, greedy, beam_search
    coherence_weight: float = 0.3     # Вес когерентности
    relevance_weight: float = 0.7     # Вес релевантности
    min_phrase_length: int = 3        # Минимальная длина фразы (слова)
    max_phrase_length: int = 50       # Максимальная длина фразы (слова)

class TextAssembler:
    """Сборщик текста из найденных фраз"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
    
    def assemble_weighted(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Взвешенная сборка на основе similarity scores"""
        if not candidates:
            return "No suitable phrases found."
        
        # Простая стратегия: берем лучшую фразу
        best_phrase, best_similarity = candidates[0]
        
        # Проверка качества
        if best_similarity < self.config.similarity_threshold:
            return "Low confidence phrase match."
        
        return best_phrase.text
    
    def assemble_greedy(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Жадная сборка - просто лучший кандидат"""
        if not candidates:
            return "No phrases available."
        
        best_phrase, _ = candidates[0]
        return best_phrase.text
    
    def assemble_beam_search(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Beam search сборка (упрощенная версия)"""
        if not candidates:
            return "No beam candidates."
        
        # Пока простая реализация - можно расширить для multi-phrase assembly
        filtered_candidates = [
            (phrase, similarity) for phrase, similarity in candidates
            if self.config.min_phrase_length <= phrase.length <= self.config.max_phrase_length
        ]
        
        if not filtered_candidates:
            # Fallback к первому кандидату
            return candidates[0][0].text
        
        # Возьмем лучший отфильтрованный
        best_phrase, _ = filtered_candidates[0]
        return best_phrase.text
    
    def assemble(self, candidates: List[Tuple[PhraseEntry, float]]) -> str:
        """Главный метод сборки"""
        if self.config.assembly_method == "weighted":
            return self.assemble_weighted(candidates)
        elif self.config.assembly_method == "greedy":
            return self.assemble_greedy(candidates)
        elif self.config.assembly_method == "beam_search":
            return self.assemble_beam_search(candidates)
        else:
            logging.warning(f"Unknown assembly method: {self.config.assembly_method}")
            return self.assemble_greedy(candidates)

class QualityAssessor:
    """Оценщик качества декодирования"""
    
    def __init__(self, config: DecodingConfig):
        self.config = config
    
    def assess_candidates(self, candidates: List[Tuple[PhraseEntry, float]]) -> Dict:
        """Оценка качества кандидатов"""
        if not candidates:
            return {
                'quality_score': 0.0,
                'confidence': 0.0,
                'coherence': 0.0,
                'diversity': 0.0
            }
        
        # Базовые метрики
        similarities = [similarity for _, similarity in candidates]
        
        quality_score = np.mean(similarities)
        confidence = max(similarities)
        coherence = self._assess_coherence(candidates)
        diversity = self._assess_diversity(candidates)
        
        return {
            'quality_score': float(quality_score),
            'confidence': float(confidence),
            'coherence': float(coherence),
            'diversity': float(diversity)
        }
    
    def _assess_coherence(self, candidates: List[Tuple[PhraseEntry, float]]) -> float:
        """Оценка когерентности кандидатов"""
        if len(candidates) <= 1:
            return 1.0
        
        # Простая метрика: схожесть категорий
        categories = [phrase.category for phrase, _ in candidates]
        unique_categories = set(categories)
        
        # Больше одинаковых категорий = выше когерентность
        coherence = 1.0 - (len(unique_categories) - 1) / len(candidates)
        return max(0.0, coherence)
    
    def _assess_diversity(self, candidates: List[Tuple[PhraseEntry, float]]) -> float:
        """Оценка разнообразия кандидатов"""
        if len(candidates) <= 1:
            return 0.0
        
        # Простая метрика: разнообразие длин фраз
        lengths = [phrase.length for phrase, _ in candidates]
        diversity = np.std(lengths) / max(lengths) if max(lengths) > 0 else 0.0
        
        return min(1.0, diversity)

class PhraseBankDecoder:
    """Основной декодер на основе phrase bank"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 phrase_bank_size: int = 50000,
                 similarity_threshold: float = 0.8,
                 config: Optional[DecodingConfig] = None):
        
        self.embedding_dim = embedding_dim
        self.phrase_bank_size = phrase_bank_size
        self.similarity_threshold = similarity_threshold
        
        # Конфигурация
        self.config = config or DecodingConfig(similarity_threshold=similarity_threshold)
        
        # Компоненты
        self.phrase_bank = PhraseBank(
            embedding_dim=embedding_dim,
            similarity_threshold=similarity_threshold,
            max_phrases=phrase_bank_size
        )
        
        self.assembler = TextAssembler(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        
        # Статистика
        self.stats = {
            'total_decodings': 0,
            'successful_decodings': 0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0
        }
        
        self.ready = False
        
        logging.info(f"PhraseBankDecoder initialized: dim={embedding_dim}, threshold={similarity_threshold}")
    
    def load_phrase_bank(self, embedding_loader=None, bank_path: Optional[str] = None):
        """Загрузка phrase bank"""
        if bank_path:
            # Загрузка из файла
            self.phrase_bank.load_bank(bank_path)
        elif embedding_loader:
            # Создание sample bank для тестирования
            self.phrase_bank.load_sample_bank(embedding_loader)
        else:
            raise ValueError("Either embedding_loader or bank_path must be provided")
        
        self.ready = True
        logging.info("Phrase bank loaded successfully")
    
    def decode(self, embedding: torch.Tensor) -> str:
        """Основной метод декодирования"""
        if not self.ready:
            raise ValueError("Phrase bank not loaded. Call load_phrase_bank() first.")
        
        if embedding.dim() != 1 or embedding.size(0) != self.embedding_dim:
            raise ValueError(f"Expected embedding shape ({self.embedding_dim},), got {embedding.shape}")
        
        try:
            # Поиск кандидатов
            candidates = self.phrase_bank.search_phrases(
                embedding, 
                k=self.config.max_candidates,
                min_similarity=self.config.similarity_threshold
            )
            
            # Оценка качества
            quality_metrics = self.quality_assessor.assess_candidates(candidates)
            
            # Сборка текста
            decoded_text = self.assembler.assemble(candidates)
            
            # Обновление статистики
            self._update_stats(quality_metrics, len(candidates) > 0)
            
            logging.debug(f"Decoded: {decoded_text} (confidence: {quality_metrics['confidence']:.3f})")
            
            return decoded_text
            
        except Exception as e:
            logging.error(f"Decoding failed: {e}")
            self._update_stats({'quality_score': 0.0, 'confidence': 0.0}, False)
            return "Decoding error occurred."
    
    def decode_with_metrics(self, embedding: torch.Tensor) -> Tuple[str, Dict]:
        """Декодирование с подробными метриками"""
        if not self.ready:
            raise ValueError("Phrase bank not loaded.")
        
        # Поиск кандидатов
        candidates = self.phrase_bank.search_phrases(
            embedding, 
            k=self.config.max_candidates,
            min_similarity=self.config.similarity_threshold
        )
        
        # Оценка качества
        quality_metrics = self.quality_assessor.assess_candidates(candidates)
        
        # Сборка текста
        decoded_text = self.assembler.assemble(candidates)
        
        # Дополнительные метрики
        detailed_metrics = {
            **quality_metrics,
            'num_candidates': len(candidates),
            'top_similarity': candidates[0][1] if candidates else 0.0,
            'phrase_categories': [phrase.category for phrase, _ in candidates[:3]],
            'phrase_bank_stats': self.phrase_bank.get_statistics()
        }
        
        self._update_stats(quality_metrics, len(candidates) > 0)
        
        return decoded_text, detailed_metrics
    
    def batch_decode(self, embeddings: torch.Tensor) -> List[str]:
        """Batch декодирование для эффективности"""
        if embeddings.dim() != 2 or embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"Expected embeddings shape (N, {self.embedding_dim}), got {embeddings.shape}")
        
        results = []
        for i, embedding in enumerate(embeddings):
            try:
                result = self.decode(embedding)
                results.append(result)
            except Exception as e:
                logging.warning(f"Batch decode failed for item {i}: {e}")
                results.append("Batch decode error.")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Статистика декодера"""
        success_rate = (
            self.stats['successful_decodings'] / max(self.stats['total_decodings'], 1) * 100
        )
        
        return {
            **self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'phrase_bank_stats': self.phrase_bank.get_statistics(),
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'assembly_method': self.config.assembly_method,
                'max_candidates': self.config.max_candidates
            }
        }
    
    def _update_stats(self, quality_metrics: Dict, success: bool):
        """Обновление статистики"""
        self.stats['total_decodings'] += 1
        
        if success:
            self.stats['successful_decodings'] += 1
        
        # Скользящее среднее для метрик
        total = self.stats['total_decodings']
        
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + quality_metrics.get('confidence', 0.0)) / total
        )
        
        self.stats['avg_quality'] = (
            (self.stats['avg_quality'] * (total - 1) + quality_metrics.get('quality_score', 0.0)) / total
        )
    
    def set_config(self, **kwargs):
        """Обновление конфигурации"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logging.info(f"Updated config: {key} = {value}")
            else:
                logging.warning(f"Unknown config parameter: {key}")
        
        # Пересоздание компонентов при необходимости
        self.assembler = TextAssembler(self.config)
        self.quality_assessor = QualityAssessor(self.config)

# Логирование
logger = logging.getLogger(__name__) 