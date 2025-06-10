"""
ProcessingMetrics - Метрики качества обработки эмбедингов
=======================================================

Отслеживает все ключевые метрики Phase 2.5:
- Cosine similarity (главная метрика - цель >90%)
- Время обработки 
- Пропускная способность
- Качество семантического сохранения
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass 
class ProcessingMetrics:
    """Метрики качества обработки эмбедингов"""
    
    # === ОСНОВНЫЕ МЕТРИКИ ===
    similarities: List[float] = field(default_factory=list)      # Cosine similarities
    processing_times: List[float] = field(default_factory=list)  # Времена обработки (сек)
    batch_sizes: List[int] = field(default_factory=list)         # Размеры батчей
    
    # === СТАТИСТИКИ ===
    total_processed: int = 0                                     # Всего обработано эмбедингов
    start_time: float = field(default_factory=time.time)        # Время начала измерений
    
    # === КОНФИГУРАЦИЯ ===
    target_similarity: float = 0.90                             # Целевая схожесть Phase 2.5
    max_history_size: int = 1000                                 # Максимум записей в истории
    
    def update(self, similarity: float, processing_time: float, batch_size: int = 1):
        """
        Обновить метрики новыми значениями
        
        Args:
            similarity: Cosine similarity между входом и выходом
            processing_time: Время обработки в секундах
            batch_size: Размер обработанного батча
        """
        # Добавляем новые значения
        self.similarities.append(similarity)
        self.processing_times.append(processing_time)
        self.batch_sizes.append(batch_size)
        self.total_processed += batch_size
        
        # Ограничиваем размер истории для экономии памяти
        if len(self.similarities) > self.max_history_size:
            self.similarities = self.similarities[-self.max_history_size:]
            self.processing_times = self.processing_times[-self.max_history_size:]
            self.batch_sizes = self.batch_sizes[-self.max_history_size:]
    
    def get_similarity_stats(self) -> Dict[str, float]:
        """Получить статистики cosine similarity"""
        if not self.similarities:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        
        similarities = np.array(self.similarities)
        
        return {
            "mean": float(np.mean(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "std": float(np.std(similarities)),
            "median": float(np.median(similarities)),
            "target_achievement": float(np.mean(similarities >= self.target_similarity))
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Получить статистики производительности"""
        if not self.processing_times:
            return {"mean_time": 0.0, "throughput": 0.0}
        
        times = np.array(self.processing_times)
        batches = np.array(self.batch_sizes)
        
        total_time = time.time() - self.start_time
        
        return {
            "mean_processing_time": float(np.mean(times)),
            "min_processing_time": float(np.min(times)),
            "max_processing_time": float(np.max(times)),
            "total_time": float(total_time),
            "throughput_embeddings_per_sec": float(self.total_processed / max(total_time, 0.001)),
            "mean_batch_size": float(np.mean(batches))
        }
    
    def get_quality_assessment(self) -> Dict[str, Any]:
        """Получить оценку качества процессора"""
        similarity_stats = self.get_similarity_stats()
        
        # Оценка качества
        quality_score = similarity_stats["mean"]
        target_achievement = similarity_stats["target_achievement"]
        
        # Классификация качества
        if quality_score >= 0.95:
            quality_level = "excellent"
        elif quality_score >= 0.90:
            quality_level = "good"
        elif quality_score >= 0.80:
            quality_level = "acceptable"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "target_achievement_rate": target_achievement,
            "phase_2_5_ready": quality_score >= self.target_similarity,
            "consistency": 1.0 - similarity_stats["std"]  # Более низкий std = больше консистентности
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить полную сводку метрик"""
        return {
            "similarity": self.get_similarity_stats(),
            "performance": self.get_performance_stats(),
            "quality": self.get_quality_assessment(),
            "total_processed": self.total_processed,
            "measurements_count": len(self.similarities)
        }
    
    def reset(self):
        """Сбросить все метрики"""
        self.similarities.clear()
        self.processing_times.clear()
        self.batch_sizes.clear()
        self.total_processed = 0
        self.start_time = time.time()
        
        logger.info("[DATA] Метрики сброшены")
    
    def log_current_stats(self):
        """Логировать текущие статистики"""
        similarity_stats = self.get_similarity_stats()
        quality = self.get_quality_assessment()
        performance = self.get_performance_stats()
        
        logger.info("=== ТЕКУЩИЕ МЕТРИКИ EMBEDDINGPROCESSOR ===")
        logger.info(f"[DATA] Средняя схожесть: {similarity_stats['mean']:.3f} (цель: {self.target_similarity:.3f})")
        logger.info(f"[TARGET] Достижение цели: {quality['target_achievement_rate']:.1%}")
        logger.info(f"[STAR] Уровень качества: {quality['quality_level']}")
        logger.info(f"[FAST] Пропускная способность: {performance['throughput_embeddings_per_sec']:.1f} эмб/сек")
        logger.info(f"🔢 Обработано всего: {self.total_processed} эмбедингов")
        logger.info(f"[OK] Phase 2.5 готовность: {'ДА' if quality['phase_2_5_ready'] else 'НЕТ'}")


def calculate_processing_quality(input_embeddings: torch.Tensor, 
                                output_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Вычислить качество обработки для батча эмбедингов
    
    Args:
        input_embeddings: Входные эмбединги [batch_size, embedding_dim]
        output_embeddings: Выходные эмбединги [batch_size, embedding_dim]
        
    Returns:
        Dict с метриками качества
    """
    batch_size = input_embeddings.shape[0]
    
    # Cosine similarities для каждой пары
    similarities = []
    for i in range(batch_size):
        similarity = torch.nn.functional.cosine_similarity(
            input_embeddings[i], output_embeddings[i], dim=0
        ).item()
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    # L2 distances
    l2_distances = []
    for i in range(batch_size):
        l2_dist = torch.norm(input_embeddings[i] - output_embeddings[i], p=2).item()
        l2_distances.append(l2_dist)
    
    l2_distances = np.array(l2_distances)
    
    # Статистики
    return {
        "mean_cosine_similarity": float(np.mean(similarities)),
        "min_cosine_similarity": float(np.min(similarities)),
        "max_cosine_similarity": float(np.max(similarities)),
        "std_cosine_similarity": float(np.std(similarities)),
        "mean_l2_distance": float(np.mean(l2_distances)),
        "median_cosine_similarity": float(np.median(similarities)),
        "batch_size": batch_size
    }


def evaluate_semantic_preservation(input_embeddings: torch.Tensor,
                                  output_embeddings: torch.Tensor,
                                  threshold: float = 0.90) -> Dict[str, Any]:
    """
    Оценить качество семантического сохранения
    
    Args:
        input_embeddings: Входные эмбединги
        output_embeddings: Выходные эмбединги  
        threshold: Порог приемлемой схожести
        
    Returns:
        Dict с результатами оценки
    """
    quality_metrics = calculate_processing_quality(input_embeddings, output_embeddings)
    
    mean_similarity = quality_metrics["mean_cosine_similarity"]
    preservation_rate = float(np.mean([
        torch.nn.functional.cosine_similarity(
            input_embeddings[i], output_embeddings[i], dim=0
        ).item() >= threshold
        for i in range(input_embeddings.shape[0])
    ]))
    
    return {
        "semantic_preservation_score": mean_similarity,
        "preservation_rate": preservation_rate,
        "threshold": threshold,
        "quality_passed": mean_similarity >= threshold,
        "batch_preservation_rate": preservation_rate,
        "detailed_metrics": quality_metrics
    }


class QualityMonitor:
    """Монитор качества для длительного отслеживания"""
    
    def __init__(self, target_similarity: float = 0.90):
        self.target_similarity = target_similarity
        self.metrics = ProcessingMetrics(target_similarity=target_similarity)
        self.quality_history = []
        
    def monitor_batch(self, input_batch: torch.Tensor, output_batch: torch.Tensor, 
                     processing_time: float):
        """Мониторить обработку батча"""
        
        # Оценка качества
        quality = calculate_processing_quality(input_batch, output_batch)
        self.quality_history.append(quality)
        
        # Обновление основных метрик
        self.metrics.update(
            similarity=quality["mean_cosine_similarity"],
            processing_time=processing_time,
            batch_size=input_batch.shape[0]
        )
    
    def get_trend_analysis(self, window_size: int = 10) -> Dict[str, Any]:
        """Анализ трендов качества"""
        if len(self.quality_history) < window_size:
            return {"trend": "insufficient_data"}
        
        recent_scores = [q["mean_cosine_similarity"] for q in self.quality_history[-window_size:]]
        earlier_scores = [q["mean_cosine_similarity"] for q in self.quality_history[-2*window_size:-window_size]]
        
        if not earlier_scores:
            return {"trend": "insufficient_data"}
        
        recent_mean = np.mean(recent_scores)
        earlier_mean = np.mean(earlier_scores)
        
        trend_direction = "improving" if recent_mean > earlier_mean else "declining"
        trend_magnitude = abs(recent_mean - earlier_mean)
        
        return {
            "trend": trend_direction,
            "trend_magnitude": trend_magnitude,
            "recent_mean": recent_mean,
            "earlier_mean": earlier_mean,
            "stability": 1.0 - np.std(recent_scores)
        } 