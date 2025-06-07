"""
Advanced Loss Functions для Stage 2.3
Система продвинутых функций потерь для улучшения Q→A similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class LossFunctionConfig:
    """Конфигурация для продвинутых loss functions"""
    # Базовые loss functions
    use_cosine_loss: bool = True
    use_mse_loss: bool = True
    cosine_weight: float = 0.7
    mse_weight: float = 0.3
    
    # Curriculum learning
    use_curriculum_loss: bool = True
    curriculum_warmup_epochs: int = 5
    easy_weight_start: float = 0.8      # Начальный вес легких примеров
    easy_weight_end: float = 0.3        # Конечный вес легких примеров
    
    # Triplet loss
    use_triplet_loss: bool = True
    triplet_margin: float = 0.2
    triplet_weight: float = 0.1
    
    # Contrastive learning
    use_contrastive_loss: bool = True
    contrastive_temperature: float = 0.5
    contrastive_weight: float = 0.15
    
    # Multi-objective optimization
    diversity_weight: float = 0.05      # Штраф за низкое разнообразие
    semantic_alignment_weight: float = 0.85  # Главная цель - semantic alignment


# Alias для обратной совместимости
AdvancedLossConfig = LossFunctionConfig


class AdvancedLossFunction(nn.Module):
    """
    Продвинутая система loss functions для Stage 2.3
    
    Включает:
    - Curriculum learning loss (easy→hard progression)
    - Triplet loss для enhanced semantic alignment
    - Contrastive learning approaches
    - Multi-objective optimization (similarity + diversity)
    """
    
    def __init__(self, config: Optional[LossFunctionConfig] = None):
        super().__init__()
        self.config = config or LossFunctionConfig()
        
        # Базовые loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = CosineSimilarityLoss()
        
        # Продвинутые loss functions
        self.triplet_loss = TripletMarginLoss(margin=self.config.triplet_margin)
        self.contrastive_loss = ContrastiveLoss(temperature=self.config.contrastive_temperature)
        
        # Curriculum learning state
        self.current_epoch = 0
        self.total_epochs = 100  # Will be updated during training
        
        print(f"🚀 AdvancedLossFunction initialized")
        print(f"   Curriculum learning: {self.config.use_curriculum_loss}")
        print(f"   Triplet loss: {self.config.use_triplet_loss}")
        print(f"   Contrastive loss: {self.config.use_contrastive_loss}")
    
    def forward(self, 
                input_embeddings: torch.Tensor,
                target_embeddings: torch.Tensor,
                output_embeddings: torch.Tensor,
                difficulty_scores: Optional[torch.Tensor] = None,
                negative_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Вычисление продвинутой loss function
        
        Args:
            input_embeddings: Входные эмбединги (batch_size, embedding_dim)
            target_embeddings: Целевые эмбединги (batch_size, embedding_dim)
            output_embeddings: Выходные эмбединги от модели (batch_size, embedding_dim)
            difficulty_scores: Scores сложности для curriculum learning (batch_size,)
            negative_embeddings: Негативные примеры для contrastive learning (batch_size, embedding_dim)
            
        Returns:
            Dict с компонентами loss и общим loss
        """
        # Приведение всех основных тензоров к одному типу (float32) для совместимости
        input_embeddings = input_embeddings.float()
        target_embeddings = target_embeddings.float()
        output_embeddings = output_embeddings.float()
        
        if negative_embeddings is not None:
            negative_embeddings = negative_embeddings.float()
        if difficulty_scores is not None:
            difficulty_scores = difficulty_scores.float()
        
        losses = {}
        
        # 1. Базовые loss functions
        if self.config.use_cosine_loss:
            cosine_loss = self.cosine_loss(output_embeddings, target_embeddings)
            losses['cosine_loss'] = cosine_loss
        
        if self.config.use_mse_loss:
            mse_loss = self.mse_loss(output_embeddings, target_embeddings)
            losses['mse_loss'] = mse_loss
        
        # 2. Curriculum learning loss
        if self.config.use_curriculum_loss and difficulty_scores is not None:
            curriculum_loss = self._compute_curriculum_loss(
                output_embeddings, target_embeddings, difficulty_scores
            )
            losses['curriculum_loss'] = curriculum_loss
        
        # 3. Triplet loss для semantic alignment
        if self.config.use_triplet_loss and negative_embeddings is not None:
            triplet_loss = self._compute_triplet_loss(
                input_embeddings, output_embeddings, negative_embeddings
            )
            losses['triplet_loss'] = triplet_loss
        
        # 4. Contrastive learning loss
        if self.config.use_contrastive_loss and negative_embeddings is not None:
            contrastive_loss = self._compute_contrastive_loss(
                output_embeddings, target_embeddings, negative_embeddings
            )
            losses['contrastive_loss'] = contrastive_loss
        
        # 5. Diversity penalty
        diversity_loss = self._compute_diversity_loss(output_embeddings)
        losses['diversity_loss'] = diversity_loss
        
        # 6. Комбинирование всех loss components
        total_loss = self._combine_losses(losses)
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_curriculum_loss(self, 
                               output_embeddings: torch.Tensor,
                               target_embeddings: torch.Tensor, 
                               difficulty_scores: torch.Tensor) -> torch.Tensor:
        """Curriculum learning loss с прогрессивным увеличением сложности"""
        
        # Вычисление текущего curriculum weight
        progress = self.current_epoch / max(self.total_epochs, 1)
        progress = min(1.0, progress)
        
        # Linear interpolation от easy_weight_start к easy_weight_end
        easy_weight = (self.config.easy_weight_start * (1 - progress) + 
                      self.config.easy_weight_end * progress)
        
        # Инвертируем difficulty scores: легкие примеры = высокий вес
        easy_weights = 1.0 - difficulty_scores  # 0-1, где 1 = легкий
        hard_weights = difficulty_scores        # 0-1, где 1 = сложный
        
        # Adaptive weighting based on curriculum phase
        sample_weights = easy_weights * easy_weight + hard_weights * (1 - easy_weight)
        
        # Weighted cosine similarity loss
        cosine_similarities = F.cosine_similarity(output_embeddings, target_embeddings, dim=1)
        cosine_losses = 1 - cosine_similarities
        
        # Apply curriculum weights
        weighted_losses = cosine_losses * sample_weights
        
        return weighted_losses.mean()
    
    def _compute_triplet_loss(self,
                            anchor_embeddings: torch.Tensor,
                            positive_embeddings: torch.Tensor,
                            negative_embeddings: torch.Tensor) -> torch.Tensor:
        """Triplet loss для enhanced semantic alignment"""
        
        # Приведение всех тензоров к одному типу (float32)
        anchor_embeddings = anchor_embeddings.float()
        positive_embeddings = positive_embeddings.float()
        negative_embeddings = negative_embeddings.float()
        
        batch_size = anchor_embeddings.size(0)
        embedding_dim = anchor_embeddings.size(1)
        
        # Обработка множественных негативных примеров
        if negative_embeddings.size(0) != batch_size:
            # Предполагаем, что negative_embeddings содержит num_negatives на каждый anchor
            num_negatives = negative_embeddings.size(0) // batch_size
            
            # Reshape negatives: (batch_size * num_negatives, embedding_dim) -> (batch_size, num_negatives, embedding_dim)
            negative_embeddings = negative_embeddings.view(batch_size, num_negatives, embedding_dim)
            
            # Expand anchors and positives для каждого негативного примера
            anchor_expanded = anchor_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)  # (batch_size, num_negatives, embedding_dim)
            positive_expanded = positive_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
            
            # Distances для всех комбинаций
            pos_distance = F.pairwise_distance(
                anchor_expanded.reshape(-1, embedding_dim), 
                positive_expanded.reshape(-1, embedding_dim)
            )  # (batch_size * num_negatives,)
            
            neg_distance = F.pairwise_distance(
                anchor_expanded.reshape(-1, embedding_dim), 
                negative_embeddings.reshape(-1, embedding_dim)
            )  # (batch_size * num_negatives,)
            
        else:
            # Стандартный случай: один негативный пример на anchor
            pos_distance = F.pairwise_distance(anchor_embeddings, positive_embeddings)
            neg_distance = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        
        # Triplet margin loss
        triplet_loss = F.relu(pos_distance - neg_distance + self.config.triplet_margin)
        
        return triplet_loss.mean()
    
    def _compute_contrastive_loss(self,
                                output_embeddings: torch.Tensor,
                                target_embeddings: torch.Tensor,
                                negative_embeddings: torch.Tensor) -> torch.Tensor:
        """Contrastive learning loss"""
        
        batch_size = output_embeddings.size(0)
        
        # Приведение всех тензоров к одному типу (float32)
        output_embeddings = output_embeddings.float()
        target_embeddings = target_embeddings.float()
        negative_embeddings = negative_embeddings.float()
        
        # Positive pairs (output vs target)
        pos_similarities = F.cosine_similarity(output_embeddings, target_embeddings, dim=1)
        pos_similarities = pos_similarities / self.config.contrastive_temperature
        
        # Negative pairs (output vs negatives)
        # Обработка множественных негативных примеров
        if negative_embeddings.size(0) == batch_size:
            # Стандартный случай: один негативный пример на anchor
            neg_similarities = F.cosine_similarity(
                output_embeddings, negative_embeddings, dim=1
            ).unsqueeze(1)
        else:
            # Множественные негативные примеры
            neg_similarities = torch.mm(output_embeddings, negative_embeddings.T)
        
        neg_similarities = neg_similarities / self.config.contrastive_temperature
        
        # Contrastive loss (InfoNCE)
        logits = torch.cat([pos_similarities.unsqueeze(1), neg_similarities], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=output_embeddings.device)
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def _compute_diversity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Штраф за низкое разнообразие в batch"""
        
        # Pairwise cosine similarities within batch
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)
        
        # Убираем диагональ (self-similarities)
        mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        
        # Штраф за высокую similarity (низкое разнообразие)
        diversity_penalty = similarity_matrix.abs().mean()
        
        return diversity_penalty
    
    def _combine_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Комбинирование всех компонентов loss"""
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        # Базовые losses
        if 'cosine_loss' in losses:
            total_loss += self.config.cosine_weight * losses['cosine_loss']
        
        if 'mse_loss' in losses:
            total_loss += self.config.mse_weight * losses['mse_loss']
        
        # Продвинутые losses
        if 'curriculum_loss' in losses:
            total_loss += self.config.semantic_alignment_weight * losses['curriculum_loss']
        
        if 'triplet_loss' in losses:
            total_loss += self.config.triplet_weight * losses['triplet_loss']
        
        if 'contrastive_loss' in losses:
            total_loss += self.config.contrastive_weight * losses['contrastive_loss']
        
        if 'diversity_loss' in losses:
            total_loss += self.config.diversity_weight * losses['diversity_loss']
        
        return total_loss
    
    def update_epoch(self, current_epoch: int, total_epochs: int):
        """Обновление информации об эпохах для curriculum learning"""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
    
    def get_curriculum_progress(self) -> float:
        """Получение прогресса curriculum learning (0-1)"""
        return min(1.0, self.current_epoch / max(self.total_epochs, 1))
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Получение текущих весов loss components"""
        progress = self.get_curriculum_progress()
        easy_weight = (self.config.easy_weight_start * (1 - progress) + 
                      self.config.easy_weight_end * progress)
        
        return {
            'cosine_weight': self.config.cosine_weight,
            'mse_weight': self.config.mse_weight,
            'triplet_weight': self.config.triplet_weight,
            'contrastive_weight': self.config.contrastive_weight,
            'diversity_weight': self.config.diversity_weight,
            'current_easy_weight': easy_weight,
            'curriculum_progress': progress
        }


class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss (1 - cosine_similarity)"""
    
    def forward(self, input_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(input_embeddings, target_embeddings, dim=1)
        return (1 - cosine_sim).mean()


class TripletMarginLoss(nn.Module):
    """Triplet margin loss для semantic alignment"""
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        triplet_loss = F.relu(pos_distance - neg_distance + self.margin)
        return triplet_loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive learning loss (InfoNCE)"""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                query: torch.Tensor,
                positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        
        # Positive similarities
        pos_sim = F.cosine_similarity(query, positive, dim=1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.mm(query, negatives.T) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits, labels)


class NegativeSampler:
    """Система для генерации негативных примеров"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def sample_random_negatives(self, 
                              positive_embeddings: torch.Tensor,
                              num_negatives: int = 5) -> torch.Tensor:
        """Генерация случайных негативных примеров"""
        batch_size = positive_embeddings.size(0)
        device = positive_embeddings.device
        
        # Random negatives
        negatives = torch.randn(
            batch_size, num_negatives, self.embedding_dim, 
            device=device
        )
        
        # Normalize
        negatives = F.normalize(negatives, p=2, dim=-1)
        
        return negatives.reshape(-1, self.embedding_dim)
    
    def sample_hard_negatives(self,
                            anchor_embeddings: torch.Tensor,
                            all_embeddings: torch.Tensor,
                            num_negatives: int = 5) -> torch.Tensor:
        """Генерация сложных негативных примеров (близких к anchor)"""
        
        batch_size = anchor_embeddings.size(0)
        
        # Compute similarities with all embeddings
        similarities = torch.mm(anchor_embeddings, all_embeddings.T)
        
        # Select hard negatives (highest similarities excluding self)
        _, indices = similarities.topk(num_negatives + 1, dim=1)
        
        # Exclude self (first index is usually self if anchor is in all_embeddings)
        negative_indices = indices[:, 1:num_negatives+1]
        
        # Gather hard negatives
        hard_negatives = all_embeddings[negative_indices.flatten()]
        
        return hard_negatives


# ================================
# HELPER FUNCTIONS
# ================================

def create_advanced_loss_function(
    use_curriculum: bool = True,
    use_triplet: bool = True,
    use_contrastive: bool = True,
    curriculum_warmup_epochs: int = 5
) -> AdvancedLossFunction:
    """
    Удобная функция для создания продвинутой loss function
    
    Args:
        use_curriculum: Использовать curriculum learning
        use_triplet: Использовать triplet loss
        use_contrastive: Использовать contrastive loss
        curriculum_warmup_epochs: Количество эпох для curriculum warmup
        
    Returns:
        Настроенная AdvancedLossFunction
    """
    config = LossFunctionConfig(
        use_curriculum_loss=use_curriculum,
        use_triplet_loss=use_triplet,
        use_contrastive_loss=use_contrastive,
        curriculum_warmup_epochs=curriculum_warmup_epochs
    )
    
    return AdvancedLossFunction(config)


def compute_loss_with_negatives(
    loss_function: AdvancedLossFunction,
    input_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    output_embeddings: torch.Tensor,
    difficulty_scores: Optional[torch.Tensor] = None,
    negative_sampler: Optional[NegativeSampler] = None
) -> Dict[str, torch.Tensor]:
    """
    Вычисление loss с автоматической генерацией негативных примеров
    
    Returns:
        Dict с компонентами loss
    """
    negative_embeddings = None
    
    if negative_sampler is not None:
        negative_embeddings = negative_sampler.sample_random_negatives(
            target_embeddings, num_negatives=5
        )
    
    return loss_function(
        input_embeddings=input_embeddings,
        target_embeddings=target_embeddings,
        output_embeddings=output_embeddings,
        difficulty_scores=difficulty_scores,
        negative_embeddings=negative_embeddings
    )


if __name__ == "__main__":
    # Демонстрация системы
    print("🚀 Testing Advanced Loss Functions...")
    
    # Создание loss function
    loss_fn = create_advanced_loss_function()
    
    # Тестовые данные
    batch_size, embedding_dim = 8, 768
    input_emb = torch.randn(batch_size, embedding_dim)
    target_emb = torch.randn(batch_size, embedding_dim)
    output_emb = torch.randn(batch_size, embedding_dim)
    difficulty = torch.rand(batch_size)  # 0-1 difficulty scores
    
    # Negative sampler
    neg_sampler = NegativeSampler(embedding_dim)
    
    # Вычисление loss
    losses = compute_loss_with_negatives(
        loss_fn, input_emb, target_emb, output_emb, difficulty, neg_sampler
    )
    
    print(f"📊 Loss Components:")
    for name, loss in losses.items():
        print(f"   {name}: {loss.item():.4f}")
    
    print(f"\n🎯 Current loss weights:")
    weights = loss_fn.get_loss_weights()
    for name, weight in weights.items():
        print(f"   {name}: {weight:.3f}")
    
    print("\n✅ Advanced Loss Functions system ready!") 