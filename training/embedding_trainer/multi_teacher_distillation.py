"""
Multi-Teacher Knowledge Distillation для Stage 2.3
Система ensemble обучения от множественных Teacher LLM моделей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

# Импорты из нашей системы
from data.embedding_loader import EmbeddingLoader
from .dialogue_dataset import DialogueDataset, DialogueConfig
from utils.config_loader import get_multi_teacher_config, get_available_teacher_models


@dataclass
class MultiTeacherConfig:
    """Конфигурация для multi-teacher knowledge distillation"""
    # Teacher models
    teacher_models: List[str] = None               # Список teacher моделей
    teacher_weights: List[float] = None            # Веса для каждой teacher модели
    adaptive_weighting: bool = True               # Адаптивное взвешивание учителей
    confidence_threshold: float = 0.7             # Порог уверенности для teacher agreement
    
    # Knowledge distillation
    distillation_temperature: float = 3.0         # Температура для знания distillation
    student_loss_weight: float = 0.3              # Вес loss студента
    distillation_loss_weight: float = 0.7         # Вес distillation loss
    
    # Ensemble optimization
    use_teacher_agreement: bool = True            # Учитывать согласие между учителями
    agreement_weight: float = 0.2                 # Вес для teacher agreement
    diversity_penalty: float = 0.1                # Штраф за low diversity между учителями
    
    # Performance tracking
    teacher_performance_tracking: bool = True      # Отслеживание performance учителей
    performance_window: int = 100                  # Окно для вычисления performance
    
    def __post_init__(self):
        if self.teacher_models is None:
            # Загружаем из центрального конфига
            try:
                config = get_multi_teacher_config()
                self.teacher_models = config.get('models', ['distilbert'])
                print(f"[INFO] Loaded teacher models from config: {self.teacher_models}")
            except Exception:
                self.teacher_models = ["llama3-8b-local", "distilbert", "roberta"]  # Fallback
                print(f"[WARNING] Using fallback teacher models: {self.teacher_models}")
        
        if self.teacher_weights is None:
            # Равные веса по умолчанию
            self.teacher_weights = [1.0 / len(self.teacher_models)] * len(self.teacher_models)
        
        # Нормализация весов
        total_weight = sum(self.teacher_weights)
        self.teacher_weights = [w / total_weight for w in self.teacher_weights]


class MultiTeacherDistillation:
    """
    Система multi-teacher knowledge distillation
    
    Возможности:
    - Ensemble learning от множественных Teacher LLM
    - Adaptive teacher weighting на основе performance
    - Teacher agreement analysis
    - Knowledge distillation с temperature scaling
    """
    
    def __init__(self, config: Optional[MultiTeacherConfig] = None):
        self.config = config or MultiTeacherConfig()
        
        # Создание teacher embedding loaders
        self.teachers = {}
        for model_name in self.config.teacher_models:
            # Используем базовый EmbeddingLoader без model_name параметра
            self.teachers[model_name] = EmbeddingLoader()
        
        # Performance tracking
        self.teacher_performances = {name: [] for name in self.config.teacher_models}
        self.teacher_confidences = {name: [] for name in self.config.teacher_models}
        
        # Adaptive weights
        self.current_teacher_weights = self.config.teacher_weights.copy()
        
        print(f"[START] MultiTeacherDistillation initialized")
        print(f"   Teachers: {len(self.config.teacher_models)}")
        print(f"   Models: {self.config.teacher_models}")
        print(f"   Adaptive weighting: {self.config.adaptive_weighting}")
    
    def create_ensemble_dataset(self, 
                              dialogue_pairs: List[Dict],
                              validation_split: float = 0.2) -> Dict[str, torch.Tensor]:
        """
        Создание ensemble dataset с эмбедингами от всех teachers
        
        Args:
            dialogue_pairs: Список Q&A пар
            validation_split: Доля для валидации
            
        Returns:
            Dict с ensemble эмбедингами и метаданными
        """
        print("[TARGET] Creating multi-teacher ensemble dataset...")
        
        # Получение эмбедингов от каждого teacher
        teacher_embeddings = {}
        teacher_confidences = {}
        
        for teacher_name, teacher_loader in self.teachers.items():
            print(f"   Processing with {teacher_name}...")
            
            # Question embeddings
            questions = [pair["question"] for pair in dialogue_pairs]
            question_embeddings = teacher_loader.load_from_llm(
                texts=questions, 
                model_key=teacher_name,
                pooling_strategy="mean"
            )
            
            # Answer embeddings
            answers = [pair["answer"] for pair in dialogue_pairs]
            answer_embeddings = teacher_loader.load_from_llm(
                texts=answers, 
                model_key=teacher_name,
                pooling_strategy="mean"
            )
            
            # Вычисление teacher confidence (Q-A semantic similarity)
            q_tensors = torch.tensor(question_embeddings)
            a_tensors = torch.tensor(answer_embeddings)
            confidences = F.cosine_similarity(q_tensors, a_tensors, dim=1)
            
            teacher_embeddings[teacher_name] = {
                "questions": question_embeddings,
                "answers": answer_embeddings
            }
            teacher_confidences[teacher_name] = confidences.tolist()
        
        print(f"   Generated embeddings from {len(self.teachers)} teachers")
        
        # Создание ensemble эмбедингов
        ensemble_data = self._create_ensemble_embeddings(
            teacher_embeddings, teacher_confidences
        )
        
        # Train/validation split
        num_samples = len(dialogue_pairs)
        split_idx = int(num_samples * (1 - validation_split))
        
        train_data = {
            "question_embeddings": ensemble_data["question_embeddings"][:split_idx],
            "answer_embeddings": ensemble_data["answer_embeddings"][:split_idx],
            "teacher_weights": ensemble_data["teacher_weights"][:split_idx],
            "confidence_scores": ensemble_data["confidence_scores"][:split_idx]
        }
        
        val_data = {
            "question_embeddings": ensemble_data["question_embeddings"][split_idx:],
            "answer_embeddings": ensemble_data["answer_embeddings"][split_idx:],
            "teacher_weights": ensemble_data["teacher_weights"][split_idx:],
            "confidence_scores": ensemble_data["confidence_scores"][split_idx:]
        }
        
        print(f"[SUCCESS] Ensemble dataset created!")
        print(f"   Train samples: {len(train_data['question_embeddings'])}")
        print(f"   Validation samples: {len(val_data['question_embeddings'])}")
        print(f"   Average confidence: {np.mean(ensemble_data['confidence_scores']):.3f}")
        
        return {
            "train": train_data,
            "validation": val_data,
            "metadata": {
                "teacher_models": self.config.teacher_models,
                "ensemble_method": "weighted_average",
                "adaptive_weighting": self.config.adaptive_weighting
            }
        }
    
    def compute_distillation_loss(self,
                                student_embeddings: torch.Tensor,
                                teacher_ensemble_embeddings: torch.Tensor,
                                target_embeddings: torch.Tensor,
                                teacher_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Вычисление multi-teacher distillation loss
        
        Args:
            student_embeddings: Выходы студента (batch_size, embedding_dim)
            teacher_ensemble_embeddings: Ensemble эмбединги от teachers (batch_size, embedding_dim)
            target_embeddings: Целевые эмбединги (batch_size, embedding_dim)
            teacher_weights: Веса teachers для каждого примера (batch_size, num_teachers)
            
        Returns:
            Dict с компонентами loss
        """
        losses = {}
        
        # 1. Student loss (обычная loss студента с targets)
        student_loss = F.mse_loss(student_embeddings, target_embeddings)
        losses["student_loss"] = student_loss
        
        # 2. Distillation loss (студент учится у ensemble teachers)
        distillation_loss = self._compute_knowledge_distillation_loss(
            student_embeddings, teacher_ensemble_embeddings
        )
        losses["distillation_loss"] = distillation_loss
        
        # 3. Teacher agreement loss (поощряем согласие между teachers)
        if self.config.use_teacher_agreement and teacher_weights is not None:
            agreement_loss = self._compute_teacher_agreement_loss(teacher_weights)
            losses["agreement_loss"] = agreement_loss
        
        # 4. Комбинированная loss
        total_loss = (self.config.student_loss_weight * student_loss +
                     self.config.distillation_loss_weight * distillation_loss)
        
        if "agreement_loss" in losses:
            total_loss += self.config.agreement_weight * losses["agreement_loss"]
        
        losses["total_loss"] = total_loss
        
        return losses
    
    def get_teacher_statistics(self) -> Dict[str, Dict]:
        """Получение статистики teachers"""
        stats = {}
        
        for teacher_name in self.config.teacher_models:
            teacher_stats = {
                "current_weight": self.current_teacher_weights[
                    self.config.teacher_models.index(teacher_name)
                ],
                "performance_history_length": len(self.teacher_performances[teacher_name]),
                "average_confidence": np.mean(self.teacher_confidences[teacher_name]) 
                    if self.teacher_confidences[teacher_name] else 0.0
            }
            
            if self.teacher_performances[teacher_name]:
                teacher_stats.update({
                    "average_performance": np.mean(self.teacher_performances[teacher_name]),
                    "recent_performance": np.mean(
                        self.teacher_performances[teacher_name][-10:]
                    ) if len(self.teacher_performances[teacher_name]) >= 10 else 
                        np.mean(self.teacher_performances[teacher_name])
                })
            
            stats[teacher_name] = teacher_stats
        
        return stats
    
    def _create_ensemble_embeddings(self,
                                  teacher_embeddings: Dict[str, Dict],
                                  teacher_confidences: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Создание ensemble эмбедингов от всех teachers"""
        
        # Получение размеров
        num_samples = len(teacher_embeddings[list(teacher_embeddings.keys())[0]]["questions"])
        embedding_dim = len(teacher_embeddings[list(teacher_embeddings.keys())[0]]["questions"][0])
        
        # Инициализация результатов
        ensemble_question_embeddings = []
        ensemble_answer_embeddings = []
        ensemble_teacher_weights = []
        ensemble_confidence_scores = []
        
        for i in range(num_samples):
            # Получение эмбедингов от всех teachers для i-го примера
            sample_question_embeddings = []
            sample_answer_embeddings = []
            sample_confidences = []
            
            for teacher_name in self.config.teacher_models:
                q_emb = teacher_embeddings[teacher_name]["questions"][i]
                a_emb = teacher_embeddings[teacher_name]["answers"][i]
                conf = teacher_confidences[teacher_name][i]
                
                sample_question_embeddings.append(q_emb)
                sample_answer_embeddings.append(a_emb)
                sample_confidences.append(conf)
            
            # Конвертация в тензоры
            # Сначала конвертируем каждый эмбединг в тензор, если он ещё не тензор
            q_tensors = [torch.tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in sample_question_embeddings]
            a_tensors = [torch.tensor(emb) if not isinstance(emb, torch.Tensor) else emb for emb in sample_answer_embeddings]
            
            # Приводим все эмбединги к одинаковой размерности (берем минимальную)
            min_dim = min(t.shape[-1] for t in q_tensors)
            q_tensors = [t[..., :min_dim] for t in q_tensors]
            a_tensors = [t[..., :min_dim] for t in a_tensors]
            
            q_stack = torch.stack(q_tensors)  # (num_teachers, min_embedding_dim)
            a_stack = torch.stack(a_tensors)
            conf_tensor = torch.tensor(sample_confidences)
            
            # Adaptive weighting на основе confidence
            if self.config.adaptive_weighting:
                weights = F.softmax(conf_tensor, dim=0)  # Normalize confidences
            else:
                weights = torch.tensor(self.current_teacher_weights)
            
            # Weighted ensemble
            ensemble_q = torch.sum(q_stack * weights.unsqueeze(1), dim=0)
            ensemble_a = torch.sum(a_stack * weights.unsqueeze(1), dim=0)
            
            ensemble_question_embeddings.append(ensemble_q.tolist())
            ensemble_answer_embeddings.append(ensemble_a.tolist())
            ensemble_teacher_weights.append(weights.tolist())
            ensemble_confidence_scores.append(conf_tensor.mean().item())
        
        return {
            "question_embeddings": ensemble_question_embeddings,
            "answer_embeddings": ensemble_answer_embeddings,
            "teacher_weights": ensemble_teacher_weights,
            "confidence_scores": ensemble_confidence_scores
        }
    
    def _compute_knowledge_distillation_loss(self,
                                           student_embeddings: torch.Tensor,
                                           teacher_embeddings: torch.Tensor) -> torch.Tensor:
        """Вычисление knowledge distillation loss с temperature scaling"""
        
        # Temperature scaling
        temperature = self.config.distillation_temperature
        
        # Compute similarity distributions
        student_logits = F.normalize(student_embeddings, dim=1) / temperature
        teacher_logits = F.normalize(teacher_embeddings, dim=1) / temperature
        
        # KL divergence loss
        student_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        return kl_loss
    
    def _compute_teacher_agreement_loss(self, teacher_weights: torch.Tensor) -> torch.Tensor:
        """Вычисление loss для teacher agreement"""
        
        # Encourage agreement among teachers (low variance in weights)
        weight_variance = torch.var(teacher_weights, dim=1)
        agreement_loss = weight_variance.mean()
        
        return agreement_loss


# ================================
# HELPER FUNCTIONS
# ================================

def create_multi_teacher_system(
    teacher_models: Optional[List[str]] = None,
    adaptive_weighting: bool = True,
    distillation_temperature: float = 3.0
) -> MultiTeacherDistillation:
    """
    Удобная функция для создания multi-teacher system
    
    Args:
        teacher_models: Список teacher моделей
        adaptive_weighting: Использовать adaptive weighting
        distillation_temperature: Температура для knowledge distillation
        
    Returns:
        Настроенная MultiTeacherDistillation система
    """
    config = MultiTeacherConfig(
        teacher_models=teacher_models,
        adaptive_weighting=adaptive_weighting,
        distillation_temperature=distillation_temperature
    )
    
    return MultiTeacherDistillation(config)


if __name__ == "__main__":
    # Демонстрация системы
    print("[START] Testing Multi-Teacher Knowledge Distillation...")
    
    # Создание multi-teacher system
    multi_teacher = create_multi_teacher_system()
    
    # Тестовые dialogue pairs
    test_pairs = [
        {"question": "What is machine learning?", 
         "answer": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed."},
        {"question": "Explain deep learning", 
         "answer": "Deep learning uses neural networks with multiple layers to learn complex patterns in data."},
        {"question": "What is supervised learning?", 
         "answer": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs."}
    ]
    
    print(f"[DATA] Multi-Teacher Statistics:")
    teacher_stats = multi_teacher.get_teacher_statistics()
    for teacher, stats in teacher_stats.items():
        print(f"   {teacher}:")
        print(f"     Weight: {stats['current_weight']:.3f}")
        print(f"     Avg confidence: {stats['average_confidence']:.3f}")
    
    print("\n[OK] Multi-Teacher Knowledge Distillation system ready!") 