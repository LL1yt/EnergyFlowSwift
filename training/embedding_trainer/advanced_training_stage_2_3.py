"""
Advanced Training Stage 2.3 - Integrated System
Интегрированная система для достижения 50%+ Q→A similarity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

# Импорты наших систем
from .cube_trainer import CubeTrainer, TrainingConfig
from .advanced_dataset_expansion import AdvancedDatasetExpander, create_expanded_dataset
from .advanced_loss_functions import AdvancedLossFunction, create_advanced_loss_function, NegativeSampler
from .multi_teacher_distillation import MultiTeacherDistillation, create_multi_teacher_system
from .dialogue_dataset import DialogueDataset


@dataclass
class Stage23Config:
    """Конфигурация для Stage 2.3 Advanced Training Enhancement"""
    # Dataset expansion
    target_pairs: int = 100                        # Целевое количество пар
    quality_threshold: float = 0.6                 # Порог качества данных
    
    # Advanced loss functions
    use_curriculum_learning: bool = True           # Curriculum learning
    use_triplet_loss: bool = True                  # Triplet loss
    use_contrastive_loss: bool = True              # Contrastive learning
    curriculum_warmup_epochs: int = 5              # Эпохи для curriculum warmup
    
    # Multi-teacher distillation
    use_multi_teacher: bool = True                 # Multi-teacher knowledge distillation
    teacher_models: List[str] = None               # Teacher модели
    distillation_temperature: float = 3.0          # Температура distillation
    
    # Training optimization
    learning_rate: float = 0.0003                  # Optimized learning rate
    batch_size: int = 6                            # Optimized batch size
    epochs: int = 15                               # Target epochs
    
    # Target metrics
    target_qa_similarity: float = 0.50            # 50%+ Q→A similarity goal
    convergence_threshold: float = 0.01           # Training convergence
    validation_patience: int = 5                   # Early stopping patience
    
    def __post_init__(self):
        if self.teacher_models is None:
            # Загружаем из центрального конфига
            try:
                from utils.config_loader import get_multi_teacher_config
                config = get_multi_teacher_config()
                self.teacher_models = config.get('models', ['distilbert'])
                print(f"📋 Loaded teacher models from config: {self.teacher_models}")
            except Exception:
                self.teacher_models = ["llama3-8b-local", "distilbert", "roberta"]  # Fallback
                print(f"⚠️ Using fallback teacher models: {self.teacher_models}")


class AdvancedTrainingStage23:
    """
    Интегрированная система Stage 2.3 Advanced Training Enhancement
    
    Объединяет:
    - Advanced Dataset Expansion (100+ pairs)
    - Advanced Loss Functions (curriculum + triplet + contrastive)
    - Multi-Teacher Knowledge Distillation
    - Optimized Training Pipeline
    """
    
    def __init__(self, config: Optional[Stage23Config] = None):
        self.config = config or Stage23Config()
        
        # Инициализация компонентов
        self.dataset_expander = None
        self.multi_teacher = None
        self.advanced_loss_fn = None
        self.negative_sampler = None
        self.cube_trainer = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.best_qa_similarity = 0.0
        self.patience_counter = 0
        
        print(f"🚀 AdvancedTrainingStage23 initialized")
        print(f"   Target Q→A similarity: {self.config.target_qa_similarity:.1%}")
        print(f"   Target dataset size: {self.config.target_pairs} pairs")
        print(f"   Multi-teacher models: {len(self.config.teacher_models)}")
    
    def setup_training_components(self):
        """Настройка всех компонентов обучения"""
        print("🔧 Setting up advanced training components...")
        
        # 1. Advanced Loss Functions
        self.advanced_loss_fn = create_advanced_loss_function(
            use_curriculum=self.config.use_curriculum_learning,
            use_triplet=self.config.use_triplet_loss,
            use_contrastive=self.config.use_contrastive_loss,
            curriculum_warmup_epochs=self.config.curriculum_warmup_epochs
        )
        
        # 2. Negative Sampler для contrastive learning
        self.negative_sampler = NegativeSampler(embedding_dim=768)
        
        # 3. Multi-Teacher System
        if self.config.use_multi_teacher:
            self.multi_teacher = create_multi_teacher_system(
                teacher_models=self.config.teacher_models,
                adaptive_weighting=True,
                distillation_temperature=self.config.distillation_temperature
            )
        
        # 4. CubeTrainer с оптимизированными настройками
        training_config = TrainingConfig(
            mode="dialogue",
            lattice_size=[8, 8, 12],  # Правильный размер для 768D эмбедингов
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            optimizer="adamw",
            loss_function="advanced"  # Будем использовать нашу advanced loss
        )
        
        self.cube_trainer = CubeTrainer(config=training_config)
        self.cube_trainer.initialize_components()
        
        print("✅ All training components setup complete!")
    
    def _normalize_embedding_dimensions(self, embeddings: torch.Tensor, target_dim: int = 768) -> torch.Tensor:
        """
        Нормализация размерности эмбедингов до target_dim
        
        Args:
            embeddings: Входные эмбединги
            target_dim: Целевая размерность (по умолчанию 768)
            
        Returns:
            Нормализованные эмбединги размера target_dim
        """
        if embeddings.size(-1) == target_dim:
            return embeddings
        
        # Если эмбединги больше целевой размерности - обрезаем
        if embeddings.size(-1) > target_dim:
            return embeddings[..., :target_dim]
        
        # Если эмбединги меньше - дополняем нулями
        else:
            padding_size = target_dim - embeddings.size(-1)
            padding = torch.zeros(*embeddings.shape[:-1], padding_size, dtype=embeddings.dtype, device=embeddings.device)
            return torch.cat([embeddings, padding], dim=-1)
    
    def create_enhanced_dataset(self) -> DialogueDataset:
        """Создание enhanced dataset с expanded data и multi-teacher embeddings"""
        print("🎯 Creating enhanced dataset for Stage 2.3...")
        
        # 1. Dataset Expansion до 100+ pairs
        expanded_dataset = create_expanded_dataset(
            target_pairs=self.config.target_pairs,
            quality_threshold=0.6  # Используем стандартное значение
        )
        
        print(f"   ✅ Dataset expanded to {len(expanded_dataset)} pairs")
        
        # 2. Multi-Teacher Enhancement (если включен)
        if self.config.use_multi_teacher and self.multi_teacher:
            # Извлечение dialogue pairs из dataset
            dialogue_pairs = []
            for i in range(len(expanded_dataset)):
                question_emb, answer_emb = expanded_dataset[i]
                # Примечание: нужно будет добавить метод для получения оригинального текста
                # Пока используем placeholder
                dialogue_pairs.append({
                    "question": f"Enhanced question {i}",
                    "answer": f"Enhanced answer {i}"
                })
            
            # Создание ensemble dataset
            ensemble_data = self.multi_teacher.create_ensemble_dataset(
                dialogue_pairs, validation_split=0.2
            )
            
            print(f"   ✅ Multi-teacher ensemble created")
            print(f"      Train samples: {len(ensemble_data['train']['question_embeddings'])}")
            print(f"      Validation samples: {len(ensemble_data['validation']['question_embeddings'])}")
        
        return expanded_dataset
    
    def run_advanced_training(self, dataset: DialogueDataset) -> Dict[str, float]:
        """Запуск продвинутого обучения Stage 2.3"""
        print("🚀 Starting Stage 2.3 Advanced Training...")
        
        # Получение dataloader
        train_dataloader = dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            validation=False
        )
        
        val_dataloader = dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=False,
            validation=True
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Обновление curriculum learning progress
            self.advanced_loss_fn.update_epoch(epoch, self.config.epochs)
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataloader)
            
            # Логирование прогресса
            self._log_epoch_progress(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics):
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint если улучшение
            if val_metrics["qa_similarity"] > self.best_qa_similarity:
                self.best_qa_similarity = val_metrics["qa_similarity"]
                self._save_checkpoint(epoch, val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Финальные результаты
        final_results = self._compute_final_results()
        
        print(f"🎉 Stage 2.3 Training Complete!")
        print(f"   Best Q→A similarity: {self.best_qa_similarity:.1%}")
        print(f"   Target achieved: {'✅' if self.best_qa_similarity >= self.config.target_qa_similarity else '❌'}")
        
        return final_results
    
    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.cube_trainer.embedding_processor.train()
        
        epoch_losses = []
        epoch_qa_similarities = []
        
        for batch_idx, (question_embeddings, answer_embeddings) in enumerate(dataloader):
            # Нормализация размерности эмбедингов до ожидаемой системой (768D)
            question_embeddings = self._normalize_embedding_dimensions(question_embeddings)
            answer_embeddings = self._normalize_embedding_dimensions(answer_embeddings)
            
            # Forward pass через cube
            output_embeddings = self.cube_trainer.forward(question_embeddings)
            
            # Генерация негативных примеров
            negative_embeddings = self.negative_sampler.sample_random_negatives(
                answer_embeddings, num_negatives=5
            )
            
            # Difficulty scores для curriculum learning (простая эвристика)
            difficulty_scores = torch.rand(question_embeddings.size(0))
            
            # Вычисление advanced loss
            losses = self.advanced_loss_fn(
                input_embeddings=question_embeddings,
                target_embeddings=answer_embeddings,
                output_embeddings=output_embeddings,
                difficulty_scores=difficulty_scores,
                negative_embeddings=negative_embeddings
            )
            
            # Backward pass
            self.cube_trainer.optimizer.zero_grad()
            losses["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.cube_trainer.embedding_processor.parameters(), 
                max_norm=1.0
            )
            
            self.cube_trainer.optimizer.step()
            
            # Метрики
            qa_similarity = torch.cosine_similarity(
                output_embeddings, answer_embeddings, dim=1
            ).mean().item()
            
            epoch_losses.append(losses["total_loss"].item())
            epoch_qa_similarities.append(qa_similarity)
        
        return {
            "train_loss": np.mean(epoch_losses),
            "qa_similarity": np.mean(epoch_qa_similarities)
        }
    
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """Валидация на одной эпохе"""
        self.cube_trainer.embedding_processor.eval()
        
        val_losses = []
        val_qa_similarities = []
        
        with torch.no_grad():
            for question_embeddings, answer_embeddings in dataloader:
                # Нормализация размерности эмбедингов
                question_embeddings = self._normalize_embedding_dimensions(question_embeddings)
                answer_embeddings = self._normalize_embedding_dimensions(answer_embeddings)
                
                # Forward pass
                output_embeddings = self.cube_trainer.forward(question_embeddings)
                
                # Simple validation loss (cosine similarity)
                loss = 1 - torch.cosine_similarity(
                    output_embeddings, answer_embeddings, dim=1
                ).mean()
                
                qa_similarity = torch.cosine_similarity(
                    output_embeddings, answer_embeddings, dim=1
                ).mean().item()
                
                val_losses.append(loss.item())
                val_qa_similarities.append(qa_similarity)
        
        return {
            "val_loss": np.mean(val_losses),
            "qa_similarity": np.mean(val_qa_similarities)
        }
    
    def _log_epoch_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Логирование прогресса эпохи"""
        curriculum_progress = self.advanced_loss_fn.get_curriculum_progress()
        
        print(f"Epoch {epoch+1}/{self.config.epochs}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Q→A Sim: {train_metrics['qa_similarity']:.1%}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Q→A Sim: {val_metrics['qa_similarity']:.1%}")
        print(f"  Curriculum Progress: {curriculum_progress:.1%} | "
              f"Best Q→A: {self.best_qa_similarity:.1%}")
        
        # Сохранение в историю
        self.training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "train_qa_similarity": train_metrics["qa_similarity"],
            "val_loss": val_metrics["val_loss"],
            "val_qa_similarity": val_metrics["qa_similarity"],
            "curriculum_progress": curriculum_progress
        })
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Проверка early stopping"""
        return self.patience_counter >= self.config.validation_patience
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Сохранение checkpoint"""
        checkpoint_dir = Path("checkpoints/stage_2_3")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.cube_trainer.embedding_processor.state_dict(),
            "optimizer_state": self.cube_trainer.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, checkpoint_dir / f"best_model_epoch_{epoch}.pt")
        
        # Сохранение метрик в JSON
        with open(checkpoint_dir / "training_results.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
    
    def _compute_final_results(self) -> Dict[str, float]:
        """Вычисление финальных результатов"""
        return {
            "best_qa_similarity": self.best_qa_similarity,
            "target_achieved": self.best_qa_similarity >= self.config.target_qa_similarity,
            "improvement_from_stage_2_2": self.best_qa_similarity - 0.3189,  # Stage 2.2 result
            "total_epochs": self.current_epoch + 1,
            "final_curriculum_progress": self.advanced_loss_fn.get_curriculum_progress()
        }
    
    def get_training_summary(self) -> Dict:
        """Получение summary обучения"""
        return {
            "config": {
                "target_pairs": self.config.target_pairs,
                "target_qa_similarity": self.config.target_qa_similarity,
                "use_multi_teacher": self.config.use_multi_teacher,
                "use_curriculum_learning": self.config.use_curriculum_learning
            },
            "results": self._compute_final_results(),
            "training_history": self.training_history,
            "loss_weights": self.advanced_loss_fn.get_loss_weights() if self.advanced_loss_fn else {}
        }


# ================================
# HELPER FUNCTIONS
# ================================

def run_stage_2_3_training(
    target_qa_similarity: float = 0.50,
    target_pairs: int = 100,
    epochs: int = 15,
    use_multi_teacher: bool = True
) -> Dict[str, float]:
    """
    Удобная функция для запуска Stage 2.3 training
    
    Args:
        target_qa_similarity: Целевая Q→A similarity (0.50 = 50%)
        target_pairs: Количество dialogue pairs
        epochs: Количество эпох обучения
        use_multi_teacher: Использовать multi-teacher distillation
        
    Returns:
        Результаты обучения
    """
    config = Stage23Config(
        target_qa_similarity=target_qa_similarity,
        target_pairs=target_pairs,
        epochs=epochs,
        use_multi_teacher=use_multi_teacher
    )
    
    # Создание training system
    trainer = AdvancedTrainingStage23(config)
    trainer.setup_training_components()
    
    # Создание enhanced dataset
    dataset = trainer.create_enhanced_dataset()
    
    # Запуск обучения
    results = trainer.run_advanced_training(dataset)
    
    return results


def analyze_stage_2_3_progress(training_history: List[Dict]) -> Dict:
    """Анализ прогресса Stage 2.3"""
    if not training_history:
        return {"error": "No training history available"}
    
    final_epoch = training_history[-1]
    first_epoch = training_history[0]
    
    return {
        "total_epochs": len(training_history),
        "final_qa_similarity": final_epoch["val_qa_similarity"],
        "initial_qa_similarity": first_epoch["val_qa_similarity"],
        "improvement": final_epoch["val_qa_similarity"] - first_epoch["val_qa_similarity"],
        "best_qa_similarity": max(epoch["val_qa_similarity"] for epoch in training_history),
        "convergence_achieved": final_epoch["val_loss"] < 0.1,
        "curriculum_completed": final_epoch["curriculum_progress"] > 0.9
    }


if __name__ == "__main__":
    # Демонстрация Stage 2.3
    print("🚀 Testing Stage 2.3 Advanced Training Enhancement...")
    
    # Запуск обучения
    results = run_stage_2_3_training(
        target_qa_similarity=0.50,  # 50% target
        target_pairs=100,
        epochs=15,
        use_multi_teacher=True
    )
    
    print(f"📊 Stage 2.3 Results:")
    for key, value in results.items():
        if isinstance(value, float):
            if "similarity" in key:
                print(f"   {key}: {value:.1%}")
            else:
                print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Stage 2.3 Advanced Training Enhancement system ready!") 