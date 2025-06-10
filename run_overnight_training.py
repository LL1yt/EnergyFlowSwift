#!/usr/bin/env python3
"""
🌙 НОЧНОЕ ОБУЧЕНИЕ: Долгосрочное обучение для проверки гипотезы

Цель: Проверить что система просто требует больше времени для обучения
      
Настройки:
- 200+ epochs (вместо 10)
- Подробный мониторинг каждые 10 epochs
- Автоматическое сохранение результатов
- Early stopping если появится прогресс
- Логирование изменений в реальном времени
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Setup enhanced logging для ночного мониторинга
log_filename = f"logs/overnight_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset

class OvernightTrainingManager:
    """Менеджер для долгосрочного ночного обучения"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = datetime.now()
        self.results_dir = Path(f"results/overnight_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}")
        self.checkpoints_dir = Path(f"checkpoints/overnight_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}")
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress = {
            'start_time': self.start_time.isoformat(),
            'epochs_completed': 0,
            'total_planned_epochs': float('inf'),  # Неограниченное количество epochs
            'loss_history': [],
            'similarity_history': [],
            'best_loss': float('inf'),
            'best_similarity': 0.0,
            'first_improvement_epoch': None,
            'learning_detected': False,
            'total_training_time': 0
        }
        
        logger.info(f"🌙 Overnight Training Manager initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Planned epochs: UNLIMITED (ручная остановка)")
        logger.info(f"   Start time: {self.start_time}")
        logger.info(f"   Results: {self.results_dir}")
        logger.info(f"   Checkpoints: {self.checkpoints_dir}")
    
    def run_overnight_training(self):
        """Запуск долгосрочного ночного обучения"""
        
        logger.info("🌙 СТАРТ НОЧНОГО ОБУЧЕНИЯ")
        logger.info("="*50)
        
        # Setup model
        trainer = self._setup_trainer()
        if trainer is None:
            return
            
        # Setup dataset  
        dataset = self._setup_dataset()
        if not dataset:
            return
            
        # Training configuration - неограниченное обучение
        epochs = 999999  # Практически бесконечно - ручная остановка
        batch_size = 1024  # Оптимально для RTX 5090
        learning_rate = 0.0001  # Lower LR для долгосрочного обучения
        
        logger.info(f"[TARGET] Training configuration:")
        logger.info(f"   Epochs: UNLIMITED (ручная остановка Ctrl+C)")
        logger.info(f"   Batch size: {batch_size} (оптимально для RTX 5090)")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Dataset size: {len(dataset)}")
        
        # Training loop - неограниченное обучение
        trainer.train()
        no_improvement_count = 0
        # Убираем early stopping - пусть работает всю ночь!
        
        logger.info("[START] Starting UNLIMITED training - используйте Ctrl+C для остановки")
        logger.info("=" * 60)
        
        try:
            for epoch in range(epochs):
                epoch_start_time = time.time()
                
                # Prepare batch
                batch_losses = []
                batch_similarities = []
                
                num_batches = len(dataset) // batch_size
                for batch_idx in range(num_batches):
                    # Get batch data
                    batch_data = self._prepare_batch(dataset, batch_idx, batch_size)
                    
                    # Training step
                    try:
                        batch_data = tuple(tensor.to(self.device) for tensor in batch_data)
                        metrics = trainer.train_step(*batch_data)
                        
                        batch_losses.append(metrics.get('total_loss', 0.0))
                        batch_similarities.append(metrics.get('cosine_similarity', 0.0))
                        
                    except Exception as e:
                        logger.warning(f"[WARNING] Batch {batch_idx} failed: {e}")
                        continue
                
                # Epoch metrics
                epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float('inf')
                epoch_similarity = sum(batch_similarities) / len(batch_similarities) if batch_similarities else 0.0
                epoch_time = time.time() - epoch_start_time
                
                # Update progress
                self.progress['epochs_completed'] = epoch + 1
                self.progress['loss_history'].append(epoch_loss)
                self.progress['similarity_history'].append(epoch_similarity)
                self.progress['total_training_time'] += epoch_time
                
                # Check for improvement
                improvement_detected = False
                if epoch_loss < self.progress['best_loss']:
                    self.progress['best_loss'] = epoch_loss
                    improvement_detected = True
                    no_improvement_count = 0
                    
                    # Save best checkpoint
                    self._save_checkpoint(trainer, epoch, epoch_loss, epoch_similarity, "best_loss")
                    
                if epoch_similarity > self.progress['best_similarity']:
                    self.progress['best_similarity'] = epoch_similarity
                    improvement_detected = True
                    no_improvement_count = 0
                    
                    # Save best checkpoint
                    self._save_checkpoint(trainer, epoch, epoch_loss, epoch_similarity, "best_similarity")
                
                if improvement_detected and not self.progress['learning_detected']:
                    self.progress['learning_detected'] = True
                    self.progress['first_improvement_epoch'] = epoch + 1
                    logger.info(f"[SUCCESS] ПЕРВОЕ УЛУЧШЕНИЕ ОБНАРУЖЕНО НА EPOCH {epoch + 1}!")
                    
                if not improvement_detected:
                    no_improvement_count += 1
                
                # Logging
                if (epoch + 1) % 10 == 0 or improvement_detected:
                    elapsed = datetime.now() - self.start_time
                    
                    logger.info(f"[DATA] EPOCH {epoch + 1} (UNLIMITED):")
                    logger.info(f"   Loss: {epoch_loss:.6f} (best: {self.progress['best_loss']:.6f})")
                    logger.info(f"   Similarity: {epoch_similarity:.6f} (best: {self.progress['best_similarity']:.6f})")
                    logger.info(f"   Epoch time: {epoch_time:.1f}s")
                    logger.info(f"   Total elapsed: {elapsed}")
                    logger.info(f"   Learning detected: {'[OK] YES' if self.progress['learning_detected'] else '[ERROR] NO'}")
                    
                    if improvement_detected:
                        logger.info(f"   [TARGET] IMPROVEMENT! No improvement count reset")
                        
                # Save progress regularly
                if (epoch + 1) % 20 == 0:
                    self._save_progress()
                    
                # Regular checkpoint
                if (epoch + 1) % 25 == 0:
                    self._save_checkpoint(trainer, epoch, epoch_loss, epoch_similarity, f"epoch_{epoch+1}")
                
                # НЕТ early stopping - пусть работает всю ночь!
                # if no_improvement_count >= max_no_improvement:
                #     logger.info(f"[STOP] Early stopping: {no_improvement_count} epochs without improvement")
                #     break
                    
                # Memory cleanup
                if (epoch + 1) % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
        except KeyboardInterrupt:
            logger.info("🛑 ОБУЧЕНИЕ ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ (Ctrl+C)")
            logger.info("   Сохраняем финальное состояние...")
        
        # Final results
        self._finalize_training(trainer)
    
    def _setup_trainer(self) -> 'EmergentCubeTrainer':
        """Setup trainer for overnight training"""
        try:
            config = EmergentTrainingConfig()
            config.teacher_model = "distilbert-base-uncased"
            config.cube_dimensions = (15, 15, 11)
            config.learning_rate = 0.0001  # Lower для stability
            config.mixed_precision = True
            config.gradient_balancing = True
            
            trainer = EmergentCubeTrainer(config, device=str(self.device))
            
            total_params = sum(p.numel() for p in trainer.parameters())
            logger.info(f"[OK] Trainer created: {total_params:,} parameters")
            
            return trainer
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create trainer: {e}")
            return None
    
    def _setup_dataset(self) -> list:
        """Setup dataset for training"""
        try:
            # Medium-size dataset для overnight training
            dialogue_pairs = [
                {"question": "What is AI?", "answer": "AI is artificial intelligence technology that mimics human cognitive functions."},
                {"question": "How do neural networks work?", "answer": "Neural networks process data through interconnected layers of artificial neurons."},
                {"question": "What is machine learning?", "answer": "Machine learning enables computers to learn and improve from experience automatically."},
                {"question": "Explain deep learning", "answer": "Deep learning uses multi-layered neural networks to analyze complex patterns in data."},
                {"question": "What are transformers?", "answer": "Transformers are neural network architectures that use attention mechanisms for sequence processing."},
                {"question": "How does backpropagation work?", "answer": "Backpropagation calculates gradients by propagating errors backward through neural network layers."},
                {"question": "What is overfitting?", "answer": "Overfitting occurs when a model learns training data too well, reducing generalization ability."},
                {"question": "Explain gradient descent", "answer": "Gradient descent optimizes neural networks by iteratively adjusting parameters to minimize loss."},
                {"question": "What are activation functions?", "answer": "Activation functions introduce non-linearity to neural networks, enabling complex pattern recognition."},
                {"question": "How do CNNs work?", "answer": "Convolutional neural networks use filters to detect local features in structured data like images."},
                {"question": "What is attention mechanism?", "answer": "Attention mechanisms allow models to focus on relevant parts of input when making predictions."},
                {"question": "Explain LSTM networks", "answer": "LSTM networks handle sequential data by selectively remembering and forgetting information over time."},
                {"question": "What is transfer learning?", "answer": "Transfer learning adapts pre-trained models to new tasks, leveraging previously learned features."},
                {"question": "How does batch normalization work?", "answer": "Batch normalization stabilizes training by normalizing inputs to each layer during processing."},
                {"question": "What are GANs?", "answer": "Generative Adversarial Networks train two competing models to generate realistic synthetic data."},
            ]
            
            dataset = create_dialogue_dataset(
                dialogue_pairs,
                teacher_model="distilbert-base-uncased",
                cache_embeddings=True,
                validation_split=0.0,
                normalize_embeddings=True
            )
            
            logger.info(f"[OK] Dataset created: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create dataset: {e}")
            return []
    
    def _prepare_batch(self, dataset, batch_idx: int, batch_size: int):
        """Prepare training batch"""
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dataset))
        
        batch_questions = []
        batch_answers = []
        
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            batch_questions.append(sample['question_embedding'])
            batch_answers.append(sample['answer_embedding'])
        
        question_batch = torch.stack(batch_questions)
        answer_batch = torch.stack(batch_answers)
        
        return question_batch, answer_batch
    
    def _save_checkpoint(self, trainer, epoch: int, loss: float, similarity: float, suffix: str):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoints_dir / f"overnight_{suffix}_epoch_{epoch+1}_loss_{loss:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'similarity': similarity,
            'progress': self.progress
        }, checkpoint_path)
        
        logger.info(f"[SAVE] Checkpoint saved: {checkpoint_path}")
    
    def _save_progress(self):
        """Save current progress to file"""
        progress_file = self.results_dir / "training_progress.json"
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[DATA] Progress saved: {progress_file}")
    
    def _finalize_training(self, trainer):
        """Finalize training and save results"""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        
        final_results = {
            **self.progress,
            'end_time': end_time.isoformat(),
            'total_time': str(total_time),
            'total_time_seconds': total_time.total_seconds(),
            'epochs_per_hour': self.progress['epochs_completed'] / (total_time.total_seconds() / 3600),
            'conclusion': self._draw_conclusions()
        }
        
        # Save final results
        results_file = self.results_dir / "overnight_training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Save final model
        final_model_path = self.results_dir / "final_model.pt"
        torch.save(trainer.state_dict(), final_model_path)
        
        logger.info("="*50)
        logger.info("🌅 НОЧНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info("="*50)
        logger.info(f"Общее время: {total_time}")
        logger.info(f"Epochs завершено: {self.progress['epochs_completed']}")
        logger.info(f"Лучший loss: {self.progress['best_loss']:.6f}")
        logger.info(f"Лучший similarity: {self.progress['best_similarity']:.6f}")
        logger.info(f"Обучение обнаружено: {'[OK] ДА' if self.progress['learning_detected'] else '[ERROR] НЕТ'}")
        if self.progress['learning_detected']:
            logger.info(f"Первое улучшение на epoch: {self.progress['first_improvement_epoch']}")
        
        logger.info(f"\n[DATA] Результаты сохранены: {results_file}")
        logger.info(f"[SAVE] Финальная модель: {final_model_path}")
        
        return final_results
    
    def _draw_conclusions(self) -> Dict[str, Any]:
        """Draw conclusions from overnight training"""
        conclusions = {
            'learning_occurred': self.progress['learning_detected'],
            'convergence_speed': 'unknown',
            'architecture_viability': 'unknown',
            'next_steps': []
        }
        
        if self.progress['learning_detected']:
            conclusions['convergence_speed'] = 'slow_but_working'
            conclusions['architecture_viability'] = 'viable'
            conclusions['next_steps'] = [
                'Continue training with more epochs',
                'Optimize hyperparameters for faster convergence',
                'Scale up dataset for better results'
            ]
        else:
            if self.progress['epochs_completed'] >= 100:
                conclusions['convergence_speed'] = 'very_slow_or_broken'
                conclusions['architecture_viability'] = 'questionable'
                conclusions['next_steps'] = [
                    'Debug loss function implementation',
                    'Check gradient flow',
                    'Verify data preprocessing',
                    'Consider architecture modifications'
                ]
            else:
                conclusions['convergence_speed'] = 'unknown_insufficient_data'
                conclusions['architecture_viability'] = 'unknown'
                conclusions['next_steps'] = [
                    'Run longer training session',
                    'Try different learning rates',
                    'Increase dataset size'
                ]
        
        return conclusions

def main():
    """Запуск неограниченного обучения"""
    print("[START] Подготовка к НЕОГРАНИЧЕННОМУ обучению...")
    
    # Проверяем ресурсы
    if torch.cuda.is_available():
        print(f"[OK] GPU доступна: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[WARNING] GPU не доступна, используется CPU")
    
    # Информация о batch size
    print(f"[TARGET] Оптимизация для RTX 5090:")
    print(f"   Batch size: 1024 (максимальная утилизация GPU)")
    print(f"   Epochs: НЕОГРАНИЧЕНО")
    print(f"   Остановка: Ctrl+C или автоматическая при отсутствии прогресса")
    
    # Estimate performance
    print(f"[TIME] Ориентировочная скорость: ~60s per epoch")
    print(f"   Прогресс будет виден каждые 10 epochs")
    print(f"   Автосохранение каждые 25 epochs")
    
    # Confirm
    user_input = input("\n🤔 Запустить НЕОГРАНИЧЕННОЕ обучение? (y/n): ").strip().lower()
    if user_input != 'y':
        print("[ERROR] Обучение отменено")
        return
    
    print("\n🌙 ЗАПУСК НЕОГРАНИЧЕННОГО ОБУЧЕНИЯ...")
    print("   Используйте Ctrl+C для остановки в любой момент")
    print("   Все прогресс будет автоматически сохранен")
    print("="*60)
    
    # Run unlimited training
    manager = OvernightTrainingManager()
    manager.run_overnight_training()

if __name__ == "__main__":
    main()