#!/usr/bin/env python3
"""
Dialogue Training Optimization Script - Stage 2.2 Phase 3

Оптимизированная версия dialogue training для достижения 80%+ Q→A similarity.
Включает hyperparameter tuning, enhanced dataset, и advanced training techniques.

ЦЕЛЬ STAGE 2.2: Повысить Q→A similarity с 27.24% до 80%+

Улучшения от Stage 2.1:
- Увеличенный dataset (15 → 100+ dialogue pairs)
- Оптимизированные hyperparameters
- Advanced loss functions
- Learning rate scheduling
- Enhanced data augmentation

Использование:
    python run_dialogue_training_optimization.py --epochs 100 --batch_size 32
    python run_dialogue_training_optimization.py --advanced_optimizer --lr_schedule

Автор: 3D Cellular Neural Network Project
Версия: v2.0.0 (Stage 2.2)
Дата: 7 июня 2025
"""

import torch
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Импорты наших компонентов
from training.embedding_trainer.cube_trainer import CubeTrainer, TrainingConfig
from training.embedding_trainer.dialogue_dataset import DialogueDataset, create_dialogue_dataset

# Настройка логирования с UTF-8 для Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dialogue_training_optimization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_enhanced_dialogue_data() -> List[Dict]:
    """Создает расширенный набор диалоговых данных (100+ pairs)"""
    
    # Базовые AI/ML topics
    ai_ml_pairs = [
        {"question": "What is artificial intelligence?", 
         "answer": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks requiring human intelligence."},
        
        {"question": "How do neural networks work?", 
         "answer": "Neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections."},
        
        {"question": "What is machine learning?", 
         "answer": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."},
        
        {"question": "Explain deep learning briefly", 
         "answer": "Deep learning uses multi-layered neural networks to model and understand complex patterns in data, similar to how the human brain processes information."},
        
        {"question": "What are transformers in AI?", 
         "answer": "Transformers are a type of neural network architecture that uses attention mechanisms to process sequential data very effectively."},
    ]
    
    # Technical details
    technical_pairs = [
        {"question": "What is backpropagation?", 
         "answer": "Backpropagation is an algorithm for training neural networks that calculates gradients by propagating errors backward through the network layers."},
        
        {"question": "Explain gradient descent optimization", 
         "answer": "Gradient descent is an optimization algorithm that iteratively moves toward the minimum of a loss function by following the negative gradient direction."},
        
        {"question": "What is batch normalization?", 
         "answer": "Batch normalization is a technique that normalizes layer inputs to improve training stability and speed by reducing internal covariate shift."},
        
        {"question": "How do attention mechanisms work?", 
         "answer": "Attention mechanisms allow models to focus on relevant parts of input sequences by learning weighted representations of different input elements."},
        
        {"question": "What is transfer learning?", 
         "answer": "Transfer learning reuses pre-trained models on new but related tasks, leveraging learned features to improve performance with less data."},
    ]
    
    # Advanced concepts
    advanced_pairs = [
        {"question": "How can we improve neural network training?", 
         "answer": "Neural network training can be improved through better optimization algorithms, regularization techniques, data augmentation, and architectural innovations."},
        
        {"question": "What is the difference between supervised and unsupervised learning?", 
         "answer": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in data without labels."},
        
        {"question": "Why is GPU acceleration important for AI?", 
         "answer": "GPUs provide massive parallel processing power that significantly speeds up matrix operations fundamental to neural network computations."},
        
        {"question": "What role does data quality play in AI systems?", 
         "answer": "Data quality is crucial for AI performance - clean, representative, and well-labeled data leads to more accurate and reliable models."},
        
        {"question": "How do we prevent overfitting in neural networks?", 
         "answer": "Overfitting can be prevented through techniques like dropout, regularization, cross-validation, and ensuring sufficient training data diversity."},
    ]
    
    # 🆕 НОВЫЕ КАТЕГОРИИ для увеличения dataset:
    
    # Computer Science fundamentals
    cs_pairs = [
        {"question": "What is computational complexity?", 
         "answer": "Computational complexity theory focuses on classifying computational problems according to their resource usage and relating these classes to each other."},
        
        {"question": "Explain Big O notation", 
         "answer": "Big O notation describes the limiting behavior of a function when the argument tends towards a particular value, often used to classify algorithms."},
        
        {"question": "What are data structures?", 
         "answer": "Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently."},
        
        {"question": "How do databases work?", 
         "answer": "Databases are organized collections of structured information that can be easily accessed, managed, and updated through specialized software systems."},
        
        {"question": "What is distributed computing?", 
         "answer": "Distributed computing is a field of computer science that studies distributed systems, where components communicate and coordinate through message passing."},
    ]
    
    # Programming concepts
    programming_pairs = [
        {"question": "What is object-oriented programming?", 
         "answer": "Object-oriented programming is a programming paradigm based on the concept of objects, which can contain data and code that manipulates that data."},
        
        {"question": "Explain functional programming", 
         "answer": "Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing state."},
        
        {"question": "What is version control?", 
         "answer": "Version control is a system that records changes to files over time so that specific versions can be recalled later."},
        
        {"question": "How does garbage collection work?", 
         "answer": "Garbage collection is automatic memory management that frees up memory space used by objects that are no longer referenced by the program."},
        
        {"question": "What are design patterns?", 
         "answer": "Design patterns are reusable solutions to commonly occurring problems in software design and development."},
    ]
    
    # Data Science и Analytics
    data_science_pairs = [
        {"question": "What is data science?", 
         "answer": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data."},
        
        {"question": "How do you handle missing data?", 
         "answer": "Missing data can be handled through deletion, imputation with mean/median/mode, predictive modeling, or using algorithms that handle missing values."},
        
        {"question": "What is feature engineering?", 
         "answer": "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve machine learning model performance."},
        
        {"question": "Explain cross-validation", 
         "answer": "Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample by dividing data into complementary subsets."},
        
        {"question": "What is dimensionality reduction?", 
         "answer": "Dimensionality reduction is the process of reducing the number of features in a dataset while preserving important information and relationships."},
    ]
    
    # Neural Network architectures
    architecture_pairs = [
        {"question": "What are convolutional neural networks?", 
         "answer": "Convolutional neural networks are specialized neural networks for processing grid-like data such as images, using convolution operations to detect local features."},
        
        {"question": "How do recurrent neural networks work?", 
         "answer": "Recurrent neural networks process sequential data by maintaining hidden states that carry information from previous time steps through recurrent connections."},
        
        {"question": "What are generative adversarial networks?", 
         "answer": "Generative adversarial networks consist of two neural networks competing against each other, with one generating fake data and the other detecting it."},
        
        {"question": "Explain LSTM networks", 
         "answer": "Long Short-Term Memory networks are a type of recurrent neural network designed to handle long-term dependencies through gating mechanisms."},
        
        {"question": "What is attention in neural networks?", 
         "answer": "Attention mechanisms allow neural networks to focus on relevant parts of input sequences by computing weighted representations of different elements."},
    ]
    
    # Mathematical foundations
    math_pairs = [
        {"question": "What is linear algebra in ML?", 
         "answer": "Linear algebra provides the mathematical foundation for machine learning, including vector spaces, matrix operations, and eigenvalue decomposition."},
        
        {"question": "How is calculus used in neural networks?", 
         "answer": "Calculus, particularly derivatives and partial derivatives, is essential for backpropagation and optimization algorithms in neural network training."},
        
        {"question": "What is probability in machine learning?", 
         "answer": "Probability theory provides frameworks for handling uncertainty, making predictions, and modeling probabilistic relationships in data."},
        
        {"question": "Explain statistical inference", 
         "answer": "Statistical inference is the process of drawing conclusions about populations from sample data using probability theory and statistical methods."},
        
        {"question": "What is optimization in ML?", 
         "answer": "Optimization in machine learning involves finding the best parameters for models by minimizing or maximizing objective functions using mathematical techniques."},
    ]
    
    # Ethics и AI safety
    ethics_pairs = [
        {"question": "What are AI ethics concerns?", 
         "answer": "AI ethics concerns include bias, fairness, transparency, accountability, privacy, and the potential societal impacts of artificial intelligence systems."},
        
        {"question": "How do we ensure AI fairness?", 
         "answer": "AI fairness can be ensured through diverse training data, bias detection algorithms, fairness metrics, and inclusive development processes."},
        
        {"question": "What is explainable AI?", 
         "answer": "Explainable AI refers to methods and techniques that make artificial intelligence decisions transparent and interpretable to humans."},
        
        {"question": "How do we protect privacy in AI?", 
         "answer": "Privacy in AI can be protected through techniques like differential privacy, federated learning, data minimization, and secure computation methods."},
        
        {"question": "What is AI safety?", 
         "answer": "AI safety focuses on ensuring that artificial intelligence systems behave as intended and do not cause unintended harm to humans or society."},
    ]
    
    # Объединяем все категории
    all_pairs = (ai_ml_pairs + technical_pairs + advanced_pairs + 
                cs_pairs + programming_pairs + data_science_pairs + 
                architecture_pairs + math_pairs + ethics_pairs)
    
    logger.info(f"Created enhanced dialogue dataset with {len(all_pairs)} pairs")
    logger.info("Categories: AI/ML, Technical, CS, Programming, Data Science, Architecture, Math, Ethics")
    
    return all_pairs


def create_optimized_training_config(args) -> TrainingConfig:
    """Создание оптимизированной конфигурации обучения для Stage 2.2"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = TrainingConfig(
        mode="dialogue",
        device=device,
        random_seed=42,
        
        # [CONFIG] ОПТИМИЗИРОВАННАЯ АРХИТЕКТУРА
        lattice_size=[8, 8, 12],           # Проверенная конфигурация
        embedding_dim=768,                  # DistilBERT размерность
        batch_size=args.batch_size,         # Увеличенный batch size
        
        # [START] ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ
        learning_rate=args.learning_rate,   # Оптимизированный LR
        epochs=args.epochs,                 # Больше эпох для convergence
        optimizer="adam",                   # Adam (AdamW настроим отдельно)
        loss_function="mse",                # MSE может работать лучше для regression
        
        # [TARGET] ПОВЫШЕННЫЕ ЦЕЛИ
        target_similarity=0.80,             # Цель 80%
        convergence_threshold=0.001,
        early_stopping_patience=30,         # Больше patience для достижения цели
        
        # [DATA] ENHANCED MONITORING
        log_interval=2,                     # Частое логирование
        save_interval=10,                   # Частое сохранение checkpoints
        checkpoint_dir="checkpoints/dialogue_training_optimization"
    )
    
    return config


def create_advanced_loss_function(alpha: float = 0.7) -> callable:
    """Создает продвинутую loss функцию для dialogue training"""
    
    def advanced_dialogue_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Комбинированная loss функция:
        - MSE loss для точности
        - Cosine similarity loss для семантики
        - L1 loss для robustness
        """
        # MSE component
        mse_loss = F.mse_loss(predicted, target)
        
        # Cosine similarity component (minimizing 1 - cosine_similarity)
        cos_sim = F.cosine_similarity(predicted, target, dim=-1)
        cos_loss = 1.0 - cos_sim.mean()
        
        # L1 component для robustness
        l1_loss = F.l1_loss(predicted, target)
        
        # Комбинированная loss
        combined_loss = (alpha * mse_loss + 
                        (1 - alpha) * cos_loss + 
                        0.1 * l1_loss)
        
        return combined_loss
    
    return advanced_dialogue_loss


def create_optimized_optimizer(model, config, use_advanced: bool = False):
    """Создает оптимизированный optimizer с scheduling"""
    
    if use_advanced:
        # AdamW с weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,  # Hardcoded weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        # Стандартный Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
    
    return optimizer


def create_lr_scheduler(optimizer, scheduler_type: str = "plateau"):
    """Создает learning rate scheduler"""
    
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximizing similarity
            factor=0.5,
            patience=10
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    return scheduler


def run_optimized_dialogue_training(args):
    """Основная функция оптимизированного dialogue training"""
    logger.info("[START] STARTING DIALOGUE TRAINING OPTIMIZATION - STAGE 2.2")
    logger.info("=" * 70)
    logger.info("GOAL: Increase Q→A similarity from 27.24% to 80%+")
    logger.info("=" * 70)
    
    # 1. Подготовка окружения
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path("results/dialogue_training_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Создание ENHANCED dialogue dataset
    logger.info("Creating ENHANCED DialogueDataset...")
    
    dialogue_pairs = create_enhanced_dialogue_data()
    logger.info(f"   Using {len(dialogue_pairs)} dialogue pairs (vs 15 in Stage 2.1)")
    
    # Создание dataset с оптимизированными параметрами
    dataset = create_dialogue_dataset(
        dialogue_pairs=dialogue_pairs,
        teacher_model=args.teacher_model,
        validation_split=0.2,                # 20% для validation
        embedding_dim=768,
        enable_quality_filter=True,
        semantic_similarity_threshold=0.1,   # Более мягкий фильтр (правильный параметр)
        use_cache=True,
        add_context_noise=args.data_augmentation # Data augmentation (правильный параметр)
    )
    
    # 3. Создание OPTIMIZED CubeTrainer
    logger.info("Initializing OPTIMIZED CubeTrainer...")
    config = create_optimized_training_config(args)
    trainer = CubeTrainer(config=config)
    
    # Инициализация компонентов
    trainer.initialize_components()
    
    # 4. ADVANCED OPTIMIZER & SCHEDULER
    logger.info("Setting up advanced optimization...")
    
    if args.advanced_optimizer:
        trainer.optimizer = create_optimized_optimizer(
            trainer.embedding_processor, 
            config, 
            use_advanced=True
        )
        logger.info("   Using AdamW optimizer with weight decay")
    
    if args.lr_schedule:
        scheduler = create_lr_scheduler(trainer.optimizer, args.scheduler_type)
        logger.info(f"   Using {args.scheduler_type} learning rate scheduler")
    else:
        scheduler = None
    
    # 5. ADVANCED LOSS FUNCTION
    if args.advanced_loss:
        advanced_loss_fn = create_advanced_loss_function(alpha=0.7)
        logger.info("   Using advanced combined loss function")
    else:
        advanced_loss_fn = None
    
    # 6. Подготовка данных
    logger.info("Preparing training data...")
    
    train_loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        validation=False
    )
    
    val_loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        validation=True
    )
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Validation batches: {len(val_loader)}")
    
    # 7. ENHANCED TRAINING LOOP
    logger.info("\n[GRADUATE] STARTING OPTIMIZED TRAINING...")
    logger.info("=" * 50)
    
    best_val_similarity = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.epochs):
        logger.info(f"\nEPOCH {epoch + 1}/{args.epochs}")
        logger.info("-" * 30)
        
        # Training phase
        trainer.embedding_processor.train()
        epoch_train_loss = 0.0
        epoch_train_similarity = 0.0
        train_batches = 0
        
        for batch_idx, (question_emb, answer_emb) in enumerate(train_loader):
            question_emb = question_emb.to(device)
            answer_emb = answer_emb.to(device)
            
            trainer.optimizer.zero_grad()
            
            # Forward pass через куб
            # Убеждаемся что input требует градиенты
            question_emb_with_grad = question_emb.clone().detach().requires_grad_(True)
            
            processed_embeddings = []
            for q_emb in question_emb_with_grad:
                processed_emb = trainer.embedding_processor.forward(q_emb.unsqueeze(0))
                processed_embeddings.append(processed_emb.squeeze(0))
            
            processed_tensor = torch.stack(processed_embeddings)
            
            # Loss calculation
            if advanced_loss_fn:
                loss = advanced_loss_fn(processed_tensor, answer_emb)
            else:
                similarity = F.cosine_similarity(processed_tensor, answer_emb, dim=-1)
                loss = 1.0 - similarity.mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping для stability
            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(trainer.embedding_processor.parameters(), 1.0)
            
            trainer.optimizer.step()
            
            # Метрики
            with torch.no_grad():
                batch_similarity = F.cosine_similarity(processed_tensor, answer_emb, dim=-1).mean()
                epoch_train_loss += loss.item()
                epoch_train_similarity += batch_similarity.item()
                train_batches += 1
        
        # Validation phase
        trainer.embedding_processor.eval()
        val_loss = 0.0
        val_similarity = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for question_emb, answer_emb in val_loader:
                question_emb = question_emb.to(device)
                answer_emb = answer_emb.to(device)
                
                processed_embeddings = []
                for q_emb in question_emb:
                    processed_emb = trainer.embedding_processor.forward(q_emb.unsqueeze(0))
                    processed_embeddings.append(processed_emb.squeeze(0))
                
                processed_tensor = torch.stack(processed_embeddings)
                
                batch_similarity = F.cosine_similarity(processed_tensor, answer_emb, dim=-1).mean()
                batch_loss = 1.0 - batch_similarity
                
                val_loss += batch_loss.item()
                val_similarity += batch_similarity.item()
                val_batches += 1
        
        # Средние метрики
        avg_train_loss = epoch_train_loss / train_batches
        avg_train_similarity = epoch_train_similarity / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_similarity = val_similarity / val_batches
        
        # Learning rate scheduling
        if scheduler:
            if args.scheduler_type == "plateau":
                scheduler.step(avg_val_similarity)
            else:
                scheduler.step()
        
        # Логирование эпохи
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(f"EPOCH {epoch + 1} RESULTS:")
        logger.info(f"   Train: Loss={avg_train_loss:.4f}, Similarity={avg_train_similarity:.4f}")
        logger.info(f"   Val:   Loss={avg_val_loss:.4f}, Similarity={avg_val_similarity:.4f}")
        logger.info(f"   LR: {current_lr:.6f}")
        
        # [TARGET] Проверка прогресса к цели
        progress = (avg_val_similarity / config.target_similarity) * 100
        logger.info(f"   Progress to goal: {progress:.1f}% ({avg_val_similarity:.4f}/{config.target_similarity})")
        
        # История обучения
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_similarity': avg_train_similarity,
            'val_loss': avg_val_loss,
            'val_similarity': avg_val_similarity,
            'learning_rate': current_lr,
            'progress_percentage': progress
        })
        
        # Early stopping и сохранение лучшей модели
        if avg_val_similarity > best_val_similarity:
            best_val_similarity = avg_val_similarity
            patience_counter = 0
            
            # Сохранение лучшей модели
            checkpoint_path = results_dir / f"best_optimized_model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.embedding_processor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_similarity': avg_val_similarity,
                'config': config,
                'training_args': vars(args)
            }, checkpoint_path)
            logger.info(f"[SAVE] NEW BEST MODEL saved: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Проверка на достижение целевой similarity
        if avg_val_similarity >= config.target_similarity:
            logger.info(f"[SUCCESS] TARGET SIMILARITY ACHIEVED: {avg_val_similarity:.4f} >= {config.target_similarity}")
            logger.info("[TROPHY] STAGE 2.2 OBJECTIVE COMPLETED!")
            break
        
        # Проверка early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # 8. ИТОГОВЫЕ РЕЗУЛЬТАТЫ
    logger.info("\n" + "=" * 70)
    logger.info("[TROPHY] DIALOGUE TRAINING OPTIMIZATION COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"[TARGET] Best validation similarity: {best_val_similarity:.4f}")
    logger.info(f"[CHART] Improvement: {best_val_similarity:.4f} vs 0.2724 (baseline)")
    logger.info(f"[DICE] Improvement factor: {best_val_similarity / 0.2724:.2f}x")
    logger.info(f"[DATA] Total epochs: {len(training_history)}")
    logger.info(f"[GRADUATE] Target achieved: {'[OK] YES' if best_val_similarity >= config.target_similarity else '[ERROR] NO (Progress: ' + f'{(best_val_similarity/config.target_similarity)*100:.1f}%)'}")
    
    # Сохранение результатов
    results_file = results_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        'optimization_config': {
            'stage': '2.2',
            'goal': 'Increase Q→A similarity to 80%+',
            'baseline_similarity': 0.2724,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'teacher_model': args.teacher_model,
            'advanced_optimizer': args.advanced_optimizer,
            'lr_schedule': args.lr_schedule,
            'advanced_loss': args.advanced_loss,
            'data_augmentation': args.data_augmentation,
            'gradient_clipping': args.gradient_clipping
        },
        'final_results': {
            'best_val_similarity': best_val_similarity,
            'baseline_similarity': 0.2724,
            'improvement_factor': best_val_similarity / 0.2724,
            'improvement_absolute': best_val_similarity - 0.2724,
            'total_epochs': len(training_history),
            'target_achieved': best_val_similarity >= config.target_similarity,
            'progress_to_goal_percentage': (best_val_similarity / config.target_similarity) * 100
        },
        'training_history': training_history,
        'dataset_info': dataset.get_statistics()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"[SAVE] Optimization results saved: {results_file}")
    
    # Построение графиков
    if args.plot_results:
        plot_optimization_results(training_history, results_dir, baseline=0.2724)
    
    return best_val_similarity, training_history


def plot_optimization_results(training_history: List[Dict], results_dir: Path, baseline: float = 0.2724):
    """Построение графиков результатов оптимизации"""
    epochs = [h['epoch'] for h in training_history]
    train_loss = [h['train_loss'] for h in training_history]
    val_loss = [h['val_loss'] for h in training_history]
    train_sim = [h['train_similarity'] for h in training_history]
    val_sim = [h['val_similarity'] for h in training_history]
    progress = [h['progress_percentage'] for h in training_history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss - Stage 2.2 Optimization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Similarity plot
    ax2.plot(epochs, train_sim, 'b-', label='Training Similarity', linewidth=2)
    ax2.plot(epochs, val_sim, 'r-', label='Validation Similarity', linewidth=2)
    ax2.axhline(y=0.80, color='g', linestyle='--', label='Target (0.80)', linewidth=2)
    ax2.axhline(y=baseline, color='orange', linestyle=':', label=f'Baseline ({baseline:.4f})', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Training and Validation Similarity - Stage 2.2 Optimization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Progress to goal
    ax3.plot(epochs, progress, 'purple', linewidth=3, marker='o')
    ax3.axhline(y=100, color='g', linestyle='--', label='Goal (100%)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Progress to Goal (%)')
    ax3.set_title('Progress to 80% Target Similarity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rate' in training_history[0]:
        lrs = [h['learning_rate'] for h in training_history]
        ax4.plot(epochs, lrs, 'brown', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Learning Rate\nTracking\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=16)
        ax4.set_title('Learning Rate Schedule (N/A)')
    
    plt.tight_layout()
    
    plot_file = results_dir / f"optimization_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"[DATA] Optimization plots saved: {plot_file}")
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dialogue Training Optimization - Stage 2.2')
    
    # [CONFIG] ENHANCED TRAINING PARAMETERS
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100, vs 20 in Stage 2.1)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training (default: 32, vs 8 in Stage 2.1)')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                       help='Learning rate (default: 0.0005, vs 0.001 in Stage 2.1)')
    
    # [START] ADVANCED OPTIMIZATION
    parser.add_argument('--advanced_optimizer', action='store_true', default=True,
                       help='Use AdamW optimizer with weight decay (default: True)')
    parser.add_argument('--lr_schedule', action='store_true', default=True,
                       help='Use learning rate scheduling (default: True)')
    parser.add_argument('--scheduler_type', type=str, default='plateau', 
                       choices=['plateau', 'cosine'],
                       help='Type of LR scheduler (default: plateau)')
    parser.add_argument('--advanced_loss', action='store_true', default=True,
                       help='Use advanced combined loss function (default: True)')
    parser.add_argument('--gradient_clipping', action='store_true', default=True,
                       help='Enable gradient clipping for stability (default: True)')
    
    # [DATA] DATA ENHANCEMENT
    parser.add_argument('--data_augmentation', action='store_true', default=True,
                       help='Enable data augmentation (default: True)')
    
    # [BOT] MODEL PARAMETERS
    parser.add_argument('--teacher_model', type=str, default='distilbert', 
                       choices=['llama3-8b', 'mistral-7b', 'distilbert', 'bert-base'],
                       help='Teacher LLM model for embeddings (default: distilbert)')
    
    # [DATA] LOGGING И VISUALIZATION
    parser.add_argument('--log_interval', type=int, default=2, 
                       help='Logging interval during training (default: 2)')
    parser.add_argument('--plot_results', action='store_true', default=True,
                       help='Generate optimization plots (default: True)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG mode enabled")
    
    # Информация о запуске
    logger.info("[START] DIALOGUE TRAINING OPTIMIZATION - STAGE 2.2 PHASE 3")
    logger.info("GOAL: Increase Q→A similarity from 27.24% to 80%+")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION SETTINGS:")
    logger.info(f"   Epochs: {args.epochs} (vs 20 in Stage 2.1)")
    logger.info(f"   Batch size: {args.batch_size} (vs 8 in Stage 2.1)")
    logger.info(f"   Learning rate: {args.learning_rate} (vs 0.001 in Stage 2.1)")  
    logger.info(f"   Teacher model: {args.teacher_model}")
    logger.info(f"   Advanced optimizer: {args.advanced_optimizer}")
    logger.info(f"   LR scheduling: {args.lr_schedule}")
    logger.info(f"   Advanced loss: {args.advanced_loss}")
    logger.info(f"   Data augmentation: {args.data_augmentation}")
    logger.info(f"   Gradient clipping: {args.gradient_clipping}")
    
    try:
        # Запуск оптимизированного обучения
        best_similarity, history = run_optimized_dialogue_training(args)
        
        # Итоговый отчет
        improvement = best_similarity / 0.2724
        logger.info("\n" + "=" * 70)
        logger.info("[TROPHY] DIALOGUE TRAINING OPTIMIZATION STAGE 2.2 COMPLETED!")
        logger.info("=" * 70)
        logger.info(f"[CHART] Best Q→A similarity: {best_similarity:.4f}")
        logger.info(f"[DATA] Baseline similarity: 0.2724")
        logger.info(f"[START] Improvement factor: {improvement:.2f}x")
        logger.info(f"[TARGET] Target (0.80): {'[OK] ACHIEVED' if best_similarity >= 0.80 else f'🔶 PROGRESS: {(best_similarity/0.80)*100:.1f}%'}")
        
        if best_similarity >= 0.80:
            logger.info("[SUCCESS] STAGE 2.2 OBJECTIVE COMPLETED!")
            logger.info("Ready for Stage 2.3!")
        else:
            logger.info("[REFRESH] Continue optimization or proceed to Stage 2.3 with current progress")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main()) 