#!/usr/bin/env python3
"""
Dialogue Training Script - Stage 2.1 Phase 3

Скрипт для запуска реального dialogue training с Teacher LLM архитектурой.
Обучает 3D Cubic Core на Q→A трансформациях через готовые компоненты.

Использование:
    python run_dialogue_training.py --epochs 50 --batch_size 32
    python run_dialogue_training.py --model distilbert --debug

Автор: 3D Cellular Neural Network Project
Версия: v1.0.0 (Stage 2.1)
Дата: 6 июня 2025
"""

import torch
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Импорты наших компонентов
from training.embedding_trainer.cube_trainer import CubeTrainer, TrainingConfig
from training.embedding_trainer.dialogue_dataset import DialogueDataset, create_dialogue_dataset

# Настройка логирования с UTF-8 для Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dialogue_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_sample_dialogue_data() -> List[Dict]:
    """Создает sample диалоговые данные для тестирования"""
    return [
        # Простые Q&A пары
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
        
        # Более сложные пары
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
        
        # Технические пары
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


def setup_training_environment() -> Tuple[str, Path]:
    """Настройка окружения для обучения"""
    # Определение устройства
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"[START] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("[COMPUTER] Using CPU for training")
    
    # Создание директорий
    log_dir = Path("logs")
    checkpoint_dir = Path("checkpoints/dialogue_training")
    results_dir = Path("results/dialogue_training")
    
    for directory in [log_dir, checkpoint_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return device, results_dir


def create_training_config(args) -> TrainingConfig:
    """Создание конфигурации обучения"""
    device, _ = setup_training_environment()
    
    config = TrainingConfig(
        mode="dialogue",                    # DIALOGUE РЕЖИМ
        device=device,
        random_seed=42,
        
        # Архитектура куба
        lattice_size=[8, 8, 12],           # [8,8,12] = 768D для DistilBERT совместимости
        embedding_dim=768,                  # DistilBERT размерность
        batch_size=args.batch_size,
        
        # Параметры обучения
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        optimizer="adam",
        loss_function="cosine",             # Cosine similarity для dialogue качества
        
        # Диалоговые цели
        target_similarity=0.80,             # Q→A similarity цель 80%
        convergence_threshold=0.001,
        early_stopping_patience=15,         # Больше patience для dialogue
        
        # Логирование
        log_interval=5,                     # Частое логирование для мониторинга
        save_interval=20,
        checkpoint_dir="checkpoints/dialogue_training"
    )
    
    return config


def run_dialogue_training(args):
    """Основная функция dialogue training"""
    logger.info("STARTING DIALOGUE TRAINING - STAGE 2.1")
    logger.info("=" * 60)
    
    # 1. Подготовка окружения
    device, results_dir = setup_training_environment()
    
    # 2. Создание dialogue dataset
    logger.info("Creating DialogueDataset...")
    
    if args.use_sample_data:
        # Используем sample данные для тестирования
        dialogue_pairs = create_sample_dialogue_data()
        logger.info(f"   Using {len(dialogue_pairs)} sample dialogue pairs")
    else:
        # TODO: Загрузка реальных диалоговых данных
        dialogue_pairs = create_sample_dialogue_data()  # Пока используем sample
        logger.warning("   Real dialogue data loading not implemented yet - using sample data")
    
    # Создание dataset
    dataset = create_dialogue_dataset(
        dialogue_pairs=dialogue_pairs,
        teacher_model=args.teacher_model,
        validation_split=0.2,
        embedding_dim=768,
        enable_quality_filter=True,
        use_cache=True
    )
    
    # 3. Создание CubeTrainer
    logger.info("Initializing CubeTrainer...")
    config = create_training_config(args)
    trainer = CubeTrainer(config=config)
    
    # Инициализация компонентов
    trainer.initialize_components()
    
    # 4. Подготовка данных
    logger.info("Preparing training data...")
    
    # Training DataLoader
    train_loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        validation=False
    )
    
    # Validation DataLoader
    val_loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        validation=True
    )
    
    logger.info(f"   Training batches: {len(train_loader)}")
    logger.info(f"   Validation batches: {len(val_loader)}")
    
    # 5. Training Loop
    logger.info("Starting dialogue training loop...")
    
    training_history = []
    best_val_similarity = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\n[CHART] EPOCH {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Training phase
        trainer.embedding_processor.train()
        epoch_train_loss = 0.0
        epoch_train_similarity = 0.0
        train_batches = 0
        
        for batch_idx, (question_emb, answer_emb) in enumerate(train_loader):
            batch_size = question_emb.size(0)
            
            # Обрабатываем каждый элемент batch'а отдельно
            batch_processed_embs = []
            batch_losses = []
            
            for i in range(batch_size):
                 # Forward pass для одного эмбединга: question_embedding → processed_embedding
                 single_question = question_emb[i:i+1].squeeze(0).to(device)  # [768]
                 single_answer = answer_emb[i:i+1].squeeze(0).to(device)      # [768]
                 
                 # Убеждаемся что requires_grad=True
                 single_question = single_question.requires_grad_(True)
                 
                 processed_emb = trainer.forward(single_question)  # [768]
                 batch_processed_embs.append(processed_emb)
                 
                 # Loss для одного примера
                 cosine_sim = torch.nn.functional.cosine_similarity(
                     processed_emb.unsqueeze(0), single_answer.unsqueeze(0), dim=1
                 )
                 single_loss = 1.0 - cosine_sim
                 batch_losses.append(single_loss)
            
            # Объединяем результаты batch'а
            batch_processed_tensor = torch.stack(batch_processed_embs)  # [batch_size, 768]
            batch_loss_tensor = torch.stack(batch_losses).mean()        # средний loss по batch'у
            
            # Backward pass
            trainer.optimizer.zero_grad()
            batch_loss_tensor.backward()
            trainer.optimizer.step()
            
            # Метрики для логирования
            with torch.no_grad():
                batch_similarity = trainer.metrics.calculate_cosine_similarity(
                    batch_processed_tensor, answer_emb.to(device)
                )
            
            epoch_train_loss += batch_loss_tensor.item()
            epoch_train_similarity += batch_similarity
            train_batches += 1
            
            if batch_idx % args.log_interval == 0:
                logger.info(f"   Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={batch_loss_tensor.item():.4f}, Similarity={batch_similarity:.4f}")
        
        # Validation phase
        trainer.embedding_processor.eval()
        val_loss = 0.0
        val_similarity = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for question_emb, answer_emb in val_loader:
                batch_size = question_emb.size(0)
                
                # Обрабатываем каждый элемент batch'а отдельно
                batch_processed_embs = []
                
                for i in range(batch_size):
                    single_question = question_emb[i:i+1].squeeze(0).to(device)  # [768]
                    processed_emb = trainer.forward(single_question)  # [768]
                    batch_processed_embs.append(processed_emb)
                
                # Объединяем результаты batch'а
                batch_processed_tensor = torch.stack(batch_processed_embs)  # [batch_size, 768]
                
                # Метрики для validation
                batch_similarity = trainer.metrics.calculate_cosine_similarity(
                    batch_processed_tensor, answer_emb.to(device)
                )
                batch_loss = 1.0 - batch_similarity
                
                val_loss += batch_loss
                val_similarity += batch_similarity
                val_batches += 1
        
        # Средние метрики
        avg_train_loss = epoch_train_loss / train_batches
        avg_train_similarity = epoch_train_similarity / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_similarity = val_similarity / val_batches
        
        # Логирование эпохи
        logger.info(f"EPOCH {epoch + 1} RESULTS:")
        logger.info(f"   Train: Loss={avg_train_loss:.4f}, Similarity={avg_train_similarity:.4f}")
        logger.info(f"   Val:   Loss={avg_val_loss:.4f}, Similarity={avg_val_similarity:.4f}")
        
        # История обучения
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_similarity': avg_train_similarity,
            'val_loss': avg_val_loss,
            'val_similarity': avg_val_similarity
        })
        
        # Early stopping
        if avg_val_similarity > best_val_similarity:
            best_val_similarity = avg_val_similarity
            patience_counter = 0
            
            # Сохранение лучшей модели
            checkpoint_path = results_dir / f"best_dialogue_model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.embedding_processor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_similarity': avg_val_similarity,
                'config': config
            }, checkpoint_path)
            logger.info(f"[SAVE] Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Проверка на достижение целевой similarity
        if avg_val_similarity >= config.target_similarity:
            logger.info(f"TARGET SIMILARITY ACHIEVED: {avg_val_similarity:.4f} >= {config.target_similarity}")
            break
        
        # Проверка early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # 6. Результаты обучения
    logger.info("\n[TROPHY] DIALOGUE TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"[TARGET] Best validation similarity: {best_val_similarity:.4f}")
    logger.info(f"[DATA] Total epochs: {len(training_history)}")
    logger.info(f"[GRADUATE] Target achieved: {'[OK] YES' if best_val_similarity >= config.target_similarity else '[ERROR] NO'}")
    
    # Сохранение результатов
    results_file = results_dir / f"dialogue_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'teacher_model': args.teacher_model,
            'target_similarity': config.target_similarity
        },
        'final_results': {
            'best_val_similarity': best_val_similarity,
            'total_epochs': len(training_history),
            'target_achieved': best_val_similarity >= config.target_similarity
        },
        'training_history': training_history,
        'dataset_info': dataset.get_statistics()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"[SAVE] Results saved: {results_file}")
    
    # Построение graphиков
    if args.plot_results:
        plot_training_results(training_history, results_dir)
    
    return best_val_similarity, training_history


def plot_training_results(training_history: List[Dict], results_dir: Path):
    """Построение графиков результатов обучения"""
    epochs = [h['epoch'] for h in training_history]
    train_loss = [h['train_loss'] for h in training_history]
    val_loss = [h['val_loss'] for h in training_history]
    train_sim = [h['train_similarity'] for h in training_history]
    val_sim = [h['val_similarity'] for h in training_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Similarity plot
    ax2.plot(epochs, train_sim, 'b-', label='Training Similarity', linewidth=2)
    ax2.plot(epochs, val_sim, 'r-', label='Validation Similarity', linewidth=2)
    ax2.axhline(y=0.80, color='g', linestyle='--', label='Target (0.80)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Training and Validation Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = results_dir / f"dialogue_training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"[DATA] Training plots saved: {plot_file}")
    plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Dialogue Training Script - Stage 2.1')
    
    # Training параметры
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate (default: 0.001)')
    
    # Модель parameters
    parser.add_argument('--teacher_model', type=str, default='distilbert', 
                       choices=['llama3-8b', 'mistral-7b', 'distilbert', 'bert-base'],
                       help='Teacher LLM model for embeddings (default: distilbert)')
    
    # Data параметры
    parser.add_argument('--use_sample_data', action='store_true', default=True,
                       help='Use sample dialogue data for testing (default: True)')
    
    # Logging и visualization
    parser.add_argument('--log_interval', type=int, default=5, 
                       help='Logging interval during training (default: 5)')
    parser.add_argument('--plot_results', action='store_true', default=True,
                       help='Generate training plots (default: True)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG mode enabled")
    
    # Информация о запуске
    logger.info("DIALOGUE TRAINING - STAGE 2.1 PHASE 3")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")  
    logger.info(f"   Teacher model: {args.teacher_model}")
    logger.info(f"   Sample data: {args.use_sample_data}")
    
    try:
        # Запуск обучения
        best_similarity, history = run_dialogue_training(args)
        
        # Итоговый отчет
        logger.info("\nDIALOGUE TRAINING STAGE 2.1 COMPLETED!")
        logger.info(f"Best Q->A similarity: {best_similarity:.4f}")
        logger.info(f"Target (0.80): {'ACHIEVED' if best_similarity >= 0.80 else 'NOT ACHIEVED'}")
        logger.info("Ready for Stage 2.2!")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main()) 