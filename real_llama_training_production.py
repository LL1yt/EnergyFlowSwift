#!/usr/bin/env python3
"""
🎯 Production Real Training: LLaMA-3-8B + 3D Cellular Neural Network

ЦЕЛЬ: Полноценное реальное обучение системы с:
- Поэтапным прогрессом (validation → training → production)
- Чекпоинтами и восстановлением
- Комплексными метриками и диагностикой  
- Долгосрочным обучением с мониторингом
- Автоматическим анализом результатов

ЭТАПЫ:
1. System Validation - проверка что все работает
2. Convergence Testing - проверка схождения на малых данных  
3. Production Training - долгосрочное обучение
4. Result Analysis - анализ и принятие решений
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_training_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from training.embedding_trainer.emergent_training_stage_3_1_4_1 import (
    EmergentCubeTrainer, EmergentTrainingConfig
)
from training.embedding_trainer.dialogue_dataset import DialogueDataset, create_dialogue_dataset
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter
from utils.llm_handler import LLMHandler

@dataclass
class TrainingStage:
    """Represents a training stage with its configuration and targets"""
    name: str
    description: str
    epochs: int
    target_loss: float
    target_similarity: float
    batch_size: int
    learning_rate: float
    save_checkpoints: bool = True
    early_stopping_patience: int = 10

class ProductionTrainingManager:
    """
    Управляет полным циклом production обучения:
    - Валидация системы
    - Поэтапное обучение  
    - Чекпоинты и восстановление
    - Метрики и диагностика
    """
    
    def __init__(self, config_path: str = "config/emergent_training_3_1_4_1.yaml"):
        self.config_path = config_path
        self.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(f"checkpoints/real_training_{self.training_id}")
        self.results_dir = Path(f"results/real_training_{self.training_id}")
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training stages configuration
        self.stages = self._configure_training_stages()
        
        # Results tracking
        self.training_history = {
            'stages': [],
            'best_metrics': {'loss': float('inf'), 'similarity': 0.0},
            'total_time': 0,
            'decisions': []
        }
        
        logger.info(f"🚀 Production Training Manager initialized")
        logger.info(f"Training ID: {self.training_id}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        logger.info(f"Results: {self.results_dir}")
    
    def _configure_training_stages(self) -> List[TrainingStage]:
        """Конфигурация этапов обучения"""
        return [
            TrainingStage(
                name="validation",
                description="System validation - проверка что все компоненты работают",
                epochs=3,
                target_loss=1.0,  # Достижимая для validation
                target_similarity=0.15,  # Минимальный прогресс
                batch_size=2,
                learning_rate=0.001
            ),
            TrainingStage(
                name="convergence",
                description="Convergence testing - проверка схождения на средних данных",
                epochs=10,
                target_loss=0.5,
                target_similarity=0.25,
                batch_size=4,
                learning_rate=0.0005
            ),
            TrainingStage(
                name="production",
                description="Production training - долгосрочное обучение для достижения целей",
                epochs=50,
                target_loss=0.3,
                target_similarity=0.40,  # Realistic target based on current 38.5%
                batch_size=8,
                learning_rate=0.0003,
                early_stopping_patience=15
            ),
            TrainingStage(
                name="optimization",
                description="Final optimization - fine-tuning для максимальной производительности",
                epochs=25,
                target_loss=0.2,
                target_similarity=0.45,
                batch_size=6,
                learning_rate=0.0001,
                early_stopping_patience=10
            )
        ]
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """
        Запускает полный pipeline обучения:
        1. System validation
        2. Progressive training через все stages
        3. Results analysis и decision making
        """
        logger.info("🎯 Starting Full Production Training Pipeline")
        
        try:
            # 1. System Validation
            logger.info("=" * 60)
            logger.info("STAGE 1: SYSTEM VALIDATION")
            logger.info("=" * 60)
            
            validation_success = self._validate_system()
            if not validation_success:
                logger.error("❌ System validation failed. Aborting training.")
                return {'status': 'failed', 'stage': 'validation'}
            
            # 2. Progressive Training
            logger.info("=" * 60)
            logger.info("STAGE 2: PROGRESSIVE TRAINING")
            logger.info("=" * 60)
            
            for stage in self.stages:
                logger.info(f"🚀 Starting stage: {stage.name}")
                logger.info(f"📋 {stage.description}")
                
                stage_result = self._run_training_stage(stage)
                
                if not stage_result['success']:
                    logger.warning(f"⚠️  Stage {stage.name} did not meet targets")
                    decision = self._analyze_stage_failure(stage, stage_result)
                    
                    if decision == 'abort':
                        logger.error(f"❌ Training aborted at stage {stage.name}")
                        return {'status': 'aborted', 'stage': stage.name, 'results': self.training_history}
                    elif decision == 'continue':
                        logger.info(f"▶️  Continuing to next stage despite targets not met")
                    elif decision == 'retry':
                        logger.info(f"🔄 Retrying stage {stage.name} with adjusted parameters")
                        # TODO: Implement retry logic
                        pass
                
                logger.info(f"✅ Stage {stage.name} completed")
            
            # 3. Results Analysis
            logger.info("=" * 60)
            logger.info("STAGE 3: RESULTS ANALYSIS")
            logger.info("=" * 60)
            
            final_analysis = self._analyze_final_results()
            
            # 4. Save complete results
            self._save_complete_results(final_analysis)
            
            logger.info("🎉 Full training pipeline completed successfully!")
            return {
                'status': 'completed',
                'training_id': self.training_id,
                'final_metrics': final_analysis['metrics'],
                'recommendations': final_analysis['recommendations']
            }
            
        except Exception as e:
            logger.error(f"💥 Training pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    def _validate_system(self) -> bool:
        """Валидация всех компонентов системы"""
        logger.info("🔍 Validating system components...")
        
        try:
            # 1. Check LLaMA-3-8B availability
            logger.info("1️⃣ Testing LLaMA-3-8B model access...")
            llm_handler = LLMHandler('llama3-8b-local')
            test_embedding = llm_handler.generate_embedding("Test validation text")
            logger.info(f"✅ LLaMA-3-8B working: {test_embedding.shape}")
            
            # 2. Check Universal Adapter
            logger.info("2️⃣ Testing Universal Adapter...")
            adapter = UniversalEmbeddingAdapter(
                input_dim=4096,
                output_shape=(15, 15),
                strategy='hierarchical'
            )
            test_surface = adapter.forward(test_embedding.unsqueeze(0))
            logger.info(f"✅ Universal Adapter working: {test_surface.shape}")
            
            # 3. Check EmergentCubeTrainer
            logger.info("3️⃣ Testing EmergentCubeTrainer...")
            config = EmergentTrainingConfig()
            config.cube_dimensions = (15, 15, 1)  # Minimal для validation
            config.batch_size = 1
            config.epochs = 1
            
            trainer = EmergentCubeTrainer(config, device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"✅ EmergentCubeTrainer initialized on {trainer.device}")
            
            # 4. Check dataset creation
            logger.info("4️⃣ Testing dataset creation...")
            test_dialogue = [
                {"question": "What is AI?", "answer": "Artificial Intelligence is machine learning."}
            ]
            
            dataset = create_dialogue_dataset(
                test_dialogue,
                teacher_model="llama3-8b-local",
                cache_embeddings=False,
                validation_split=0.0
            )
            logger.info(f"✅ Dataset created: {len(dataset)} pairs")
            
            # 5. Test mini training step
            logger.info("5️⃣ Testing mini training step...")
            sample = dataset[0]
            
            # Convert to appropriate format for training
            if isinstance(sample, tuple):
                input_emb, target_emb = sample
                # Adapt to surface format
                input_surface = adapter.forward(input_emb.unsqueeze(0))
                target_surface = adapter.forward(target_emb.unsqueeze(0))
                
                # Single training step
                metrics = trainer.train_step(input_surface, target_surface)
                logger.info(f"✅ Training step successful: loss = {metrics.get('loss', 'N/A')}")
            
            logger.info("🎉 System validation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_training_stage(self, stage: TrainingStage) -> Dict[str, Any]:
        """Запуск одного этапа обучения"""
        logger.info(f"🏃 Running training stage: {stage.name}")
        
        stage_start_time = time.time()
        
        # Setup configuration for this stage
        config = EmergentTrainingConfig()
        config.epochs = stage.epochs
        config.batch_size = stage.batch_size
        config.learning_rate = stage.learning_rate
        config.cube_dimensions = (15, 15, 11)  # Full size для real training
        config.mixed_precision = torch.cuda.is_available()
        config.enable_nca = True  # Enable NCA для emergent behavior
        
        # Initialize trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = EmergentCubeTrainer(config, device=device)
        
        # Create enhanced dataset for this stage
        dataset = self._create_enhanced_dataset(stage)
        
        # Initialize tracking
        stage_metrics = {
            'losses': [],
            'similarities': [],
            'epoch_times': [],
            'best_loss': float('inf'),
            'best_similarity': 0.0,
            'converged': False,
            'early_stopped': False
        }
        
        # Training loop with checkpointing
        patience_counter = 0
        
        for epoch in range(stage.epochs):
            epoch_start = time.time()
            epoch_losses = []
            epoch_similarities = []
            
            logger.info(f"  📈 Epoch {epoch + 1}/{stage.epochs}")
            
            # Training batches
            num_batches = len(dataset) // stage.batch_size
            
            for batch_idx in range(num_batches):
                try:
                    # Prepare batch
                    batch_data = self._prepare_batch(dataset, batch_idx, stage.batch_size)
                    
                    # Training step
                    metrics = trainer.train_step(*batch_data)
                    
                    epoch_losses.append(metrics.get('loss', 0.0))
                    epoch_similarities.append(metrics.get('similarity', 0.0))
                    
                except Exception as e:
                    logger.warning(f"    ⚠️  Batch {batch_idx} failed: {e}")
                    continue
            
            # Epoch metrics
            epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            epoch_similarity = np.mean(epoch_similarities) if epoch_similarities else 0.0
            epoch_time = time.time() - epoch_start
            
            stage_metrics['losses'].append(epoch_loss)
            stage_metrics['similarities'].append(epoch_similarity)
            stage_metrics['epoch_times'].append(epoch_time)
            
            # Track best metrics
            if epoch_loss < stage_metrics['best_loss']:
                stage_metrics['best_loss'] = epoch_loss
                patience_counter = 0
                
                # Save checkpoint
                if stage.save_checkpoints:
                    self._save_checkpoint(trainer, stage, epoch, epoch_loss, epoch_similarity)
            else:
                patience_counter += 1
            
            if epoch_similarity > stage_metrics['best_similarity']:
                stage_metrics['best_similarity'] = epoch_similarity
            
            logger.info(f"    📊 Loss: {epoch_loss:.4f}, Similarity: {epoch_similarity:.4f}, Time: {epoch_time:.1f}s")
            
            # Early stopping check
            if patience_counter >= stage.early_stopping_patience:
                logger.info(f"    ⏹️  Early stopping triggered at epoch {epoch + 1}")
                stage_metrics['early_stopped'] = True
                break
            
            # Convergence check
            if epoch_loss <= stage.target_loss and epoch_similarity >= stage.target_similarity:
                logger.info(f"    🎯 Stage targets achieved at epoch {epoch + 1}!")
                stage_metrics['converged'] = True
                break
        
        # Stage completion
        stage_time = time.time() - stage_start_time
        stage_metrics['total_time'] = stage_time
        
        # Determine success
        success = (
            stage_metrics['best_loss'] <= stage.target_loss or
            stage_metrics['best_similarity'] >= stage.target_similarity
        )
        
        stage_result = {
            'stage': stage.name,
            'success': success,
            'metrics': stage_metrics,
            'config': asdict(stage)
        }
        
        # Update training history
        self.training_history['stages'].append(stage_result)
        
        if stage_metrics['best_loss'] < self.training_history['best_metrics']['loss']:
            self.training_history['best_metrics']['loss'] = stage_metrics['best_loss']
        
        if stage_metrics['best_similarity'] > self.training_history['best_metrics']['similarity']:
            self.training_history['best_metrics']['similarity'] = stage_metrics['best_similarity']
        
        self.training_history['total_time'] += stage_time
        
        logger.info(f"✅ Stage {stage.name} completed in {stage_time:.1f}s")
        logger.info(f"    🎯 Best Loss: {stage_metrics['best_loss']:.4f} (target: {stage.target_loss})")
        logger.info(f"    🎯 Best Similarity: {stage_metrics['best_similarity']:.4f} (target: {stage.target_similarity})")
        
        return stage_result
    
    def _create_enhanced_dataset(self, stage: TrainingStage) -> List:
        """Создает dataset подходящий для данного этапа"""
        
        if stage.name == "validation":
            # Minimal dataset для validation
            dialogue_pairs = [
                {"question": "What is AI?", "answer": "AI is artificial intelligence technology."},
                {"question": "How do neural networks work?", "answer": "Neural networks process data through connected layers."}
            ]
        elif stage.name == "convergence":
            # Medium dataset для convergence testing
            dialogue_pairs = self._get_medium_dialogue_dataset()
        else:  # production, optimization
            # Full enhanced dataset
            dialogue_pairs = self._get_full_dialogue_dataset()
        
        return create_dialogue_dataset(
            dialogue_pairs,
            teacher_model="llama3-8b-local",
            cache_embeddings=True,  # Enable caching для production
            validation_split=0.0,
            normalize_embeddings=True
        )
    
    def _get_medium_dialogue_dataset(self) -> List[Dict[str, str]]:
        """Medium-size dataset для convergence testing"""
        return [
            # Neural Networks & Deep Learning
            {"question": "What is a neural network?", "answer": "A computational model inspired by biological neural networks with interconnected processing nodes."},
            {"question": "How does backpropagation work?", "answer": "An algorithm that calculates gradients by propagating errors backward through network layers."},
            {"question": "What is overfitting?", "answer": "When a model learns training data too well, including noise, reducing generalization ability."},
            {"question": "What are activation functions?", "answer": "Mathematical functions that determine neuron output, introducing non-linearity to networks."},
            {"question": "How do convolutional layers work?", "answer": "They apply filters to detect local features in input data through convolution operations."},
            
            # 3D & Cellular Processing
            {"question": "What are cellular neural networks?", "answer": "Arrays of locally connected processing units performing complex computations through emergent behavior."},
            {"question": "How do 3D neural networks differ?", "answer": "They process volumetric data with depth information for spatial reasoning in three dimensions."},
            {"question": "What is emergent behavior?", "answer": "Complex patterns arising from simple interactions between individual network components."},
            {"question": "What is spatial reasoning?", "answer": "The ability to understand and manipulate spatial relationships between objects."},
            {"question": "How do cellular automata work?", "answer": "Discrete models with cells evolving based on local rules and neighbor states."},
            
            # Training & Optimization
            {"question": "Why use GPU training?", "answer": "GPUs have thousands of parallel cores optimized for matrix operations and simultaneous computation."},
            {"question": "What is mixed precision?", "answer": "Training technique using both 16-bit and 32-bit floating point to reduce memory and accelerate training."},
            {"question": "How does batch size affect training?", "answer": "Larger batches provide stable gradients and better GPU utilization but require more memory."},
            {"question": "What is learning rate scheduling?", "answer": "Adjusting learning rate during training to improve convergence and avoid local minima."},
            {"question": "What are optimization algorithms?", "answer": "Methods like Adam, SGD that update network parameters to minimize training loss."}
        ]
    
    def _get_full_dialogue_dataset(self) -> List[Dict[str, str]]:
        """Full comprehensive dataset для production training"""
        medium_dataset = self._get_medium_dialogue_dataset()
        
        # Add advanced topics
        advanced_pairs = [
            # Advanced AI & ML
            {"question": "What are attention mechanisms?", "answer": "Neural network components that focus on relevant input parts, enabling better sequence processing."},
            {"question": "How do transformers work?", "answer": "Models using self-attention to process sequences in parallel, capturing long-range dependencies."},
            {"question": "What is transfer learning?", "answer": "Using pre-trained models as starting points for new tasks, leveraging learned features."},
            {"question": "What are generative models?", "answer": "AI systems that learn to create new data samples similar to training data."},
            {"question": "How does reinforcement learning work?", "answer": "Learning through interaction with environment, receiving rewards and penalties for actions."},
            
            # Research & Advanced Topics
            {"question": "What is meta-learning?", "answer": "Learning algorithms that learn how to learn, adapting quickly to new tasks."},
            {"question": "What are neural architecture search?", "answer": "Automated methods for discovering optimal neural network architectures."},
            {"question": "How do graph neural networks work?", "answer": "Networks processing graph-structured data by passing messages between connected nodes."},
            {"question": "What is continual learning?", "answer": "Learning new tasks without forgetting previously learned knowledge, avoiding catastrophic forgetting."},
            {"question": "What are foundation models?", "answer": "Large pre-trained models serving as basis for various downstream applications."},
            
            # Practical Applications
            {"question": "How is AI used in healthcare?", "answer": "Medical diagnosis, drug discovery, treatment planning, and personalized medicine applications."},
            {"question": "What is computer vision?", "answer": "AI field enabling machines to interpret and understand visual information from images and videos."},
            {"question": "How does natural language processing work?", "answer": "Computational techniques for analyzing, understanding, and generating human language text."},
            {"question": "What are recommendation systems?", "answer": "AI systems that predict user preferences and suggest relevant content or products."},
            {"question": "How is AI used in robotics?", "answer": "Enabling robots to perceive, plan, learn, and interact with physical environments."}
        ]
        
        return medium_dataset + advanced_pairs
    
    def _prepare_batch(self, dataset, batch_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Подготовка batch данных для training"""
        
        # Collect batch samples
        batch_inputs = []
        batch_targets = []
        
        for i in range(batch_size):
            sample_idx = (batch_idx * batch_size + i) % len(dataset)
            sample = dataset[sample_idx]
            
            if isinstance(sample, tuple) and len(sample) == 2:
                input_emb, target_emb = sample
                batch_inputs.append(input_emb)
                batch_targets.append(target_emb)
        
        # Convert to tensors and adapt to surface format
        batch_input_tensor = torch.stack(batch_inputs)
        batch_target_tensor = torch.stack(batch_targets)
        
        # Use Universal Adapter для conversion to surface format
        adapter = UniversalEmbeddingAdapter(
            input_dim=batch_input_tensor.shape[-1],
            output_shape=(15, 15),
            strategy='hierarchical'
        )
        
        input_surfaces = adapter.forward(batch_input_tensor)
        target_surfaces = adapter.forward(batch_target_tensor)
        
        return input_surfaces, target_surfaces
    
    def _save_checkpoint(self, trainer, stage: TrainingStage, epoch: int, loss: float, similarity: float):
        """Сохранение checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{stage.name}_epoch_{epoch}_loss_{loss:.4f}.pt"
        
        checkpoint = {
            'training_id': self.training_id,
            'stage': stage.name,
            'epoch': epoch,
            'loss': loss,
            'similarity': similarity,
            'model_state_dict': trainer.state_dict(),
            'config': asdict(stage),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _analyze_stage_failure(self, stage: TrainingStage, result: Dict[str, Any]) -> str:
        """Анализ неудачного этапа и принятие решения"""
        logger.info(f"🔍 Analyzing stage failure: {stage.name}")
        
        metrics = result['metrics']
        best_loss = metrics['best_loss']
        best_similarity = metrics['best_similarity']
        
        # Decision logic
        if stage.name == "validation":
            # Validation failure is critical
            if best_loss > 2.0 or best_similarity < 0.05:
                logger.error("❌ Critical validation failure - system not functional")
                return 'abort'
            else:
                logger.warning("⚠️  Validation targets not met but system functional")
                return 'continue'
        
        elif stage.name == "convergence":
            # Check if there's any learning happening
            if len(metrics['losses']) > 5:
                initial_loss = np.mean(metrics['losses'][:3])
                final_loss = np.mean(metrics['losses'][-3:])
                improvement = (initial_loss - final_loss) / initial_loss
                
                if improvement > 0.1:  # 10% improvement
                    logger.info(f"📈 Learning detected ({improvement:.1%} loss improvement)")
                    return 'continue'
                else:
                    logger.warning("⚠️  No significant learning detected")
                    return 'continue'  # Still continue но с warning
            else:
                return 'continue'
        
        else:  # production, optimization
            # More lenient for later stages
            return 'continue'
    
    def _analyze_final_results(self) -> Dict[str, Any]:
        """Финальный анализ результатов обучения"""
        logger.info("📊 Analyzing final training results...")
        
        final_metrics = self.training_history['best_metrics']
        total_time = self.training_history['total_time']
        
        # Performance analysis
        performance_analysis = {
            'final_loss': final_metrics['loss'],
            'final_similarity': final_metrics['similarity'],
            'total_training_time': total_time,
            'stages_completed': len(self.training_history['stages']),
            'convergence_achieved': final_metrics['loss'] < 0.3,
            'similarity_target_met': final_metrics['similarity'] >= 0.40
        }
        
        # Generate recommendations
        recommendations = []
        
        if performance_analysis['similarity_target_met']:
            recommendations.append("🎉 Similarity target achieved! System ready for production.")
        elif final_metrics['similarity'] >= 0.35:
            recommendations.append("📈 Good progress made. Consider extended training or hyperparameter tuning.")
        else:
            recommendations.append("🔧 Limited progress. Review architecture, dataset quality, or training approach.")
        
        if performance_analysis['convergence_achieved']:
            recommendations.append("✅ Loss convergence successful.")
        else:
            recommendations.append("⚠️  Loss convergence incomplete. Consider longer training or learning rate adjustment.")
        
        # Create visualizations
        self._create_training_visualizations()
        
        return {
            'metrics': performance_analysis,
            'recommendations': recommendations,
            'training_history': self.training_history
        }
    
    def _create_training_visualizations(self):
        """Создание графиков обучения"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect all metrics
        all_losses = []
        all_similarities = []
        stage_boundaries = [0]
        
        for stage_result in self.training_history['stages']:
            stage_losses = stage_result['metrics']['losses']
            stage_similarities = stage_result['metrics']['similarities']
            
            all_losses.extend(stage_losses)
            all_similarities.extend(stage_similarities)
            stage_boundaries.append(len(all_losses))
        
        # Plot 1: Loss over time
        axes[0, 0].plot(all_losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add stage boundaries
        for boundary in stage_boundaries[1:-1]:
            axes[0, 0].axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Similarity over time
        axes[0, 1].plot(all_similarities, 'g-', linewidth=2)
        axes[0, 1].set_title('Q→A Similarity Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Similarity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add stage boundaries
        for boundary in stage_boundaries[1:-1]:
            axes[0, 1].axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
        
        # Plot 3: Stage comparison
        stage_names = [stage['stage'] for stage in self.training_history['stages']]
        best_losses = [stage['metrics']['best_loss'] for stage in self.training_history['stages']]
        best_similarities = [stage['metrics']['best_similarity'] for stage in self.training_history['stages']]
        
        x_pos = np.arange(len(stage_names))
        
        axes[1, 0].bar(x_pos, best_losses, alpha=0.7, color='blue')
        axes[1, 0].set_title('Best Loss by Stage')
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Best Loss')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(stage_names, rotation=45)
        
        axes[1, 1].bar(x_pos, best_similarities, alpha=0.7, color='green')
        axes[1, 1].set_title('Best Similarity by Stage')
        axes[1, 1].set_xlabel('Stage')
        axes[1, 1].set_ylabel('Best Similarity')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(stage_names, rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'training_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Training visualization saved: {plot_path}")
        
        plt.close()
    
    def _save_complete_results(self, analysis: Dict[str, Any]):
        """Сохранение полных результатов"""
        results_file = self.results_dir / 'complete_results.json'
        
        complete_results = {
            'training_id': self.training_id,
            'timestamp': datetime.now().isoformat(),
            'final_analysis': analysis,
            'training_history': self.training_history,
            'system_info': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Complete results saved: {results_file}")

def main():
    """Main entry point для production training"""
    logger.info("🚀 Starting Production Real Training")
    
    # Initialize training manager
    manager = ProductionTrainingManager()
    
    # Run full training pipeline
    results = manager.run_full_training_pipeline()
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Status: {results['status']}")
    logger.info(f"Training ID: {results.get('training_id', 'N/A')}")
    
    if 'final_metrics' in results:
        metrics = results['final_metrics']
        logger.info(f"Final Loss: {metrics['final_loss']:.4f}")
        logger.info(f"Final Similarity: {metrics['final_similarity']:.4f}")
        logger.info(f"Total Time: {metrics['total_training_time']:.1f}s")
    
    if 'recommendations' in results:
        logger.info("Recommendations:")
        for rec in results['recommendations']:
            logger.info(f"  {rec}")
    
    return results

if __name__ == "__main__":
    main() 