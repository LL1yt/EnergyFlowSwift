#!/usr/bin/env python3
"""
Simple Training Example - демонстрация EnergyTrainer
====================================================

Полный workflow обучения energy_flow архитектуры:
1. Конфигурация debug режима
2. Подготовка простых данных
3. Инициализация EnergyTrainer с text_bridge
4. Обучение и валидация
5. Анализ результатов

Пример использования:
python energy_flow/examples/simple_training.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import logging

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.training.energy_trainer import EnergyTrainer
from energy_flow.utils.logging import get_logger, DEBUG_TRAINING

logger = get_logger(__name__)


class SimpleTextDataset(Dataset):
    """
    Простой датасет для демонстрации
    Использует базовые пары вопрос-ответ для обучения text_bridge
    """
    
    def __init__(self, max_samples: int = 50):
        """
        Args:
            max_samples: Максимальное количество примеров
        """
        # Простые пары для обучения (вопрос -> ответ)
        self.data_pairs = [
            ("What is AI?", "Artificial Intelligence is the simulation of human intelligence."),
            ("How does machine learning work?", "Machine learning uses algorithms to find patterns in data."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers."),
            ("What is a neural network?", "A neural network is inspired by biological neural systems."),
            ("How do computers learn?", "Computers learn by processing data and adjusting parameters."),
            ("What is training data?", "Training data is used to teach machine learning models."),
            ("What is an algorithm?", "An algorithm is a set of rules for solving problems."),
            ("How does AI work?", "AI works by processing information and making decisions."),
            ("What is data science?", "Data science extracts insights from structured and unstructured data."),
            ("What is natural language processing?", "NLP helps computers understand and process human language."),
            ("What is computer vision?", "Computer vision enables machines to interpret visual information."),
            ("What is reinforcement learning?", "Reinforcement learning learns through interaction and rewards."),
            ("What is supervised learning?", "Supervised learning uses labeled examples to train models."),
            ("What is unsupervised learning?", "Unsupervised learning finds patterns in data without labels."),
            ("What is a model?", "A model is a mathematical representation of a process."),
            ("What is prediction?", "Prediction is forecasting future outcomes based on data."),
            ("What is classification?", "Classification assigns data points to predefined categories."),
            ("What is regression?", "Regression predicts continuous numerical values."),
            ("What is feature extraction?", "Feature extraction identifies relevant data characteristics."),
            ("What is optimization?", "Optimization finds the best solution to a problem."),
        ]
        
        # Дублируем данные если нужно больше примеров
        while len(self.data_pairs) < max_samples:
            self.data_pairs.extend(self.data_pairs[:min(20, max_samples - len(self.data_pairs))])
        
        # Ограничиваем до max_samples
        self.data_pairs = self.data_pairs[:max_samples]
        
        logger.log(DEBUG_TRAINING, f"SimpleTextDataset initialized with {len(self.data_pairs)} pairs")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_text, target_text = self.data_pairs[idx]
        return input_text, target_text


def create_simple_dataloader(batch_size: int = 4, max_samples: int = 50) -> DataLoader:
    """Создание DataLoader с простыми данными"""
    dataset = SimpleTextDataset(max_samples=max_samples)
    
    # Создаем CUDA generator заранее для правильной работы shuffle
    cuda_generator = torch.Generator(device='cuda') if torch.cuda.is_available() else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=cuda_generator,  # Передаем CUDA generator напрямую!
        num_workers=0,  # Избегаем multiprocessing для простоты
        pin_memory=False  # Отключаем для упрощения отладки
    )
    
    # Логирование для подтверждения исправления
    if torch.cuda.is_available() and hasattr(dataloader, 'generator') and dataloader.generator:
        print(f"✅ DataLoader generator device: {dataloader.generator.device}")
    
    return dataloader


def create_teacher_embeddings_loader(batch_size: int = 4, max_samples: int = 50):
    """Создает итератор с парами teacher embeddings для обучения куба"""
    
    class TeacherEmbeddingsDataset(Dataset):
        def __init__(self, max_samples: int):
            self.max_samples = max_samples
        
        def __len__(self):
            return self.max_samples
        
        def __getitem__(self, idx):
            # Генерируем пары teacher embeddings (768D)
            input_embedding = torch.randn(768, dtype=torch.float32)
            target_embedding = torch.randn(768, dtype=torch.float32)
            return input_embedding, target_embedding
    
    dataset = TeacherEmbeddingsDataset(max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device='cuda') if torch.cuda.is_available() else None,
        num_workers=0,
        pin_memory=False  # Отключаем для упрощения отладки
    )
    
    if torch.cuda.is_available() and hasattr(dataloader, 'generator') and dataloader.generator:
        print(f"✅ Teacher embeddings DataLoader generator device: {dataloader.generator.device}")
    
    return dataloader


def run_simple_training():
    """Запуск простого обучения для демонстрации"""
    print("🚀 Starting Simple Energy Flow Training Demo")
    print("=" * 50)
    
    # 1. Конфигурация debug режима
    config = create_debug_config()
    set_energy_config(config)
    
    print(f"📊 Configuration:")
    print(f"  - Lattice size: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    print(f"  - Text bridge: {config.text_bridge_enabled}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Device: {config.device}")
    print()
    
    # 2. Подготовка данных
    print("📁 Preparing data...")
    train_dataloader = create_simple_dataloader(
        batch_size=config.batch_size, 
        max_samples=40  # Небольшой датасет для debug
    )
    val_dataloader = create_simple_dataloader(
        batch_size=config.batch_size,
        max_samples=10  # Еще меньше для валидации
    )
    
    # Teacher embeddings для основного обучения куба
    train_teacher_loader = create_teacher_embeddings_loader(
        batch_size=config.batch_size,
        max_samples=40
    )
    val_teacher_loader = create_teacher_embeddings_loader(
        batch_size=config.batch_size,
        max_samples=10
    )
    
    print(f"  - Training batches: {len(train_dataloader)}")
    print(f"  - Validation batches: {len(val_dataloader)}")
    print(f"  - Teacher embeddings batches: {len(train_teacher_loader)}")
    print()
    
    # 3. Инициализация тренера
    print("🤖 Initializing EnergyTrainer...")
    try:
        trainer = EnergyTrainer(config)
        model_info = trainer.get_model_info()
        
        print(f"  - Flow processor parameters: {model_info.get('flow_processor_parameters', 0):,}")
        if config.text_bridge_enabled:
            print(f"  - Text encoder parameters: {model_info.get('text_encoder_parameters', 0):,}")
            print(f"  - Text decoder parameters: {model_info.get('text_decoder_parameters', 0):,}")
        print(f"  - Device: {model_info['device']}")
        print()
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return False
    
    # 4. Валидация перед обучением
    print("🔍 Initial validation...")
    try:
        # Берем первый batch для валидации
        val_batch = next(iter(val_dataloader))
        val_teacher_batch = next(iter(val_teacher_loader))
        val_metrics = trainer.validate(val_batch[0], val_batch[1], val_teacher_batch[0], val_teacher_batch[1])
        
        print(f"  - Initial loss: {val_metrics.get('total_loss', 0):.4f}")
        if val_metrics.get('examples'):
            example = val_metrics['examples'][0]
            print(f"  - Example input: '{example['input'][:40]}...'")
            if config.text_bridge_enabled:
                print(f"  - Example predicted: '{example.get('predicted', 'N/A')[:40]}...'")
        print()
        
    except Exception as e:
        print(f"⚠️ Initial validation failed: {e}")
        val_metrics = None
    
    # 5. Обучение
    print("🎯 Starting training...")
    try:
        num_epochs = 3  # Небольшое количество эпох для demo
        training_history = trainer.train(train_dataloader, train_teacher_loader, num_epochs=num_epochs)
        
        print(f"✅ Training completed!")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Final loss: {training_history.get('total_loss', [0])[-1]:.4f}")
        print(f"  - Best loss: {trainer.best_loss:.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Финальная валидация
    print("🔍 Final validation...")
    try:
        final_val_metrics = trainer.validate(val_batch[0], val_batch[1], val_teacher_batch[0], val_teacher_batch[1])
        
        print(f"  - Final loss: {final_val_metrics.get('total_loss', 0):.4f}")
        print(f"  - Energy loss: {final_val_metrics.get('energy_loss', 0):.4f}")
        print(f"  - Text loss: {final_val_metrics.get('text_loss', 0):.4f}")
        
        if final_val_metrics.get('examples') and config.text_bridge_enabled:
            example = final_val_metrics['examples'][0]
            print(f"\n📝 Example after training:")
            print(f"  Input:     '{example['input']}'")
            print(f"  Target:    '{example['target']}'")
            print(f"  Predicted: '{example.get('predicted', 'N/A')}'")
        print()
        
    except Exception as e:
        print(f"⚠️ Final validation failed: {e}")
    
    # 7. Сохранение модели
    print("💾 Saving model...")
    try:
        checkpoint_path = "simple_training_demo.pt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"  - Model saved: {checkpoint_path}")
        print()
        
    except Exception as e:
        print(f"⚠️ Model saving failed: {e}")
    
    # 8. Итоговая статистика
    print("📊 Training Summary:")
    print("=" * 50)
    if training_history:
        print(f"Total losses: {[f'{loss:.4f}' for loss in training_history.get('total_loss', [])]}")
        print(f"Energy losses: {[f'{loss:.4f}' for loss in training_history.get('energy_loss', [])]}")
        if config.text_bridge_enabled:
            print(f"Text losses: {[f'{loss:.4f}' for loss in training_history.get('text_loss', [])]}")
    
    print(f"Best loss achieved: {trainer.best_loss:.4f}")
    print(f"Configuration used: {config.lattice_width}x{config.lattice_height}x{config.lattice_depth}")
    print("✅ Demo completed successfully!")
    
    return True


def run_interactive_demo():
    """Интерактивная демонстрация с пользовательским вводом"""
    print("\n🎮 Interactive Demo Mode")
    print("=" * 30)
    
    # Быстрая инициализация для интерактивности
    config = create_debug_config()
    config.lattice_width = 10  # Еще меньше для скорости
    config.lattice_height = 10
    config.lattice_depth = 5
    set_energy_config(config)
    
    try:
        trainer = EnergyTrainer(config)
        print("🤖 EnergyTrainer initialized!")
        
        while True:
            print("\nEnter input text (or 'quit' to exit):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                # Простая проверка forward pass
                result = trainer.validate([user_input], ["Sample target response"])
                
                print(f"✅ Processing completed!")
                print(f"  - Loss: {result.get('total_loss', 0):.4f}")
                if result.get('examples') and config.text_bridge_enabled:
                    predicted = result['examples'][0].get('predicted', 'N/A')
                    print(f"  - Model output: '{predicted}'")
                
            except Exception as e:
                print(f"❌ Processing failed: {e}")
    
    except Exception as e:
        print(f"❌ Interactive demo initialization failed: {e}")
    
    print("👋 Interactive demo ended!")


if __name__ == "__main__":
    """Главная функция запуска"""
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🌟 Energy Flow Training Demo")
    print("=" * 40)
    print("Choose mode:")
    print("1. Simple Training Demo (recommended)")
    print("2. Interactive Demo")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice in ['1', '3']:
            success = run_simple_training()
            if not success:
                print("❌ Simple training demo failed!")
        
        if choice in ['2', '3']:
            run_interactive_demo()
        
        if choice not in ['1', '2', '3']:
            print("Invalid choice, running simple training demo...")
            run_simple_training()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()