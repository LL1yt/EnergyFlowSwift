#!/usr/bin/env python3
"""
Обучение с готовым датасетом
===========================

Демонстрирует использование предварительно сгенерированного датасета
для обучения EnergyTrainer без сложной интеграции.
"""

import sys
from pathlib import Path
import torch

# Добавляем корень проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.training import EnergyTrainer
from energy_flow.utils.logging import get_logger

logger = get_logger(__name__)

# Путь к готовому датасету
DATASET_PATH = "data/energy_flow/active/debug_precomputed_30pairs_20250729_110314.pt"


def load_dataset(dataset_path: str):
    """Загрузка готового датасета"""
    print(f"📁 Loading dataset from {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Загружаем датасет
    dataset = torch.load(dataset_path, map_location='cuda', weights_only=False)
    
    print(f"✅ Dataset loaded:")
    print(f"   Total samples: {len(dataset['text_pairs'])}")
    print(f"   Embedding dimension: {dataset['input_embeddings'].shape[1]}")
    print(f"   Generated: {dataset['generation_info']['generation_timestamp']}")
    print(f"   Sources: {', '.join(dataset['generation_info']['sources'])}")
    
    return dataset


def create_training_setup():
    """Создание настройки для обучения"""
    print("🔧 Setting up training environment...")
    
    # Energy конфигурация (debug режим)
    energy_config = create_debug_config()
    set_energy_config(energy_config)
    
    # EnergyTrainer
    trainer = EnergyTrainer(energy_config)
    
    return energy_config, trainer


def create_dataloader_from_dataset(dataset, batch_size: int = 4, shuffle: bool = True):
    """Создание DataLoader из готового датасета"""
    from torch.utils.data import DataLoader
    
    # Создаем wrapper для доступа к текстовым парам
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.length = len(dataset['text_pairs'])
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            input_text, target_text = self.dataset['text_pairs'][idx]
            return {
                'input_embedding': self.dataset['input_embeddings'][idx],
                'target_embedding': self.dataset['target_embeddings'][idx],
                'input_text': input_text,
                'target_text': target_text
            }
    
    wrapped_dataset = DatasetWrapper(dataset)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device=torch.get_default_device()) if shuffle else None,
        collate_fn=lambda batch: {
            'input_embedding': torch.stack([item['input_embedding'] for item in batch]),
            'target_embedding': torch.stack([item['target_embedding'] for item in batch]),
            'input_text': [item['input_text'] for item in batch],
            'target_text': [item['target_text'] for item in batch]
        }
    )
    
    return dataloader


def run_training_demo(dataset_path: str = DATASET_PATH, num_epochs: int = 2):
    """Демонстрация обучения с готовым датасетом"""
    print(f"🚀 Starting training demo ({num_epochs} epochs)")
    print("=" * 60)
    
    try:
        # Загрузка датасета
        print("\n1️⃣ Loading dataset...")
        dataset = load_dataset(dataset_path)
        
        # Настройка тренера
        print("\n2️⃣ Setting up trainer...")
        energy_config, trainer = create_training_setup()
        
        # Создание DataLoader
        print("\n3️⃣ Creating DataLoader...")
        dataloader = create_dataloader_from_dataset(
            dataset, 
            batch_size=energy_config.batch_size,
            shuffle=True
        )
        print(f"✅ DataLoader ready: {len(dataloader)} batches")
        
        # Тренировка
        print(f"\n4️⃣ Starting training ({num_epochs} epochs)...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            epoch_metrics = {
                'total_loss': 0.0,
                'energy_loss': 0.0,
                'text_loss': 0.0,
                'batches_processed': 0
            }
            
            # Проходим по батчам
            for batch_idx, batch in enumerate(dataloader):
                # Извлекаем данные
                input_texts = batch['input_text']
                target_texts = batch['target_text']
                input_embeddings = batch['input_embedding']
                target_embeddings = batch['target_embedding']
                
                # Один шаг обучения
                step_metrics = trainer.train_step(
                    input_texts=input_texts,
                    target_texts=target_texts,
                    teacher_input_embeddings=input_embeddings,
                    teacher_target_embeddings=target_embeddings
                )
                
                # Аккумулируем метрики
                epoch_metrics['total_loss'] += step_metrics.get('total_loss', 0)
                epoch_metrics['energy_loss'] += step_metrics.get('energy_loss', 0)
                epoch_metrics['text_loss'] += step_metrics.get('text_loss', 0)
                epoch_metrics['batches_processed'] += 1
                
                # Логирование каждого батча
                if batch_idx % 5 == 0:  # Каждый 5-й батч
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                          f"loss={step_metrics.get('total_loss', 0):.4f}")
                
                # Ограничиваем количество батчей для демо
                if batch_idx >= 10:  # Максимум 10 батчей
                    break
            
            # Усредняем метрики по эпохе
            if epoch_metrics['batches_processed'] > 0:
                for key in ['total_loss', 'energy_loss', 'text_loss']:
                    epoch_metrics[key] /= epoch_metrics['batches_processed']
            
            print(f"  Epoch {epoch + 1} completed:")
            print(f"    Total loss: {epoch_metrics['total_loss']:.4f}")
            print(f"    Energy loss: {epoch_metrics['energy_loss']:.4f}")
            print(f"    Text loss: {epoch_metrics['text_loss']:.4f}")
            print(f"    Batches processed: {epoch_metrics['batches_processed']}")
        
        # Валидация после обучения
        print(f"\n5️⃣ Running post-training validation...")
        
        # Берем несколько примеров для валидации
        val_batch = next(iter(dataloader))
        val_input_texts = val_batch['input_text'][:3]  # Первые 3 примера
        val_target_texts = val_batch['target_text'][:3]
        val_input_embeddings = val_batch['input_embedding'][:3]
        val_target_embeddings = val_batch['target_embedding'][:3]
        
        val_results = trainer.validate(
            input_texts=val_input_texts,
            target_texts=val_target_texts,
            teacher_input_embeddings=val_input_embeddings,
            teacher_target_embeddings=val_target_embeddings
        )
        
        print(f"✅ Validation completed:")
        print(f"   Validation loss: {val_results.get('total_loss', 'N/A'):.4f}")
        print(f"   Examples generated: {len(val_results.get('examples', []))}")
        
        # Показываем примеры предсказаний
        if val_results.get('examples'):
            print(f"\n📝 Prediction examples:")
            for i, example in enumerate(val_results['examples'][:2]):
                print(f"   Example {i+1}:")
                print(f"     Input: '{example['input'][:80]}...'")
                print(f"     Target: '{example['target'][:80]}...'")
                print(f"     Predicted: '{example['predicted'][:80]}...'")
        
        print(f"\n🎉 Training demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading_only(dataset_path: str = DATASET_PATH):
    """Тест только загрузки датасета без обучения"""
    print("\n🧪 Testing Dataset Loading Only")
    print("-" * 40)
    
    try:
        # Загрузка датасета
        dataset = load_dataset(dataset_path)
        
        # Создание DataLoader
        dataloader = create_dataloader_from_dataset(dataset, batch_size=2)
        
        # Получаем один батч
        batch = next(iter(dataloader))
        
        print(f"✅ Sample batch loaded:")
        print(f"   Batch size: {len(batch['input_text'])}")
        print(f"   Input embeddings shape: {batch['input_embedding'].shape}")
        print(f"   Target embeddings shape: {batch['target_embedding'].shape}")
        print(f"   Sample input text: '{batch['input_text'][0][:50]}...'")
        print(f"   Sample target text: '{batch['target_text'][0][:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training with pre-generated dataset")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH,
                       help="Path to dataset file")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test dataset loading without training")
    
    args = parser.parse_args()
    
    try:
        if args.test_only:
            # Только тест загрузки
            success = test_dataset_loading_only(args.dataset)
        else:
            # Полное обучение
            success = run_training_demo(args.dataset, args.epochs)
            
            if success:
                # Дополнительный тест загрузки
                test_dataset_loading_only(args.dataset)
                
                print(f"\n✨ Training completed successfully!")
                print(f"   Dataset: {args.dataset}")
                print(f"   Epochs: {args.epochs}")
        
        if success:
            print(f"\n🎯 Ready for production use!")
        else:
            print(f"\n⚠️  Issues found, check logs for debugging.")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()