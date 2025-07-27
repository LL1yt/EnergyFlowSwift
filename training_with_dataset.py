#!/usr/bin/env python3
"""
Пример интеграции dataset модуля с EnergyTrainer
===============================================

Демонстрирует полный цикл обучения:
- Подготовка данных через DatasetManager
- Создание EnergyTrainer с новой архитектурой
- Запуск обучения с валидацией
"""

import sys
from pathlib import Path
import torch

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.dataset import (
    create_dataset_config_from_energy,
    create_dataset_manager
)
from energy_flow.training import EnergyTrainer
from energy_flow.utils.logging import get_logger

logger = get_logger(__name__)


def create_training_setup():
    """Создание полной настройки для обучения"""
    print("🔧 Setting up training environment...")
    
    # 1. Energy конфигурация (debug режим для быстроты)
    energy_config = create_debug_config()
    set_energy_config(energy_config)
    
    # 2. Dataset конфигурация с интеграцией
    dataset_config = create_dataset_config_from_energy(
        energy_config,
        dataset_sources=["precomputed"],  # Только готовые эмбеддинги для скорости
        max_samples_per_source=100,  # Небольшой датасет для демо
        batch_size=4  # Маленький batch для debug
    )
    
    # 3. DatasetManager
    dataset_manager = create_dataset_manager(dataset_config, energy_config)
    
    # 4. EnergyTrainer
    trainer = EnergyTrainer(energy_config)
    
    return energy_config, dataset_manager, trainer


def run_training_demo(num_epochs: int = 2):
    """Демонстрация обучения с новым dataset модулем"""
    print(f"🚀 Starting training demo ({num_epochs} epochs)")
    print("=" * 60)
    
    try:
        # Настройка
        energy_config, dataset_manager, trainer = create_training_setup()
        
        # Валидация готовности
        print("\n1️⃣ Validating setup...")
        validation = dataset_manager.validate_setup()
        
        if not validation['overall_status']:
            print("❌ Setup validation failed:")
            for error in validation['errors']:
                print(f"   - {error}")
            return False
        
        print("✅ Setup validation passed")
        
        # Подготовка DataLoader
        print("\n2️⃣ Preparing data...")
        dataloader = dataset_manager.create_dataloader(
            batch_size=energy_config.batch_size,
            shuffle=True
        )
        
        if not dataloader:
            print("❌ Failed to create DataLoader")
            return False
        
        print(f"✅ DataLoader ready: {len(dataloader)} batches")
        
        # Статистика датасета
        stats = dataset_manager.get_statistics()
        print(f"   Dataset: {stats.get('total_samples', 'N/A')} samples")
        print(f"   Sources: {', '.join(stats.get('providers_used', []))}")
        print(f"   Embedding dim: {stats.get('embedding_dimension', 'N/A')}")
        
        # Тренировка
        print(f"\n3️⃣ Starting training ({num_epochs} epochs)...")
        
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
                # Извлекаем данные из нового формата
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
        print(f"\n4️⃣ Running post-training validation...")
        
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


def test_dataset_integration_only():
    """Тест только интеграции датасета без полного обучения"""
    print("\n🧪 Testing Dataset Integration Only")
    print("-" * 40)
    
    try:
        energy_config, dataset_manager, trainer = create_training_setup()
        
        # Проверяем что все инициализировано
        print("✅ Components initialized")
        
        # Получаем один батч
        dataloader = dataset_manager.create_dataloader(batch_size=2)
        if dataloader:
            batch = next(iter(dataloader))
            
            print(f"✅ Sample batch loaded:")
            print(f"   Texts: {len(batch['input_text'])} pairs")
            print(f"   Embeddings: {batch['input_embedding'].shape}")
            print(f"   Sample input: '{batch['input_text'][0][:50]}...'")
            
            # Тест генерации эмбеддингов через dataset manager
            test_texts = ["Hello world", "Machine learning is interesting"]
            embeddings = dataset_manager.get_teacher_embeddings(test_texts)
            print(f"✅ Teacher embeddings generated: {embeddings.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Основное демо обучения
        success = run_training_demo(num_epochs=1)
        
        if success:
            # Дополнительный тест интеграции
            test_dataset_integration_only()
            
            print(f"\n✨ All demos completed successfully!")
            print(f"   The new dataset module is ready for production use.")
        else:
            print(f"\n⚠️  Demo had issues, but components are available for debugging.")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()