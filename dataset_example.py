#!/usr/bin/env python3
"""
Пример использования нового dataset модуля
==========================================

Демонстрирует основные возможности:
- Создание DatasetManager с автоматической настройкой
- Проверка наличия модели-учителя
- Подготовка унифицированного датасета
- Интеграция с EnergyTrainer
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.dataset import (
    DatasetConfig, 
    DatasetManager,
    create_dataset_config_from_energy,
    create_dataset_manager
)
from energy_flow.dataset.utils import create_dataset_summary_report
from energy_flow.training import EnergyTrainer
from energy_flow.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Основная демонстрация dataset модуля"""
    print("🚀 Energy Flow Dataset Module Demo")
    print("=" * 50)
    
    # 1. Создаем энергетическую конфигурацию
    print("\n1️⃣ Creating energy configuration...")
    energy_config = create_debug_config()
    set_energy_config(energy_config)
    
    # 2. Создаем конфигурацию датасета (автоматическая адаптация)
    print("\n2️⃣ Creating dataset configuration...")
    dataset_config = create_dataset_config_from_energy(
        energy_config,
        dataset_sources=["precomputed", "snli"],  # Переопределяем источники
        max_samples_per_source=50  # Ограничиваем для демо
    )
    
    print(f"Dataset config: sources={dataset_config.dataset_sources}, "
          f"batch_size={dataset_config.batch_size}")
    
    # 3. Создаем DatasetManager
    print("\n3️⃣ Creating DatasetManager...")
    dataset_manager = create_dataset_manager(dataset_config, energy_config)
    
    # 4. Запускаем полную диагностику
    print("\n4️⃣ Running comprehensive validation...")
    validation = dataset_manager.validate_setup()
    
    print(f"Validation results:")
    print(f"  Teacher model: {'✅' if validation['teacher_model'] else '❌'}")
    print(f"  Providers: {sum(validation['providers'].values())}/{len(validation['providers'])} available")
    print(f"  Dataset prep: {'✅' if validation['dataset_preparation'] else '❌'}")
    print(f"  Overall: {'🎉 READY' if validation['overall_status'] else '⚠️ ISSUES'}")
    
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    
    # 5. Если все готово, создаем DataLoader
    if validation['overall_status']:
        print("\n5️⃣ Creating DataLoader...")
        dataloader = dataset_manager.create_dataloader(batch_size=8, shuffle=True)
        
        if dataloader:
            print(f"✅ DataLoader created: {len(dataloader)} batches")
            
            # Тестируем один батч
            print("\n6️⃣ Testing batch loading...")
            for i, batch in enumerate(dataloader):
                print(f"Batch {i+1}:")
                print(f"  Input texts: {len(batch['input_text'])}")
                print(f"  Target texts: {len(batch['target_text'])}")
                print(f"  Input embeddings: {batch['input_embedding'].shape}")
                print(f"  Target embeddings: {batch['target_embedding'].shape}")
                
                # Показываем примеры текстов
                print(f"  Example input: '{batch['input_text'][0][:50]}...'")
                print(f"  Example target: '{batch['target_text'][0][:50]}...'")
                
                if i >= 2:  # Только первые 3 батча
                    break
            
            # 7. Интеграция с EnergyTrainer (если возможно)
            print("\n7️⃣ Testing EnergyTrainer integration...")
            try:
                trainer = EnergyTrainer(energy_config)
                
                # Получаем один батч для тестирования
                test_batch = next(iter(dataloader))
                input_texts = test_batch['input_text']
                target_texts = test_batch['target_text']
                input_embeddings = test_batch['input_embedding']
                target_embeddings = test_batch['target_embedding']
                
                print(f"✅ EnergyTrainer initialized successfully")
                print(f"   Ready for training with {len(input_texts)} samples per batch")
                
                # Можно протестировать один шаг обучения (опционально)
                # step_metrics = trainer.train_step(input_texts, target_texts, input_embeddings, target_embeddings)
                # print(f"   Test training step completed: loss={step_metrics.get('total_loss', 'N/A')}")
                
            except Exception as e:
                print(f"❌ EnergyTrainer integration failed: {e}")
        
        # 8. Генерируем итоговый отчет
        print("\n8️⃣ Generating summary report...")
        report = create_dataset_summary_report(dataset_manager)
        print(report)
        
    else:
        print("\n❌ Setup validation failed - cannot proceed with training")
        print("Please check the errors above and resolve them.")
    
    print("\n🎉 Demo completed!")


def test_teacher_model_only():
    """Тест только модели-учителя без полного датасета"""
    print("\n🧪 Testing Teacher Model Only")
    print("-" * 30)
    
    # Простая конфигурация только для teacher model
    from energy_flow.dataset.config import DatasetConfig
    from energy_flow.dataset.providers import create_teacher_model_provider
    
    config = DatasetConfig(
        teacher_model="distilbert-base-uncased",
        use_local_model=True
    )
    
    teacher_provider = create_teacher_model_provider(config)
    
    # Проверяем доступность
    if teacher_provider.is_available():
        print("✅ Teacher model available")
        
        if teacher_provider.ensure_initialized():
            print("✅ Teacher model initialized")
            
            # Тестируем генерацию эмбеддингов
            test_texts = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "The weather is nice today."
            ]
            
            embeddings = teacher_provider.encode_texts(test_texts)
            print(f"✅ Generated embeddings: {embeddings.shape}")
            print(f"   Embedding norms: {embeddings.norm(dim=1).tolist()}")
            
            # Статистика кэша
            cache_info = teacher_provider.get_cache_info()
            print(f"   Cache info: {cache_info}")
        else:
            print("❌ Teacher model initialization failed")
    else:
        print("❌ Teacher model not available")
        
        # Предлагаем загрузить
        if teacher_provider.download_model_if_needed():
            print("✅ Model downloaded successfully, try again")
        else:
            print("❌ Model download failed")


if __name__ == "__main__":
    try:
        main()
        
        # Дополнительный тест teacher model
        test_teacher_model_only()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()