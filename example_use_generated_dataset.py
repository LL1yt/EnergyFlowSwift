#!/usr/bin/env python3
"""
Пример использования сгенерированного датасета в обучении
========================================================

Демонстрирует простую загрузку готового файла датасета 
и его использование в training loop без сложной интеграции.
"""

import sys
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from energy_flow.config import create_debug_config, set_energy_config
from energy_flow.training import EnergyTrainer


class SimpleDatasetLoader:
    """Простой загрузчик для готовых файлов датасетов"""
    
    @staticmethod
    def load_dataset(filepath: str) -> dict:
        """Загрузка готового файла датасета"""
        print(f"📥 Loading dataset: {Path(filepath).name}")
        
        try:
            data = torch.load(filepath)
            
            # Проверяем формат
            required_keys = ['input_embeddings', 'target_embeddings']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Missing keys in dataset: {missing_keys}")
            
            # Информация о датасете
            sample_count = data['input_embeddings'].shape[0]
            embedding_dim = data['input_embeddings'].shape[1]
            device = data['input_embeddings'].device
            
            print(f"✅ Dataset loaded successfully:")
            print(f"   Samples: {sample_count:,}")
            print(f"   Embedding dimension: {embedding_dim}")
            print(f"   Device: {device}")
            
            # Метаданные если есть
            if 'generation_info' in data:
                gen_info = data['generation_info']
                print(f"   Mode: {gen_info.get('mode', 'unknown')}")
                print(f"   Sources: {', '.join(gen_info.get('sources', []))}")
                
            return data
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
    
    @staticmethod
    def create_dataloader(data: dict, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Создание DataLoader из загруженных данных"""
        input_embeddings = data['input_embeddings']
        target_embeddings = data['target_embeddings']
        
        # Простой TensorDataset
        dataset = TensorDataset(input_embeddings, target_embeddings)
        
        # Создаем generator на правильном устройстве если используем shuffle
        generator = None
        if shuffle:
            device = input_embeddings.device
            generator = torch.Generator(device=device)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            generator=generator
        )
        
        print(f"📦 DataLoader created: {len(dataloader)} batches, batch_size={batch_size}")
        
        return dataloader


def find_latest_dataset(active_dir: str = "data/energy_flow/active") -> str:
    """Найти самый новый датасет в активной директории"""
    active_path = Path(active_dir)
    
    if not active_path.exists():
        raise FileNotFoundError(f"Active directory not found: {active_dir}")
    
    # Ищем .pt файлы
    dataset_files = list(active_path.glob("*.pt"))
    
    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found in {active_dir}")
    
    # Сортируем по времени модификации (новые первыми)
    latest_file = max(dataset_files, key=lambda f: f.stat().st_mtime)
    
    return str(latest_file)


def training_example_with_generated_dataset():
    """Пример обучения с использованием готового датасета"""
    print("🚀 Training Example with Generated Dataset")
    print("=" * 50)
    
    try:
        # 1. Ищем самый новый датасет
        print("\n1️⃣ Finding latest dataset...")
        try:
            dataset_file = find_latest_dataset()
            print(f"📁 Using: {Path(dataset_file).name}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 First generate a dataset using: python generate_energy_dataset.py --mode debug")
            return
        
        # 2. Загружаем датасет
        print("\n2️⃣ Loading dataset...")
        dataset_data = SimpleDatasetLoader.load_dataset(dataset_file)
        
        # 3. Создаем DataLoader
        print("\n3️⃣ Creating DataLoader...")
        dataloader = SimpleDatasetLoader.create_dataloader(
            dataset_data, 
            batch_size=8, 
            shuffle=True
        )
        
        # 4. Настраиваем обучение
        print("\n4️⃣ Setting up training...")
        energy_config = create_debug_config()
        set_energy_config(energy_config)
        
        trainer = EnergyTrainer(energy_config)
        print("✅ EnergyTrainer initialized")
        
        # 5. Тренировочный цикл (упрощенный)
        print("\n5️⃣ Running training loop...")
        
        for epoch in range(2):  # Только 2 эпохи для демо
            print(f"\nEpoch {epoch + 1}/2:")
            print("-" * 20)
            
            epoch_loss = 0.0
            batches_processed = 0
            
            for batch_idx, (input_embeddings, target_embeddings) in enumerate(dataloader):
                
                # Создаем заглушки для текстов (в реальном обучении могут быть из метаданных)
                batch_size = input_embeddings.shape[0]
                input_texts = [f"input_sample_{i}" for i in range(batch_size)]
                target_texts = [f"target_sample_{i}" for i in range(batch_size)]
                
                # Один шаг обучения
                step_metrics = trainer.train_step(
                    input_texts=input_texts,
                    target_texts=target_texts,
                    teacher_input_embeddings=input_embeddings,
                    teacher_target_embeddings=target_embeddings
                )
                
                current_loss = step_metrics.get('total_loss', 0)
                epoch_loss += current_loss
                batches_processed += 1
                
                # Логирование каждого 5-го батча
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={current_loss:.4f}")
                
                # Ограничиваем для демо
                if batch_idx >= 15:  # Максимум 15 батчей
                    break
            
            # Статистика эпохи
            avg_loss = epoch_loss / batches_processed if batches_processed > 0 else 0
            print(f"  Epoch {epoch + 1} completed: avg_loss={avg_loss:.4f}, batches={batches_processed}")
        
        # 6. Результат
        print(f"\n🎉 Training completed successfully!")
        print(f"   Dataset: {Path(dataset_file).name}")
        print(f"   Samples: {dataset_data['input_embeddings'].shape[0]:,}")
        print(f"   Final loss: {avg_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_dataset_example():
    """Пример анализа содержимого датасета"""
    print("\n🔍 Dataset Inspection Example")
    print("-" * 30)
    
    try:
        dataset_file = find_latest_dataset()
        data = SimpleDatasetLoader.load_dataset(dataset_file)
        
        # Анализируем содержимое
        print(f"\n📊 Dataset Analysis:")
        print(f"   Main keys: {list(data.keys())}")
        
        # Статистика эмбеддингов
        input_emb = data['input_embeddings']
        target_emb = data['target_embeddings']
        
        print(f"\n📈 Embedding Statistics:")
        print(f"   Input norms: mean={input_emb.norm(dim=1).mean():.4f}, "
              f"std={input_emb.norm(dim=1).std():.4f}")
        print(f"   Target norms: mean={target_emb.norm(dim=1).mean():.4f}, "
              f"std={target_emb.norm(dim=1).std():.4f}")
        
        # Метаданные генерации
        if 'generation_info' in data:
            gen_info = data['generation_info']
            print(f"\n🏷️ Generation Info:")
            print(f"   Generated: {gen_info.get('generation_timestamp', 'unknown')}")
            print(f"   Generation time: {gen_info.get('generation_time', 0):.1f}s")
            print(f"   Target pairs: {gen_info.get('target_pairs', 'unknown'):,}")
            print(f"   Actual pairs: {gen_info.get('actual_pairs', 'unknown'):,}")
        
        # Примеры текстов если есть
        if 'text_pairs' in data and data['text_pairs']:
            print(f"\n📝 Sample Text Pairs:")
            for i, (input_text, target_text) in enumerate(data['text_pairs'][:3]):
                print(f"   {i+1}. Input: '{input_text[:50]}...'")
                print(f"      Target: '{target_text[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        return False


def main():
    """Главная функция демонстрации"""
    try:
        # Основной пример обучения
        success = training_example_with_generated_dataset()
        
        if success:
            # Дополнительный анализ датасета
            inspect_dataset_example()
            
            print(f"\n✨ All examples completed successfully!")
            print(f"\n💡 Key takeaways:")
            print(f"   - Загрузка датасета: torch.load(filepath)")
            print(f"   - Создание DataLoader: TensorDataset + DataLoader")
            print(f"   - Прямое использование в trainer.train_step()")
            print(f"   - Никаких сложных интеграций!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    main()