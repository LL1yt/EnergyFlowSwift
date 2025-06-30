#!/usr/bin/env python3
"""Real training with minimal settings"""

import time
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.core.training.utils import create_training_dataloader

print("[START] Real training with minimal settings...")

config = SimpleProjectConfig()
print(f"[DATA] Config: test_dataset_size={config.training_embedding.test_dataset_size}")
print(f"[DATA] Config: num_epochs={config.training_embedding.num_epochs}")
print(f"[DATA] Config: batch_size={config.training_embedding.embedding_batch_size}")

# Создаем минимальный dataloader
dataloader, stats = create_training_dataloader(
    config, 
    max_samples_per_source=config.training_embedding.test_dataset_size,
    shuffle=True
)

print(f"\n[DATA] Dataset loaded: {stats.total_samples} samples")
print(f"[DATA] Batches per epoch: {len(dataloader)}")

# Создаем trainer
trainer = EmbeddingTrainer(config)

# Обучение одной эпохи
start_time = time.time()
print("\n[TIME] Starting training epoch...")

try:
    losses = trainer.train_epoch(dataloader)
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Epoch completed in {elapsed:.1f} seconds!")
    print(f"[UP] Losses:")
    for key, value in losses.items():
        if key != 'count' and isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
    
    # Средняя скорость на батч
    if 'count' in losses:
        avg_time_per_batch = elapsed / losses['count']
        print(f"\n⏱️ Average time per batch: {avg_time_per_batch:.2f} seconds")
        print(f"📦 Total batches processed: {losses['count']}")
        
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()