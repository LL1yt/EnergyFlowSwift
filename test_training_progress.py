#!/usr/bin/env python3
"""Test training with progress tracking"""

# ВАЖНО: Настраиваем логирование ПЕРЕД импортом других модулей
from new_rebuild.utils.logging import setup_logging
setup_logging(debug_mode=False, level="INFO", enable_context=True)

import torch
import time
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training.utils import create_training_dataloader

print("[START] Starting training test...")

config = SimpleProjectConfig()
trainer = EmbeddingTrainer(config)

# Create minimal dataloader
dataloader, stats = create_training_dataloader(
    config, 
    max_samples_per_source=50,  # Небольшое количество для теста
    shuffle=True
)

print(f"[DATA] Dataset: {stats.total_samples} samples")
print(f"📦 Batch size: {config.training_embedding.embedding_batch_size}")
print(f"[SYNC] Total batches: {len(dataloader)}")

# Test one epoch with timing
start_time = time.time()
print("\n[TIME] Starting epoch...")

try:
    losses = trainer.train_epoch(dataloader)
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Epoch completed in {elapsed:.1f} seconds!")
    print(f"[UP] Losses:")
    for key, value in losses.items():
        if key != 'count':
            print(f"   {key}: {value:.6f}")
    
except Exception as e:
    print(f"\n[ERROR] Error during training: {e}")
    import traceback
    traceback.print_exc()