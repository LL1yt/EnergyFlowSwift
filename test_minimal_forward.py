#!/usr/bin/env python3
"""Minimal test - just one forward pass"""

import torch
import time
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config import SimpleProjectConfig

print("🚀 Minimal forward pass test...")

# Создаем trainer
config = SimpleProjectConfig()
trainer = EmbeddingTrainer(config)

# Создаем минимальный тестовый батч
batch_size = 1  # Только 1 сэмпл
embedding_dim = 768
test_embeddings = torch.randn(batch_size, embedding_dim).to(trainer.device)

print(f"📊 Test input shape: {test_embeddings.shape}")

# Время forward pass
start_time = time.time()

try:
    # Напрямую вызываем _forward_pass
    losses = trainer._forward_pass(
        input_embeddings=test_embeddings,
        target_embeddings=test_embeddings,
        texts=None
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Forward pass completed in {elapsed:.2f} seconds!")
    print(f"📈 Losses:")
    for key, value in losses.items():
        print(f"   {key}: {value.item():.6f}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()