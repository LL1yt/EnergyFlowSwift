#!/usr/bin/env python3
"""Краткий тест классификации для отладки кэша"""

import torch
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config.simple_config import SimpleProjectConfig
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("🔍 Debug cache classification...")

# Создаем trainer
config = SimpleProjectConfig()
trainer = EmbeddingTrainer(config)

# Создаем простой тест
test_embeddings = torch.randn(1, 768).to(trainer.device)

logger.info("🚀 Starting single forward pass for debugging...")

try:
    # Один forward pass для отладки
    losses = trainer._forward_pass(
        input_embeddings=test_embeddings,
        target_embeddings=test_embeddings,
        texts=None
    )
    
    logger.info(f"✅ Forward pass completed, total loss: {losses['total'].item():.6f}")
    
except Exception as e:
    logger.error(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

logger.info("✅ Debug completed!")