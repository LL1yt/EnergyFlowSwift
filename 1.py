import torch
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training.utils import create_training_dataloader

config = SimpleProjectConfig()
trainer = EmbeddingTrainer(config)

# Create a small test dataloader
dataloader, stats = create_training_dataloader(config, max_samples_per_source=10, shuffle=False)

# Test one batch
for batch in dataloader:
    print(f'Testing batch with shape: {batch[embedding].shape}')
    losses = trainer.train_epoch(dataloader)
    print(f'[OK] Training successful! Losses: {losses}')
    break
