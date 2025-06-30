#!/usr/bin/env python3
"""Multiple forward pass test - проверка стабильности системы"""

import torch
import time
import gc
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.config import SimpleProjectConfig

print("[START] Multiple forward pass test...")

# Создаем trainer
config = SimpleProjectConfig()
trainer = EmbeddingTrainer(config)

# Параметры теста
num_passes = 5  # Количество forward pass'ов
batch_size = 1  # Только 1 сэмпл
embedding_dim = 768

print(f"[DATA] Running {num_passes} forward passes...")
print(f"[DATA] Batch size: {batch_size}, Embedding dim: {embedding_dim}")

# Статистика
total_time = 0
all_losses = []

for i in range(num_passes):
    print(f"\n[SYNC] Forward pass #{i+1}:")
    
    # Создаем новый тестовый батч для каждого прохода
    test_embeddings = torch.randn(batch_size, embedding_dim).to(trainer.device)
    
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
        total_time += elapsed
        
        print(f"  [OK] Completed in {elapsed:.2f} seconds")
        print(f"  [UP] Total loss: {losses['total'].item():.6f}")
        
        # Сохраняем потери для анализа
        loss_dict = {k: v.item() for k, v in losses.items()}
        all_losses.append(loss_dict)
        
        # Очистка памяти между проходами
        del test_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        break

# Финальная статистика
print(f"\n[DATA] Summary:")
print(f"  Total time: {total_time:.2f} seconds")
print(f"  Average time per pass: {total_time/len(all_losses):.2f} seconds")
print(f"  Successful passes: {len(all_losses)}/{num_passes}")

if all_losses:
    print(f"\n[UP] Loss evolution:")
    for i, losses in enumerate(all_losses):
        print(f"  Pass {i+1}: total={losses['total']:.6f}, reconstruction={losses['reconstruction']:.6f}")
        
    # Проверим стабильность
    total_losses = [l['total'] for l in all_losses]
    max_loss = max(total_losses)
    min_loss = min(total_losses)
    loss_variation = (max_loss - min_loss) / min_loss if min_loss > 0 else 0
    
    print(f"\n[SEARCH] Stability analysis:")
    print(f"  Min total loss: {min_loss:.6f}")
    print(f"  Max total loss: {max_loss:.6f}")
    print(f"  Loss variation: {loss_variation:.2%}")
    
    if loss_variation < 0.1:
        print("  [OK] System is stable (variation < 10%)")
    else:
        print("  [WARN] System shows instability (variation >= 10%)")

print("\n[OK] Test completed!")