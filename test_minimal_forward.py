#!/usr/bin/env python3
"""Multiple forward pass test - проверка стабильности системы"""

import torch
import time
import gc

# Сначала настраиваем конфигурацию ДО всех остальных импортов
from new_rebuild.config import (
    set_project_config, 
    create_debug_config
)

# Создаем и устанавливаем конфиг глобально ДО импорта других модулей
config = create_debug_config()
set_project_config(config)

# Теперь импортируем остальные модули
from new_rebuild.core.training import EmbeddingTrainer
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Основная функция теста"""
    logger.info("🚀 Multiple forward pass test...")
    
    # Конфиг уже установлен глобально выше
    # Создаем trainer - он автоматически получит глобальную конфигурацию
    trainer = EmbeddingTrainer(config)
    
    # Получаем параметры из централизованного конфига
    num_passes = config.validation.num_forward_passes
    batch_size = config.training.batch_size
    embedding_dim = config.embeddings.teacher_dim  # 768 для distilbert
    
    logger.info(f"📊 Running {num_passes} forward passes...")
    logger.info(f"📊 Batch size: {batch_size}, Embedding dim: {embedding_dim}")
    logger.info(f"🎯 Config mode: {config.mode.mode.name}")
    
    # Статистика
    total_time = 0
    all_losses = []
    
    for i in range(num_passes):
        logger.info(f"\n🔄 Forward pass #{i+1}:")
        
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
            
            logger.info(f"  ✅ Completed in {elapsed:.2f} seconds")
            logger.info(f"  📈 Total loss: {losses['total'].item():.6f}")
            
            # Сохраняем потери для анализа
            loss_dict = {k: v.item() for k, v in losses.items()}
            all_losses.append(loss_dict)
            
            # Очистка памяти между проходами
            del test_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Финальная статистика
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    
    if all_losses:
        logger.info(f"  Average time per pass: {total_time/len(all_losses):.2f} seconds")
        logger.info(f"  Successful passes: {len(all_losses)}/{num_passes}")
        
        logger.info(f"\n📈 Loss evolution:")
        for i, losses in enumerate(all_losses):
            logger.info(f"  Pass {i+1}: total={losses['total']:.6f}, reconstruction={losses['reconstruction']:.6f}")
            
        # Проверим стабильность
        total_losses = [l['total'] for l in all_losses]
        if len(total_losses) > 1:  # Нужно минимум 2 значения для анализа стабильности
            max_loss = max(total_losses)
            min_loss = min(total_losses)
            loss_variation = (max_loss - min_loss) / min_loss if min_loss > 0 else 0
            
            logger.info(f"\n🔍 Stability analysis:")
            logger.info(f"  Min total loss: {min_loss:.6f}")
            logger.info(f"  Max total loss: {max_loss:.6f}")
            logger.info(f"  Loss variation: {loss_variation:.2%}")
            
            # Используем порог стабильности из централизованного конфига
            stability_threshold = config.validation.stability_threshold
            if loss_variation < stability_threshold:
                logger.info(f"  ✅ System is stable (variation < {stability_threshold:.0%})")
            else:
                logger.warning(f"  ⚠️ System shows instability (variation >= {stability_threshold:.0%})")
    
    logger.info("\n✅ Test completed!")


if __name__ == "__main__":
    main()