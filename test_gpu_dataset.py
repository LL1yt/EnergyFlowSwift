#!/usr/bin/env python3
"""
Тест GPU-ускоренного UnifiedEmbeddingDataset
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training.utils import create_training_dataloader
from new_rebuild.utils.logging import get_logger
from new_rebuild.utils.device_manager import get_device_manager

logger = get_logger(__name__)

def test_gpu_dataset():
    """Тестируем GPU-ускоренный датасет"""
    
    logger.info("🧪 TESTING GPU-ACCELERATED DATASET")
    logger.info("=" * 60)
    
    # Создаем конфигурацию
    config = SimpleProjectConfig()
    device_manager = get_device_manager()
    
    # Проверяем GPU
    logger.info(f"🔍 GPU Available: {device_manager.is_cuda()}")
    if device_manager.is_cuda():
        logger.info(f"💾 Total GPU Memory: {device_manager.get_available_memory_gb():.1f}GB")
    
    # Тест 1: Небольшой датасет для проверки функциональности
    logger.info("\n🔬 Test 1: Small dataset (100 samples)")
    config.training_embedding.max_total_samples = 100
    
    start_time = time.time()
    dataloader, stats = create_training_dataloader(
        config=config,
        shuffle=True,
        num_workers=0  # Начинаем с 0 для простоты
    )
    load_time = time.time() - start_time
    
    logger.info(f"⏱️ Load time: {load_time:.2f}s")
    logger.info(f"📊 Dataset size: {stats.total_samples}")
    logger.info(f"📊 Sources: {stats.source_distribution}")
    
    # Тест работы с батчами
    logger.info("\n🔄 Testing batch loading:")
    batch_times = []
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        embeddings = batch['embedding']
        metadata = batch['metadata']
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        logger.info(f"  Batch {i+1}: {embeddings.shape}, device: {embeddings.device}, time: {batch_time:.4f}s")
        
        if i >= 2:  # Только первые 3 батча
            break
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    logger.info(f"📈 Average batch time: {avg_batch_time:.4f}s")
    
    # Тест памяти
    if device_manager.is_cuda():
        memory_stats = device_manager.get_memory_stats()
        logger.info(f"\n💾 GPU Memory after test:")
        logger.info(f"  Allocated: {memory_stats.get('allocated_mb', 0):.1f}MB")
        logger.info(f"  Available: {device_manager.get_available_memory_gb():.1f}GB")
    
    # Тест 2: Автоматическое управление памятью
    logger.info("\n🔬 Test 2: Automatic memory management")
    config.training_embedding.max_total_samples = None  # Автоматический лимит
    
    start_time = time.time()
    dataloader2, stats2 = create_training_dataloader(
        config=config,
        shuffle=False,
        num_workers=2  # Тест с воркерами
    )
    auto_load_time = time.time() - start_time
    
    logger.info(f"⏱️ Auto load time: {auto_load_time:.2f}s")
    logger.info(f"📊 Auto dataset size: {stats2.total_samples}")
    
    # Сравнение производительности
    logger.info(f"\n📈 Performance Summary:")
    logger.info(f"  Manual limit (100): {load_time:.2f}s")
    logger.info(f"  Auto limit ({stats2.total_samples}): {auto_load_time:.2f}s")
    logger.info(f"  Samples per second: {stats2.total_samples / auto_load_time:.0f}")
    
    logger.info("\n✅ GPU Dataset test completed!")
    
    return {
        'small_dataset_time': load_time,
        'small_dataset_size': stats.total_samples,
        'auto_dataset_time': auto_load_time,
        'auto_dataset_size': stats2.total_samples,
        'avg_batch_time': avg_batch_time,
        'gpu_available': device_manager.is_cuda()
    }

if __name__ == "__main__":
    results = test_gpu_dataset()
    print("\n📊 Test Results Summary:")
    for key, value in results.items():
        print(f"  {key}: {value}")