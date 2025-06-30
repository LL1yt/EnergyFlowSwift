#!/usr/bin/env python3
"""
Быстрый тест GPU-only режима с ограничением сэмплов
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

def test_gpu_only_mode():
    """Быстрый тест GPU-only режима"""
    
    logger.info("[FAST] TESTING STRICT GPU-ONLY MODE")
    logger.info("=" * 50)
    
    # Создаем конфигурацию с принудительным GPU режимом
    config = SimpleProjectConfig()
    config.device.fallback_cpu = False  # Строгий GPU-only режим
    
    device_manager = get_device_manager()
    logger.info(f"[SEARCH] GPU Status: {device_manager.is_cuda()}")
    
    if not device_manager.is_cuda():
        logger.error("[ALERT] GPU не доступен! Тест не может продолжиться в GPU-only режиме")
        return False
    
    # Тест 1: Малый датасет для скорости
    logger.info("\n[SCIENCE] Test 1: Small GPU-only dataset (50 samples)")
    config.training_embedding.max_total_samples = 50
    
    try:
        logger.info("[SYNC] Creating dataloader...")
        start_time = time.time()
        dataloader, stats = create_training_dataloader(
            config=config,
            shuffle=True,
            num_workers=0  # Без воркеров для Windows
        )
        load_time = time.time() - start_time
        
        logger.info(f"[FAST] SUCCESS: Load time: {load_time:.2f}s")
        logger.info(f"[DATA] Dataset size: {stats.total_samples}")
        
        # Быстрый тест батчей
        logger.info("\n[SYNC] Testing first batch:")
        for i, batch in enumerate(dataloader):
            embeddings = batch['embedding']
            logger.info(f"  Batch shape: {embeddings.shape}")
            logger.info(f"  Device: {embeddings.device}")
            logger.info(f"  Dtype: {embeddings.dtype}")
            break  # Только первый батч
            
        # Проверка памяти
        memory_stats = device_manager.get_memory_stats()
        logger.info(f"\n[DISK] GPU Memory:")
        logger.info(f"  Allocated: {memory_stats.get('allocated_mb', 0):.1f}MB")
        
        logger.info("[OK] GPU-only mode test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"[ALERT] GPU-only test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_only_mode()
    print(f"\n{'[OK] SUCCESS' if success else '[ERROR] FAILED'}: GPU-only mode test")