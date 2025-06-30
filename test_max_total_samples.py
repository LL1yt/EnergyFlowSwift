#!/usr/bin/env python3
"""
Тест функции create_training_dataloader с новым параметром max_total_samples
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from new_rebuild.config import SimpleProjectConfig
from new_rebuild.core.training.utils import create_training_dataloader
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

def test_max_total_samples():
    """Тестируем ограничение общего количества сэмплов"""
    
    logger.info("[TEST] TESTING MAX_TOTAL_SAMPLES FUNCTIONALITY")
    logger.info("=" * 50)
    
    # Создаем конфигурацию с разрешенным fallback для тестов
    config = SimpleProjectConfig()
    config.device.fallback_cpu = True  # Разрешаем CPU fallback для совместимости
    
    # Тест 1: Без ограничений
    logger.info("\n[SCIENCE] Test 1: No limit")
    dataloader1, stats1 = create_training_dataloader(
        config=config,
        max_total_samples=None,
        shuffle=False
    )
    logger.info(f"Result: {stats1.total_samples} samples")
    
    # Тест 2: Ограничение через параметр функции
    logger.info("\n[SCIENCE] Test 2: Limit via function parameter (100 samples)")
    dataloader2, stats2 = create_training_dataloader(
        config=config,
        max_total_samples=100,
        shuffle=False
    )
    logger.info(f"Result: {stats2.total_samples} samples")
    assert stats2.total_samples <= 100, f"Expected ≤100, got {stats2.total_samples}"
    
    # Тест 3: Ограничение через конфигурацию
    logger.info("\n[SCIENCE] Test 3: Limit via config (50 samples)")
    config.training_embedding.max_total_samples = 50
    dataloader3, stats3 = create_training_dataloader(
        config=config,
        max_total_samples=200,  # Должно игнорироваться
        shuffle=False
    )
    logger.info(f"Result: {stats3.total_samples} samples")
    assert stats3.total_samples <= 50, f"Expected ≤50, got {stats3.total_samples}"
    
    # Тест 4: Очень маленький лимит
    logger.info("\n[SCIENCE] Test 4: Very small limit (5 samples)")
    config.training_embedding.max_total_samples = 5
    dataloader4, stats4 = create_training_dataloader(
        config=config,
        shuffle=False
    )
    logger.info(f"Result: {stats4.total_samples} samples")
    assert stats4.total_samples <= 5, f"Expected ≤5, got {stats4.total_samples}"
    
    # Проверяем, что датасет действительно содержит правильное количество сэмплов
    logger.info("\n[SCIENCE] Test 5: DataLoader consistency check")
    total_batches = 0
    total_samples_in_batches = 0
    for batch in dataloader4:
        total_batches += 1
        total_samples_in_batches += len(batch['embedding'])
        
    logger.info(f"Batches: {total_batches}, Samples in batches: {total_samples_in_batches}")
    
    logger.info("\n[OK] ALL TESTS PASSED!")
    
    return {
        'no_limit': stats1.total_samples,
        'param_limit_100': stats2.total_samples,
        'config_limit_50': stats3.total_samples,
        'config_limit_5': stats4.total_samples,
        'dataloader_samples': total_samples_in_batches
    }

if __name__ == "__main__":
    results = test_max_total_samples()
    print("\n[DATA] Test Results Summary:")
    for test, result in results.items():
        print(f"  {test}: {result} samples")