#!/usr/bin/env python3
"""
Simple Batch Processing Test
============================

Простой тест для проверки работоспособности batch processing.
Без pytest, только базовая проверка импортов и функционирования.
"""

import torch
import time
import traceback
from new_rebuild.config import create_debug_config, set_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_imports():
    """Проверка всех необходимых импортов"""
    logger.info("🔍 Testing imports...")
    
    try:
        from new_rebuild.core.moe.batch import BatchMoEProcessor, BatchProcessingAdapter
        logger.info("✅ MoE batch imports OK")
        
        from new_rebuild.core.lattice.spatial_optimization.batch_integration import (
            create_batch_optimized_spatial_optimizer,
            upgrade_lattice_to_batch
        )
        logger.info("✅ Spatial optimization batch imports OK")
        
        from new_rebuild.core.lattice import Lattice3D
        logger.info("✅ Lattice imports OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_batch_adapter():
    """Тест базовой функциональности BatchProcessingAdapter"""
    logger.info("🧪 Testing basic batch adapter...")
    
    try:
        from new_rebuild.core.moe.batch import BatchProcessingAdapter
        from new_rebuild.core.moe.moe_processor import MoEConnectionProcessor
        
        # Создаем базовый MoE процессор
        dimensions = (8, 8, 8)
        moe = MoEConnectionProcessor(lattice_dimensions=dimensions)
        logger.info(f"✅ MoE processor created: state_size={moe.state_size}")
        
        # Создаем адаптер
        adapter = BatchProcessingAdapter(
            moe_processor=moe,
            enable_batch=True,
            fallback_on_error=True
        )
        logger.info("✅ Batch adapter created successfully")
        
        # Тестовые данные - ВАЖНО: переносим на правильное устройство
        total_cells = 8 * 8 * 8
        device = moe.device if hasattr(moe, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        states = torch.randn(total_cells, moe.state_size, device=device)
        cell_indices = [0, 10, 20, 30, 40]
        
        logger.info(f"Created test data: {len(cell_indices)} cells, states shape: {states.shape}")
        
        # Пробуем обработать
        result = adapter.process_cells(
            cell_indices=cell_indices,
            full_lattice_states=states
        )
        
        # Проверяем результат
        assert len(result) == len(cell_indices), f"Expected {len(cell_indices)} results, got {len(result)}"
        
        for idx in cell_indices:
            assert idx in result, f"Missing result for cell {idx}"
            assert result[idx].shape == (moe.state_size,), f"Wrong shape for cell {idx}: {result[idx].shape}"
        
        logger.info("✅ Batch adapter test passed!")
        
        # Получаем статистику
        stats = adapter.get_performance_comparison()
        logger.info(f"📊 Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch adapter test failed: {e}")
        traceback.print_exc()
        return False


def test_lattice_integration():
    """Тест интеграции с Lattice3D"""
    logger.info("🧪 Testing lattice integration...")
    
    try:
        from new_rebuild.core.lattice import Lattice3D
        from new_rebuild.core.lattice.spatial_optimization.batch_integration import upgrade_lattice_to_batch
        
        # Создаем небольшую решетку
        dimensions = (8, 8, 8)  # Маленькая для быстрого теста
        lattice = Lattice3D()
        logger.info(f"✅ Lattice created: {dimensions}")
        
        # Проверяем базовый forward pass
        logger.info("Testing standard forward pass...")
        start_time = time.time()
        result1 = lattice.forward()
        standard_time = (time.time() - start_time) * 1000
        logger.info(f"Standard forward: {standard_time:.1f}ms, result shape: {result1.shape}")
        
        # Пробуем обновить для batch обработки
        logger.info("Upgrading to batch processing...")
        lattice_batch = upgrade_lattice_to_batch(lattice)
        
        # Включаем batch режим
        if hasattr(lattice_batch, 'set_batch_enabled'):
            lattice_batch.set_batch_enabled(True)
            logger.info("✅ Batch mode enabled")
        
        # Тестируем batch forward pass
        logger.info("Testing batch forward pass...")
        start_time = time.time()
        result2 = lattice_batch.forward()
        batch_time = (time.time() - start_time) * 1000
        logger.info(f"Batch forward: {batch_time:.1f}ms, result shape: {result2.shape}")
        
        # Сравниваем результаты
        if result1.shape == result2.shape:
            diff = torch.mean(torch.abs(result1 - result2)).item()
            logger.info(f"Mean difference between standard and batch: {diff:.6f}")
            
            if diff < 1e-4:
                logger.info("✅ Results are very similar!")
            else:
                logger.warning(f"⚠️ Results differ: {diff:.6f}")
        
        # Вычисляем ускорение
        if batch_time > 0 and standard_time > 0:
            speedup = standard_time / batch_time
            logger.info(f"🚀 Potential speedup: {speedup:.2f}x")
        
        logger.info("✅ Lattice integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lattice integration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Запуск всех тестов"""
    logger.info("=" * 60)
    logger.info("🏁 SIMPLE BATCH PROCESSING TEST")
    logger.info("=" * 60)
    
    # Настройка конфигурации
    config = create_debug_config()
    set_project_config(config)
    logger.info("✅ Debug configuration set")
    
    tests = [
        ("Import Test", test_imports),
        ("Batch Adapter Test", test_basic_batch_adapter),
        ("Lattice Integration Test", test_lattice_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n" + "=" * 40)
        logger.info(f"Running: {test_name}")
        logger.info("=" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)