#!/usr/bin/env python3
"""
Новый тест forward pass с фокусом на batch обработку
Тестирует нашу новую batch архитектуру напрямую
"""

import torch
import time
import logging
from pathlib import Path
import sys

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent / "new_rebuild"
sys.path.insert(0, str(project_root))

from new_rebuild.config import create_debug_config, set_project_config
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.lattice.spatial_optimization.batch_integration import upgrade_lattice_to_batch
from new_rebuild.utils.logging import get_logger


def setup_logging():
    """Настройка детального логирования для отладки batch обработки"""
    # Включаем debug_verbose уровень
    logger = get_logger()
    logger.setLevel(11)  # DEBUG_VERBOSE = 11
    
    # Также для root logger
    logging.getLogger().setLevel(11)
    
    # Специальные уровни для batch диагностики
    logging.getLogger("new_rebuild.core.moe.batch").setLevel(11)
    logging.getLogger("new_rebuild.core.lattice.spatial_optimization").setLevel(11)
    
    print("🔍 Verbose logging enabled for batch processing diagnostics")


def test_direct_batch_processing():
    """Тест прямой batch обработки без chunking"""
    print("\n" + "="*80)
    print("🚀 TESTING DIRECT BATCH PROCESSING")
    print("="*80)
    
    # Создаем конфигурацию для batch режима
    config = create_debug_config()
    # Проверяем и настраиваем batch обработку
    if hasattr(config, 'performance') and config.performance is not None:
        config.performance.enable_batch_processing = True
        print("✅ Batch processing enabled via config")
    else:
        print("⚠️ Performance config not available, using defaults")
    set_project_config(config)
    
    # Создаем решетку 10x10x10 (1000 клеток) для тестирования
    dimensions = (10, 10, 10)
    print(f"📊 Creating lattice: {dimensions} = {dimensions[0]*dimensions[1]*dimensions[2]} cells")
    
    # Обновляем конфигурацию с нужными размерами
    config.lattice.dimensions = dimensions
    set_project_config(config)
    
    lattice = Lattice3D()
    
    # Обновляем для batch режима
    print("🔧 Upgrading to batch processing...")
    lattice = upgrade_lattice_to_batch(lattice)
    lattice.set_batch_enabled(True)
    
    # Входные данные
    batch_size = 1
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    feature_size = 24
    
    input_states = torch.randn(batch_size, total_cells, feature_size, dtype=torch.float32)
    print(f"📊 Input shape: {input_states.shape}")
    
    # Тест forward pass
    print("\n🔄 Running batch forward pass...")
    start_time = time.time()
    
    try:
        output = lattice.forward(input_states)
        elapsed = time.time() - start_time
        
        print(f"✅ Forward pass completed in {elapsed:.3f}s")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output stats: mean={output.mean():.6f}, std={output.std():.6f}")
        
        # Проверяем, что выход имеет правильную форму
        expected_shape = input_states.shape
        if output.shape == expected_shape:
            print("✅ Output shape is correct")
        else:
            print(f"❌ Output shape mismatch: expected {expected_shape}, got {output.shape}")
            
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Forward pass failed after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def test_batch_consistency():
    """Тест консистентности batch обработки"""
    print("\n" + "="*80)
    print("🔄 TESTING BATCH CONSISTENCY")
    print("="*80)
    
    config = create_debug_config()
    if hasattr(config, 'performance') and config.performance is not None:
        config.performance.enable_batch_processing = True
    set_project_config(config)
    
    # Маленькая решетка для быстрого тестирования
    dimensions = (5, 5, 5)
    config.lattice.dimensions = dimensions
    set_project_config(config)
    
    lattice = Lattice3D()
    lattice = upgrade_lattice_to_batch(lattice)
    lattice.set_batch_enabled(True)
    
    total_cells = dimensions[0] * dimensions[1] * dimensions[2]
    input_states = torch.randn(1, total_cells, 24, dtype=torch.float32)
    
    print(f"📊 Testing consistency with {total_cells} cells")
    
    # Запускаем несколько раз с одним и тем же входом
    outputs = []
    times = []
    
    for i in range(3):
        print(f"\n--- Run {i+1}/3 ---")
        start_time = time.time()
        
        try:
            output = lattice.forward(input_states)
            elapsed = time.time() - start_time
            
            outputs.append(output)
            times.append(elapsed)
            
            print(f"✅ Run {i+1} completed in {elapsed:.3f}s")
            print(f"📊 Output stats: mean={output.mean():.6f}, std={output.std():.6f}")
            
        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            return False
    
    # Проверяем консистентность
    if len(outputs) >= 2:
        diff = torch.abs(outputs[0] - outputs[1]).max().item()
        print(f"\n🔍 Max difference between runs: {diff:.8f}")
        
        if diff < 1e-6:
            print("✅ Outputs are consistent!")
        else:
            print("⚠️ Outputs have significant differences")
    
    avg_time = sum(times) / len(times)
    print(f"📊 Average time: {avg_time:.3f}s")
    
    return True


def test_performance_comparison():
    """Сравнение производительности разных размеров решеток"""
    print("\n" + "="*80)
    print("⚡ PERFORMANCE COMPARISON")
    print("="*80)
    
    config = create_debug_config()
    if hasattr(config, 'performance') and config.performance is not None:
        config.performance.enable_batch_processing = True
    set_project_config(config)
    
    test_sizes = [
        (20, 20, 20), # 1728 клеток
    ]
    
    results = []
    
    for dimensions in test_sizes:
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]
        print(f"\n📊 Testing {dimensions} = {total_cells} cells")
        
        try:
            config.lattice.dimensions = dimensions
            set_project_config(config)
            
            lattice = Lattice3D()
            lattice = upgrade_lattice_to_batch(lattice)
            lattice.set_batch_enabled(True)
            
            input_states = torch.randn(1, total_cells, 24, dtype=torch.float32)
            
            # Прогрев
            _ = lattice.forward(input_states)
            
            # Измерение
            start_time = time.time()
            output = lattice.forward(input_states)
            elapsed = time.time() - start_time
            
            cells_per_second = total_cells / elapsed
            
            print(f"✅ {total_cells:4d} cells in {elapsed:.3f}s = {cells_per_second:.0f} cells/sec")
            
            results.append({
                'dimensions': dimensions,
                'total_cells': total_cells,
                'time': elapsed,
                'cells_per_second': cells_per_second
            })
            
        except Exception as e:
            print(f"❌ Failed for {dimensions}: {e}")
            results.append({
                'dimensions': dimensions,
                'total_cells': total_cells,
                'time': None,
                'cells_per_second': 0
            })
    
    # Сводка результатов
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("Dimensions     | Cells | Time    | Cells/sec")
    print("-" * 45)
    
    for result in results:
        dims = f"{result['dimensions'][0]}×{result['dimensions'][1]}×{result['dimensions'][2]}"
        cells = result['total_cells']
        time_str = f"{result['time']:.3f}s" if result['time'] else "FAILED"
        cps = int(result['cells_per_second'])
        
        print(f"{dims:<14} | {cells:5d} | {time_str:<7} | {cps:8d}")


def main():
    """Основная функция тестирования"""
    print("🚀 BATCH PROCESSING FORWARD PASS TEST V3")
    print("=" * 80)
    
    setup_logging()
    
    # Тест 1: Прямая batch обработка
    success, time_taken = test_direct_batch_processing()
    
    if not success:
        print("❌ Direct batch processing test failed. Stopping.")
        return False
    
    # Тест 2: Консистентность
    if not test_batch_consistency():
        print("❌ Consistency test failed.")
        return False
    
    # Тест 3: Сравнение производительности
    test_performance_comparison()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)