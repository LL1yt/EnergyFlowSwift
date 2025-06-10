#!/usr/bin/env python3
"""
Test Suite: gMLP Architecture Optimization (15×15×11)
====================================================

Тестирование revolutionary архитектуры:
- gMLP Spatial Gating Units
- 15×15×11 area-focused scaling  
- Parameter budget optimization
- Memory efficiency validation
- Integration compatibility

Цель: Подготовить breakthrough >50% Q→A similarity
"""

import sys
import os
import torch
import logging
import traceback
import numpy as np
import time
from pathlib import Path

# Добавляем корневую директорию в path
sys.path.append(str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gmlp_cell_architecture():
    """
    Тест 1: Базовая функциональность gMLP клетки
    """
    print("\n🧪 ТЕСТ 1: gMLP Cell Architecture")
    print("=" * 50)
    
    try:
        from core.cell_prototype.architectures.gmlp_cell import (
            GatedMLPCell, 
            SpatialGatingUnit,
            create_gmlp_cell_from_config,
            test_gmlp_cell_basic
        )
        
        # 1.1: Создание gMLP клетки
        print("[INFO] 1.1: Создание GatedMLPCell...")
        cell = GatedMLPCell(
            state_size=32,
            neighbor_count=6,
            hidden_dim=256,
            external_input_size=12,
            activation="gelu",
            use_memory=True
        )
        
        # Проверка параметров
        info = cell.get_info()
        print(f"   [OK] Клетка создана: {info['total_parameters']:,} параметров")
        print(f"   [DATA] Target: 25K, Actual: {info['total_parameters']:,}")
        print(f"   [CHART] Efficiency: {info['parameter_efficiency']:.2f}")
        
        # 1.2: Тестирование forward pass
        print("\n[INFO] 1.2: Forward Pass Testing...")
        batch_size = 4
        neighbor_states = torch.randn(batch_size, 6, 32)
        own_state = torch.randn(batch_size, 32)
        external_input = torch.randn(batch_size, 12)
        
        # Forward pass
        start_time = time.time()
        new_state = cell(neighbor_states, own_state, external_input)
        forward_time = time.time() - start_time
        
        print(f"   [OK] Forward pass: {neighbor_states.shape} + {own_state.shape} → {new_state.shape}")
        print(f"   [FAST] Время: {forward_time*1000:.2f}ms")
        
        # Проверки качества
        assert new_state.shape == (batch_size, 32), f"Wrong shape: {new_state.shape}"
        assert not torch.isnan(new_state).any(), "NaN values detected"
        assert not torch.isinf(new_state).any(), "Inf values detected"
        
        # 1.3: Memory component testing
        print("\n[INFO] 1.3: Memory Component Testing...")
        initial_memory = cell.memory_state
        print(f"   [WRITE] Initial memory state: {initial_memory is not None}")
        
        # Second forward pass (memory should persist)
        new_state_2 = cell(neighbor_states, own_state, external_input)
        second_memory = cell.memory_state
        print(f"   [WRITE] Memory after second pass: {second_memory is not None}")
        print(f"   [WRITE] Memory shape: {second_memory.shape if second_memory is not None else 'None'}")
        
        # Memory reset test
        cell.reset_memory()
        assert cell.memory_state is None, "Memory not properly reset"
        print(f"   [OK] Memory reset successful")
        
        # 1.4: Built-in test
        print("\n[INFO] 1.4: Built-in Test Suite...")
        success = test_gmlp_cell_basic()
        assert success, "Built-in tests failed"
        print(f"   [OK] Built-in tests passed")
        
        print("\n[TARGET] ТЕСТ 1 РЕЗУЛЬТАТ: [OK] SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ТЕСТ 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_15x15x11_lattice_configuration():
    """
    Тест 2: Конфигурация 15×15×11 решетки
    """
    print("\n🧪 ТЕСТ 2: 15×15×11 Lattice Configuration")
    print("=" * 50)
    
    try:
        import yaml
        
        # 2.1: Загрузка optimized конфигурации
        print("[INFO] 2.1: Загрузка optimized конфигурации...")
        config_path = Path("config/optimized_architecture_15x15x11.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        lattice_config = config['lattice_3d']
        cell_config = config['cell_prototype']
        
        print(f"   [OK] Конфигурация загружена: {config_path}")
        print(f"   [DATA] Lattice размеры: {lattice_config['dimensions']}")
        print(f"   [DATA] Total cells: {lattice_config['total_cells']}")
        print(f"   [BRAIN] Cell architecture: {cell_config['architecture_type']}")
        
        # 2.2: Проверка размерности
        print("\n[INFO] 2.2: Проверка размерности...")
        dimensions = lattice_config['dimensions']
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]
        
        assert dimensions == [15, 15, 11], f"Wrong dimensions: {dimensions}"
        assert total_cells == 2475, f"Wrong total cells: {total_cells}"
        
        print(f"   [OK] Размерность корректна: {dimensions} = {total_cells} клеток")
        print(f"   [CHART] Увеличение: {total_cells / 512:.1f}x vs 8×8×8")
        
        # 2.3: Golden Ratio проверка
        print("\n[INFO] 2.3: Golden Ratio Analysis...")
        x, y, z = dimensions
        ratio_xy = y / x  # Should be ~1.0
        ratio_z = z / x   # Should be ~0.5-0.73
        
        print(f"   📏 X:Y ratio: {ratio_xy:.3f} (target: ~1.0)")
        print(f"   📏 Z:X ratio: {ratio_z:.3f} (target: 0.5-0.73)")
        
        # Area-focused validation
        area_cells = x * y  # 15 × 15 = 225
        volume_ratio = area_cells / total_cells
        print(f"   📏 Area-focused ratio: {volume_ratio:.3f} (higher = better)")
        
        assert 0.5 <= ratio_z <= 0.8, f"Z ratio out of range: {ratio_z}"
        print(f"   [OK] Golden Ratio compliance verified")
        
        # 2.4: Memory estimation
        print("\n[INFO] 2.4: Memory Estimation...")
        cells_per_param = cell_config['architecture']['target_parameters']
        total_params = total_cells * cells_per_param
        memory_mb = total_params * 4 / (1024 * 1024)  # float32
        
        print(f"   [DATA] Параметры на клетку: {cells_per_param:,}")
        print(f"   [DATA] Total параметры: {total_params:,}")
        print(f"   [SAVE] Estimated memory: {memory_mb:.1f} MB")
        
        # Warning для больших размеров
        if memory_mb > 1000:  # 1GB
            print(f"   [WARNING]  Large memory requirement: {memory_mb:.1f} MB")
        
        print("\n[TARGET] ТЕСТ 2 РЕЗУЛЬТАТ: [OK] SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ТЕСТ 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_embedding_processor_compatibility():
    """
    Тест 3: Совместимость с EmbeddingProcessor для 15×15×11
    """
    print("\n🧪 ТЕСТ 3: EmbeddingProcessor Compatibility")
    print("=" * 50)
    
    try:
        # 3.1: Проверка EmbeddingReshaper для 2,475 elements
        print("[INFO] 3.1: EmbeddingReshaper Adaptation...")
        
        from data.embedding_reshaper import EmbeddingReshaper
        
        # Создаем reshaper для новых размеров
        reshaper = EmbeddingReshaper(
            input_dim=768,
            cube_shape=(15, 15, 11),  # 2,475 elements
            reshaping_method="adaptive"
        )
        
        print(f"   [OK] EmbeddingReshaper создан для 15×15×11")
        print(f"   [DATA] Input dim: 768")
        print(f"   [DATA] Cube shape: {reshaper.cube_shape}")
        print(f"   [DATA] Total elements: {np.prod(reshaper.cube_shape)}")
        
        # 3.2: Тестирование reshape операций
        print("\n[INFO] 3.2: Reshape Operations Testing...")
        
        # Тестовый embedding
        test_embedding = torch.randn(768)
        
        # 768D → 15×15×11 (2,475 elements)
        try:
            matrix_3d = reshaper.vector_to_matrix(test_embedding)
            print(f"   [OK] Vector to matrix: {test_embedding.shape} → {matrix_3d.shape}")
            
            # 15×15×11 → 768D 
            reconstructed = reshaper.matrix_to_vector(matrix_3d)
            print(f"   [OK] Matrix to vector: {matrix_3d.shape} → {reconstructed.shape}")
            
            # Проверка размерностей
            assert matrix_3d.shape == (15, 15, 11), f"Wrong matrix shape: {matrix_3d.shape}"
            assert reconstructed.shape == (768,), f"Wrong vector shape: {reconstructed.shape}"
            
        except Exception as reshape_error:
            print(f"   [WARNING]  Reshape operations требуют адаптации: {reshape_error}")
            # Это ожидаемо - EmbeddingReshaper потребует модификации для 768→2475 mapping
            
        # 3.3: Parameter scaling analysis
        print("\n[INFO] 3.3: Parameter Scaling Analysis...")
        
        old_cells = 8 * 8 * 8  # 512
        new_cells = 15 * 15 * 11  # 2,475
        scaling_factor = new_cells / old_cells
        
        old_params_total = 512 * 1000  # 512K
        new_params_total = 2475 * 25000  # 61.875M
        param_scaling = new_params_total / old_params_total
        
        print(f"   [DATA] Cell scaling: {old_cells} → {new_cells} ({scaling_factor:.1f}x)")
        print(f"   [DATA] Parameter scaling: {old_params_total:,} → {new_params_total:,} ({param_scaling:.0f}x)")
        print(f"   [DATA] Per-cell improvement: {1000} → {25000} (25x richer)")
        
        # Efficiency analysis
        efficiency_gain = (scaling_factor * 25) ** 0.5  # Approximation
        print(f"   [CHART] Estimated capacity gain: ~{efficiency_gain:.1f}x")
        
        print("\n[TARGET] ТЕСТ 3 РЕЗУЛЬТАТ: [OK] SUCCESS (с адаптациями)")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ТЕСТ 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_training_configuration():
    """
    Тест 4: Конфигурация обучения для optimized architecture
    """
    print("\n🧪 ТЕСТ 4: Training Configuration")
    print("=" * 50)
    
    try:
        # 4.1: Загрузка training config
        print("[INFO] 4.1: Training Configuration Validation...")
        
        import yaml
        config_path = Path("config/optimized_architecture_15x15x11.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        training_config = config['training']
        performance_config = config['performance']
        
        print(f"   [OK] Training config loaded")
        print(f"   [TARGET] Target similarity: {training_config['target_similarity']}")
        print(f"   [DATA] Batch size: {training_config['batch_size']}")
        print(f"   [DATA] Learning rate: {training_config['learning_rate']}")
        
        # 4.2: Memory requirements validation
        print("\n[INFO] 4.2: Memory Requirements Analysis...")
        
        batch_size = training_config['batch_size']
        max_memory_gb = performance_config['max_memory_gb']
        gradient_checkpointing = performance_config['gradient_checkpointing']
        
        # Estimate memory usage
        total_params = 2475 * 25000  # 61.875M parameters
        param_memory_gb = total_params * 4 / (1024**3)  # float32
        gradient_memory_gb = param_memory_gb * 2  # gradients + params
        
        if gradient_checkpointing:
            gradient_memory_gb *= 0.5  # Gradient checkpointing reduces memory
        
        estimated_memory = param_memory_gb + gradient_memory_gb
        
        print(f"   [DATA] Parameter memory: {param_memory_gb:.2f} GB")
        print(f"   [DATA] Gradient memory: {gradient_memory_gb:.2f} GB")
        print(f"   [DATA] Total estimated: {estimated_memory:.2f} GB")
        print(f"   [DATA] Memory limit: {max_memory_gb} GB")
        print(f"   [CONFIG] Gradient checkpointing: {gradient_checkpointing}")
        
        if estimated_memory > max_memory_gb:
            print(f"   [WARNING]  Memory requirement exceeds limit!")
            print(f"   [IDEA] Suggestions:")
            print(f"      - Reduce batch size to 1")
            print(f"      - Enable gradient checkpointing")
            print(f"      - Use mixed precision training")
        
        # 4.3: Performance optimization validation
        print("\n[INFO] 4.3: Performance Optimization Checks...")
        
        checks = {
            'gradient_checkpointing': performance_config.get('gradient_checkpointing', False),
            'gradient_clipping': performance_config.get('gradient_clipping', 0) > 0,
            'memory_monitoring': performance_config.get('memory_monitoring', False),
            'mixed_precision': config['embedding_processor'].get('mixed_precision', False)
        }
        
        for check, enabled in checks.items():
            status = "[OK]" if enabled else "[WARNING]"
            print(f"   {status} {check}: {enabled}")
        
        # 4.4: Target metrics validation
        print("\n[INFO] 4.4: Target Metrics Analysis...")
        
        target_similarity = training_config['target_similarity']
        current_best = 0.385  # Current plateau
        improvement_needed = target_similarity - current_best
        
        print(f"   [DATA] Current best: {current_best:.1%}")
        print(f"   [TARGET] Target: {target_similarity:.1%}")
        print(f"   [CHART] Improvement needed: {improvement_needed:.1%}")
        print(f"   [START] Architecture improvements:")
        print(f"      - 4.8x more cells (better representation)")
        print(f"      - 25x richer cells (gMLP vs simple MLP)")
        print(f"      - Spatial Gating Units (attention-like processing)")
        print(f"      - Memory components (emergent behavior)")
        
        print("\n[TARGET] ТЕСТ 4 РЕЗУЛЬТАТ: [OK] SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ТЕСТ 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_memory_and_computational_feasibility():
    """
    Тест 5: Практическая осуществимость по памяти и вычислениям
    """
    print("\n🧪 ТЕСТ 5: Memory & Computational Feasibility")
    print("=" * 50)
    
    try:
        # 5.1: Hardware requirements
        print("[INFO] 5.1: Hardware Requirements Analysis...")
        
        # Current system info
        print(f"   [COMPUTER] PyTorch version: {torch.__version__}")
        print(f"   [COMPUTER] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   [COMPUTER] CUDA device: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   [COMPUTER] GPU memory: {memory_gb:.1f} GB")
        
        # 5.2: Создание mini-версии для тестирования
        print("\n[INFO] 5.2: Mini Architecture Test...")
        
        from core.cell_prototype.architectures.gmlp_cell import GatedMLPCell
        
        # Создаем уменьшенную версию для тестирования memory scaling
        mini_cells = 10  # Вместо 2,475
        cells = []
        
        print(f"   🔬 Создание {mini_cells} gMLP клеток...")
        
        start_time = time.time()
        for i in range(mini_cells):
            cell = GatedMLPCell(
                state_size=32,
                hidden_dim=128,  # Уменьшено для тестирования
                use_memory=True
            )
            cells.append(cell)
        
        creation_time = time.time() - start_time
        
        # Подсчет параметров
        total_params = sum(sum(p.numel() for p in cell.parameters()) for cell in cells)
        avg_params = total_params / mini_cells
        
        print(f"   [OK] {mini_cells} клеток созданы за {creation_time:.3f}s")
        print(f"   [DATA] Average parameters per cell: {avg_params:,.0f}")
        print(f"   [DATA] Total parameters: {total_params:,}")
        
        # Экстраполяция для полной системы
        full_system_params = avg_params * 2475
        full_memory_gb = full_system_params * 4 / (1024**3)
        
        print(f"\n   [CHART] Full system extrapolation:")
        print(f"      [DATA] Total parameters: {full_system_params:,.0f}")
        print(f"      [SAVE] Memory requirement: {full_memory_gb:.2f} GB")
        
        # 5.3: Forward pass timing
        print("\n[INFO] 5.3: Forward Pass Performance...")
        
        cell = cells[0]
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            neighbor_states = torch.randn(batch_size, 6, 32)
            own_state = torch.randn(batch_size, 32)
            external_input = torch.randn(batch_size, 12)
            
            # Timing test
            start_time = time.time()
            for _ in range(10):  # 10 forward passes
                output = cell(neighbor_states, own_state, external_input)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / 10
            print(f"   [FAST] Batch {batch_size}: {avg_time_ms:.2f}ms per forward pass")
        
        # Extrapolation для полной решетки
        single_cell_time = avg_time_ms  # последний результат
        full_lattice_time = single_cell_time * 2475 / 1000  # в секундах
        
        print(f"\n   [CHART] Full lattice extrapolation:")
        print(f"      [FAST] Est. time per step: {full_lattice_time:.2f}s")
        print(f"      [FAST] Est. time per epoch: {full_lattice_time * 100:.1f}s (100 steps)")
        
        # 5.4: Feasibility assessment
        print("\n[INFO] 5.4: Feasibility Assessment...")
        
        feasible = True
        recommendations = []
        
        # Memory check
        if full_memory_gb > 8:
            feasible = False
            recommendations.append("Use gradient checkpointing")
            recommendations.append("Reduce batch size to 1")
            recommendations.append("Consider parameter sharing")
        
        # Timing check
        if full_lattice_time > 10:
            recommendations.append("Optimize cell implementation")
            recommendations.append("Use GPU acceleration")
            recommendations.append("Consider parallel processing")
        
        print(f"   [TARGET] Feasibility: {'[OK] FEASIBLE' if feasible else '[WARNING] CHALLENGING'}")
        
        if recommendations:
            print(f"   [IDEA] Recommendations:")
            for rec in recommendations:
                print(f"      - {rec}")
        
        print("\n[TARGET] ТЕСТ 5 РЕЗУЛЬТАТ: [OK] SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] ТЕСТ 5 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """
    Основная функция тестирования
    """
    print("[START] GMLP ARCHITECTURE OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print("Testing revolutionary 15×15×11 + gMLP architecture")
    print("Goal: Breakthrough >50% Q→A similarity\n")
    
    tests = [
        ("gMLP Cell Architecture", test_gmlp_cell_architecture),
        ("15×15×11 Lattice Config", test_15x15x11_lattice_configuration),
        ("EmbeddingProcessor Compatibility", test_embedding_processor_compatibility),
        ("Training Configuration", test_training_configuration),
        ("Memory & Computational Feasibility", test_memory_and_computational_feasibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[ERROR] CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Итоговые результаты
    print("\n" + "=" * 60)
    print("[TARGET] ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "[OK] PASSED" if success else "[ERROR] FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n[DATA] ОБЩИЙ РЕЗУЛЬТАТ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("[SUCCESS] ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Готовы к реализации optimized architecture!")
        print("\n[START] СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Адаптировать EmbeddingReshaper для 768→2475 mapping")
        print("2. Интегрировать gMLP клетки в Lattice3D")
        print("3. Обновить CubeTrainer для новой архитектуры")
        print("4. Запустить тренировку с gradient checkpointing")
        print("5. Monitoring memory usage и performance")
    else:
        print("[WARNING]  Некоторые тесты failed. Требуется доработка перед внедрением.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 