#!/usr/bin/env python3
"""
🧪 TEST: Surface-Only Processing Integration
Тестирование интеграции surface-only режима в EmbeddingProcessor для Universal Adapter

Проверяет:
1. Создание surface-only конфигурации
2. Инициализация EmbeddingProcessor в SURFACE_ONLY режиме
3. Обработка surface embeddings (225D)
4. Batch processing
5. Gradient flow для training готовности
6. Совместимость с Universal Adapter

Автор: 3D Cellular Neural Network Project
Дата: 7 июня 2025
"""

import torch
import numpy as np
import logging
from typing import Dict, Any
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_surface_only_config():
    """Тест создания surface-only конфигурации"""
    print("[CONFIG] Testing surface-only configuration creation...")
    
    try:
        from core.embedding_processor import create_surface_only_config, ProcessingMode
        
        # Создание конфигурации
        config = create_surface_only_config(
            surface_size=225,
            surface_dims=(15, 15)
        )
        
        # Проверка параметров
        assert config.processing_mode == ProcessingMode.SURFACE_ONLY, "Неверный режим"
        assert config.input_dim == 225, f"Неверный input_dim: {config.input_dim}"
        assert config.output_dim == 225, f"Неверный output_dim: {config.output_dim}"
        assert config.surface_dimensions == (15, 15), f"Неверные surface_dimensions: {config.surface_dimensions}"
        assert config.surface_processing_depth == 11, f"Неверный processing_depth: {config.surface_processing_depth}"
        
        print("[OK] Surface-only конфигурация создана успешно")
        print(f"   Mode: {config.processing_mode.value}")
        print(f"   Dimensions: {config.input_dim}D → {config.output_dim}D")
        print(f"   Surface: {config.surface_dimensions}")
        print(f"   Depth: {config.surface_processing_depth}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка в surface-only конфигурации: {e}")
        return False


def test_surface_only_processor_initialization():
    """Тест инициализации EmbeddingProcessor в surface-only режиме"""
    print("\n[START] Testing EmbeddingProcessor initialization for surface-only mode...")
    
    try:
        from core.embedding_processor import EmbeddingProcessor, create_surface_only_config
        
        # Создание конфигурации
        config = create_surface_only_config(surface_size=225, surface_dims=(15, 15))
        
        # Инициализация процессора
        processor = EmbeddingProcessor(config)
        
        # Проверка компонентов
        assert processor.reshaper is None, "EmbeddingReshaper должен быть None для surface-only"
        assert processor.lattice is None, "Lattice3D должен быть None для surface-only"
        assert processor.metrics is not None, "ProcessingMetrics должны быть созданы"
        
        # Проверка настроек
        assert processor.config.processing_mode.value == "surface_only", "Неверный режим процессора"
        
        print("[OK] EmbeddingProcessor инициализирован успешно")
        print(f"   Режим: {processor.config.processing_mode.value}")
        print(f"   EmbeddingReshaper: {'[ERROR] Пропущен' if processor.reshaper is None else '[OK] Создан'}")
        print(f"   Lattice3D: {'[ERROR] Пропущен' if processor.lattice is None else '[OK] Создан'}")
        print(f"   Metrics: {'[OK] Созданы' if processor.metrics is not None else '[ERROR] Отсутствуют'}")
        
        return processor
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации процессора: {e}")
        return None


def test_surface_embedding_processing(processor):
    """Тест обработки surface embeddings"""
    print("\n[BRAIN] Testing surface embedding processing...")
    
    try:
        # Создание тестового surface embedding
        surface_embedding = torch.randn(225, dtype=torch.float32, requires_grad=True)
        
        print(f"📥 Input surface: {surface_embedding.shape}")
        
        # Обработка через processor
        start_time = time.time()
        processed_surface = processor.forward(surface_embedding)
        processing_time = time.time() - start_time
        
        print(f"📤 Output surface: {processed_surface.shape}")
        print(f"⏱️  Processing time: {processing_time:.4f}s")
        
        # Проверка результата
        assert processed_surface.shape == surface_embedding.shape, f"Shape mismatch: {processed_surface.shape} != {surface_embedding.shape}"
        assert processed_surface.dtype == torch.float32, f"Dtype mismatch: {processed_surface.dtype}"
        assert processed_surface.requires_grad, "Output должен поддерживать градиенты"
        
        # Проверка качества
        similarity = torch.cosine_similarity(surface_embedding, processed_surface, dim=0).item()
        print(f"[DATA] Cosine similarity: {similarity:.4f}")
        
        # Проверка что это не identity transformation
        l2_distance = torch.norm(surface_embedding - processed_surface, p=2).item()
        print(f"📏 L2 distance: {l2_distance:.4f}")
        
        assert l2_distance > 0.01, "Слишком малое изменение - возможно identity transformation"
        
        print("[OK] Surface embedding обработка успешна")
        
        return True, similarity
        
    except Exception as e:
        print(f"[ERROR] Ошибка обработки surface embedding: {e}")
        return False, 0.0


def test_batch_processing(processor):
    """Тест batch обработки surface embeddings"""
    print("\n[PACKAGE] Testing batch processing...")
    
    try:
        # Создание batch surface embeddings
        batch_size = 4
        batch_surfaces = torch.randn(batch_size, 225, dtype=torch.float32, requires_grad=True)
        
        print(f"📥 Input batch: {batch_surfaces.shape}")
        
        # Batch обработка
        start_time = time.time()
        batch_processed = processor.forward(batch_surfaces)
        processing_time = time.time() - start_time
        
        print(f"📤 Output batch: {batch_processed.shape}")
        print(f"⏱️  Batch processing time: {processing_time:.4f}s")
        print(f"[FAST] Throughput: {batch_size / processing_time:.1f} samples/sec")
        
        # Проверка результата
        assert batch_processed.shape == batch_surfaces.shape, f"Batch shape mismatch: {batch_processed.shape} != {batch_surfaces.shape}"
        assert batch_processed.requires_grad, "Batch output должен поддерживать градиенты"
        
        # Проверка качества для каждого примера
        similarities = []
        for i in range(batch_size):
            sim = torch.cosine_similarity(batch_surfaces[i], batch_processed[i], dim=0).item()
            similarities.append(sim)
            print(f"   Sample {i}: similarity = {sim:.3f}")
        
        avg_similarity = np.mean(similarities)
        print(f"[DATA] Average batch similarity: {avg_similarity:.4f}")
        
        print("[OK] Batch processing успешен")
        
        return True, avg_similarity
        
    except Exception as e:
        print(f"[ERROR] Ошибка batch processing: {e}")
        return False, 0.0


def test_gradient_flow(processor):
    """Тест gradient flow для training готовности"""
    print("\n[REFRESH] Testing gradient flow for training readiness...")
    
    try:
        # Создание тестовых данных с градиентами
        surface_input = torch.randn(2, 225, dtype=torch.float32, requires_grad=True)
        target_surface = torch.randn(2, 225, dtype=torch.float32)
        
        # Forward pass
        output_surface = processor.forward(surface_input)
        
        # Простая loss function для тестирования
        loss = torch.nn.functional.mse_loss(output_surface, target_surface)
        
        print(f"[DATA] Test loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Проверка gradients
        assert surface_input.grad is not None, "Input gradients отсутствуют"
        grad_norm = torch.norm(surface_input.grad).item()
        print(f"[REFRESH] Gradient norm: {grad_norm:.6f}")
        
        assert grad_norm > 1e-8, f"Слишком малые градиенты: {grad_norm}"
        
        print("[OK] Gradient flow работает корректно")
        print("[OK] Система готова к training")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка gradient flow: {e}")
        return False


def test_universal_adapter_compatibility():
    """Тест совместимости с Universal Adapter размерами"""
    print("\n[LINK] Testing Universal Adapter compatibility...")
    
    try:
        from core.embedding_processor import EmbeddingProcessor, create_surface_only_config
        
        # Тестирование различных размеров от Universal Adapter
        test_cases = [
            {"name": "LLaMA-3-8B", "surface_size": 225, "dims": (15, 15)},     # 4096D → 225D
            {"name": "Custom-512", "surface_size": 256, "dims": (16, 16)},      # 512D → 256D  
            {"name": "Large-1024", "surface_size": 400, "dims": (20, 20)},     # 1024D → 400D
        ]
        
        for test_case in test_cases:
            print(f"\n   Testing {test_case['name']}: {test_case['surface_size']}D surface...")
            
            # Создание конфигурации
            config = create_surface_only_config(
                surface_size=test_case['surface_size'], 
                surface_dims=test_case['dims']
            )
            
            # Создание процессора
            processor = EmbeddingProcessor(config)
            
            # Тестирование обработки
            test_surface = torch.randn(test_case['surface_size'], dtype=torch.float32, requires_grad=True)
            processed = processor.forward(test_surface)
            
            # Проверка
            assert processed.shape == test_surface.shape, f"Shape mismatch для {test_case['name']}"
            
            similarity = torch.cosine_similarity(test_surface, processed, dim=0).item()
            print(f"     [OK] {test_case['name']}: similarity = {similarity:.3f}")
        
        print("[OK] Universal Adapter compatibility подтверждена")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка Universal Adapter compatibility: {e}")
        return False


def run_comprehensive_test():
    """Запуск полного набора тестов"""
    print("🧪 COMPREHENSIVE SURFACE-ONLY INTEGRATION TEST")
    print("=" * 60)
    
    results = {}
    
    # 1. Тест конфигурации
    results['config'] = test_surface_only_config()
    
    # 2. Тест инициализации
    processor = test_surface_only_processor_initialization()
    results['initialization'] = processor is not None
    
    if processor is not None:
        # 3. Тест обработки single embedding
        success, similarity = test_surface_embedding_processing(processor)
        results['single_processing'] = success
        results['single_similarity'] = similarity
        
        # 4. Тест batch processing
        success, batch_similarity = test_batch_processing(processor)
        results['batch_processing'] = success
        results['batch_similarity'] = batch_similarity
        
        # 5. Тест gradient flow
        results['gradient_flow'] = test_gradient_flow(processor)
    
    # 6. Тест Universal Adapter compatibility
    results['adapter_compatibility'] = test_universal_adapter_compatibility()
    
    # Финальный отчет
    print("\n" + "=" * 60)
    print("[DATA] FINAL TEST RESULTS:")
    print("=" * 60)
    
    total_tests = len([k for k in results.keys() if k.endswith('_similarity')==False])
    passed_tests = sum([1 for k, v in results.items() if k.endswith('_similarity')==False and v])
    
    for test_name, result in results.items():
        if not test_name.endswith('_similarity'):
            status = "[OK] PASS" if result else "[ERROR] FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
    
    if 'single_similarity' in results:
        print(f"[DATA] Single processing similarity: {results['single_similarity']:.3f}")
    if 'batch_similarity' in results:
        print(f"[DATA] Batch processing similarity: {results['batch_similarity']:.3f}")
    
    print(f"\n[TARGET] OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("[SUCCESS] ALL TESTS PASSED! Surface-only integration готова к использованию!")
        print("[START] Ready for Stage 3.1.2 integration with Universal Adapter!")
    else:
        print("[WARNING]  Some tests failed. Review errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1) 