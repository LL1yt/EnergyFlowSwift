#!/usr/bin/env python3
"""
Test Batch Processing
====================

Тесты для проверки корректности batch обработки и сравнения производительности.
"""

import torch
import pytest
import time
import numpy as np
from typing import Dict, List

from new_rebuild.core.moe.batch import BatchMoEProcessor, BatchProcessingAdapter
from new_rebuild.core.lattice import Lattice3D
from new_rebuild.core.lattice.spatial_optimization.batch_integration import (
    create_batch_optimized_spatial_optimizer,
    upgrade_lattice_to_batch
)
from new_rebuild.config import create_debug_config, set_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


class TestBatchProcessing:
    """Тесты для batch обработки"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Настройка для каждого теста"""
        # Создаем debug конфигурацию
        config = create_debug_config()
        config.effective_max_chunk_size = 8  # Оптимальный размер chunk'а
        set_project_config(config)
        
    def test_batch_vs_percell_correctness(self):
        """Проверка что batch и per-cell дают одинаковые результаты"""
        logger.info("🧪 Testing batch vs per-cell correctness...")
        
        # Создаем небольшую решетку для теста
        dimensions = (8, 8, 8)
        lattice = Lattice3D(dimensions=dimensions)
        
        # Сохраняем начальные состояния
        initial_states = lattice.get_states().clone()
        
        # Forward pass с per-cell обработкой
        lattice.spatial_optimizer.set_batch_enabled(False) if hasattr(lattice.spatial_optimizer, 'set_batch_enabled') else None
        percell_result = lattice.forward()
        percell_states = percell_result.clone()
        
        # Сбрасываем состояния
        lattice.set_states(initial_states)
        
        # Обновляем решетку для batch обработки
        lattice = upgrade_lattice_to_batch(lattice)
        lattice.set_batch_enabled(True)
        
        # Forward pass с batch обработкой
        batch_result = lattice.forward()
        batch_states = batch_result.clone()
        
        # Сравниваем результаты
        max_diff = torch.max(torch.abs(percell_states - batch_states)).item()
        mean_diff = torch.mean(torch.abs(percell_states - batch_states)).item()
        
        logger.info(f"Max difference: {max_diff:.6f}")
        logger.info(f"Mean difference: {mean_diff:.6f}")
        
        # Проверяем что различия минимальны (с учетом floating point ошибок)
        assert max_diff < 1e-5, f"Results differ too much: max_diff={max_diff}"
        assert mean_diff < 1e-6, f"Results differ too much: mean_diff={mean_diff}"
        
        logger.info("✅ Batch and per-cell processing produce identical results!")
    
    def test_batch_performance_improvement(self):
        """Тест улучшения производительности с batch обработкой"""
        logger.info("🧪 Testing batch performance improvement...")
        
        # Тестируем на решетке среднего размера
        dimensions = (15, 15, 15)
        lattice = Lattice3D(dimensions=dimensions)
        
        # Измеряем per-cell производительность
        logger.info("Testing per-cell performance...")
        percell_times = []
        for i in range(3):
            start = time.time()
            lattice.forward()
            percell_times.append(time.time() - start)
        
        avg_percell_time = np.mean(percell_times[1:])  # Пропускаем первый (прогрев)
        logger.info(f"Per-cell average time: {avg_percell_time*1000:.1f}ms")
        
        # Обновляем для batch обработки
        lattice = upgrade_lattice_to_batch(lattice)
        lattice.set_batch_enabled(True)
        
        # Измеряем batch производительность
        logger.info("Testing batch performance...")
        batch_times = []
        for i in range(3):
            start = time.time()
            lattice.forward()
            batch_times.append(time.time() - start)
        
        avg_batch_time = np.mean(batch_times[1:])  # Пропускаем первый (прогрев)
        logger.info(f"Batch average time: {avg_batch_time*1000:.1f}ms")
        
        # Вычисляем ускорение
        speedup = avg_percell_time / avg_batch_time if avg_batch_time > 0 else 0
        logger.info(f"🚀 Speedup: {speedup:.2f}x")
        
        # Получаем детальную статистику
        perf_report = lattice.get_batch_performance()
        logger.info(f"Performance report: {perf_report}")
        
        # Проверяем что есть улучшение
        assert speedup > 1.5, f"Expected significant speedup, got only {speedup:.2f}x"
        
        logger.info("✅ Batch processing shows significant performance improvement!")
    
    def test_batch_adapter_fallback(self):
        """Тест fallback механизма при ошибках"""
        logger.info("🧪 Testing batch adapter fallback...")
        
        from new_rebuild.core.moe.moe_processor import MoEConnectionProcessor
        
        # Создаем MoE процессор
        dimensions = (8, 8, 8)
        moe = MoEConnectionProcessor(lattice_dimensions=dimensions)
        
        # Создаем адаптер с включенным fallback
        adapter = BatchProcessingAdapter(
            moe_processor=moe,
            enable_batch=True,
            fallback_on_error=True
        )
        
        # Создаем тестовые данные
        total_cells = np.prod(dimensions)
        states = torch.randn(total_cells, moe.state_size)
        cell_indices = [0, 10, 20, 30]  # Тестовые индексы
        
        # Обрабатываем клетки
        result = adapter.process_cells(
            cell_indices=cell_indices,
            full_lattice_states=states
        )
        
        # Проверяем результат
        assert len(result) == len(cell_indices), "Should process all cells"
        for idx in cell_indices:
            assert idx in result, f"Cell {idx} not processed"
            assert result[idx].shape == (moe.state_size,), f"Wrong shape for cell {idx}"
        
        # Проверяем статистику
        stats = adapter.get_performance_comparison()
        logger.info(f"Adapter stats: {stats}")
        
        logger.info("✅ Batch adapter works correctly with fallback!")
    
    def test_different_chunk_sizes(self):
        """Тест производительности с разными размерами chunk'ов"""
        logger.info("🧪 Testing different chunk sizes...")
        
        dimensions = (15, 15, 15)
        results = []
        
        for chunk_size in [4, 8, 16, 32]:
            logger.info(f"\nTesting chunk size: {chunk_size}")
            
            # Создаем конфигурацию с заданным размером chunk'а
            config = create_debug_config()
            config.effective_max_chunk_size = chunk_size
            set_project_config(config)
            
            # Создаем решетку с batch оптимизацией
            lattice = Lattice3D(dimensions=dimensions)
            lattice = upgrade_lattice_to_batch(lattice)
            lattice.set_batch_enabled(True)
            
            # Измеряем производительность
            times = []
            for _ in range(3):
                start = time.time()
                lattice.forward()
                times.append(time.time() - start)
            
            avg_time = np.mean(times[1:])  # Пропускаем первый
            
            results.append({
                "chunk_size": chunk_size,
                "avg_time_ms": avg_time * 1000,
                "performance": lattice.get_batch_performance()
            })
            
            logger.info(f"Chunk size {chunk_size}: {avg_time*1000:.1f}ms")
        
        # Анализируем результаты
        best_result = min(results, key=lambda x: x["avg_time_ms"])
        logger.info(f"\n🏆 Best chunk size: {best_result['chunk_size']} "
                   f"with {best_result['avg_time_ms']:.1f}ms")
        
        logger.info("✅ Chunk size analysis completed!")


def run_performance_benchmark():
    """Запуск benchmark'а производительности"""
    logger.info("=" * 60)
    logger.info("🏁 BATCH PROCESSING PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    # Настройка
    config = create_debug_config()
    set_project_config(config)
    
    test_dimensions = [(8, 8, 8), (15, 15, 15), (20, 20, 20)]
    results = []
    
    for dims in test_dimensions:
        logger.info(f"\n📊 Testing lattice {dims}...")
        
        # Per-cell baseline
        lattice = Lattice3D(dimensions=dims)
        percell_times = []
        for _ in range(5):
            start = time.time()
            lattice.forward()
            percell_times.append(time.time() - start)
        avg_percell = np.mean(percell_times[1:]) * 1000  # ms
        
        # Batch optimized
        lattice = upgrade_lattice_to_batch(lattice)
        lattice.set_batch_enabled(True)
        batch_times = []
        for _ in range(5):
            start = time.time()
            lattice.forward()
            batch_times.append(time.time() - start)
        avg_batch = np.mean(batch_times[1:]) * 1000  # ms
        
        speedup = avg_percell / avg_batch if avg_batch > 0 else 0
        
        results.append({
            "dimensions": dims,
            "total_cells": np.prod(dims),
            "percell_ms": avg_percell,
            "batch_ms": avg_batch,
            "speedup": speedup
        })
        
        logger.info(f"Per-cell: {avg_percell:.1f}ms, Batch: {avg_batch:.1f}ms, Speedup: {speedup:.2f}x")
    
    # Итоговая таблица
    logger.info("\n" + "=" * 60)
    logger.info("📈 BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Lattice':<15} {'Cells':<10} {'Per-cell':<12} {'Batch':<12} {'Speedup':<10}")
    logger.info("-" * 60)
    
    for r in results:
        dims_str = f"{r['dimensions'][0]}x{r['dimensions'][1]}x{r['dimensions'][2]}"
        logger.info(
            f"{dims_str:<15} {r['total_cells']:<10} "
            f"{r['percell_ms']:<12.1f} {r['batch_ms']:<12.1f} "
            f"{r['speedup']:<10.2f}x"
        )
    
    avg_speedup = np.mean([r["speedup"] for r in results])
    logger.info(f"\n🎯 Average speedup: {avg_speedup:.2f}x")
    

if __name__ == "__main__":
    # Запуск тестов
    test = TestBatchProcessing()
    test.setup()
    
    try:
        test.test_batch_vs_percell_correctness()
        test.test_batch_performance_improvement()
        test.test_batch_adapter_fallback()
        test.test_different_chunk_sizes()
        
        # Запуск полного benchmark'а
        run_performance_benchmark()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise