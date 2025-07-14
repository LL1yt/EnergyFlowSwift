#!/usr/bin/env python3
"""
Test GPU Utilization После Оптимизации
=====================================

Этот тест проверяет улучшение GPU утилизации после применения оптимизаций:
1. Исправлены memory leaks в training loop (векторизованная spatial consistency)
2. Убраны блокирующие torch.cuda.synchronize() операции
3. Увеличены workers до 16 для RTX 5090 (pin_memory=True, prefetch=16)
4. GPU-direct data loading (загрузка сразу на GPU)
5. Векторизованные spatial operations (убраны nested loops)
6. Параллельная обработка экспертов с CUDA streams
7. Отключение debug режима

Ожидаемый результат: GPU утилизация должна увеличиться с 25% до 80-90%
"""

import torch
import time
import psutil
import os
from typing import Dict, Any
import threading
import sys
import subprocess

# Добавляем путь к проекту
sys.path.append('/mnt/c/Users/n0n4a/projects/AA')

from new_rebuild.config.simple_config import SimpleProjectConfig
from new_rebuild.config.config_components import ConfigMode, ModeSettings
from new_rebuild.core.training.embedding_trainer import EmbeddingTrainer
from new_rebuild.core.training.utils.unified_dataset_loader import create_training_dataloader
from new_rebuild.utils.device_manager import get_device_manager
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)

class GPUMonitor:
    """Мониторинг GPU утилизации в реальном времени"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'gpu_utilization': [],
            'memory_usage': [],
            'timestamps': []
        }
        self.thread = None
        
    def start_monitoring(self):
        """Запускает мониторинг GPU"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
            
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.monitoring:
            try:
                # Используем nvidia-smi для получения статистики GPU
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    line = result.stdout.strip()
                    gpu_util, mem_used, mem_total = map(int, line.split(', '))
                    
                    self.stats['gpu_utilization'].append(gpu_util)
                    self.stats['memory_usage'].append((mem_used / mem_total) * 100)
                    self.stats['timestamps'].append(time.time())
                    
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
                
            time.sleep(1.0)  # Обновляем каждую секунду
            
    def get_average_stats(self) -> Dict[str, float]:
        """Возвращает средние показатели"""
        if not self.stats['gpu_utilization']:
            return {'avg_gpu_util': 0.0, 'avg_memory_util': 0.0, 'max_gpu_util': 0.0}
            
        return {
            'avg_gpu_util': sum(self.stats['gpu_utilization']) / len(self.stats['gpu_utilization']),
            'avg_memory_util': sum(self.stats['memory_usage']) / len(self.stats['memory_usage']),
            'max_gpu_util': max(self.stats['gpu_utilization']),
            'samples': len(self.stats['gpu_utilization'])
        }

def test_gpu_utilization_optimized():
    """Тест GPU утилизации после оптимизаций"""
    
    logger.info("🚀 Testing GPU Utilization After Optimizations")
    
    # Проверяем наличие GPU
    device_manager = get_device_manager()
    if not device_manager.is_cuda():
        logger.error("❌ CUDA not available - cannot test GPU utilization")
        return
        
    logger.info(f"📊 GPU Device: {device_manager.get_device()}")
    
    # Создаем конфигурацию в режиме OPTIMIZED
    config = SimpleProjectConfig(mode=ModeSettings(mode=ConfigMode.OPTIMIZED))
    
    logger.info("🔧 Configuration settings:")
    logger.info(f"  - Training batch size: {config.training.batch_size}")
    logger.info(f"  - Embedding batch size: {config.training_embedding.embedding_batch_size}")
    logger.info(f"  - Debug mode: {config.logging.debug_mode}")
    logger.info(f"  - Fallback CPU: {config.device.fallback_cpu}")
    
    # Создаем тренер
    trainer = EmbeddingTrainer(config)
    
    # Создаем небольшой тестовый датасет
    logger.info("📦 Creating test dataset...")
    dataset_loader, dataset_stats = create_training_dataloader(
        config=config,
        batch_size=config.training_embedding.embedding_batch_size,
        num_workers=8,  # Используем оптимизированное количество воркеров
        shuffle=True
    )
    
    logger.info(f"📊 Dataset stats: {dataset_stats['total_samples']} samples")
    
    # Запускаем мониторинг GPU
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        logger.info("🔥 Starting training to test GPU utilization...")
        
        # Засекаем время
        start_time = time.time()
        
        # Запускаем несколько итераций тренировки
        trainer.train(dataset_loader, num_epochs=1, max_batches=20)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"⏱️ Training time: {training_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return
    finally:
        # Останавливаем мониторинг
        monitor.stop_monitoring()
        
    # Анализируем результаты
    stats = monitor.get_average_stats()
    
    logger.info("📈 GPU Utilization Results:")
    logger.info(f"  - Average GPU Utilization: {stats['avg_gpu_util']:.1f}%")
    logger.info(f"  - Maximum GPU Utilization: {stats['max_gpu_util']:.1f}%")
    logger.info(f"  - Average Memory Usage: {stats['avg_memory_util']:.1f}%")
    logger.info(f"  - Monitoring samples: {stats['samples']}")
    
    # Оценка результатов
    if stats['avg_gpu_util'] >= 80:
        logger.info("🎉 EXCELLENT: GPU utilization dramatically improved!")
        logger.info(f"🎯 Target exceeded: {stats['avg_gpu_util']:.1f}% >= 80%")
    elif stats['avg_gpu_util'] >= 60:
        logger.info("✅ SUCCESS: GPU utilization significantly improved!")
        logger.info(f"🎯 Target achieved: {stats['avg_gpu_util']:.1f}% >= 60%")
    elif stats['avg_gpu_util'] >= 40:
        logger.info("⚠️ PARTIAL SUCCESS: GPU utilization improved but not optimal")
        logger.info(f"🎯 Improvement noted: {stats['avg_gpu_util']:.1f}% (was ~25%)")
    else:
        logger.warning("❌ NO IMPROVEMENT: GPU utilization still low")
        logger.warning(f"🎯 Still low: {stats['avg_gpu_util']:.1f}% (target: 80%+)")
        
    # Анализ памяти
    if stats['avg_memory_util'] >= 80:
        logger.info("✅ GPU memory well utilized")
    elif stats['avg_memory_util'] >= 50:
        logger.info("⚠️ GPU memory moderately utilized")
    else:
        logger.warning("❌ GPU memory underutilized - data may be on CPU")
        
    # Рекомендации
    if stats['avg_gpu_util'] < 80:
        logger.info("💡 Recommendations for further optimization:")
        logger.info("  - Consider mixed precision training (AMP)")
        logger.info("  - Profile remaining CPU bottlenecks")
        logger.info("  - Check memory allocation patterns")
        logger.info("  - Consider tensor fusion optimizations")
        
    if stats['avg_memory_util'] < 50:
        logger.info("💡 GPU memory recommendations:")
        logger.info("  - Ensure all data loading uses GPU")
        logger.info("  - Check for CPU tensor operations")
        logger.info("  - Verify strict GPU-only mode is enabled")
        
    return stats

if __name__ == "__main__":
    # Запускаем тест
    stats = test_gpu_utilization_optimized()
    
    # Выводим финальный результат
    if stats:
        print(f"\n🎯 FINAL RESULT: {stats['avg_gpu_util']:.1f}% average GPU utilization")
    else:
        print("\n❌ TEST FAILED: Could not measure GPU utilization")