#!/usr/bin/env python3
"""
Final GPU Optimization Test
==========================

Тестирует все GPU оптимизации после исправления критичных проблем:

1. ✅ Memory leaks исправлены (tensor reuse)
2. ✅ Blocking synchronization отключен
3. ✅ CPU tensor operations убраны (.cpu() calls)
4. ✅ Векторизованные операции (убраны list comprehensions)
5. ✅ GPU-direct data loading (map_location='cuda')
6. ✅ Strict GPU-only mode добавлен
7. ✅ In-place operations исправлены (checkpoint compatibility)

Ожидаемый результат:
- GPU утилизация: 60-90% (было 20-25%)
- GPU память: 50-80% (было 10%)
- Нет in-place operation errors
- Стабильная работа без memory leaks
"""

import torch
import time
import sys
import subprocess
import psutil
import threading
from pathlib import Path

# Добавляем путь к проекту
sys.path.append("/mnt/c/Users/n0n4a/projects/AA")

from new_rebuild.config.simple_config import SimpleProjectConfig
from new_rebuild.config.config_components import ConfigMode, ModeSettings
from new_rebuild.utils.device_manager import get_device_manager
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


class GPUMemoryMonitor:
    """Расширенный мониторинг GPU и CPU памяти"""

    def __init__(self):
        self.monitoring = False
        self.stats = {
            "gpu_utilization": [],
            "gpu_memory_usage": [],
            "cpu_memory_usage": [],
            "timestamps": [],
        }
        self.thread = None

    def start_monitoring(self):
        """Запускает мониторинг"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("🔍 GPU/CPU memory monitoring started")

    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        logger.info("⏹️ GPU/CPU memory monitoring stopped")

    def _monitor_loop(self):
        """Оптимизированный цикл мониторинга с минимальным CPU использованием"""
        while self.monitoring:
            try:
                # Используем PyTorch для GPU статистики (без subprocess)
                if torch.cuda.is_available():
                    try:
                        # Пытаемся получить GPU utilization через PyTorch
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        gpu_util = util.gpu
                        gpu_mem_used = (mem_info.used / mem_info.total) * 100
                        
                        self.stats["gpu_utilization"].append(gpu_util)
                        self.stats["gpu_memory_usage"].append(gpu_mem_used)
                    except:
                        # Fallback если pynvml недоступен
                        self.stats["gpu_utilization"].append(0)
                        self.stats["gpu_memory_usage"].append(0)
                else:
                    self.stats["gpu_utilization"].append(0)
                    self.stats["gpu_memory_usage"].append(0)

                # CPU память с минимальным overhead
                try:
                    import psutil
                    cpu_memory = psutil.virtual_memory()
                    self.stats["cpu_memory_usage"].append(cpu_memory.percent)
                except:
                    self.stats["cpu_memory_usage"].append(0)
                    
                self.stats["timestamps"].append(time.time())

            except Exception:
                # Минимальное логирование для производительности
                pass

            time.sleep(0.5)  # Уменьшаем интервал для точности

    def get_comprehensive_stats(self):
        """Возвращает подробную статистику"""
        if not self.stats["gpu_utilization"]:
            return {
                "avg_gpu_util": 0.0,
                "max_gpu_util": 0.0,
                "avg_gpu_memory": 0.0,
                "max_gpu_memory": 0.0,
                "avg_cpu_memory": 0.0,
                "max_cpu_memory": 0.0,
                "samples": 0,
            }

        return {
            "avg_gpu_util": sum(self.stats["gpu_utilization"])
            / len(self.stats["gpu_utilization"]),
            "max_gpu_util": max(self.stats["gpu_utilization"]),
            "avg_gpu_memory": sum(self.stats["gpu_memory_usage"])
            / len(self.stats["gpu_memory_usage"]),
            "max_gpu_memory": max(self.stats["gpu_memory_usage"]),
            "avg_cpu_memory": sum(self.stats["cpu_memory_usage"])
            / len(self.stats["cpu_memory_usage"]),
            "max_cpu_memory": max(self.stats["cpu_memory_usage"]),
            "samples": len(self.stats["gpu_utilization"]),
        }


def test_gpu_optimizations_comprehensive():
    """Комплексный тест GPU оптимизаций"""

    logger.info("🚀 Testing Final GPU Optimizations")
    logger.info("=" * 50)

    # Проверяем CUDA
    device_manager = get_device_manager()
    if not device_manager.is_cuda():
        logger.error("❌ CUDA not available - cannot test GPU optimizations")
        return None

    logger.info(f"📊 GPU Device: {device_manager.get_device()}")

    # Создаем конфигурацию в OPTIMIZED режиме
    config = SimpleProjectConfig(mode=ModeSettings(mode=ConfigMode.OPTIMIZED))

    # Активируем strict GPU mode
    device_manager.enable_strict_gpu_mode()

    logger.info("🔧 Optimization settings:")
    logger.info(f"  - Mode: {config.mode.mode}")
    logger.info(f"  - Training batch size: {config.training.batch_size}")
    logger.info(
        f"  - Embedding batch size: {config.training_embedding.embedding_batch_size}"
    )
    logger.info(f"  - Debug mode: {config.logging.debug_mode}")
    logger.info(f"  - CPU fallback: {config.device.fallback_cpu}")
    logger.info(f"  - Strict GPU mode: {device_manager.strict_gpu}")

    # Запускаем мониторинг
    monitor = GPUMemoryMonitor()
    monitor.start_monitoring()

    success = False
    error_info = None

    try:
        logger.info("🔥 Starting optimized forward pass test...")

        # Импортируем и запускаем тест
        sys.path.append("/mnt/c/Users/n0n4a/projects/AA")

        # Запуск теста forward pass
        start_time = time.time()

        # Эмулируем простой forward pass
        from new_rebuild.core.lattice.lattice import Lattice3D
        from new_rebuild.core.moe.moe_processor import MoEConnectionProcessor

        # Создаем тестовую решетку
        lattice = Lattice3D()

        # Инициализируем состояния на GPU
        with torch.no_grad():
            initial_states = torch.randn(
                3375, config.model.state_size, device=device_manager.get_device()
            )
            lattice.states = initial_states

            # Выполняем несколько forward passes
            for i in range(3):
                logger.info(f"--- Forward pass {i+1}/3 ---")
                lattice.forward()

                # Проверяем, что данные остались на GPU
                if lattice.states.device.type != "cuda":
                    raise RuntimeError(f"States moved to CPU: {lattice.states.device}")

            logger.info("✅ Forward passes completed successfully")

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")
        success = True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        error_info = str(e)
        success = False

    finally:
        # Останавливаем мониторинг
        monitor.stop_monitoring()

    # Анализируем результаты
    stats = monitor.get_comprehensive_stats()

    logger.info("\n" + "=" * 50)
    logger.info("📈 FINAL OPTIMIZATION RESULTS:")
    logger.info("=" * 50)

    logger.info(f"  - Average GPU Utilization: {stats['avg_gpu_util']:.1f}%")
    logger.info(f"  - Maximum GPU Utilization: {stats['max_gpu_util']:.1f}%")
    logger.info(f"  - Average GPU Memory: {stats['avg_gpu_memory']:.1f}%")
    logger.info(f"  - Maximum GPU Memory: {stats['max_gpu_memory']:.1f}%")
    logger.info(f"  - Average CPU Memory: {stats['avg_cpu_memory']:.1f}%")
    logger.info(f"  - Maximum CPU Memory: {stats['max_cpu_memory']:.1f}%")
    logger.info(f"  - Monitoring samples: {stats['samples']}")
    logger.info(f"  - Test success: {'✅' if success else '❌'}")

    # Оценка GPU утилизации
    if stats["avg_gpu_util"] >= 80:
        logger.info("🎉 EXCELLENT: GPU utilization dramatically improved!")
        result_level = "EXCELLENT"
    elif stats["avg_gpu_util"] >= 60:
        logger.info("✅ SUCCESS: GPU utilization significantly improved!")
        result_level = "SUCCESS"
    elif stats["avg_gpu_util"] >= 40:
        logger.info("⚠️ PARTIAL: GPU utilization improved but not optimal")
        result_level = "PARTIAL"
    else:
        logger.info("❌ POOR: GPU utilization still low")
        result_level = "POOR"

    # Оценка GPU памяти
    if stats["avg_gpu_memory"] >= 70:
        logger.info("✅ GPU memory well utilized")
    elif stats["avg_gpu_memory"] >= 30:
        logger.info("⚠️ GPU memory moderately utilized")
    else:
        logger.info("❌ GPU memory underutilized - data may still be on CPU")

    # Оценка CPU памяти
    if stats["avg_cpu_memory"] <= 30:
        logger.info("✅ CPU memory usage optimized")
    elif stats["avg_cpu_memory"] <= 50:
        logger.info("⚠️ CPU memory usage moderate")
    else:
        logger.info("❌ CPU memory usage high - potential memory leaks")

    # Финальная рекомендация
    if success and result_level in ["EXCELLENT", "SUCCESS"]:
        logger.info("🎯 OPTIMIZATION COMPLETE: GPU utilization optimized successfully!")
    elif success and result_level == "PARTIAL":
        logger.info("🔧 OPTIMIZATION PARTIAL: Some improvements needed")
    else:
        logger.info("❗ OPTIMIZATION FAILED: Requires further investigation")

    if error_info:
        logger.info(f"🐛 Error details: {error_info}")

    return {
        "success": success,
        "result_level": result_level,
        "stats": stats,
        "error": error_info,
    }


if __name__ == "__main__":
    result = test_gpu_optimizations_comprehensive()

    if result:
        print(f"\n🎯 FINAL RESULT: {result['result_level']}")
        print(f"📊 GPU Utilization: {result['stats']['avg_gpu_util']:.1f}%")
        print(f"💾 GPU Memory: {result['stats']['avg_gpu_memory']:.1f}%")
        print(f"🧠 CPU Memory: {result['stats']['avg_cpu_memory']:.1f}%")
        print(f"✅ Success: {result['success']}")
    else:
        print("\n❌ TEST FAILED: Could not complete optimization test")
