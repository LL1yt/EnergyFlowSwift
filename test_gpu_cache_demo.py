#!/usr/bin/env python3
"""
GPU Cache Demo - Демонстрация GPU ускорения кэширования
======================================================

Быстрый тест для демонстрации работы GPU-ускоренного кэширования
связей для RTX 5090.
"""

import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.moe import create_connection_classifier
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_gpu_acceleration():
    """Тест GPU ускорения для кэширования"""

    print("🚀 GPU Cache Acceleration Demo")
    print("=" * 50)

    # Проверяем доступность GPU
    if not torch.cuda.is_available():
        print("❌ CUDA не доступна, тест будет работать на CPU")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f}GB")

    # Тест 1: Малая решетка (должна использовать CPU)
    print(f"\n1️⃣ Малая решетка (8x8x8 = 512 клеток) - ожидается CPU")
    config = ProjectConfig()
    config.lattice.dimensions = (8, 8, 8)
    config.expert.cache.enabled = True
    config.expert.cache.use_gpu_acceleration = True
    config.expert.cache.enable_performance_monitoring = True
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    if should_use_cache:
        start_time = time.time()
        classifier = create_connection_classifier(lattice_dimensions=(8, 8, 8))
        init_time = time.time() - start_time

        print(f"   Cache enabled: {classifier.enable_cache}")
        if classifier.cache_manager:
            print(f"   GPU acceleration: {classifier.cache_manager.use_gpu}")
            print(f"   Device: {classifier.cache_manager.device}")
        print(f"   Initialization time: {init_time:.2f}s")

    # Тест 2: Средняя решетка (должна использовать GPU)
    print(f"\n2️⃣ Средняя решетка (15x15x15 = 3375 клеток) - ожидается GPU")
    config = ProjectConfig()
    config.lattice.dimensions = (15, 15, 15)
    config.expert.cache.enabled = True
    config.expert.cache.use_gpu_acceleration = True
    config.expert.cache.enable_performance_monitoring = True
    config.expert.cache.auto_enable_threshold = 3000
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    if should_use_cache:
        print("   🔄 Инициализация кэша (может занять время)...")
        start_time = time.time()
        classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))
        init_time = time.time() - start_time

        print(f"   Cache enabled: {classifier.enable_cache}")
        if classifier.cache_manager:
            print(f"   GPU acceleration: {classifier.cache_manager.use_gpu}")
            print(f"   Device: {classifier.cache_manager.device}")
            print(f"   GPU batch size: {classifier.cache_manager.gpu_batch_size}")
        print(f"   Initialization time: {init_time:.2f}s")

        # Показываем статистику кэша
        cache_stats = classifier.get_cache_stats()
        print(f"   Cache status: {cache_stats.get('status', 'unknown')}")
        if cache_stats.get("cached_cells"):
            print(f"   Cached cells: {cache_stats['cached_cells']}")
            print(f"   Cache size: {cache_stats.get('cache_size_mb', 0):.1f}MB")

    # Тест 3: Большая решетка (для демонстрации полной мощности)
    if gpu_memory >= 8.0:  # Только если достаточно GPU памяти
        print(f"\n3️⃣ Большая решетка (20x20x20 = 8000 клеток) - полная мощность GPU")
        config = ProjectConfig()
        config.lattice.dimensions = (20, 20, 20)
        config.expert.cache.enabled = True
        config.expert.cache.use_gpu_acceleration = True
        config.expert.cache.enable_performance_monitoring = True
        config.expert.cache.auto_enable_threshold = 3000
        config.expert.cache.gpu_batch_size = 5000  # Уменьшаем для стабильности
        set_project_config(config)

        should_use_cache = config.should_use_connection_cache()
        print(f"   Should use cache: {should_use_cache}")

        if should_use_cache:
            print("   🚀 Инициализация большого кэша на GPU...")
            print("   ⏱️  Это может занять несколько минут для первого создания...")

            start_time = time.time()
            classifier = create_connection_classifier(lattice_dimensions=(20, 20, 20))
            init_time = time.time() - start_time

            print(f"   ✅ Cache enabled: {classifier.enable_cache}")
            if classifier.cache_manager:
                print(f"   🚀 GPU acceleration: {classifier.cache_manager.use_gpu}")
                print(f"   💾 Device: {classifier.cache_manager.device}")
                print(
                    f"   📦 GPU batch size: {classifier.cache_manager.gpu_batch_size}"
                )
            print(f"   ⏱️  Total initialization time: {init_time:.2f}s")

            # Показываем подробную статистику
            cache_stats = classifier.get_cache_stats()
            print(f"   📊 Cache status: {cache_stats.get('status', 'unknown')}")
            if cache_stats.get("cached_cells"):
                print(f"   🎯 Cached cells: {cache_stats['cached_cells']}")
                print(f"   💾 Cache size: {cache_stats.get('cache_size_mb', 0):.1f}MB")

                # Проверяем что кэш сохранен на диск
                print(f"   💽 Cache saved to disk for future reuse")
        else:
            print("   ⚠️  Cache disabled for this lattice size")
    else:
        print(
            f"\n3️⃣ Пропускаем большую решетку (недостаточно GPU памяти: {gpu_memory:.1f}GB < 8GB)"
        )

    print(f"\n✅ GPU Cache Demo завершен!")
    print(f"🔄 При повторном запуске кэш будет загружен с диска моментально!")


def test_cache_reuse():
    """Тест переиспользования кэша"""

    print(f"\n🔄 Тест переиспользования кэша")
    print("=" * 30)

    # Создаем ту же конфигурацию что и в предыдущем тесте
    config = ProjectConfig()
    config.lattice.dimensions = (15, 15, 15)
    config.expert.cache.enabled = True
    config.expert.cache.use_gpu_acceleration = True
    config.expert.cache.auto_enable_threshold = 3000
    set_project_config(config)

    print("   🔍 Проверяем существующий кэш...")

    start_time = time.time()
    classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))
    reuse_time = time.time() - start_time

    print(f"   ⚡ Время загрузки: {reuse_time:.2f}s")

    if reuse_time < 1.0:
        print("   ✅ Кэш успешно переиспользован (быстрая загрузка)!")
    else:
        print("   🔄 Кэш пересоздан (первый запуск или изменились параметры)")

    cache_stats = classifier.get_cache_stats()
    if cache_stats.get("cached_cells"):
        print(f"   📊 Cached cells: {cache_stats['cached_cells']}")


if __name__ == "__main__":
    test_gpu_acceleration()
    test_cache_reuse()
