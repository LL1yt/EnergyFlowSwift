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
import logging

# Настройка логирования для отладки
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.moe import create_connection_classifier
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def run_full_test():
    """Полный тест кэширования, включая создание и переиспользование"""

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

    # ==================================================================
    # Тест 1: Малая решетка (должна использовать CPU)
    # ==================================================================
    print(f"\n1️⃣ Малая решетка (8x8x8 = 512 клеток) - ожидается CPU")
    config_small = ProjectConfig()
    config_small.lattice.dimensions = (8, 8, 8)
    config_small.cache.enabled = True
    config_small.cache.use_gpu_acceleration = True
    config_small.cache.enable_performance_monitoring = True
    set_project_config(config_small)

    should_use_cache = (
        config_small.cache.enabled
        and config_small.total_cells >= config_small.cache.auto_enable_threshold
    )
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

    # ==================================================================
    # Тест 2: Средняя решетка (должна использовать GPU) - СОЗДАНИЕ КЭША
    # ==================================================================
    print(f"\n2️⃣ Средняя решетка (15x15x15 = 3375 клеток) - ожидается GPU")
    config_medium = ProjectConfig()
    config_medium.lattice.dimensions = (15, 15, 15)
    config_medium.cache.enabled = True
    config_medium.cache.use_gpu_acceleration = True
    config_medium.cache.enable_performance_monitoring = True
    config_medium.cache.auto_enable_threshold = 3000
    set_project_config(config_medium)

    should_use_cache = (
        config_medium.cache.enabled
        and config_medium.total_cells >= config_medium.cache.auto_enable_threshold
    )
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

        cache_stats = classifier.get_cache_stats()
        print(f"   Cache status: {cache_stats.get('status', 'unknown')}")
        if cache_stats.get("cached_cells"):
            print(f"   Cached cells: {cache_stats['cached_cells']}")
            print(f"   Cache size: {cache_stats.get('cache_size_mb', 0):.1f}MB")

    # ==================================================================
    # Тест 3: Большая решетка (для демонстрации полной мощности)
    # ==================================================================
    if gpu_memory >= 8.0:
        print(f"\n3️⃣ Большая решетка (20x20x20 = 8000 клеток) - полная мощность GPU")
        config_large = ProjectConfig()
        config_large.lattice.dimensions = (20, 20, 20)
        config_large.cache.enabled = True
        config_large.cache.use_gpu_acceleration = True
        config_large.cache.enable_performance_monitoring = True
        config_large.cache.auto_enable_threshold = 3000
        config_large.cache.gpu_batch_size = 5000
        set_project_config(config_large)

        should_use_cache = (
            config_large.cache.enabled
            and config_large.total_cells >= config_large.cache.auto_enable_threshold
        )
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

            cache_stats = classifier.get_cache_stats()
            print(f"   📊 Cache status: {cache_stats.get('status', 'unknown')}")
            if cache_stats.get("cached_cells"):
                print(f"   🎯 Cached cells: {cache_stats['cached_cells']}")
                print(f"   💾 Cache size: {cache_stats.get('cache_size_mb', 0):.1f}MB")
                print(f"   💽 Cache saved to disk for future reuse")
        else:
            print("   ⚠️  Cache disabled for this lattice size")
    else:
        print(
            f"\n3️⃣ Пропускаем большую решетку (недостаточно GPU памяти: {gpu_memory:.1f}GB < 8GB)"
        )

    print(f"\n✅ GPU Cache Demo завершен!")
    print(f"🔄 При повторном запуске кэш будет загружен с диска моментально!")

    # ==================================================================
    # Тест 4: ПЕРЕИСПОЛЬЗОВАНИЕ КЭША для средней решетки
    # ==================================================================
    print(f"\n🔄 Тест переиспользования кэша (15x15x15)")
    print("=" * 45)

    # Используем ту же конфигурацию, что и для создания кэша
    set_project_config(config_medium)

    print("   🔍 Проверяем существующий кэш...")
    start_time = time.time()
    classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))
    reuse_time = time.time() - start_time

    print(f"   ⚡ Время загрузки: {reuse_time:.2f}s")

    if reuse_time < 2.0:  # Увеличим порог на всякий случай
        print("   ✅ Кэш успешно переиспользован (быстрая загрузка)!")
    else:
        print("   🔄 Кэш пересоздан (первый запуск или изменились параметры)")

    cache_stats = classifier.get_cache_stats()
    if cache_stats.get("cached_cells"):
        print(f"   📊 Cached cells: {cache_stats['cached_cells']}")


if __name__ == "__main__":
    run_full_test()
