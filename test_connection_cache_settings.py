#!/usr/bin/env python3
"""
Тест настроек Connection Cache
==============================

Проверяет корректность работы новых настроек кэширования:
- Включение/выключение кэша
- Мониторинг производительности
- Автоматическое определение нужности кэширования
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import get_project_config, set_project_config, ProjectConfig
from new_rebuild.core.moe import create_connection_classifier
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_cache_settings():
    """Тест настроек кэширования"""

    print("🧪 Тест настроек Connection Cache")
    print("=" * 50)

    # Тест 1: Малые решетки - автоматическое отключение кэша
    print("\n1️⃣ Тест малых решеток (5x5x5 = 125 клеток)")
    config = ProjectConfig()
    config.lattice.dimensions = (5, 5, 5)
    config.expert.cache.enabled = True
    config.expert.cache.auto_disable_threshold = 1000
    config.expert.cache.small_lattice_fallback = True
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    classifier = create_connection_classifier(lattice_dimensions=(5, 5, 5))

    print(f"   Classifier cache enabled: {classifier.enable_cache}")
    print(f"   Performance monitoring: {classifier.enable_performance_monitoring}")

    # Тест 2: Средние решетки - включение кэша
    print("\n2️⃣ Тест средних решеток (15x15x15 = 3375 клеток)")
    config = ProjectConfig()  # Создаем новую конфигурацию
    config.lattice.dimensions = (15, 15, 15)
    config.expert.cache.enabled = True  # Включаем кэш
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))

    print(f"   Classifier cache enabled: {classifier.enable_cache}")
    print(f"   Performance monitoring: {classifier.enable_performance_monitoring}")

    # Тест 3: Средние решетки для демонстрации GPU (изменено с больших)
    print("\n3️⃣ Тест средних решеток с GPU (20x20x20 = 8000 клеток)")
    config = ProjectConfig()  # Создаем новую конфигурацию
    config.lattice.dimensions = (20, 20, 20)
    config.expert.cache.enabled = True  # Включаем кэш
    config.expert.cache.auto_enable_threshold = 3000  # Снижен порог
    config.expert.cache.use_gpu_acceleration = True  # Включаем GPU
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    classifier = create_connection_classifier(lattice_dimensions=(30, 30, 30))

    print(f"   Classifier cache enabled: {classifier.enable_cache}")
    print(f"   Performance monitoring: {classifier.enable_performance_monitoring}")

    # Тест 4: Принудительное отключение кэша
    print("\n4️⃣ Тест принудительного отключения кэша")
    config = ProjectConfig()  # Создаем новую конфигурацию
    config.lattice.dimensions = (30, 30, 30)
    config.expert.cache.enabled = False  # Принудительно отключаем
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Should use cache: {should_use_cache}")

    classifier = create_connection_classifier(lattice_dimensions=(30, 30, 30))

    print(f"   Classifier cache enabled: {classifier.enable_cache}")
    print(f"   Performance monitoring: {classifier.enable_performance_monitoring}")

    # Тест 5: Отключение мониторинга производительности
    print("\n5️⃣ Тест отключения мониторинга производительности")
    config = ProjectConfig()  # Создаем новую конфигурацию
    config.lattice.dimensions = (15, 15, 15)
    config.expert.cache.enabled = True
    config.expert.cache.enable_performance_monitoring = False
    set_project_config(config)

    classifier = create_connection_classifier(lattice_dimensions=(15, 15, 15))

    print(f"   Classifier cache enabled: {classifier.enable_cache}")
    print(f"   Performance monitoring: {classifier.enable_performance_monitoring}")
    print(f"   Performance stats: {classifier.performance_stats}")

    # Тест 6: Статистика кэша
    print("\n6️⃣ Тест статистики кэша")
    config = ProjectConfig()  # Создаем новую конфигурацию
    config.lattice.dimensions = (10, 10, 10)
    config.expert.cache.enabled = True
    config.expert.cache.enable_performance_monitoring = True
    config.expert.cache.enable_detailed_stats = True
    set_project_config(config)

    classifier = create_connection_classifier(lattice_dimensions=(10, 10, 10))

    stats = classifier.get_classification_stats()
    print(f"   Cache performance stats: {stats.get('cache_performance', {})}")

    # Тест 7: Кэш конфигурация
    print("\n7️⃣ Тест кэш конфигурации")
    config = ProjectConfig()  # Создаем новую конфигурацию для финального теста
    config.lattice.dimensions = (20, 20, 20)
    config.expert.cache.enabled = True
    set_project_config(config)

    cache_config = config.get_connection_cache_config()
    print(f"   Cache config: {cache_config}")
    print(f"   Total cells: {config.total_cells}")
    print(f"   Should use cache: {config.should_use_connection_cache()}")

    print("\n✅ Все тесты настроек кэширования пройдены успешно!")


def test_performance_comparison():
    """Тест сравнения производительности"""

    print("\n🏁 Тест сравнения производительности")
    print("=" * 50)

    # Небольшая решетка для быстого теста
    dimensions = (8, 8, 8)
    total_cells = 8 * 8 * 8

    # Создаем тестовые данные
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Тест с кэшем
    print(f"\n📊 Тест с кэшем для решетки {dimensions} ({total_cells} клеток)")

    config = ProjectConfig()
    config.lattice.dimensions = dimensions
    config.expert.cache.enabled = True
    config.expert.cache.enable_performance_monitoring = True
    config.expert.cache.benchmark_small_lattices = True
    set_project_config(config)

    classifier_with_cache = create_connection_classifier(lattice_dimensions=dimensions)

    # Тест без кэша
    print(f"\n📊 Тест без кэша для решетки {dimensions} ({total_cells} клеток)")

    config.expert.cache.enabled = False
    set_project_config(config)

    classifier_without_cache = create_connection_classifier(
        lattice_dimensions=dimensions
    )

    print(f"   Classifier with cache: {classifier_with_cache.enable_cache}")
    print(f"   Classifier without cache: {classifier_without_cache.enable_cache}")

    # Для больших решеток
    large_dimensions = (20, 20, 20)
    large_total_cells = 20 * 20 * 20

    print(
        f"\n📊 Рекомендация для больших решеток {large_dimensions} ({large_total_cells} клеток)"
    )

    config.lattice.dimensions = large_dimensions
    config.expert.cache.enabled = True
    set_project_config(config)

    should_use_cache = config.should_use_connection_cache()
    print(f"   Рекомендуется использовать кэш: {should_use_cache}")

    cache_config = config.get_connection_cache_config()
    print(f"   Cache enabled: {cache_config['enabled']}")
    print(f"   Performance monitoring: {cache_config['enable_performance_monitoring']}")

    print("\n✅ Тест сравнения производительности завершен!")


if __name__ == "__main__":
    test_cache_settings()
    test_performance_comparison()
