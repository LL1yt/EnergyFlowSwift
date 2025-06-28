#!/usr/bin/env python3
"""
Тест исправления кэша моделей
============================

Проверяет правильную работу определения кэшированных моделей.
"""

import logging
from new_rebuild.config.simple_config import get_project_config
from new_rebuild.utils.model_cache import get_model_cache_manager

# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_model_cache_detection():
    """Тест определения кэшированных моделей"""
    print("🔍 Тест определения кэшированных моделей")
    print("=" * 50)

    config = get_project_config()
    manager = get_model_cache_manager(config)

    model_name = "distilbert-base-uncased"

    print(f"\n📋 Проверка модели: {model_name}")

    # Проверяем кэш
    is_cached = manager.is_model_cached(model_name)
    print(f"✅ Модель в кэше: {is_cached}")

    # Получаем путь
    model_path = manager.get_model_path(model_name)
    print(f"📁 Путь к модели: {model_path}")

    # Проверяем информацию о кэше
    cache_info = manager.get_cache_info()
    print(f"\n📊 Информация о кэше:")
    print(f"  Моделей в кэше: {cache_info['models_count']}")
    print(f"  Общий размер: {cache_info['total_size_mb']:.1f} MB")
    print(f"  Директория: {cache_info['cache_dir']}")
    print(f"  Модели: {cache_info['models']}")

    if is_cached:
        print("\n✅ Тест пройден: модель правильно определена как кэшированная!")
        return True
    else:
        print("\n❌ Тест не пройден: модель не определена как кэшированная")
        return False


if __name__ == "__main__":
    success = test_model_cache_detection()
    if success:
        print("\n🎉 Все тесты пройдены!")
    else:
        print("\n💥 Тесты не пройдены!")
