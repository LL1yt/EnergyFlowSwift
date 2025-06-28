#!/usr/bin/env python3
"""
Утилита для настройки локального кэша моделей
============================================

Загружает необходимые модели в локальный кэш для работы без интернета.
Оптимизировано для RTX 5090.
"""

import logging
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from new_rebuild.utils.model_cache import (
    get_model_cache_manager, 
    setup_model_cache,
    check_model_cache_status
)
from new_rebuild.config.simple_config import get_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Основная функция настройки кэша"""
    
    logger.info("🚀 Настройка локального кэша моделей")
    logger.info("=" * 50)
    
    # Получаем конфигурацию
    config = get_project_config()
    
    logger.info(f"📁 Cache directory: {config.embedding.local_models_dir}")
    logger.info(f"🔧 Auto download: {config.embedding.auto_download_models}")
    logger.info(f"💾 Prefer local: {config.embedding.prefer_local_models}")
    
    # Проверяем текущий статус
    logger.info("\n📊 Проверка текущего статуса кэша:")
    status = check_model_cache_status()
    
    logger.info(f"  Cached models: {status['models_count']}")
    logger.info(f"  Total size: {status['total_size_mb']:.1f} MB")
    logger.info(f"  Cache dir: {status['cache_dir']}")
    
    # Показываем статус каждой модели
    for model, info in status['model_status'].items():
        cached_status = "✅ Cached" if info['cached'] else "❌ Not cached"
        logger.info(f"    {model}: {cached_status}")
    
    # Настраиваем кэш для основных моделей
    logger.info("\n🔄 Настройка кэша моделей:")
    models_to_cache = ['distilbert-base-uncased']
    
    results = setup_model_cache(models_to_cache)
    
    # Отчет о результатах
    logger.info("\n📈 Результаты настройки:")
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"  {model}: {status}")
    
    # Финальная проверка
    logger.info("\n🔍 Финальная проверка:")
    final_status = check_model_cache_status()
    
    logger.info(f"  Cached models: {final_status['models_count']}")
    logger.info(f"  Total size: {final_status['total_size_mb']:.1f} MB")
    
    # Тест загрузки модели
    logger.info("\n🧪 Тест загрузки модели:")
    test_model_loading()
    
    logger.info("\n🎉 Настройка локального кэша завершена!")


def test_model_loading():
    """Тест загрузки модели из кэша"""
    try:
        from new_rebuild.core.inference.text_decoder import SimpleTextDecoder
        
        config = get_project_config()
        decoder = SimpleTextDecoder(config)
        
        # Тестовое декодирование
        import torch
        test_embeddings = torch.randn(2, 768)
        
        logger.info("  Тестирование декодера...")
        decoded_texts = decoder.decode_embeddings(test_embeddings)
        
        logger.info("  ✅ Decoder test successful!")
        for i, text in enumerate(decoded_texts):
            logger.info(f"    Test {i}: '{text[:50]}...'")
            
        # Информация о кэше
        cache_stats = decoder.get_cache_stats()
        logger.info(f"  Cache stats: {cache_stats}")
        
    except Exception as e:
        logger.error(f"  ❌ Decoder test failed: {e}")


def show_cache_info():
    """Показать информацию о кэше"""
    status = check_model_cache_status()
    
    print("\n" + "="*50)
    print("📊 ИНФОРМАЦИЯ О КЭШЕ МОДЕЛЕЙ")
    print("="*50)
    
    print(f"📁 Директория: {status['cache_dir']}")
    print(f"📦 Моделей в кэше: {status['models_count']}")
    print(f"💾 Общий размер: {status['total_size_mb']:.1f} MB")
    
    if status['models']:
        print(f"\n📋 Кэшированные модели:")
        for model in status['models']:
            print(f"  ✅ {model}")
    
    print(f"\n🔍 Статус моделей:")
    for model, info in status['model_status'].items():
        cached = "✅ Кэшировано" if info['cached'] else "❌ Не кэшировано"
        print(f"  {model}: {cached}")
        if info['path']:
            print(f"    Путь: {info['path']}")


def clear_cache():
    """Очистка кэша моделей"""
    logger.info("🗑️ Очистка кэша моделей...")
    
    manager = get_model_cache_manager()
    manager.clear_cache()
    
    logger.info("✅ Кэш очищен!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "info":
            show_cache_info()
        elif command == "clear":
            clear_cache()
        elif command == "setup":
            main()
        else:
            print("Доступные команды:")
            print("  python setup_model_cache.py setup  - Настроить кэш")
            print("  python setup_model_cache.py info   - Показать информацию")
            print("  python setup_model_cache.py clear  - Очистить кэш")
    else:
        main()