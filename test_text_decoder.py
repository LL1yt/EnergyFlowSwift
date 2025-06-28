#!/usr/bin/env python3
"""
Тест декодера эмбедингов в текст
===============================

Проверяет работу кэширования и базового декодирования.
"""

import logging
import torch
from pathlib import Path

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from new_rebuild.config.simple_config import get_project_config
from new_rebuild.core.inference.text_decoder import (
    SimpleTextDecoder, 
    JointTextDecoder,
    EmbeddingTextCache,
    create_text_decoder
)
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_embedding_cache():
    """Тест кэша эмбединг-текст"""
    logger.info("=== Тест EmbeddingTextCache ===")
    
    cache = EmbeddingTextCache(max_size=5)
    
    # Создаем тестовые данные
    embeddings = [
        torch.randn(768),
        torch.randn(768),
        torch.randn(768)
    ]
    
    texts = [
        "Hello world!",
        "This is a test sentence.",
        "Another example text."
    ]
    
    # Тестируем сохранение
    for emb, text in zip(embeddings, texts):
        cache.put(emb, text)
        logger.info(f"  Saved: '{text}'")
    
    # Тестируем получение
    logger.info("\nТест получения из кэша:")
    for i, emb in enumerate(embeddings):
        retrieved = cache.get(emb)
        logger.info(f"  Retrieved {i}: '{retrieved}'")
        assert retrieved == texts[i], f"Cache mismatch: {retrieved} != {texts[i]}"
    
    # Тестируем похожие эмбединги
    logger.info("\nТест поиска похожих:")
    similar_emb = embeddings[0] + torch.randn(768) * 0.01  # Очень похожий
    retrieved_similar = cache.get(similar_emb)
    logger.info(f"  Similar embedding retrieved: '{retrieved_similar}'")
    
    # Тестируем переполнение кэша
    logger.info("\nТест переполнения кэша:")
    for i in range(10):
        cache.put(torch.randn(768), f"Overflow text {i}")
    
    stats = cache.get_stats()
    logger.info(f"  Cache stats: {stats}")
    
    logger.info("✅ EmbeddingTextCache test passed!")
    return cache


def test_simple_decoder():
    """Тест простого декодера"""
    logger.info("\n=== Тест SimpleTextDecoder ===")
    
    config = get_project_config()
    config.embedding.decoder_cache_enabled = True
    
    # Создаем временную директорию для кэша
    cache_dir = Path("temp_cache")
    cache_dir.mkdir(exist_ok=True)
    config.embedding.cache_dir = str(cache_dir)
    
    decoder = SimpleTextDecoder(config)
    
    # Тестовые эмбединги
    test_embeddings = torch.randn(4, 768)
    
    logger.info("Декодирование тестовых эмбедингов:")
    decoded_texts = decoder.decode_embeddings(test_embeddings)
    
    for i, text in enumerate(decoded_texts):
        logger.info(f"  Embedding {i}: '{text}'")
    
    # Тест повторного декодирования (должен использовать кэш)
    logger.info("\nПовторное декодирование (тест кэша):")
    decoded_texts_2 = decoder.decode_embeddings(test_embeddings)
    
    for i, text in enumerate(decoded_texts_2):
        logger.info(f"  Cached {i}: '{text}'")
        assert text == decoded_texts[i], "Cache consistency error"
    
    # Статистика кэша
    cache_stats = decoder.get_cache_stats()
    logger.info(f"\nCache stats: {cache_stats}")
    
    # Сохранение кэша
    decoder.save_cache()
    logger.info("Cache saved to disk")
    
    logger.info("✅ SimpleTextDecoder test passed!")
    return decoder


def test_joint_decoder():
    """Тест joint декодера"""
    logger.info("\n=== Тест JointTextDecoder ===")
    
    config = get_project_config()
    joint_decoder = JointTextDecoder(config)
    
    # Тестовые данные
    embeddings = torch.randn(3, 768)
    target_texts = ["Hello world", "Test sentence", "Another example"]
    
    # Тест в режиме обучения
    joint_decoder.train()
    joint_decoder.set_training_mode(True)
    
    results = joint_decoder(embeddings, target_texts)
    logger.info(f"Training mode results:")
    logger.info(f"  Logits shape: {results['logits'].shape if results['logits'] is not None else None}")
    logger.info(f"  Loss: {results['loss']}")
    
    for i, text in enumerate(results['decoded_texts']):
        logger.info(f"  Decoded {i}: '{text}'")
    
    # Тест в режиме inference
    joint_decoder.eval()
    joint_decoder.set_training_mode(False)
    
    results_inf = joint_decoder(embeddings)
    logger.info(f"\nInference mode results:")
    for i, text in enumerate(results_inf['decoded_texts']):
        logger.info(f"  Decoded {i}: '{text}'")
    
    logger.info("✅ JointTextDecoder test passed!")
    return joint_decoder


def test_decoder_factory():
    """Тест фабричной функции"""
    logger.info("\n=== Тест фабричной функции ===")
    
    config = get_project_config()
    
    # Простой декодер
    simple_decoder = create_text_decoder(config, joint_training=False)
    logger.info(f"Simple decoder type: {type(simple_decoder).__name__}")
    
    # Joint декодер
    joint_decoder = create_text_decoder(config, joint_training=True)
    logger.info(f"Joint decoder type: {type(joint_decoder).__name__}")
    
    logger.info("✅ Factory function test passed!")


def test_cache_persistence():
    """Тест сохранения/загрузки кэша"""
    logger.info("\n=== Тест персистентности кэша ===")
    
    cache_path = "temp_cache/test_cache.json"
    
    # Создаем кэш и заполняем данными
    cache1 = EmbeddingTextCache(max_size=10)
    
    test_data = [
        (torch.randn(768), "First test sentence"),
        (torch.randn(768), "Second test sentence"),
        (torch.randn(768), "Third test sentence")
    ]
    
    for emb, text in test_data:
        cache1.put(emb, text)
    
    logger.info(f"Created cache with {len(test_data)} items")
    
    # Сохраняем
    cache1.save(cache_path)
    
    # Создаем новый кэш и загружаем
    cache2 = EmbeddingTextCache(max_size=10)
    cache2.load(cache_path)
    
    logger.info(f"Loaded cache with {cache2.get_stats()['size']} items")
    
    # Проверяем, что данные загрузились
    for emb, original_text in test_data:
        retrieved_text = cache2.get(emb)
        logger.info(f"  Original: '{original_text}' -> Retrieved: '{retrieved_text}'")
        assert retrieved_text == original_text, "Cache persistence error"
    
    # Очистка
    Path(cache_path).unlink(missing_ok=True)
    
    logger.info("✅ Cache persistence test passed!")


def test_batch_processing():
    """Тест батчевой обработки"""
    logger.info("\n=== Тест батчевой обработки ===")
    
    config = get_project_config()
    decoder = SimpleTextDecoder(config)
    
    # Тест разных размеров батчей
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        logger.info(f"\nБатч размер: {batch_size}")
        
        embeddings = torch.randn(batch_size, 768)
        decoded_texts = decoder.decode_embeddings(embeddings)
        
        assert len(decoded_texts) == batch_size, f"Batch size mismatch: {len(decoded_texts)} != {batch_size}"
        logger.info(f"  ✓ Decoded {len(decoded_texts)} texts")
        
        # Показываем первые несколько
        for i, text in enumerate(decoded_texts[:3]):
            logger.info(f"    {i}: '{text[:50]}...'")
    
    logger.info("✅ Batch processing test passed!")


def test_gpu_performance():
    """Тест GPU производительности для RTX 5090"""
    logger.info("\n=== Тест GPU производительности (RTX 5090) ===")
    
    config = get_project_config()
    decoder = SimpleTextDecoder(config)
    
    if not decoder.use_gpu_acceleration:
        logger.info("⚠️  GPU acceleration not available, skipping GPU tests")
        return
    
    logger.info(f"🚀 GPU acceleration enabled (batch size: {decoder.gpu_batch_size})")
    
    # Тест больших батчей для использования GPU оптимизаций
    large_batch_sizes = [32, 64, 128, 256, 512]
    
    import time
    
    for batch_size in large_batch_sizes:
        logger.info(f"\nТест большого батча: {batch_size}")
        
        embeddings = torch.randn(batch_size, 768)
        
        # Замеряем время
        start_time = time.time()
        decoded_texts = decoder.decode_embeddings(embeddings)
        elapsed = time.time() - start_time
        
        assert len(decoded_texts) == batch_size
        throughput = batch_size / elapsed
        
        logger.info(f"  ✓ Decoded {batch_size} embeddings in {elapsed:.3f}s ({throughput:.1f} emb/s)")
        
        # Показываем примеры GPU-декодирования
        for i, text in enumerate(decoded_texts[:2]):
            logger.info(f"    GPU[{i}]: '{text}'")
    
    # Тест повторного декодирования (кэш)
    logger.info(f"\nТест кэширования для GPU:")
    large_embeddings = torch.randn(128, 768)
    
    # Первый раз
    start_time = time.time()
    first_decode = decoder.decode_embeddings(large_embeddings)
    first_time = time.time() - start_time
    
    # Второй раз (кэш)
    start_time = time.time()
    second_decode = decoder.decode_embeddings(large_embeddings)
    second_time = time.time() - start_time
    
    speedup = first_time / second_time if second_time > 0 else float('inf')
    logger.info(f"  First decode: {first_time:.3f}s")
    logger.info(f"  Cached decode: {second_time:.3f}s") 
    logger.info(f"  Cache speedup: {speedup:.1f}x")
    
    cache_stats = decoder.get_cache_stats()
    logger.info(f"  Cache stats: {cache_stats}")
    
    logger.info("✅ GPU performance test passed!")


if __name__ == "__main__":
    logger.info("🚀 Запуск тестов TextDecoder")
    
    # Запускаем все тесты
    test_embedding_cache()
    test_simple_decoder()
    test_joint_decoder()
    test_decoder_factory()
    test_cache_persistence()
    test_batch_processing()
    test_gpu_performance()  # Новый тест для RTX 5090
    
    logger.info("\n🎉 Все тесты TextDecoder завершены успешно!")
    
    # Очистка временных файлов
    import shutil
    shutil.rmtree("temp_cache", ignore_errors=True)
    
    logger.info("🧹 Временные файлы очищены")