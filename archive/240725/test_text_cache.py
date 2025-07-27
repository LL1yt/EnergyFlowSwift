#!/usr/bin/env python3
"""
Тест TextCache для проверки кэширования функциональности
"""

import torch
import sys
import time
from pathlib import Path

# Добавляем путь к energy_flow
sys.path.append('energy_flow')

from energy_flow.config import create_debug_config
from energy_flow.text_bridge import TextToCubeEncoder, CubeToTextDecoder
from energy_flow.text_bridge.text_cache import (
    TextCache, 
    create_text_cache,
    CachedTextToCubeEncoder,
    CachedCubeToTextDecoder
)

def test_text_cache():
    print("🧪 Тестирование TextCache...")
    
    # Создаем debug конфигурацию
    config = create_debug_config()
    print(f"📐 Размеры куба: {config.lattice_width}×{config.lattice_height}×{config.lattice_depth}")
    print(f"📏 Surface dim: {config.lattice_width * config.lattice_height}")
    
    # Создаем кэш
    cache_file = "test_cache.pt"
    cache = create_text_cache(max_size=100, cache_file=cache_file, config=config)
    
    print(f"\n1️⃣ Инициализация кэша:")
    print(f"   Максимальный размер: {cache.max_size}")
    print(f"   Включен: {cache.enabled}")
    print(f"   Файл кэша: {cache.cache_file}")
    print(f"   Surface dim: {cache.surface_dim}")
    
    # Тестовые данные
    test_texts = [
        "Hello world!",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "Neural networks process information.",
        "Energy flows through the lattice."
    ]
    
    print(f"\n2️⃣ Тест базового кэширования text → surface:")
    
    # Создаем тестовые surface embeddings
    surface_embeddings = []
    print(f"   📱 Device info - Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"   📱 CUDA available: {torch.cuda.is_available()}")
    
    for text in test_texts:
        # Генерируем детерминированные embeddings на базе текста
        torch.manual_seed(hash(text) % 2**32)
        embedding = torch.randn(cache.surface_dim) * 0.5
        embedding = torch.clamp(embedding, -1, 1)
        surface_embeddings.append(embedding)
        
        print(f"   📱 Original embedding device: {embedding.device}")
        
        # Кэшируем
        cache.put_text_to_surface(text, embedding)
        print(f"   Кэшировали: '{text}' → {embedding.shape}")
    
    # Проверяем извлечение из кэша
    print(f"\n3️⃣ Тест извлечения text → surface:")
    for i, text in enumerate(test_texts):
        cached_embedding = cache.get_surface_from_text(text)
        if cached_embedding is not None:
            original = surface_embeddings[i]
            print(f"   📱 Original device: {original.device}, Cached device: {cached_embedding.device}")
            try:
                match = torch.allclose(cached_embedding, original, atol=1e-6)
                print(f"   '{text}': {'✅ MATCH' if match else '❌ MISMATCH'}")
            except RuntimeError as e:
                print(f"   ❌ Device error for '{text}': {e}")
                # Try moving to same device for comparison
                if original.device != cached_embedding.device:
                    print(f"   📱 Attempting device alignment...")
                    if original.device.type == 'cuda':
                        cached_embedding = cached_embedding.to(original.device)
                    else:
                        cached_embedding = cached_embedding.cpu()
                    match = torch.allclose(cached_embedding, original, atol=1e-6)
                    print(f"   📱 After alignment - '{text}': {'✅ MATCH' if match else '❌ MISMATCH'}")
        else:
            print(f"   '{text}': ❌ NOT FOUND")
    
    # Тест обратного кэширования
    print(f"\n4️⃣ Тест кэширования surface → text:")
    for i, embedding in enumerate(surface_embeddings):
        text = test_texts[i]
        cache.put_surface_to_text(embedding, text)
        print(f"   Кэшировали: {embedding.shape} → '{text}'")
    
    # Проверяем обратное извлечение
    print(f"\n5️⃣ Тест извлечения surface → text:")
    for i, embedding in enumerate(surface_embeddings):
        cached_text = cache.get_text_from_surface(embedding)
        original_text = test_texts[i]
        if cached_text is not None:
            match = cached_text == original_text
            print(f"   {embedding.shape}: {'✅ MATCH' if match else '❌ MISMATCH'} '{cached_text}'")
        else:
            print(f"   {embedding.shape}: ❌ NOT FOUND")
    
    # Статистика кэша
    print(f"\n6️⃣ Статистика кэша:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Тест LRU функциональности
    print(f"\n7️⃣ Тест LRU функциональности:")
    small_cache = TextCache(max_size=3, enabled=True, config=config)
    
    # Добавляем больше элементов чем максимальный размер
    lru_texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    for i, text in enumerate(lru_texts):
        embedding = torch.randn(cache.surface_dim) * 0.5
        small_cache.put_text_to_surface(text, embedding)
        print(f"   Добавили '{text}', размер кэша: {len(small_cache)}")
    
    print(f"   Финальный размер кэша: {len(small_cache)} (макс: {small_cache.max_size})")
    
    # Проверяем что старые записи удалены
    for text in lru_texts:
        cached = small_cache.get_surface_from_text(text)
        status = "✅ FOUND" if cached is not None else "❌ EVICTED"
        print(f"   '{text}': {status}")
    
    # Тест сохранения и загрузки
    print(f"\n8️⃣ Тест персистентного хранения:")
    
    # Сохраняем кэш
    cache.save_cache()
    print(f"   Кэш сохранен в {cache.cache_file}")
    
    # Создаем новый кэш и загружаем
    new_cache = TextCache(cache_file=cache_file, config=config)
    print(f"   Новый кэш создан, размер: {len(new_cache)}")
    
    # Проверяем что данные загрузились
    for text in test_texts[:3]:  # Проверяем первые 3
        cached = new_cache.get_surface_from_text(text)
        status = "✅ LOADED" if cached is not None else "❌ NOT LOADED"
        print(f"   '{text}': {status}")
    
    # Тест интеграции с моделями
    print(f"\n9️⃣ Тест интеграции с моделями:")
    
    try:
        # Создаем модели
        encoder = TextToCubeEncoder(config)
        decoder = CubeToTextDecoder(config)
        
        # Создаем кэшированные версии
        cached_encoder = CachedTextToCubeEncoder(encoder, cache)
        cached_decoder = CachedCubeToTextDecoder(decoder, cache)
        
        # Тестовые тексты
        integration_texts = ["Integration test 1", "Integration test 2"]
        
        # Первый прогон (без кэша)
        start_time = time.time()
        embeddings1 = cached_encoder.encode_text(integration_texts)
        time1 = time.time() - start_time
        print(f"   Первый прогон encoder: {time1:.4f}s")
        
        # Второй прогон (с кэшем)
        start_time = time.time()
        embeddings2 = cached_encoder.encode_text(integration_texts)
        time2 = time.time() - start_time
        print(f"   Второй прогон encoder (кэш): {time2:.4f}s")
        
        # Проверяем что результаты идентичны
        match = torch.allclose(embeddings1, embeddings2, atol=1e-6)
        print(f"   Результаты идентичны: {'✅ YES' if match else '❌ NO'}")
        print(f"   Ускорение: {time1/time2 if time2 > 0 else 'N/A'}x")
        
        # Тест decoder кэширования
        texts1 = cached_decoder.decode_surface(embeddings1[:2])  # Первые 2
        texts2 = cached_decoder.decode_surface(embeddings1[:2])  # Те же (должны из кэша)
        
        print(f"   Decoder результаты идентичны: {'✅ YES' if texts1 == texts2 else '❌ NO'}")
        
    except Exception as e:
        print(f"   ❌ Ошибка в интеграции: {e}")
    
    # Итоговая статистика
    print(f"\n🔟 Итоговая статистика:")
    final_stats = cache.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Очистка тестовых файлов
    try:
        Path(cache_file).unlink(missing_ok=True)
        print(f"\n🧹 Тестовый файл кэша удален")
    except:
        pass
    
    print("\n✅ Тест TextCache завершен!")
    return True

if __name__ == "__main__":
    try:
        test_text_cache()
    except Exception as e:
        print(f"❌ Общая ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()