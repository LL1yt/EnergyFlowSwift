#!/usr/bin/env python3
"""
Тест CubeToTextDecoder для проверки базовой функциональности
"""

import torch
import sys
from pathlib import Path

# Добавляем путь к energy_flow
sys.path.append('energy_flow')

from energy_flow.config import create_debug_config
from energy_flow.text_bridge import TextToCubeEncoder
from energy_flow.text_bridge.cube_to_text_decoder import (
    CubeToTextDecoder, 
    create_cube_to_text_decoder,
    SyntheticTrainingDataGenerator
)

def test_cube_to_text_decoder():
    print("🧪 Тестирование CubeToTextDecoder...")
    
    # Создаем debug конфигурацию
    config = create_debug_config()
    print(f"📐 Размеры куба: {config.lattice_width}×{config.lattice_height}×{config.lattice_depth}")
    print(f"📏 Surface dim: {config.lattice_width * config.lattice_height}")
    
    # Создаем decoder
    decoder = create_cube_to_text_decoder(config)
    
    print("\n1️⃣ Анализ модели:")
    total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    print(f"   Обучаемые параметры: {total_params:,}")
    print(f"   Замороженные параметры: {frozen_params:,}")
    print(f"   Общие параметры: {total_params + frozen_params:,}")
    print(f"   Целевое количество (~60M): {'✅ OK' if total_params < 70_000_000 else '⚠️ Превышено'}")
    
    # Создаем тестовые surface embeddings
    print("\n2️⃣ Генерация тестовых surface embeddings:")
    batch_size = 3
    surface_dim = config.lattice_width * config.lattice_height
    
    # Случайные embeddings в диапазоне [-1, 1]
    test_surface_embeddings = torch.randn(batch_size, surface_dim) * 0.5
    test_surface_embeddings = torch.clamp(test_surface_embeddings, -1, 1)
    
    print(f"   Размер тестовых embeddings: {test_surface_embeddings.shape}")
    print(f"   Статистика: mean={test_surface_embeddings.mean():.4f}, std={test_surface_embeddings.std():.4f}")
    print(f"   Диапазон: [{test_surface_embeddings.min():.4f}, {test_surface_embeddings.max():.4f}]")
    
    # Тест базового декодирования
    print("\n3️⃣ Тест базового декодирования:")
    try:
        decoded_texts = decoder.decode_surface(test_surface_embeddings, max_length=32)
        print(f"   Количество декодированных текстов: {len(decoded_texts)}")
        for i, text in enumerate(decoded_texts):
            print(f"   Текст {i+1}: '{text}'")
    except Exception as e:
        print(f"   ❌ Ошибка в базовом декодировании: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Тест итеративного декодирования
    print("\n4️⃣ Тест итеративного декодирования:")
    try:
        iterative_texts = decoder.iterative_decode(
            test_surface_embeddings, 
            max_length=32, 
            correction_steps=2
        )
        print(f"   Количество итеративно декодированных текстов: {len(iterative_texts)}")
        for i, text in enumerate(iterative_texts):
            print(f"   Итеративный текст {i+1}: '{text}'")
    except Exception as e:
        print(f"   ❌ Ошибка в итеративном декодировании: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест генератора синтетических данных
    print("\n5️⃣ Тест генератора синтетических данных:")
    try:
        generator = SyntheticTrainingDataGenerator(config)
        synthetic_pairs = generator.generate_synthetic_pairs(5)
        
        print(f"   Сгенерировано пар: {len(synthetic_pairs)}")
        for i, (emb, text) in enumerate(synthetic_pairs[:3]):
            print(f"   Пара {i+1}: embedding {emb.shape} → '{text}'")
            print(f"            статистика: mean={emb.mean():.3f}, std={emb.std():.3f}")
    except Exception as e:
        print(f"   ❌ Ошибка в генераторе данных: {e}")
    
    # Тест интеграции с TextToCubeEncoder
    print("\n6️⃣ Тест интеграции с TextToCubeEncoder:")
    try:
        # Создаем encoder
        encoder = TextToCubeEncoder(config)
        
        # Тестовые тексты
        test_texts = [
            "Hello world!",
            "This is a test for integration.",
            "Neural networks are fascinating."
        ]
        
        # Кодируем в surface embeddings
        surface_embeddings = encoder.encode_text(test_texts)
        print(f"   Кодированные embeddings: {surface_embeddings.shape}")
        
        # Декодируем обратно в текст
        reconstructed_texts = decoder.decode_surface(surface_embeddings, max_length=32)
        
        print(f"   Сравнение оригинал → reconstruction:")
        for orig, recon in zip(test_texts, reconstructed_texts):
            print(f"   '{orig}' → '{recon}'")
        
    except Exception as e:
        print(f"   ❌ Ошибка в интеграции: {e}")
        import traceback
        traceback.print_exc()
    
    # Проверка устройства
    print(f"\n7️⃣ Информация об устройстве:")
    device = next(decoder.parameters()).device
    print(f"   Устройство модели: {device}")
    print(f"   CUDA доступна: {torch.cuda.is_available()}")
    print(f"   Default device: {torch.get_default_device()}")
    
    # Информация о токенизаторе
    print(f"\n8️⃣ Информация о токенизаторе:")
    tokenizer = decoder.get_tokenizer()
    print(f"   Тип токенизатора: T5Tokenizer")
    print(f"   Размер словаря: {len(tokenizer)}")
    print(f"   Pad token: {tokenizer.pad_token}")
    print(f"   EOS token: {tokenizer.eos_token}")
    
    print("\n✅ Тест CubeToTextDecoder завершен!")
    return True

if __name__ == "__main__":
    try:
        test_cube_to_text_decoder()
    except Exception as e:
        print(f"❌ Общая ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()