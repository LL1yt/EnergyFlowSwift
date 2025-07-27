#!/usr/bin/env python3
"""
Тест TextToCubeEncoder для проверки базовой функциональности
"""

import torch
import sys
from pathlib import Path

# Добавляем путь к energy_flow
sys.path.append('energy_flow')

from energy_flow.config import create_debug_config
from energy_flow.text_bridge import TextToCubeEncoder, create_text_to_cube_encoder

def test_text_to_cube_encoder():
    print("🧪 Тестирование TextToCubeEncoder...")
    
    # Создаем debug конфигурацию
    config = create_debug_config()
    print(f"📐 Размеры куба: {config.lattice_width}×{config.lattice_height}×{config.lattice_depth}")
    print(f"📏 Surface dim: {config.lattice_width * config.lattice_height}")
    
    # Создаем encoder
    encoder = create_text_to_cube_encoder(config)
    
    # Тестовые тексты
    test_texts = [
        "Hello world!",
        "This is a test sentence for energy flow cube.",
        "Machine learning and neural networks are fascinating.",
        "Короткий текст",
        "A much longer text that should test the tokenization and encoding capabilities of our TextToCubeEncoder model implementation."
    ]
    
    print(f"\n🔤 Тестируем на {len(test_texts)} текстах...")
    
    # Тест одиночного текста
    print("\n1️⃣ Тест одиночного текста:")
    single_result = encoder.encode_text(test_texts[0])
    print(f"   Входной текст: '{test_texts[0]}'")
    print(f"   Размер результата: {single_result.shape}")
    print(f"   Статистика: mean={single_result.mean():.4f}, std={single_result.std():.4f}")
    print(f"   Диапазон: [{single_result.min():.4f}, {single_result.max():.4f}]")
    
    # Тест батча текстов
    print("\n2️⃣ Тест батча текстов:")
    batch_result = encoder.encode_text(test_texts)
    print(f"   Размер батча: {len(test_texts)}")
    print(f"   Размер результата: {batch_result.shape}")
    print(f"   Статистика батча: mean={batch_result.mean():.4f}, std={batch_result.std():.4f}")
    
    # Тест reshape в 2D поверхность
    print("\n3️⃣ Тест reshape в 2D поверхность:")
    surface_2d = encoder.reshape_to_surface(batch_result)
    print(f"   Размер 2D поверхности: {surface_2d.shape}")
    print(f"   Ожидаемый размер: [{len(test_texts)}, {config.lattice_height}, {config.lattice_width}]")
    
    # Тест токенизатора
    print("\n4️⃣ Информация о токенизаторе:")
    tokenizer = encoder.get_tokenizer()
    print(f"   Размер словаря: {encoder.get_vocab_size():,}")
    print(f"   Pad token: {tokenizer.pad_token}")
    
    # Тест разных длин текста
    print("\n5️⃣ Тест разных длин текста:")
    for i, text in enumerate(test_texts):
        tokens = tokenizer.encode(text)
        result = encoder.encode_text(text)
        print(f"   Текст {i+1}: {len(tokens)} токенов → {result.shape} эмбеддинг")
    
    # Проверка параметров модели
    print("\n6️⃣ Анализ модели:")
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   Общее количество параметров: {total_params:,}")
    print(f"   Целевое количество (~5M): {'✅ OK' if total_params < 7_000_000 else '⚠️ Превышено'}")
    
    # Проверка устройства (должно быть CUDA если доступно)
    print(f"\n7️⃣ Информация об устройстве:")
    device = next(encoder.parameters()).device
    print(f"   Устройство модели: {device}")
    print(f"   CUDA доступна: {torch.cuda.is_available()}")
    print(f"   Default device: {torch.get_default_device()}")
    
    print("\n✅ Тест TextToCubeEncoder завершен!")
    return True

if __name__ == "__main__":
    try:
        test_text_to_cube_encoder()
    except Exception as e:
        print(f"❌ Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()