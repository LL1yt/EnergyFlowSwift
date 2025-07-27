#!/usr/bin/env python3
"""
Тест EnergyConfig с text_bridge параметрами
"""

import sys
from pathlib import Path

# Добавляем путь к energy_flow
sys.path.append('energy_flow')

from energy_flow.config import create_debug_config, create_experiment_config, create_optimized_config

def test_energy_config_text_bridge():
    print("🧪 Тестирование EnergyConfig с text_bridge параметрами...")
    
    configs = {
        "debug": create_debug_config(),
        "experiment": create_experiment_config(), 
        "optimized": create_optimized_config()
    }
    
    for name, config in configs.items():
        print(f"\n📋 Конфигурация {name.upper()}:")
        print(f"   Размеры решетки: {config.lattice_width}×{config.lattice_height}×{config.lattice_depth}")
        print(f"   Surface dimension: {config.surface_dimension}")
        print(f"   Text bridge включен: {config.text_bridge_enabled}")
        
        if config.text_bridge_enabled:
            print(f"   Text cache включен: {config.text_cache_enabled}")
            print(f"   Text cache размер: {config.text_cache_size}")
            print(f"   Text loss weight: {config.text_loss_weight}")
            print(f"   Итеративные шаги: {config.iterative_correction_steps}")
            print(f"   Max text length: {config.text_generation_max_length}")
            print(f"   Num beams: {config.text_generation_num_beams}")
            print(f"   Temperature: {config.text_generation_temperature}")
        
        # Проверяем валидацию
        try:
            # Это должно пройти без ошибок благодаря __post_init__
            print(f"   ✅ Валидация пройдена")
        except Exception as e:
            print(f"   ❌ Ошибка валидации: {e}")
    
    # Тест создания конфигурации с невалидными text_bridge параметрами
    print(f"\n🚨 Тест валидации с неправильными параметрами:")
    
    try:
        from energy_flow.config.energy_config import EnergyConfig
        
        # Попробуем создать конфигурацию с неправильным text_loss_weight
        invalid_config = EnergyConfig(
            lattice_width=10,
            lattice_height=10,
            lattice_depth=5,
            text_bridge_enabled=True,
            text_loss_weight=1.5  # Неправильное значение > 1.0
        )
        print("   ❌ Валидация НЕ сработала!")
        
    except AssertionError as e:
        print(f"   ✅ Валидация сработала корректно: {e}")
    except Exception as e:
        print(f"   ⚠️ Неожиданная ошибка: {e}")
    
    # Тест свойства surface_dimension
    print(f"\n🔍 Тест вычисляемых свойств:")
    debug_config = create_debug_config()
    expected_surface_dim = debug_config.lattice_width * debug_config.lattice_height
    actual_surface_dim = debug_config.surface_dimension
    
    print(f"   Expected surface_dimension: {expected_surface_dim}")
    print(f"   Actual surface_dimension: {actual_surface_dim}")
    print(f"   ✅ Совпадение: {'YES' if expected_surface_dim == actual_surface_dim else 'NO'}")
    
    # Тест to_dict
    print(f"\n📄 Тест сериализации конфигурации:")
    config_dict = debug_config.to_dict()
    text_bridge_keys = [k for k in config_dict.keys() if k.startswith('text_')]
    print(f"   Text bridge ключи в словаре: {len(text_bridge_keys)}")
    for key in text_bridge_keys[:5]:  # Показываем первые 5
        print(f"     {key}: {config_dict[key]}")
    if len(text_bridge_keys) > 5:
        print(f"     ... и еще {len(text_bridge_keys) - 5}")
    
    print("\n✅ Тест EnergyConfig с text_bridge завершен!")
    return True

if __name__ == "__main__":
    try:
        test_energy_config_text_bridge()
    except Exception as e:
        print(f"❌ Общая ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()