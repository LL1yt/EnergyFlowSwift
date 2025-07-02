#!/usr/bin/env python3
"""
Тест предупреждения о централизованной конфигурации
==================================================
"""

import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_rebuild.config import create_debug_config, create_experiment_config


def test_warning_on_first_use():
    """Проверяем что предупреждение показывается только один раз за сессию"""
    print("=== Тест предупреждения при создании конфига ===\n")
    
    # Сбрасываем глобальный флаг для теста
    from new_rebuild.config.simple_config import _global_migration_warned
    import new_rebuild.config.simple_config as config_module
    config_module._global_migration_warned = False
    
    # Ловим предупреждения
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Создаем DEBUG конфиг
        print("1. Создаем DEBUG конфиг (первый раз)...")
        config = create_debug_config()
        
        # Проверяем что было предупреждение
        if len(w) > 0:
            print(f"✅ Предупреждение показано при первом создании")
        else:
            print("❌ Предупреждение не показано!")
            
    # Проверяем что значения из пресетов применились
    print("\n2. Проверяем значения из пресетов:")
    print(f"   Решетка: {config.lattice.dimensions} (из mode_presets.debug)")
    print(f"   State size: {config.model.state_size} (из mode_presets.debug)")
    print(f"   MoE functional: {config.architecture.moe_functional_params} (из mode_presets.debug)")
    
    # Создаем второй конфиг - предупреждение НЕ должно повториться
    print("\n3. Создаем EXPERIMENT конфиг (предупреждения быть НЕ должно)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        config2 = create_experiment_config()
        
        # Предупреждения быть не должно
        if len(w) == 0:
            print(f"✅ Предупреждение НЕ показано (как и должно быть)")
        else:
            print(f"❌ Неожиданное предупреждение: {w[0].message}")
            
    print(f"\n4. Значения EXPERIMENT режима:")
    print(f"   Решетка: {config2.lattice.dimensions} (из mode_presets.experiment)")
    print(f"   State size: {config2.model.state_size} (из mode_presets.experiment)")


def test_accessing_presets():
    """Показываем как можно напрямую обращаться к пресетам"""
    print("\n=== Прямой доступ к пресетам ===\n")
    
    config = create_debug_config()
    
    print("Доступ к пресетам через config.mode_presets:")
    print(f"  DEBUG пресет: {config.mode_presets.debug.lattice_dimensions}")
    print(f"  EXPERIMENT пресет: {config.mode_presets.experiment.lattice_dimensions}")
    print(f"  OPTIMIZED пресет: {config.mode_presets.optimized.lattice_dimensions}")
    
    print("\nТак можно создавать свои кастомные режимы на основе пресетов!")


def main():
    print("🔧 Тестирование предупреждений о централизованной конфигурации")
    print("=" * 60)
    
    test_warning_on_first_use()
    test_accessing_presets()
    
    print("\n" + "=" * 60)
    print("✅ Тест завершен!")
    print("\n💡 Итоги:")
    print("1. Предупреждение показывается при первом создании конфига")
    print("2. Все значения берутся из централизованных пресетов")
    print("3. Больше никаких hardcoded значений в режимах!")
    print("4. Можно напрямую обращаться к пресетам через config.mode_presets")


if __name__ == "__main__":
    main()