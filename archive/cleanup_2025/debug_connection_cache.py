#!/usr/bin/env python3
"""Диагностический тест для отладки connection cache"""

import torch
from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.config import SimpleProjectConfig

print("🔍 Диагностика connection cache...")

# Создаем конфигурацию
config = SimpleProjectConfig()
print(f"Lattice dimensions: {config.lattice.dimensions}")
print(f"Adaptive radius: {config.calculate_adaptive_radius()}")
print(f"Local threshold: {config.lattice.local_distance_threshold}")

# Создаем классификатор
classifier = UnifiedConnectionClassifier(
    lattice_dimensions=config.lattice.dimensions,
    enable_cache=True
)

print(f"\n📊 Classifier thresholds:")
if hasattr(classifier, 'local_threshold'):
    print(f"  Local: {classifier.local_threshold:.3f}")
    print(f"  Functional: {classifier.functional_threshold:.3f}")
    print(f"  Distant: {classifier.distant_threshold:.3f}")
else:
    print("  Пороги не найдены как атрибуты, ищем в cache_manager...")
    if hasattr(classifier.cache_manager, 'local_threshold'):
        print(f"  Local: {classifier.cache_manager.local_threshold:.3f}")
        print(f"  Functional: {classifier.cache_manager.functional_threshold:.3f}")
        print(f"  Distant: {classifier.cache_manager.distant_threshold:.3f}")

# Проверяем состояние кэша
if hasattr(classifier, 'cache_manager') and classifier.cache_manager:
    cache_manager = classifier.cache_manager
    print(f"\n🗂️ Cache manager state:")
    print(f"  Total cells: {cache_manager.total_cells}")
    print(f"  Cache size: {len(cache_manager.cache) if hasattr(cache_manager, 'cache') else 'NO CACHE'}")
    
    # Проверяем несколько ключевых клеток
    test_cells = [0, 100, 200, 327, 391, 500]
    for cell_idx in test_cells:
        if cell_idx < cache_manager.total_cells:
            if hasattr(cache_manager, 'cache') and cell_idx in cache_manager.cache:
                cached_data = cache_manager.cache[cell_idx]
                local_count = len(cached_data.get('local', []))
                functional_count = len(cached_data.get('functional_candidates', []))
                distant_count = len(cached_data.get('distant', []))
                print(f"    Cell {cell_idx}: LOCAL={local_count}, FUNC={functional_count}, DIST={distant_count}")
            else:
                print(f"    Cell {cell_idx}: НЕТ В КЭШЕ")
    
    # Проверим инициализацию кэша
    if not hasattr(cache_manager, 'cache') or len(cache_manager.cache) == 0:
        print("\n⚠️ ПРОБЛЕМА: Кэш не инициализирован!")
        print("Попробуем принудительно инициализировать...")
        
        try:
            # Попробуем вызвать метод инициализации если он есть
            if hasattr(cache_manager, '_build_cache'):
                cache_manager._build_cache()
                print("✅ Кэш успешно построен")
                print(f"   Размер кэша: {len(cache_manager.cache)}")
            elif hasattr(cache_manager, 'build_cache'):
                cache_manager.build_cache()
                print("✅ Кэш успешно построен")
                print(f"   Размер кэша: {len(cache_manager.cache)}")
            else:
                print("❌ Метод инициализации кэша не найден")
                
        except Exception as e:
            print(f"❌ Ошибка при инициализации кэша: {e}")
            import traceback
            traceback.print_exc()
else:
    print("\n❌ Cache manager не найден!")

# Тестируем простую классификацию
print(f"\n🧪 Тестируем классификацию для клетки 327:")

# Создаем тестовые состояния
state_size = config.model.state_size
cell_state = torch.randn(state_size)
neighbor_indices = [327]  # Только сама клетка как сосед
neighbor_states = torch.randn(len(neighbor_indices), state_size)

try:
    result = classifier.classify_connections(
        cell_idx=327,
        neighbor_indices=neighbor_indices,
        cell_state=cell_state,
        neighbor_states=neighbor_states
    )
    
    for category, connections in result.items():
        print(f"  {category.name}: {len(connections)} соединений")
        
except Exception as e:
    print(f"❌ Ошибка при классификации: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Диагностика завершена!")