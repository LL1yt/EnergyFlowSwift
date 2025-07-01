#!/usr/bin/env python3
"""Диагностика соответствия кэша и найденных соседей"""

from new_rebuild.core.moe.connection_classifier import UnifiedConnectionClassifier
from new_rebuild.core.lattice.spatial_optimization.unified_spatial_optimizer import create_unified_spatial_optimizer
from new_rebuild.config import SimpleProjectConfig

print("🔍 Диагностика кэша vs найденных соседей...")

config = SimpleProjectConfig()
print(f"Lattice dimensions: {config.lattice.dimensions}")

# Создаем spatial optimizer 
spatial_optimizer = create_unified_spatial_optimizer(config.lattice.dimensions)

# Создаем classifier
classifier = UnifiedConnectionClassifier(
    lattice_dimensions=config.lattice.dimensions,
    enable_cache=True
)

# Тестируем клетку 447
cell_idx = 447
print(f"\n🧪 Тестируем клетку {cell_idx}:")

# Получаем соседей через spatial optimizer
neighbors_from_spatial = spatial_optimizer.find_neighbors_by_radius_safe(cell_idx)
print(f"Соседи от spatial optimizer: {len(neighbors_from_spatial)} - {neighbors_from_spatial[:10]}...")

# Проверяем что есть в кэше для этой клетки
cache_manager = classifier.cache_manager
if cell_idx in cache_manager.cache:
    cached_data = cache_manager.cache[cell_idx]
    local_cached = [conn.target_idx for conn in cached_data.get('local', [])]
    functional_cached = [conn.target_idx for conn in cached_data.get('functional_candidates', [])]
    distant_cached = [conn.target_idx for conn in cached_data.get('distant', [])]
    
    print(f"Кэш LOCAL: {len(local_cached)} - {local_cached[:10]}")
    print(f"Кэш FUNCTIONAL: {len(functional_cached)} - {functional_cached[:10]}")
    print(f"Кэш DISTANT: {len(distant_cached)} - {distant_cached[:10]}")
    
    # Проверяем пересечения
    neighbors_set = set(neighbors_from_spatial)
    local_set = set(local_cached)
    functional_set = set(functional_cached)
    distant_set = set(distant_cached)
    all_cached = local_set | functional_set | distant_set
    
    intersection = neighbors_set & all_cached
    print(f"\n📊 Пересечение spatial и кэша: {len(intersection)} из {len(neighbors_from_spatial)}")
    print(f"   Соседи НЕ в кэше: {len(neighbors_set - all_cached)}")
    print(f"   Примеры НЕ в кэше: {list(neighbors_set - all_cached)[:10]}")
    
else:
    print(f"❌ Клетка {cell_idx} НЕ найдена в кэше!")
    
print("\n✅ Диагностика завершена!")