"""Debug test for cache initialization"""
from new_rebuild.config import get_project_config
from new_rebuild.config.config_components import CacheSettings
from dataclasses import asdict

# Test CacheSettings
print("=== Testing CacheSettings ===")
cache_settings = CacheSettings()
print(f"CacheSettings fields: {asdict(cache_settings)}")

# Test project config
print("\n=== Testing Project Config ===")
config = get_project_config()
print(f"Config type: {type(config)}")
print(f"Config has cache: {hasattr(config, 'cache')}")
if hasattr(config, 'cache'):
    print(f"Cache settings: {asdict(config.cache) if config.cache else 'None'}")
    
# Test cache-specific fields
print("\n=== Testing cache-specific fields ===")
if config.cache:
    cache_dict = asdict(config.cache)
    print(f"enabled: {cache_dict.get('enabled', 'NOT FOUND')}")
    print(f"local_radius: {cache_dict.get('local_radius', 'NOT FOUND')}")
    print(f"functional_similarity_threshold: {cache_dict.get('functional_similarity_threshold', 'NOT FOUND')}")