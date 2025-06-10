#!/usr/bin/env python3
"""
Конвертер конфигурации из dict в объект для EmergentCubeTrainer
"""

class ConfigObject:
    """Простой объект для конвертации dict в атрибуты"""
    
    def __init__(self, config_dict):
        """Конвертируем словарь в атрибуты объекта"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Рекурсивно конвертируем вложенные словари
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        """Fallback для отсутствующих атрибутов с разумными значениями по умолчанию"""
        defaults = {
            'teacher_model': 'distilbert-base-uncased',
            'cube_dimensions': (15, 15, 11),
            'enable_full_cube_gradient': True,
            'spatial_propagation_depth': 11,
            'emergent_specialization': True,
            'learning_rate': 0.001,
            'batch_size': 8,
            'epochs': 15,
            'warmup_epochs': 3,
            'gradient_balancing': True,
            'adaptive_loss_weighting': True,
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'gradient_accumulation_steps': 4,
            'effective_batch_size': 32,
            'enable_nca': True,
            'gmlp_config': {
                'state_size': 32,
                'neighbor_count': 6,
                'hidden_dim': 32,
                'external_input_size': 12,
                'memory_dim': 16,
                'use_memory': True,
                'activation': 'gelu',
                'dropout': 0.1,
                'spatial_connections': True
            },
            'loss_weights': {
                'surface_reconstruction': 0.3,
                'internal_consistency': 0.3,
                'dialogue_similarity': 0.4
            },
            'nca_config': {
                'update_probability': 0.7,
                'stochastic_scheduling': True,
                'synchronization_avoidance': True,
                'residual_learning_rate': 0.1,
                'stability_threshold': 0.01,
                'max_update_magnitude': 0.5,
                'pattern_detection_enabled': True,
                'spatial_coherence_weight': 0.3,
                'temporal_consistency_weight': 0.2,
                'track_emergent_specialization': True,
                'specialization_threshold': 0.15,
                'diversity_preservation_weight': 0.25
            }
        }
        
        if name in defaults:
            value = defaults[name]
            if isinstance(value, dict):
                value = ConfigObject(value)
            setattr(self, name, value)
            return value
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __repr__(self):
        attrs = []
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                attrs.append(f"{key}=ConfigObject(...)")
            else:
                attrs.append(f"{key}={repr(value)}")
        return f"ConfigObject({', '.join(attrs)})"
    
    def __iter__(self):
        """Поддержка итерации для unpacking с **"""
        return iter(self.__dict__)
    
    def keys(self):
        """Поддержка dict.keys() для unpacking"""
        return self.__dict__.keys()
    
    def values(self):
        """Поддержка dict.values()"""
        return self.__dict__.values()
    
    def items(self):
        """Поддержка dict.items()"""
        return self.__dict__.items()
    
    def __getitem__(self, key):
        """Поддержка индексации как словарь"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Поддержка присваивания как словарь"""
        setattr(self, key, value)
    
    def to_dict(self):
        """Конвертация обратно в словарь"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __json__(self):
        """JSON сериализация"""
        return self.to_dict()
    
    def to_json(self):
        """Конвертация в JSON строку"""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

def convert_config_dict_to_object(config_dict):
    """
    Конвертирует словарь конфигурации в объект с атрибутами
    
    Args:
        config_dict: Словарь конфигурации
        
    Returns:
        ConfigObject с атрибутами
    """
    return ConfigObject(config_dict)

def test_config_converter():
    """Тест конвертера"""
    print("🧪 Testing config converter...")
    
    # Тестовая конфигурация
    test_config = {
        'teacher_model': 'distilbert-base-uncased',
        'cube_dimensions': [15, 15, 11],
        'nested': {
            'value1': 42,
            'value2': 'test'
        }
    }
    
    # Конвертируем
    config_obj = convert_config_dict_to_object(test_config)
    
    # Проверяем атрибуты
    assert config_obj.teacher_model == 'distilbert-base-uncased'
    assert config_obj.cube_dimensions == [15, 15, 11]
    assert config_obj.nested.value1 == 42
    assert config_obj.nested.value2 == 'test'
    
    # Проверяем fallback атрибуты
    assert config_obj.learning_rate == 0.001
    assert config_obj.enable_nca == True
    
    # Проверяем поддержку unpacking (для **kwargs)
    def test_unpacking(**kwargs):
        return len(kwargs)
    
    count = test_unpacking(**config_obj.nested)
    assert count == 2  # value1 и value2
    
    print("[OK] Config converter works!")
    return True

if __name__ == "__main__":
    test_config_converter() 