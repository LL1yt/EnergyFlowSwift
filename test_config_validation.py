#!/usr/bin/env python3
"""Тест для проверки validation настроек в централизованном конфиге"""

from new_rebuild.config import (
    create_debug_config,
    create_experiment_config,
    create_optimized_config
)

def test_validation_settings():
    """Проверяем, что validation настройки правильно загружаются из пресетов"""
    
    # DEBUG режим
    debug_config = create_debug_config()
    print(f"DEBUG mode:")
    print(f"  num_forward_passes: {debug_config.validation.num_forward_passes}")
    print(f"  stability_threshold: {debug_config.validation.stability_threshold}")
    assert debug_config.validation.num_forward_passes == 3
    assert debug_config.validation.stability_threshold == 0.15
    
    # EXPERIMENT режим
    exp_config = create_experiment_config()
    print(f"\nEXPERIMENT mode:")
    print(f"  num_forward_passes: {exp_config.validation.num_forward_passes}")
    print(f"  stability_threshold: {exp_config.validation.stability_threshold}")
    assert exp_config.validation.num_forward_passes == 5
    assert exp_config.validation.stability_threshold == 0.1
    
    # OPTIMIZED режим
    opt_config = create_optimized_config()
    print(f"\nOPTIMIZED mode:")
    print(f"  num_forward_passes: {opt_config.validation.num_forward_passes}")
    print(f"  stability_threshold: {opt_config.validation.stability_threshold}")
    assert opt_config.validation.num_forward_passes == 10
    assert opt_config.validation.stability_threshold == 0.05
    
    print("\n✅ Все тесты пройдены!")

if __name__ == "__main__":
    test_validation_settings()