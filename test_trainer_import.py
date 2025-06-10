#!/usr/bin/env python3
"""
Тест импорта и создания trainer
"""

import sys
from pathlib import Path

# Добавляем пути
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Тестирование импортов"""
    print("🧪 Testing imports...")
    
    try:
        from training.embedding_trainer.emergent_training_stage_3_1_4_1 import EmergentCubeTrainer
        print("[OK] EmergentCubeTrainer imported successfully")
    except Exception as e:
        print(f"[ERROR] EmergentCubeTrainer import failed: {e}")
        return False
    
    try:
        from utils.config_manager.config_manager import ConfigManager
        print("[OK] ConfigManager imported successfully")
    except Exception as e:
        print(f"[ERROR] ConfigManager import failed: {e}")
        return False
    
    try:
        from simple_embedding_fallback import create_dialogue_dataset_simple_fallback
        print("[OK] SimpleFallbackEmbeddingLoader imported successfully")
    except Exception as e:
        print(f"[ERROR] SimpleFallbackEmbeddingLoader import failed: {e}")
        return False
    
    try:
        from model_weights_manager import ModelWeightsManager
        print("[OK] ModelWeightsManager imported successfully")
    except Exception as e:
        print(f"[ERROR] ModelWeightsManager import failed: {e}")
        return False
    
    try:
        from config_converter import convert_config_dict_to_object
        print("[OK] ConfigConverter imported successfully")
    except Exception as e:
        print(f"[ERROR] ConfigConverter import failed: {e}")
        return False
    
    return True

def test_trainer_creation():
    """Тестирование создания trainer"""
    print("\n[CONFIG] Testing trainer creation...")
    
    try:
        import torch  # Добавляем импорт torch
        from training.embedding_trainer.emergent_training_stage_3_1_4_1 import EmergentCubeTrainer
        from utils.config_manager.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config_dict = config_manager.get_config()  # Получаем всю конфигурацию
        
        # Конвертируем dict в объект для EmergentCubeTrainer
        from config_converter import convert_config_dict_to_object
        config = convert_config_dict_to_object(config_dict)
        
        trainer = EmergentCubeTrainer(config)
        print(f"[OK] Trainer created successfully")
        print(f"   Parameters: {sum(p.numel() for p in trainer.parameters()):,}")
        print(f"   Device: cuda available = {torch.cuda.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция тестирования"""
    print("[TARGET] TESTING TRAINER IMPORTS AND CREATION")
    print("="*50)
    
    # Тест импортов
    if not test_imports():
        print("\n[ERROR] Import tests failed")
        return 1
    
    # Тест создания
    import torch  # Добавляем torch import здесь
    if not test_trainer_creation():
        print("\n[ERROR] Trainer creation tests failed")
        return 1
    
    print("\n[OK] All tests passed!")
    print("Ready to run overnight training")
    return 0

if __name__ == "__main__":
    exit(main()) 