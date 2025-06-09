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
        print("✅ EmergentCubeTrainer imported successfully")
    except Exception as e:
        print(f"❌ EmergentCubeTrainer import failed: {e}")
        return False
    
    try:
        from utils.config_manager.config_manager import ConfigManager
        print("✅ ConfigManager imported successfully")
    except Exception as e:
        print(f"❌ ConfigManager import failed: {e}")
        return False
    
    try:
        from simple_embedding_fallback import create_dialogue_dataset_simple_fallback
        print("✅ SimpleFallbackEmbeddingLoader imported successfully")
    except Exception as e:
        print(f"❌ SimpleFallbackEmbeddingLoader import failed: {e}")
        return False
    
    try:
        from model_weights_manager import ModelWeightsManager
        print("✅ ModelWeightsManager imported successfully")
    except Exception as e:
        print(f"❌ ModelWeightsManager import failed: {e}")
        return False
    
    try:
        from config_converter import convert_config_dict_to_object
        print("✅ ConfigConverter imported successfully")
    except Exception as e:
        print(f"❌ ConfigConverter import failed: {e}")
        return False
    
    return True

def test_trainer_creation():
    """Тестирование создания trainer"""
    print("\n🔧 Testing trainer creation...")
    
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
        print(f"✅ Trainer created successfully")
        print(f"   Parameters: {sum(p.numel() for p in trainer.parameters()):,}")
        print(f"   Device: cuda available = {torch.cuda.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция тестирования"""
    print("🎯 TESTING TRAINER IMPORTS AND CREATION")
    print("="*50)
    
    # Тест импортов
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Тест создания
    import torch  # Добавляем torch import здесь
    if not test_trainer_creation():
        print("\n❌ Trainer creation tests failed")
        return 1
    
    print("\n✅ All tests passed!")
    print("Ready to run overnight training")
    return 0

if __name__ == "__main__":
    exit(main()) 