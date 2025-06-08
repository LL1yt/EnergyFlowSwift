#!/usr/bin/env python3
"""
🧪 ADAPTER INTEGRATION TEST SUITE
Тестирование интеграции универсального адаптера с CubeTrainer
"""

import torch
import traceback
import time
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорт наших модулей
try:
    from training.embedding_trainer.adapter_integration import (
        AdapterIntegrationConfig,
        AdapterCubeTrainer,
        create_llama3_cube_trainer,
        create_distilbert_cube_trainer
    )
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    exit(1)


def test_basic_integration():
    """
    Тест 1: Базовая интеграция адаптера с CubeTrainer
    """
    print("\n🧪 ТЕСТ 1: Basic Adapter-Cube Integration")
    print("=" * 50)
    
    try:
        print("📋 1.1: Creating LLaMA-3-8B → 15×15×11 integrated trainer...")
        
        # Создание интегрированного тренера
        trainer = create_llama3_cube_trainer(
            cube_dimensions=(15, 15, 11),
            adapter_strategy="learned_linear",
            device="cpu"
        )
        
        # Проверка информации о системе
        info = trainer.get_info()
        
        print(f"   ✅ Integrated trainer created:")
        print(f"   📊 Teacher: {info['teacher_model']}")
        print(f"   🔧 Adapter: {info['adapter']['input_dim']}D → {info['adapter']['output_dim']}D")
        print(f"   📦 Cube: {info['cube_dimensions']}")
        print(f"   🎯 Compression: {info['adapter']['compression_ratio']:.3f}")
        print(f"   📈 Total parameters: {info['total_parameters']:,}")
        
        print("\n📋 1.2: Testing forward pass through full pipeline...")
        
        # Тестовый forward pass
        batch_size = 3
        teacher_embeddings = torch.randn(batch_size, 4096)  # LLaMA-3-8B размер
        
        # Full forward pass
        output = trainer.forward(teacher_embeddings)
        print(f"   ✅ Forward pass: {teacher_embeddings.shape} → {output.shape}")
        
        # Forward pass с промежуточными результатами
        results = trainer.forward(teacher_embeddings, return_intermediate=True)
        
        print(f"   📊 Pipeline details:")
        print(f"      Teacher: {results['teacher_embeddings'].shape}")
        print(f"      Surface: {results['surface_embeddings'].shape}")
        print(f"      Output: {results['output'].shape}")
        
        if results['reconstructed'] is not None:
            print(f"      Reconstructed: {results['reconstructed'].shape}")
        
        print("\n🎯 ТЕСТ 1 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_different_models():
    """
    Тест 2: Разные teacher модели
    """
    print("\n🧪 ТЕСТ 2: Different Teacher Models")
    print("=" * 50)
    
    try:
        models_to_test = [
            ("LLaMA-3-8B", create_llama3_cube_trainer),
            ("DistilBERT", create_distilbert_cube_trainer)
        ]
        
        results = {}
        
        for model_name, create_func in models_to_test:
            print(f"\n📋 2.{models_to_test.index((model_name, create_func))+1}: Testing {model_name}...")
            
            # Создание тренера для модели
            trainer = create_func(
                cube_dimensions=(15, 15, 11),
                adapter_strategy="learned_linear",
                device="cpu"
            )
            
            info = trainer.get_info()
            
            # Тестовый forward pass
            input_dim = info['adapter']['input_dim']
            test_input = torch.randn(2, input_dim)
            
            start_time = time.time()
            output = trainer.forward(test_input)
            forward_time = time.time() - start_time
            
            results[model_name] = {
                "input_dim": input_dim,
                "output_dim": info['adapter']['output_dim'],
                "compression_ratio": info['adapter']['compression_ratio'],
                "parameters": info['total_parameters'],
                "forward_time_ms": forward_time * 1000
            }
            
            print(f"   ✅ {model_name}:")
            print(f"      Input: {input_dim}D")
            print(f"      Compression: {results[model_name]['compression_ratio']:.3f}")
            print(f"      Parameters: {results[model_name]['parameters']:,}")
            print(f"      Forward time: {results[model_name]['forward_time_ms']:.2f}ms")
        
        print("\n📊 MODEL COMPARISON:")
        print("-" * 50)
        for model_name, stats in results.items():
            print(f"{model_name:15} | {stats['input_dim']:4}D → {stats['output_dim']:3}D | "
                  f"{stats['compression_ratio']:5.3f} | {stats['parameters']:7,} params")
        
        print("\n🎯 ТЕСТ 2 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_joint_training():
    """
    Тест 3: Joint training адаптера и куба
    """
    print("\n🧪 ТЕСТ 3: Joint Training Functionality")
    print("=" * 50)
    
    try:
        print("📋 3.1: Setting up joint training...")
        
        # Создание тренера с joint training
        config = AdapterIntegrationConfig(
            teacher_model="DistilBERT",  # Используем меньшую модель для быстрого тестирования
            cube_dimensions=(15, 15, 11),
            adapter_strategy="learned_linear",
            joint_training=True,
            use_reconstruction_loss=True,
            reconstruction_weight=0.1
        )
        
        trainer = AdapterCubeTrainer(config=config, device="cpu")
        trainer.initialize_components()
        
        print(f"   ✅ Joint trainer initialized")
        print(f"   🔧 Joint training: {trainer.config.joint_training}")
        print(f"   📊 Reconstruction loss: {trainer.config.use_reconstruction_loss}")
        
        print("\n📋 3.2: Testing training step...")
        
        # Создание тестовых данных (Q&A пары)
        batch_size = 2
        embedding_dim = 768  # DistilBERT
        
        question_embeddings = torch.randn(batch_size, embedding_dim)
        answer_embeddings = torch.randn(batch_size, embedding_dim)
        
        # Один шаг обучения
        metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        print(f"   ✅ Training step completed:")
        print(f"      Total loss: {metrics['total_loss']:.4f}")
        print(f"      Main loss: {metrics['main_loss']:.4f}")
        print(f"      Reconstruction loss: {metrics['reconstruction_loss']:.4f}")
        print(f"      Q→A similarity: {metrics['qa_similarity']:.4f}")
        
        print("\n📋 3.3: Testing loss computation...")
        
        # Тестирование loss computation
        losses = trainer.compute_loss(question_embeddings, answer_embeddings)
        
        print(f"   ✅ Loss computation:")
        for loss_name, loss_value in losses.items():
            print(f"      {loss_name}: {loss_value.item():.4f}")
        
        print("\n🎯 ТЕСТ 3 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_adapter_strategies():
    """
    Тест 4: Разные стратегии адаптера
    """
    print("\n🧪 ТЕСТ 4: Different Adapter Strategies")
    print("=" * 50)
    
    strategies = ["learned_linear", "hierarchical", "attention_based", "autoencoder"]
    results = {}
    
    try:
        for strategy in strategies:
            print(f"\n📋 4.{strategies.index(strategy)+1}: Testing {strategy} strategy...")
            
            # Создание тренера с данной стратегией
            config = AdapterIntegrationConfig(
                teacher_model="DistilBERT",
                cube_dimensions=(10, 10, 8),  # Меньший куб для быстрого тестирования
                adapter_strategy=strategy,
                joint_training=True
            )
            
            trainer = AdapterCubeTrainer(config=config, device="cpu")
            trainer.initialize_components()
            
            info = trainer.get_info()
            
            # Тестовый forward pass
            test_input = torch.randn(2, 768)
            
            start_time = time.time()
            output = trainer.forward(test_input)
            forward_time = time.time() - start_time
            
            results[strategy] = {
                "compression_ratio": info['adapter']['compression_ratio'],
                "parameters": info['adapter']['parameters'],
                "forward_time_ms": forward_time * 1000,
                "output_shape": output.shape
            }
            
            print(f"   ✅ Strategy: {strategy}")
            print(f"      Compression: {results[strategy]['compression_ratio']:.3f}")
            print(f"      Parameters: {results[strategy]['parameters']:,}")
            print(f"      Forward time: {results[strategy]['forward_time_ms']:.2f}ms")
        
        print("\n📊 STRATEGY COMPARISON:")
        print("-" * 70)
        print(f"{'Strategy':<15} | {'Compression':<11} | {'Parameters':<10} | {'Time (ms)':<9}")
        print("-" * 70)
        for strategy, stats in results.items():
            print(f"{strategy:<15} | {stats['compression_ratio']:>10.3f} | "
                  f"{stats['parameters']:>9,} | {stats['forward_time_ms']:>8.2f}")
        
        print("\n🎯 ТЕСТ 4 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_surface_strategies():
    """
    Тест 5: Разные стратегии surface (single/triple/full)
    """
    print("\n🧪 ТЕСТ 5: Different Surface Strategies")
    print("=" * 50)
    
    surface_strategies = ["single", "triple", "full"]
    results = {}
    
    try:
        for surface_strategy in surface_strategies:
            print(f"\n📋 5.{surface_strategies.index(surface_strategy)+1}: Testing {surface_strategy} surface...")
            
            config = AdapterIntegrationConfig(
                teacher_model="DistilBERT",
                cube_dimensions=(10, 10, 8),
                surface_strategy=surface_strategy,
                adapter_strategy="learned_linear",
                joint_training=True
            )
            
            trainer = AdapterCubeTrainer(config=config, device="cpu")
            trainer.initialize_components()
            
            info = trainer.get_info()
            
            # Тестовый forward pass
            test_input = torch.randn(2, 768)
            output = trainer.forward(test_input)
            
            surface_size = info['adapter']['output_dim']
            
            results[surface_strategy] = {
                "surface_size": surface_size,
                "compression_ratio": info['adapter']['compression_ratio'],
                "output_shape": output.shape
            }
            
            print(f"   ✅ Surface strategy: {surface_strategy}")
            print(f"      Surface size: {surface_size}")
            print(f"      Compression: {results[surface_strategy]['compression_ratio']:.3f}")
            print(f"      Output shape: {output.shape}")
        
        print("\n📊 SURFACE STRATEGY COMPARISON:")
        print("-" * 60)
        for strategy, stats in results.items():
            coverage = ""
            if strategy == "single":
                coverage = "1 face"
            elif strategy == "triple": 
                coverage = "3 faces"
            elif strategy == "full":
                coverage = "6 faces"
            
            print(f"{strategy:<10} | {coverage:<8} | {stats['surface_size']:>4} elements | "
                  f"{stats['compression_ratio']:>6.3f} compression")
        
        print("\n🎯 ТЕСТ 5 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 5 FAILED: {e}")
        traceback.print_exc()
        return False


def test_configuration_flexibility():
    """
    Тест 6: Гибкость конфигурации
    """
    print("\n🧪 ТЕСТ 6: Configuration Flexibility")
    print("=" * 50)
    
    try:
        print("📋 6.1: Testing config from dict...")
        
        # Конфигурация из словаря
        config_dict = {
            "teacher_model": "Meta-Llama-3-8B",
            "cube_dimensions": (12, 12, 10),
            "surface_strategy": "triple",
            "adapter_strategy": "hierarchical",
            "joint_training": False,
            "use_reconstruction_loss": True,
            "reconstruction_weight": 0.2
        }
        
        trainer = AdapterCubeTrainer(config=config_dict, device="cpu")
        trainer.initialize_components()
        
        info = trainer.get_info()
        
        print(f"   ✅ Config from dict:")
        print(f"      Teacher: {info['teacher_model']}")
        print(f"      Cube: {info['cube_dimensions']}")
        print(f"      Surface: {info['surface_strategy']}")
        print(f"      Joint training: {info['joint_training']}")
        
        print("\n📋 6.2: Testing forward pass...")
        
        # Тестовый forward pass
        test_input = torch.randn(1, 4096)  # LLaMA-3-8B
        output = trainer.forward(test_input)
        
        print(f"   ✅ Forward pass: {test_input.shape} → {output.shape}")
        
        print("\n📋 6.3: Testing training step...")
        
        # Тестовый training step
        question_emb = torch.randn(2, 4096)
        answer_emb = torch.randn(2, 4096)
        
        metrics = trainer.train_step(question_emb, answer_emb)
        
        print(f"   ✅ Training step:")
        print(f"      Total loss: {metrics['total_loss']:.4f}")
        print(f"      Q→A similarity: {metrics['qa_similarity']:.4f}")
        
        print("\n🎯 ТЕСТ 6 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 6 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """
    Основная функция тестирования
    """
    print("🚀 ADAPTER INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Testing integration of universal adapter with CubeTrainer")
    print("Goal: Seamless adapter+cube training pipeline\n")
    
    tests = [
        ("Basic Adapter-Cube Integration", test_basic_integration),
        ("Different Teacher Models", test_different_models),
        ("Joint Training Functionality", test_joint_training),
        ("Different Adapter Strategies", test_adapter_strategies),
        ("Different Surface Strategies", test_surface_strategies),
        ("Configuration Flexibility", test_configuration_flexibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Итоговые результаты
    print("\n" + "=" * 60)
    print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n📊 ОБЩИЙ РЕЗУЛЬТАТ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Adapter Integration готов к использованию!")
    else:
        print("⚠️  Некоторые тесты не прошли. Проверьте ошибки выше.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 