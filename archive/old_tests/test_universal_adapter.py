#!/usr/bin/env python3
"""
🧪 UNIVERSAL ADAPTER TEST SUITE
Тестирование универсального адаптера эмбедингов для разных моделей и размеров куба
"""

import torch
import torch.nn.functional as F
import traceback
import time
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорт нашего адаптера
try:
    from data.embedding_adapter.universal_adapter import (
        UniversalEmbeddingAdapter, 
        AdapterManager, 
        KNOWN_MODELS,
        create_adapter_for_cube
    )
except ImportError as e:
    logger.error(f"❌ Import failed: {e}")
    exit(1)


def test_basic_adapter_functionality():
    """
    Тест 1: Базовая функциональность адаптера
    """
    print("\n🧪 ТЕСТ 1: Basic Adapter Functionality")
    print("=" * 50)
    
    try:
        print("📋 1.1: Creating Meta-Llama-3-8B → 15×15 adapter...")
        
        # Создание адаптера для LLaMA 3
        adapter = UniversalEmbeddingAdapter(
            input_dim=4096,  # LLaMA-3-8B
            output_dim=225,  # 15×15 surface
            strategy="learned_linear"
        )
        
        print(f"   ✅ Adapter created: {adapter.input_dim}D → {adapter.output_dim}D")
        print(f"   📊 Compression ratio: {adapter.get_compression_ratio():.3f}")
        print(f"   🔧 Parameters: {adapter.get_parameter_count():,}")
        
        print("\n📋 1.2: Testing forward pass...")
        
        # Тестовый forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, 4096)
        
        # Compression
        compressed = adapter(input_tensor)
        print(f"   ✅ Input shape: {input_tensor.shape} → Output shape: {compressed.shape}")
        
        # With reconstruction
        compressed_rec, reconstructed = adapter(input_tensor, return_reconstruction=True)
        print(f"   ✅ Reconstruction shape: {reconstructed.shape}")
        
        # Reconstruction quality
        reconstruction_loss = adapter.compute_reconstruction_loss(input_tensor, reconstructed)
        print(f"   📊 Reconstruction loss: {reconstruction_loss.item():.4f}")
        
        print("\n📋 1.3: Testing single sample processing...")
        
        # Single sample
        single_input = torch.randn(4096)
        single_output = adapter(single_input)
        print(f"   ✅ Single sample: {single_input.shape} → {single_output.shape}")
        
        print("\n🎯 ТЕСТ 1 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_different_strategies():
    """
    Тест 2: Разные стратегии конвертации
    """
    print("\n🧪 ТЕСТ 2: Different Conversion Strategies")
    print("=" * 50)
    
    strategies = ["learned_linear", "hierarchical", "attention_based", "autoencoder"]
    results = {}
    
    try:
        for strategy in strategies:
            print(f"\n📋 2.{strategies.index(strategy)+1}: Testing {strategy} strategy...")
            
            adapter = UniversalEmbeddingAdapter(
                input_dim=768,   # DistilBERT для быстрого тестирования
                output_dim=225,  # 15×15 surface
                strategy=strategy
            )
            
            # Forward pass test
            test_input = torch.randn(2, 768)
            start_time = time.time()
            output = adapter(test_input)
            forward_time = time.time() - start_time
            
            # Reconstruction test
            _, reconstructed = adapter(test_input, return_reconstruction=True)
            reconstruction_loss = adapter.compute_reconstruction_loss(test_input, reconstructed)
            
            results[strategy] = {
                "parameters": adapter.get_parameter_count(),
                "forward_time_ms": forward_time * 1000,
                "reconstruction_loss": reconstruction_loss.item(),
                "compression_ratio": adapter.get_compression_ratio()
            }
            
            print(f"   ✅ Strategy: {strategy}")
            print(f"   📊 Parameters: {results[strategy]['parameters']:,}")
            print(f"   ⚡ Forward time: {results[strategy]['forward_time_ms']:.2f}ms")
            print(f"   🔧 Reconstruction loss: {results[strategy]['reconstruction_loss']:.4f}")
        
        print("\n📊 STRATEGY COMPARISON:")
        print("-" * 50)
        for strategy, metrics in results.items():
            print(f"{strategy:15s}: {metrics['parameters']:8,} params, "
                  f"{metrics['reconstruction_loss']:.4f} loss, "
                  f"{metrics['forward_time_ms']:5.1f}ms")
        
        print("\n🎯 ТЕСТ 2 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_multiple_models():
    """
    Тест 3: Разные модели и размеры
    """
    print("\n🧪 ТЕСТ 3: Multiple Models and Sizes")
    print("=" * 50)
    
    test_configs = [
        {"model": "DistilBERT", "cube": (15, 15, 11)},
        {"model": "Meta-Llama-3-8B", "cube": (15, 15, 11)},
        {"model": "BERT-large", "cube": (20, 20, 15)},
        {"model": "RoBERTa-base", "cube": (12, 12, 8)},
    ]
    
    try:
        for i, config in enumerate(test_configs):
            print(f"\n📋 3.{i+1}: Testing {config['model']} → {config['cube']} cube...")
            
            # Создание адаптера через helper function
            adapter = create_adapter_for_cube(
                cube_dimensions=config["cube"],
                teacher_model=config["model"],
                strategy="learned_linear"
            )
            
            # Получение размеров
            model_dim = KNOWN_MODELS[config["model"]]["embedding_dim"]
            surface_size = config["cube"][0] * config["cube"][1]
            
            print(f"   📏 Model dimension: {model_dim}D")
            print(f"   📏 Surface size: {surface_size}D")
            print(f"   📊 Compression ratio: {adapter.get_compression_ratio():.3f}")
            print(f"   🔧 Parameters: {adapter.get_parameter_count():,}")
            
            # Test forward pass
            test_input = torch.randn(model_dim)
            output = adapter(test_input)
            
            assert output.shape == (surface_size,), f"Wrong output shape: {output.shape} vs {(surface_size,)}"
            print(f"   ✅ Forward pass successful: {model_dim}D → {surface_size}D")
        
        print("\n🎯 ТЕСТ 3 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_adapter_manager():
    """
    Тест 4: AdapterManager функциональность
    """
    print("\n🧪 ТЕСТ 4: AdapterManager Functionality")
    print("=" * 50)
    
    try:
        print("📋 4.1: Creating AdapterManager...")
        
        manager = AdapterManager(config_dir="test_configs/adapters/")
        
        # Регистрация модели
        manager.register_model("Test-Model", embedding_dim=512)
        
        print("\n📋 4.2: Getting adapters for different configurations...")
        
        # Создание адаптеров
        adapter1 = manager.get_adapter("Test-Model", 225, "learned_linear")
        adapter2 = manager.get_adapter("Test-Model", 400, "hierarchical")
        
        print(f"   ✅ Adapter 1: Test-Model → 225D ({adapter1.strategy})")
        print(f"   ✅ Adapter 2: Test-Model → 400D ({adapter2.strategy})")
        
        print("\n📋 4.3: Listing all adapters...")
        
        adapters_list = manager.list_adapters()
        for key, info in adapters_list.items():
            print(f"   📝 {key}: {info['input_dim']}D → {info['output_dim']}D, "
                  f"{info['parameters']:,} params")
        
        print("\n📋 4.4: Testing adapter reuse...")
        
        # Повторное получение существующего адаптера
        adapter1_again = manager.get_adapter("Test-Model", 225, "learned_linear")
        assert adapter1 is adapter1_again, "Adapter not reused properly"
        print("   ✅ Adapter reuse working correctly")
        
        print("\n🎯 ТЕСТ 4 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_auto_initialization():
    """
    Тест 5: Автоматическая инициализация из данных
    """
    print("\n🧪 ТЕСТ 5: Auto-Initialization from Data")
    print("=" * 50)
    
    try:
        print("📋 5.1: Creating adapter without dimensions...")
        
        # Создание адаптера без размеров
        adapter = UniversalEmbeddingAdapter(strategy="learned_linear")
        
        assert not adapter.initialized, "Adapter should not be initialized yet"
        print("   ✅ Adapter created in uninitialized state")
        
        print("\n📋 5.2: Initializing from sample data...")
        
        # Инициализация из данных
        sample_data = torch.randn(1024)  # 1024D sample
        target_size = 225
        
        adapter.initialize_from_data(sample_data, target_size)
        
        assert adapter.initialized, "Adapter should be initialized now"
        assert adapter.input_dim == 1024, f"Wrong input dim: {adapter.input_dim}"
        assert adapter.output_dim == 225, f"Wrong output dim: {adapter.output_dim}"
        
        print(f"   ✅ Auto-initialized: {adapter.input_dim}D → {adapter.output_dim}D")
        
        print("\n📋 5.3: Testing functionality after auto-init...")
        
        # Тестирование после инициализации
        test_input = torch.randn(3, 1024)
        output = adapter(test_input)
        
        assert output.shape == (3, 225), f"Wrong output shape: {output.shape}"
        print(f"   ✅ Forward pass working: {test_input.shape} → {output.shape}")
        
        print("\n🎯 ТЕСТ 5 РЕЗУЛЬТАТ: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ ТЕСТ 5 FAILED: {e}")
        traceback.print_exc()
        return False


def test_config_save_load():
    """
    Тест 6: Сохранение и загрузка конфигурации
    """
    print("\n🧪 ТЕСТ 6: Config Save/Load")
    print("=" * 50)
    
    try:
        print("📋 6.1: Creating and saving adapter config...")
        
        # Создание адаптера
        adapter = UniversalEmbeddingAdapter(
            input_dim=768,
            output_dim=225,
            strategy="hierarchical"
        )
        
        # Сохранение конфигурации
        config_path = Path("test_configs/test_adapter_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.save_config(config_path)
        
        print(f"   ✅ Config saved to: {config_path}")
        
        print("\n📋 6.2: Loading adapter from config...")
        
        # Загрузка из конфигурации
        loaded_adapter = UniversalEmbeddingAdapter.from_config(config_path)
        
        assert loaded_adapter.input_dim == adapter.input_dim
        assert loaded_adapter.output_dim == adapter.output_dim
        assert loaded_adapter.strategy == adapter.strategy
        
        print(f"   ✅ Adapter loaded: {loaded_adapter.input_dim}D → {loaded_adapter.output_dim}D")
        
        print("\n📋 6.3: Testing loaded adapter functionality...")
        
        # Тестирование загруженного адаптера
        test_input = torch.randn(2, 768)
        output = loaded_adapter(test_input)
        
        assert output.shape == (2, 225)
        print(f"   ✅ Loaded adapter working: {test_input.shape} → {output.shape}")
        
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
    print("🚀 UNIVERSAL EMBEDDING ADAPTER TEST SUITE")
    print("=" * 60)
    print("Testing universal adapter for different models and cube sizes")
    print("Goal: Flexible embedding conversion system\n")
    
    tests = [
        ("Basic Adapter Functionality", test_basic_adapter_functionality),
        ("Different Conversion Strategies", test_different_strategies),
        ("Multiple Models and Sizes", test_multiple_models),
        ("AdapterManager Functionality", test_adapter_manager),
        ("Auto-Initialization from Data", test_auto_initialization),
        ("Config Save/Load", test_config_save_load)
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
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Universal Adapter готов к использованию!")
        print("\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Интегрировать с системой обучения")
        print("2. Тестировать на реальных данных LLaMA-3")
        print("3. Оптимизировать reconstruction loss")
        print("4. Экспериментировать с разными стратегиями")
        print("5. Scaling tests с большими моделями")
    else:
        print("⚠️  Некоторые тесты failed. Требуется доработка.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 