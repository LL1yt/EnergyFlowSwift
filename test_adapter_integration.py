#!/usr/bin/env python3
"""
🧪 TEST: AdapterCubeTrainer Integration with EmbeddingProcessor.SURFACE_ONLY
Тестирование обновленной интеграции Universal Adapter + EmbeddingProcessor

Проверяет:
1. Создание AdapterCubeTrainer с новой архитектурой
2. Universal Adapter → EmbeddingProcessor.SURFACE_ONLY pipeline
3. Training workflow (joint & separate training)
4. Gradient flow и backpropagation
5. Performance metrics
6. End-to-end functionality

Автор: 3D Cellular Neural Network Project
Дата: 7 июня 2025
"""

import torch
import numpy as np
import logging
from typing import Dict, Any
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_adapter_cube_trainer_creation():
    """Тест создания обновленного AdapterCubeTrainer"""
    print("🔧 Testing AdapterCubeTrainer creation with EmbeddingProcessor.SURFACE_ONLY...")
    
    try:
        from training.embedding_trainer.adapter_integration import (
            AdapterCubeTrainer, 
            AdapterIntegrationConfig,
            create_llama3_cube_trainer
        )
        
        # Создание конфигурации
        config = AdapterIntegrationConfig(
            teacher_model="Meta-Llama-3-8B",
            cube_dimensions=(15, 15, 11),
            surface_strategy="single",
            adapter_strategy="learned_linear",
            joint_training=True
        )
        
        # Создание тренера
        trainer = AdapterCubeTrainer(config=config, device="cpu")
        trainer.initialize_components()
        
        # Проверка компонентов
        assert trainer.adapter is not None, "UniversalEmbeddingAdapter не создан"
        assert trainer.embedding_processor is not None, "EmbeddingProcessor не создан"
        assert trainer.joint_optimizer is not None, "Joint optimizer не создан"
        
        # Проверка размеров
        expected_surface_size = 15 * 15  # single strategy
        assert trainer.adapter.output_dim == expected_surface_size, f"Неверный adapter output: {trainer.adapter.output_dim}"
        
        # Информация о системе
        info = trainer.get_info()
        print("✅ AdapterCubeTrainer создан успешно:")
        print(f"   Teacher: {info['teacher_model']}")
        print(f"   Surface size: {info['processor']['surface_size']}D")
        print(f"   Processing mode: {info['processor']['mode']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Ошибка создания AdapterCubeTrainer: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_end_to_end_pipeline(trainer):
    """Тест end-to-end pipeline: teacher embeddings → surface processing → output"""
    print("\n🚀 Testing end-to-end pipeline...")
    
    try:
        # Создание тестовых teacher embeddings (LLaMA-3-8B: 4096D)
        batch_size = 4
        teacher_dim = 4096
        teacher_embeddings = torch.randn(batch_size, teacher_dim, dtype=torch.float32)
        
        print(f"📥 Input: {teacher_embeddings.shape} (teacher embeddings)")
        
        # Forward pass
        start_time = time.time()
        
        # Тест обычного forward
        output = trainer.forward(teacher_embeddings)
        
        # Тест с промежуточными результатами
        results = trainer.forward(teacher_embeddings, return_intermediate=True)
        
        processing_time = time.time() - start_time
        
        print(f"📤 Output: {output.shape}")
        print(f"⏱️  Processing time: {processing_time:.4f}s")
        print(f"⚡ Throughput: {batch_size / processing_time:.1f} samples/sec")
        
        # Проверка результатов
        expected_surface_size = 15 * 15  # single surface strategy
        assert output.shape == (batch_size, expected_surface_size), f"Неверная output shape: {output.shape}"
        assert output.requires_grad, "Output должен поддерживать градиенты"
        
        # Проверка промежуточных результатов
        assert "surface_embeddings" in results, "surface_embeddings отсутствуют"
        assert "output" in results, "output отсутствует"
        
        surface_embeddings = results["surface_embeddings"]
        assert surface_embeddings.shape == (batch_size, expected_surface_size), f"Неверная surface shape: {surface_embeddings.shape}"
        
        print("✅ End-to-end pipeline работает корректно")
        print(f"   Teacher ({teacher_dim}D) → Adapter ({expected_surface_size}D) → Processor ({expected_surface_size}D)")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в end-to-end pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_workflows(trainer):
    """Тест различных режимов обучения"""
    print("\n🎯 Testing training workflows...")
    
    try:
        # Подготовка тестовых данных
        batch_size = 2
        teacher_dim = 4096
        
        question_embeddings = torch.randn(batch_size, teacher_dim, dtype=torch.float32)
        answer_embeddings = torch.randn(batch_size, teacher_dim, dtype=torch.float32)
        
        print(f"📥 Training data: questions {question_embeddings.shape}, answers {answer_embeddings.shape}")
        
        # Тест joint training step
        print("\n🔗 Testing joint training step...")
        trainer.config.joint_training = True
        
        joint_metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        assert "total_loss" in joint_metrics, "total_loss отсутствует"
        assert "main_loss" in joint_metrics, "main_loss отсутствует"
        assert "qa_similarity" in joint_metrics, "qa_similarity отсутствует"
        
        print(f"   Loss: {joint_metrics['total_loss']:.6f}")
        print(f"   QA similarity: {joint_metrics['qa_similarity']:.4f}")
        
        # Тест separate training
        print("\n🔀 Testing separate training workflow...")
        trainer.config.joint_training = False
        trainer.current_epoch = 0  # Reset для warmup
        trainer.adapter_warmup_complete = False
        
        # Adapter warmup step
        warmup_metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        assert "phase" in warmup_metrics, "phase информация отсутствует"
        assert warmup_metrics["phase"] == "adapter_warmup", f"Неверная фаза: {warmup_metrics['phase']}"
        
        print(f"   Warmup loss: {warmup_metrics['total_loss']:.6f}")
        
        # Processor training step
        trainer.current_epoch = 5  # Skip warmup
        processor_metrics = trainer.train_step(question_embeddings, answer_embeddings)
        
        assert processor_metrics["phase"] == "processor_training", f"Неверная фаза: {processor_metrics['phase']}"
        
        print(f"   Processor loss: {processor_metrics['total_loss']:.6f}")
        print(f"   Processor QA similarity: {processor_metrics['qa_similarity']:.4f}")
        
        print("✅ Все режимы обучения работают корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в training workflows: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(trainer):
    """Тест gradient flow для training готовности"""
    print("\n🔄 Testing gradient flow...")
    
    try:
        # Включаем anomaly detection для debugging
        torch.autograd.set_detect_anomaly(True)
        
        # Подготовка данных с градиентами
        question_embeddings = torch.randn(2, 4096, dtype=torch.float32, requires_grad=True)
        answer_embeddings = torch.randn(2, 4096, dtype=torch.float32)
        
        # Forward pass
        results = trainer.forward(question_embeddings, return_intermediate=True)
        
        # Loss computation
        target_surface = trainer.adapter(answer_embeddings)
        loss = torch.nn.functional.mse_loss(results["output"], target_surface)
        
        print(f"📊 Test loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Выключаем anomaly detection
        torch.autograd.set_detect_anomaly(False)
        
        # Проверка градиентов в adapter
        adapter_grad_norm = 0.0
        adapter_params_with_grad = 0
        for param in trainer.adapter.parameters():
            if param.grad is not None:
                adapter_grad_norm += param.grad.norm().item() ** 2
                adapter_params_with_grad += 1
        adapter_grad_norm = np.sqrt(adapter_grad_norm)
        
        # Проверка градиентов в embedding_processor
        processor_grad_norm = 0.0
        processor_params_with_grad = 0
        for param in trainer.embedding_processor.parameters():
            if param.grad is not None:
                processor_grad_norm += param.grad.norm().item() ** 2
                processor_params_with_grad += 1
        processor_grad_norm = np.sqrt(processor_grad_norm)
        
        print(f"🔗 Adapter gradients: norm={adapter_grad_norm:.6f}, params={adapter_params_with_grad}")
        print(f"🧠 Processor gradients: norm={processor_grad_norm:.6f}, params={processor_params_with_grad}")
        
        # Проверка что градиенты есть
        assert adapter_grad_norm > 0, "Adapter градиенты отсутствуют"
        assert processor_grad_norm > 0, "EmbeddingProcessor градиенты отсутствуют"
        assert adapter_params_with_grad > 0, "Нет параметров с градиентами в adapter"
        assert processor_params_with_grad > 0, "Нет параметров с градиентами в processor"
        
        print("✅ Gradient flow работает корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в gradient flow: {e}")
        # Выключаем anomaly detection в случае ошибки
        torch.autograd.set_detect_anomaly(False)
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark(trainer):
    """Тест производительности интегрированной системы"""
    print("\n⚡ Performance benchmark...")
    
    try:
        # Подготовка различных размеров batch
        teacher_dim = 4096
        batch_sizes = [1, 4, 8, 16]
        
        print("Batch Size | Forward Time | Throughput")
        print("-" * 40)
        
        for batch_size in batch_sizes:
            teacher_embeddings = torch.randn(batch_size, teacher_dim, dtype=torch.float32)
            
            # Warm-up
            for _ in range(3):
                _ = trainer.forward(teacher_embeddings)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                output = trainer.forward(teacher_embeddings)
            total_time = time.time() - start_time
            
            avg_time_per_batch = total_time / 10
            throughput = batch_size / avg_time_per_batch
            
            print(f"{batch_size:>9} | {avg_time_per_batch:>11.4f}s | {throughput:>9.1f} smp/s")
        
        print("✅ Performance benchmark завершен")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в performance benchmark: {e}")
        return False


def test_convenience_functions():
    """Тест convenience functions для создания тренеров"""
    print("\n🛠️  Testing convenience functions...")
    
    try:
        from training.embedding_trainer.adapter_integration import (
            create_llama3_cube_trainer,
            create_distilbert_cube_trainer
        )
        
        # Тест LLaMA-3 тренера
        llama_trainer = create_llama3_cube_trainer(
            cube_dimensions=(15, 15, 11),
            adapter_strategy="learned_linear",
            device="cpu"
        )
        
        llama_info = llama_trainer.get_info()
        assert llama_info["teacher_model"] == "Meta-Llama-3-8B", "Неверная teacher модель для LLaMA"
        
        print(f"✅ LLaMA-3 trainer: {llama_info['teacher_model']}, {llama_info['total_parameters']:,} params")
        
        # Тест DistilBERT тренера
        bert_trainer = create_distilbert_cube_trainer(
            cube_dimensions=(15, 15, 11),
            adapter_strategy="learned_linear",
            device="cpu"
        )
        
        bert_info = bert_trainer.get_info()
        assert bert_info["teacher_model"] == "DistilBERT", "Неверная teacher модель для BERT"
        
        print(f"✅ DistilBERT trainer: {bert_info['teacher_model']}, {bert_info['total_parameters']:,} params")
        
        print("✅ Convenience functions работают корректно")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в convenience functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Запуск полного набора тестов"""
    print("=" * 80)
    print("🧪 COMPREHENSIVE TEST: AdapterCubeTrainer + EmbeddingProcessor.SURFACE_ONLY")
    print("=" * 80)
    
    test_results = []
    
    # 1. Создание тренера  
    trainer = test_adapter_cube_trainer_creation()
    test_results.append(("Trainer Creation", trainer is not None))
    
    if trainer is not None:
        # 2. End-to-end pipeline
        pipeline_success = test_end_to_end_pipeline(trainer)
        test_results.append(("End-to-End Pipeline", pipeline_success))
        
        # 3. Training workflows
        training_success = test_training_workflows(trainer)
        test_results.append(("Training Workflows", training_success))
        
        # 4. Gradient flow
        gradient_success = test_gradient_flow(trainer)
        test_results.append(("Gradient Flow", gradient_success))
        
        # 5. Performance benchmark
        performance_success = test_performance_benchmark(trainer)
        test_results.append(("Performance Benchmark", performance_success))
    
    # 6. Convenience functions
    convenience_success = test_convenience_functions()
    test_results.append(("Convenience Functions", convenience_success))
    
    # Результаты
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS:")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! AdapterCubeTrainer integration готов к использованию.")
        return True
    else:
        print("⚠️  Some tests failed. Требует исправления.")
        return False


if __name__ == "__main__":
    run_comprehensive_test() 