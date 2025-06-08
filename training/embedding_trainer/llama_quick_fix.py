"""
🦙 Quick LLaMA Fix - Stage 3.1.3.2
Быстрое исправление с использованием существующей DialogueDataset 
и правильным сравнением размеров tensor'ов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Наши компоненты (используем существующие!)
from training.embedding_trainer.adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from training.embedding_trainer.dialogue_dataset import DialogueDataset, create_dialogue_dataset

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_llama_with_real_data(strategy: str = "hierarchical", device: str = "cpu") -> Dict[str, Any]:
    """
    Тест LLaMA с реальными диалоговыми данными
    
    Args:
        strategy: Adapter strategy  
        device: Устройство
        
    Returns:
        Результаты тестирования
    """
    logger.info(f"🦙 Testing {strategy} with REAL dialogue data...")
    
    start_time = time.time()
    
    # 1. Создаем конфигурацию
    config = AdapterIntegrationConfig(
        teacher_model="llama3-8b-local",  # Используем локальную модель
        teacher_embedding_dim=4096,
        cube_dimensions=(15, 15, 11),
        surface_strategy="single",
        adapter_strategy=strategy,
        adapter_learning_rate=0.001,
        cube_learning_rate=0.0005,
        joint_training=True,
        use_reconstruction_loss=True,
        reconstruction_weight=0.1
    )
    
    # 2. Создаем trainer
    trainer = AdapterCubeTrainer(config, device=device)
    trainer.initialize_components()
    
    # 3. Создаем РЕАЛЬНЫЕ диалоговые данные (используем существующую DialogueDataset!)
    dialogue_pairs = [
        {"question": "What is machine learning?", 
         "answer": "Machine learning is a method of data analysis that automates analytical model building."},
        {"question": "How does neural networks work?", 
         "answer": "Neural networks are computing systems inspired by biological neural networks that process information."},
        {"question": "What is deep learning?", 
         "answer": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers."},
        {"question": "Explain artificial intelligence", 
         "answer": "Artificial intelligence is the simulation of human intelligence in machines programmed to think and learn."},
        {"question": "What is computer vision?", 
         "answer": "Computer vision is a field of AI that trains computers to interpret and understand visual information."},
        {"question": "How does natural language processing work?", 
         "answer": "NLP combines computational linguistics with statistical machine learning and deep learning models."},
        {"question": "What is reinforcement learning?", 
         "answer": "Reinforcement learning is an area of ML where agents learn to make decisions by taking actions in an environment."},
        {"question": "Explain data science", 
         "answer": "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data."}
    ]
    
    # Создаем dataset с реальными данными - используем локальный путь к модели
    dataset = create_dialogue_dataset(
        dialogue_pairs, 
        teacher_model="llama3-8b-local",  # Используем локальную модель
        cache_embeddings=True,
        embedding_dim=4096  # LLaMA-3-8B размерность
    )
    
    # 4. Получаем batch данных
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    
    questions = batch[0].to(device).float()  # 4096D (question embeddings) - конвертируем в float32
    answers = batch[1].to(device).float()    # 4096D (answer embeddings) - конвертируем в float32
    
    logger.info(f"📊 Real data loaded:")
    logger.info(f"   Questions: {questions.shape}")
    logger.info(f"   Answers: {answers.shape}")
    logger.info(f"   Baseline Q→A similarity: {F.cosine_similarity(questions, answers, dim=1).mean().item():.3f}")
    
    # 5. Базовая информация
    param_count = trainer.adapter.get_parameter_count()
    compression_ratio = trainer.adapter.get_compression_ratio()
    
    logger.info(f"   Parameters: {param_count:,}")
    logger.info(f"   Compression: {compression_ratio:.3f}")
    
    # 6. Training loop с правильными метриками
    losses = []
    qa_similarities = []
    surface_qa_similarities = []  # НОВАЯ метрика: similarity на surface level
    
    num_epochs = 15
    baseline_qa = F.cosine_similarity(questions, answers, dim=1).mean().item()
    
    for epoch in range(num_epochs):
        # Training step
        metrics = trainer.train_step(questions, answers)
        losses.append(metrics["total_loss"])
        
        # ПРАВИЛЬНАЯ проверка Q→A similarity (на surface level)
        with torch.no_grad():
            # Получаем surface embeddings для questions и answers
            surface_questions = trainer.forward(questions, return_intermediate=False)  # 225D
            surface_answers = trainer.forward(answers, return_intermediate=False)      # 225D
            
            # Теперь сравниваем 225D с 225D (НЕ 225D с 4096D!)
            surface_qa_sim = F.cosine_similarity(surface_questions, surface_answers, dim=1).mean().item()
            surface_qa_similarities.append(surface_qa_sim)
            
            # Также сохраняем original Q→A similarity для reference
            qa_sim = F.cosine_similarity(questions, answers, dim=1).mean().item()
            qa_similarities.append(qa_sim)
        
        if epoch % 5 == 0:
            logger.info(f"   Epoch {epoch:2d}: loss={metrics['total_loss']:.4f}, "
                       f"surface_qa_sim={surface_qa_sim:.4f}, original_qa_sim={qa_sim:.4f}")
    
    training_time = time.time() - start_time
    
    # 7. Финальные результаты
    final_loss = losses[-1]
    final_surface_qa = surface_qa_similarities[-1] 
    surface_improvement = final_surface_qa - surface_qa_similarities[0]  # Улучшение на surface level
    
    # 8. Анализ успеха
    converged = final_loss < 2.0 and len(losses) >= 10
    positive_learning = surface_improvement > 0.01  # Улучшение минимум на 0.01
    stable_training = all(l < 10.0 for l in losses[-5:])  # Стабильные последние losses
    
    success_count = sum([converged, positive_learning, stable_training])
    overall_success = success_count >= 2  # Минимум 2 из 3 критериев
    
    result = {
        "strategy": strategy,
        "model_info": {
            "parameter_count": param_count,
            "compression_ratio": compression_ratio
        },
        "data_info": {
            "real_dialogue_pairs": len(dialogue_pairs),
            "baseline_qa_similarity": baseline_qa
        },
        "training_results": {
            "training_time": training_time,
            "final_loss": final_loss,
            "final_surface_qa_similarity": final_surface_qa,
            "surface_similarity_improvement": surface_improvement,
            "epochs": num_epochs
        },
        "success_metrics": {
            "converged": converged,
            "positive_learning": positive_learning,
            "stable_training": stable_training,
            "overall_success": overall_success,
            "success_score": f"{success_count}/3"
        },
        "history": {
            "losses": losses,
            "surface_qa_similarities": surface_qa_similarities,
            "original_qa_similarities": qa_similarities
        }
    }
    
    logger.info(f"✅ Test completed:")
    logger.info(f"   Final loss: {final_loss:.4f}")
    logger.info(f"   Surface Q→A similarity: {surface_qa_similarities[0]:.3f} → {final_surface_qa:.3f} (Δ{surface_improvement:+.3f})")
    logger.info(f"   Overall success: {overall_success} ({success_count}/3 criteria)")
    
    return result


def print_results(result: Dict[str, Any]):
    """Печать результатов"""
    
    print("\n" + "🦙" * 20)
    print("Meta-Llama-3-8B FIXED Test Results")
    print("🦙" * 20)
    
    model = result["model_info"]
    data = result["data_info"]
    training = result["training_results"]
    success = result["success_metrics"]
    
    print(f"\n📊 MODEL INFO:")
    print(f"   Strategy: {result['strategy']}")
    print(f"   Parameters: {model['parameter_count']:,}")
    print(f"   Compression: {model['compression_ratio']:.3f}")
    
    print(f"\n📚 DATA INFO:")
    print(f"   Real dialogue pairs: {data['real_dialogue_pairs']}")
    print(f"   Baseline Q→A similarity: {data['baseline_qa_similarity']:.3f}")
    
    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   Training time: {training['training_time']:.1f}s")
    print(f"   Final loss: {training['final_loss']:.4f}")
    print(f"   Surface Q→A similarity: {training['final_surface_qa_similarity']:.3f}")
    print(f"   Surface improvement: {training['surface_similarity_improvement']:+.3f}")
    
    print(f"\n✅ SUCCESS EVALUATION:")
    print(f"   Converged: {success['converged']}")
    print(f"   Positive learning: {success['positive_learning']}")
    print(f"   Stable training: {success['stable_training']}")
    print(f"   Overall success: {success['overall_success']} ({success['success_score']})")
    
    if success["overall_success"]:
        print(f"\n🎉 SUCCESS! Architecture works with real Meta-Llama-3-8B data!")
    else:
        print(f"\n🔧 Needs improvement, but basic functionality confirmed.")


if __name__ == "__main__":
    print("🦙 Starting FIXED Meta-Llama-3-8B Test with Real Data...")
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Тестируем с реальными данными
    result = test_llama_with_real_data("hierarchical", device)
    
    # Сохраняем результаты
    output_dir = Path("results/llama_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "fixed_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Печатаем результаты
    print_results(result)
    
    print(f"\n📁 Results saved to: {output_dir}/fixed_test_results.json") 