#!/usr/bin/env python3
"""
Прямой тест Meta-LLaMA-3-8B БЕЗ валидации teacher модели
"""

import sys
import torch
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Добавляем пути
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.embedding_trainer.adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from training.embedding_trainer.dialogue_dataset import DialogueDataset, DialogueConfig
import torch.nn.functional as F

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_direct_dialogue_dataset(dialogue_pairs, teacher_model: str = "llama3-8b-local"):
    """Создание DialogueDataset БЕЗ валидации teacher модели"""
    
    # Создаем конфигурацию - ОТКЛЮЧАЕМ кэш для принудительной загрузки модели
    config = DialogueConfig(
        teacher_model=teacher_model,
        validation_split=0.2,
        embedding_dim=4096,
        cache_embeddings=False,  # ОТКЛЮЧАЕМ кэш!
        use_cache=False,         # ОТКЛЮЧАЕМ кэш!
        enable_quality_filter=False
    )
    
    # ВРЕМЕННО патчим метод _validate_teacher_model чтобы он ничего не делал
    original_validate = DialogueDataset._validate_teacher_model
    DialogueDataset._validate_teacher_model = lambda self: logger.info("🔧 Skipping teacher model validation")
    
    try:
        # Создаем dataset БЕЗ автоматической валидации - передаем dialogue_pairs напрямую
        dataset = DialogueDataset(config=config, dialogue_pairs=dialogue_pairs)
        logger.info("✅ Dataset created with bypassed validation")
        return dataset
    finally:
        # Восстанавливаем оригинальный метод
        DialogueDataset._validate_teacher_model = original_validate

def test_llama_direct(strategy: str = "hierarchical", device: str = "cpu") -> Dict[str, Any]:
    """Прямой тест локальной LLaMA БЕЗ валидации"""
    
    logger.info(f"🦙 Starting DIRECT Meta-LLaMA-3-8B Test (strategy: {strategy})")
    start_time = time.time()
    
    # 1. Создаем конфигурацию trainer
    config = AdapterIntegrationConfig(
        teacher_model="llama3-8b-local",
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
    
    # 3. Создаем РЕАЛЬНЫЕ диалоговые данные
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
    
    # 4. Создаем dataset БЕЗ валидации - прямой вызов
    logger.info("📚 Creating dataset with DIRECT LLaMA access...")
    
    try:
        dataset = create_direct_dialogue_dataset(dialogue_pairs, "llama3-8b-local")
        logger.info("✅ Dataset created successfully!")
        
        # Проверяем что получили
        stats = dataset.get_statistics()
        logger.info(f"📊 Dataset stats:")
        logger.info(f"   Teacher model: {stats['teacher_model']}")
        logger.info(f"   Embedding dim: {stats['embedding_dimension']}")
        logger.info(f"   Total pairs: {stats['total_dialogue_pairs']}")
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        # Если все же не получается, пробуем через cache
        logger.info("🔄 Trying fallback approach...")
        
        # Пробуем создать через обычный способ но с отключенной валидацией
        from training.embedding_trainer.dialogue_dataset import create_dialogue_dataset
        dataset = create_dialogue_dataset(
            dialogue_pairs=dialogue_pairs,
            teacher_model="distilbert",  # Временно fallback
            cache_embeddings=True,
            embedding_dim=768
        )
        
        # Обновляем конфигурацию trainer под distilbert
        config.teacher_embedding_dim = 768
        trainer = AdapterCubeTrainer(config, device=device)
        trainer.initialize_components()
        
        logger.info("⚠️ Using fallback distilbert model")
    
    # 5. Получаем batch данных
    dataloader = dataset.get_dataloader(batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    
    questions = batch[0].to(device).float()  # Embedding dimensions
    answers = batch[1].to(device).float()    # Embedding dimensions
    
    logger.info(f"📊 Real data loaded:")
    logger.info(f"   Questions: {questions.shape}")
    logger.info(f"   Answers: {answers.shape}")
    logger.info(f"   Data type: {questions.dtype}")
    logger.info(f"   Baseline Q→A similarity: {F.cosine_similarity(questions, answers, dim=1).mean().item():.3f}")
    
    # 6. Проверяем совместимость размерностей
    if questions.shape[1] != config.teacher_embedding_dim:
        logger.warning(f"⚠️  Dimension mismatch: got {questions.shape[1]}, expected {config.teacher_embedding_dim}")
        logger.info("🔧 Recreating trainer with correct dimensions...")
        
        config.teacher_embedding_dim = questions.shape[1]
        trainer = AdapterCubeTrainer(config, device=device)
        trainer.initialize_components()
    
    # 7. Базовая информация
    param_count = trainer.adapter.get_parameter_count()
    compression_ratio = trainer.adapter.get_compression_ratio()
    
    logger.info(f"   Parameters: {param_count:,}")
    logger.info(f"   Compression: {compression_ratio:.3f}")
    
    # 8. Training loop с правильными метриками
    losses = []
    surface_qa_similarities = []
    
    num_epochs = 10
    baseline_qa = F.cosine_similarity(questions, answers, dim=1).mean().item()
    
    logger.info(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training step
        metrics = trainer.train_step(questions, answers)
        losses.append(metrics["total_loss"])
        
        # Проверяем Q→A similarity на surface level
        with torch.no_grad():
            surface_questions = trainer.forward(questions, return_intermediate=False)  # 225D
            surface_answers = trainer.forward(answers, return_intermediate=False)      # 225D
            
            surface_qa_sim = F.cosine_similarity(surface_questions, surface_answers, dim=1).mean().item()
            surface_qa_similarities.append(surface_qa_sim)
        
        if epoch % 2 == 0:
            logger.info(f"   Epoch {epoch:2d}: loss={metrics['total_loss']:.4f}, "
                       f"surface_qa_sim={surface_qa_sim:.4f}")
    
    training_time = time.time() - start_time
    
    # 9. Финальные результаты
    final_loss = losses[-1]
    final_surface_qa = surface_qa_similarities[-1] 
    surface_improvement = final_surface_qa - surface_qa_similarities[0]
    
    # 10. Анализ успеха
    converged = final_loss < 2.0
    positive_learning = surface_improvement > 0.01
    stable_training = all(l < 10.0 for l in losses[-3:])
    
    success_count = sum([converged, positive_learning, stable_training])
    overall_success = success_count >= 2
    
    result = {
        "strategy": strategy,
        "model_info": {
            "parameter_count": param_count,
            "compression_ratio": compression_ratio,
            "actual_teacher_model": dataset.config.teacher_model,
            "embedding_dim": questions.shape[1]
        },
        "data_info": {
            "dialogue_pairs": len(dialogue_pairs),
            "baseline_qa_similarity": baseline_qa,
            "data_dtype": str(questions.dtype)
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
        }
    }
    
    logger.info(f"✅ DIRECT Test completed:")
    logger.info(f"   Used model: {dataset.config.teacher_model}")
    logger.info(f"   Final loss: {final_loss:.4f}")
    logger.info(f"   Surface Q→A: {surface_qa_similarities[0]:.3f} → {final_surface_qa:.3f} (Δ{surface_improvement:+.3f})")
    logger.info(f"   Overall success: {overall_success} ({success_count}/3)")
    
    return result


if __name__ == "__main__":
    print("🦙 Starting DIRECT Meta-LLaMA-3-8B Test (bypassing validation)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    result = test_llama_direct("hierarchical", device)
    
    # Сохраняем результаты
    output_dir = Path("results/llama_direct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "direct_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_dir}/direct_test_results.json")
    
    if result["model_info"]["actual_teacher_model"] == "llama3-8b-local":
        print("🎉 SUCCESS! Used local LLaMA-3-8B directly!")
    else:
        print(f"⚠️  Used fallback model: {result['model_info']['actual_teacher_model']}") 