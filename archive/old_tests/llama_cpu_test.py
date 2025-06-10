#!/usr/bin/env python3
"""
CPU версия теста Meta-LLaMA-3-8B для обхода проблем с CUDA совместимостью
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

def create_cpu_dialogue_dataset(dialogue_pairs, teacher_model: str = "llama3-8b-local"):
    """Создание DialogueDataset для CPU с обходом валидации"""
    
    # Создаем конфигурацию - ПРИНУДИТЕЛЬНО CPU режим для LLaMA
    config = DialogueConfig(
        teacher_model=teacher_model,
        validation_split=0.2,
        embedding_dim=4096,
        cache_embeddings=True,
        enable_quality_filter=False
    )
    
    # Создаем dataset
    dataset = DialogueDataset(config=config)
    
    # Модифицируем embedding_loader чтобы использовать CPU для LLaMA
    original_device = dataset.embedding_loader._device if hasattr(dataset.embedding_loader, '_device') else "auto"
    
    # Временно переключаем на CPU
    if hasattr(dataset.embedding_loader, '_device'):
        dataset.embedding_loader._device = "cpu"
    
    try:
        # Загружаем данные
        dataset._load_from_dialogue_pairs(dialogue_pairs)
        dataset._create_train_val_split()
        logger.info("[OK] Dataset created with CPU LLaMA")
        
    except Exception as e:
        logger.error(f"[ERROR] CPU LLaMA failed: {e}")
        # Fallback на distilbert
        config.teacher_model = "distilbert"
        config.embedding_dim = 768
        dataset = DialogueDataset(config=config)
        dataset._load_from_dialogue_pairs(dialogue_pairs)
        dataset._create_train_val_split()
        logger.info("[WARNING] Fallback to distilbert")
    
    finally:
        # Восстанавливаем device
        if hasattr(dataset.embedding_loader, '_device'):
            dataset.embedding_loader._device = original_device
    
    return dataset

def test_llama_cpu(strategy: str = "hierarchical") -> Dict[str, Any]:
    """CPU тест локальной LLaMA"""
    
    logger.info(f"🦙 Starting CPU Meta-LLaMA-3-8B Test (strategy: {strategy})")
    start_time = time.time()
    
    # ПРИНУДИТЕЛЬНО используем CPU для всего
    device = "cpu"
    torch.cuda.set_device(-1) if torch.cuda.is_available() else None
    
    # 1. Создаем конфигурацию trainer для CPU
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
    
    # 2. Создаем trainer на CPU
    trainer = AdapterCubeTrainer(config, device=device)
    trainer.initialize_components()
    
    # 3. Создаем диалоговые данные
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
         "answer": "NLP combines computational linguistics with statistical machine learning and deep learning models."}
    ]
    
    # 4. Создаем dataset с CPU LLaMA
    logger.info("[BOOKS] Creating dataset with CPU LLaMA access...")
    
    dataset = create_cpu_dialogue_dataset(dialogue_pairs, "llama3-8b-local")
    
    # Проверяем результат
    stats = dataset.get_statistics()
    logger.info(f"[DATA] Dataset stats:")
    logger.info(f"   Teacher model: {stats['teacher_model']}")
    logger.info(f"   Embedding dim: {stats['embedding_dimension']}")
    logger.info(f"   Total pairs: {stats['total_dialogue_pairs']}")
    
    # Адаптируем trainer под фактическую размерность
    actual_dim = stats['embedding_dimension']
    if actual_dim != config.teacher_embedding_dim:
        logger.info(f"[CONFIG] Adapting trainer to {actual_dim}D embeddings")
        config.teacher_embedding_dim = actual_dim
        trainer = AdapterCubeTrainer(config, device=device)
        trainer.initialize_components()
    
    # 5. Получаем batch данных
    dataloader = dataset.get_dataloader(batch_size=6, shuffle=True)  # Меньший batch для CPU
    batch = next(iter(dataloader))
    
    questions = batch[0].to(device).float()
    answers = batch[1].to(device).float()
    
    logger.info(f"[DATA] Real data loaded:")
    logger.info(f"   Questions: {questions.shape}")
    logger.info(f"   Answers: {answers.shape}")
    logger.info(f"   Device: {questions.device}")
    logger.info(f"   Data type: {questions.dtype}")
    logger.info(f"   Baseline Q→A similarity: {F.cosine_similarity(questions, answers, dim=1).mean().item():.3f}")
    
    # 6. Информация о модели
    param_count = trainer.adapter.get_parameter_count()
    compression_ratio = trainer.adapter.get_compression_ratio()
    
    logger.info(f"   Parameters: {param_count:,}")
    logger.info(f"   Compression: {compression_ratio:.3f}")
    
    # 7. Короткий training loop для CPU
    losses = []
    surface_qa_similarities = []
    
    num_epochs = 5  # Меньше эпох для CPU
    baseline_qa = F.cosine_similarity(questions, answers, dim=1).mean().item()
    
    logger.info(f"[START] Starting CPU training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training step
        metrics = trainer.train_step(questions, answers)
        losses.append(metrics["total_loss"])
        
        # Surface similarity
        with torch.no_grad():
            surface_questions = trainer.forward(questions, return_intermediate=False)
            surface_answers = trainer.forward(answers, return_intermediate=False)
            
            surface_qa_sim = F.cosine_similarity(surface_questions, surface_answers, dim=1).mean().item()
            surface_qa_similarities.append(surface_qa_sim)
        
        logger.info(f"   Epoch {epoch:2d}: loss={metrics['total_loss']:.4f}, "
                   f"surface_qa_sim={surface_qa_sim:.4f}")
    
    training_time = time.time() - start_time
    
    # 8. Финальные результаты
    final_loss = losses[-1]
    final_surface_qa = surface_qa_similarities[-1] 
    surface_improvement = final_surface_qa - surface_qa_similarities[0]
    
    # 9. Анализ успеха
    converged = final_loss < 2.0
    positive_learning = surface_improvement > 0.01
    stable_training = all(l < 10.0 for l in losses[-3:])
    
    success_count = sum([converged, positive_learning, stable_training])
    overall_success = success_count >= 2
    
    result = {
        "strategy": strategy,
        "device": device,
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
    
    logger.info(f"[OK] CPU Test completed:")
    logger.info(f"   Used model: {dataset.config.teacher_model}")
    logger.info(f"   Final loss: {final_loss:.4f}")
    logger.info(f"   Surface Q→A: {surface_qa_similarities[0]:.3f} → {final_surface_qa:.3f} (Δ{surface_improvement:+.3f})")
    logger.info(f"   Overall success: {overall_success} ({success_count}/3)")
    
    return result


if __name__ == "__main__":
    print("🦙 Starting CPU Meta-LLaMA-3-8B Test...")
    print("[WARNING]  Using CPU due to CUDA compatibility issues")
    
    result = test_llama_cpu("hierarchical")
    
    # Сохраняем результаты
    output_dir = Path("results/llama_cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "cpu_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[FOLDER] Results saved to: {output_dir}/cpu_test_results.json")
    
    if result["model_info"]["actual_teacher_model"] == "llama3-8b-local":
        print("[SUCCESS] SUCCESS! Used local LLaMA-3-8B on CPU!")
    else:
        print(f"[WARNING]  Used fallback model: {result['model_info']['actual_teacher_model']}")
    
    print("\n[IDEA] To fix CUDA issue, reinstall PyTorch with proper RTX 5090 support:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128") 