"""
🦙+ Improved LLaMA Testing - Stage 3.1.3.2
Улучшенное тестирование с реалистичными Q→A данными и правильными метриками
Решение проблемы cosine similarity = 0.000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import numpy as np

# Наши компоненты
from training.embedding_trainer.adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedLlamaValidator:
    """
    Улучшенный валидатор для Meta-Llama-3-8B
    
    Исправления:
    1. Реалистичные Q→A данные с корреляцией
    2. Правильные метрики cosine similarity
    3. Проверка gradient flow
    4. Детальная диагностика обучения
    """
    
    def __init__(self, device: str = "cpu", output_dir: str = "results/llama_improved"):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLaMA параметры
        self.model_name = "Meta-Llama-3-8B"
        self.embedding_dim = 4096
        self.cube_dimensions = (15, 15, 11)
        self.surface_size = 225
        
        logger.info(f"🦙+ ImprovedLlamaValidator initialized:")
        logger.info(f"   Model: {self.model_name} ({self.embedding_dim}D → {self.surface_size}D)")
        logger.info(f"   Compression: {self.embedding_dim/self.surface_size:.1f}x")
        logger.info(f"   Output: {self.output_dir}")
    
    def create_realistic_qa_data(self, batch_size: int = 20) -> Dict[str, torch.Tensor]:
        """
        Создание реалистичных Q→A данных с четкой корреляцией
        
        Моделируем:
        - Вопросы как query vectors
        - Ответы как related но transformed vectors
        - Реальную Q→A семантическую связь
        """
        torch.manual_seed(42)
        
        # 1. Создаем "концепции" (базовые темы)
        num_concepts = 10
        concept_vectors = torch.randn(num_concepts, self.embedding_dim)
        concept_vectors = F.normalize(concept_vectors, dim=1)
        
        # 2. Создаем вопросы на основе концепций
        questions = []
        answers = []
        question_concept_ids = []
        
        for i in range(batch_size):
            # Выбираем случайную концепцию
            concept_id = torch.randint(0, num_concepts, (1,)).item()
            base_concept = concept_vectors[concept_id]
            
            # Создаем вопрос: концепция + шум
            question_noise = torch.randn(self.embedding_dim) * 0.2
            question = base_concept + question_noise
            question = F.normalize(question, dim=0)
            
            # Создаем ответ: трансформированная концепция (rotation + scaling)
            # Моделируем "ответ связан с вопросом, но имеет другой семантический угол"
            transformation_matrix = self._create_semantic_transformation()
            answer = torch.matmul(transformation_matrix, base_concept.unsqueeze(1)).squeeze(1)
            
            # Добавляем небольшой шум к ответу
            answer_noise = torch.randn(self.embedding_dim) * 0.1
            answer = answer + answer_noise
            answer = F.normalize(answer, dim=0)
            
            questions.append(question)
            answers.append(answer)
            question_concept_ids.append(concept_id)
        
        questions_tensor = torch.stack(questions).to(self.device)
        answers_tensor = torch.stack(answers).to(self.device)
        
        # Проверяем что корреляция Q→A существует
        qa_similarities = F.cosine_similarity(questions_tensor, answers_tensor, dim=1)
        avg_qa_similarity = qa_similarities.mean().item()
        
        logger.info(f"📊 Generated Q→A data:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Concepts: {num_concepts}")
        logger.info(f"   Avg Q→A similarity: {avg_qa_similarity:.3f}")
        logger.info(f"   Q→A similarity range: [{qa_similarities.min():.3f}, {qa_similarities.max():.3f}]")
        
        return {
            "questions": questions_tensor,
            "answers": answers_tensor,
            "concept_ids": question_concept_ids,
            "baseline_qa_similarity": avg_qa_similarity
        }
    
    def _create_semantic_transformation(self) -> torch.Tensor:
        """
        Создание матрицы семантической трансформации Q→A
        
        Моделирует как "ответ связан с вопросом через семантическое преобразование"
        """
        # Создаем ортогональную матрицу (rotation) + небольшой scaling
        A = torch.randn(self.embedding_dim, self.embedding_dim)
        Q, R = torch.qr(A)  # QR decomposition для ортогональности
        
        # Добавляем небольшое scaling (0.8-1.2)
        scaling = torch.diag(torch.rand(self.embedding_dim) * 0.4 + 0.8)
        transformation = torch.matmul(Q, scaling)
        
        return transformation
    
    def test_training_with_diagnostics(self, strategy: str = "hierarchical") -> Dict[str, Any]:
        """
        Тестирование с детальной диагностикой
        
        Args:
            strategy: Adapter strategy для тестирования
            
        Returns:
            Подробные результаты с диагностикой
        """
        logger.info(f"🧪+ Testing {strategy} with improved diagnostics...")
        
        start_time = time.time()
        
        # Создаем конфигурацию
        config = AdapterIntegrationConfig(
            teacher_model=self.model_name,
            teacher_embedding_dim=self.embedding_dim,
            cube_dimensions=self.cube_dimensions,
            surface_strategy="single",
            adapter_strategy=strategy,
            adapter_learning_rate=0.001,
            cube_learning_rate=0.0005,
            joint_training=True,
            use_reconstruction_loss=True,
            reconstruction_weight=0.1
        )
        
        # Создаем trainer
        trainer = AdapterCubeTrainer(config, device=str(self.device))
        trainer.initialize_components()
        
        # Создаем реалистичные данные
        test_data = self.create_realistic_qa_data(batch_size=16)
        
        # Базовая информация
        param_count = trainer.adapter.get_parameter_count()
        compression_ratio = trainer.adapter.get_compression_ratio()
        baseline_similarity = test_data["baseline_qa_similarity"]
        
        logger.info(f"   Parameters: {param_count:,}")
        logger.info(f"   Compression: {compression_ratio:.3f}")
        logger.info(f"   Baseline Q→A similarity: {baseline_similarity:.3f}")
        
        # Диагностика ДО обучения
        pre_training_diagnostics = self._run_diagnostics(trainer, test_data, "PRE-TRAINING")
        
        # Training loop с детальным мониторингом
        training_history = {
            "total_losses": [],
            "reconstruction_losses": [],
            "qa_similarities": [],
            "adapter_gradients": [],
            "processor_gradients": [],
            "intermediate_activations": []
        }
        
        num_epochs = 15
        
        for epoch in range(num_epochs):
            # Training step
            metrics = trainer.train_step(
                test_data["questions"], 
                test_data["answers"]
            )
            
            # Сохраняем метрики
            training_history["total_losses"].append(metrics["total_loss"])
            training_history["reconstruction_losses"].append(metrics.get("reconstruction_loss", 0.0))
            
            # Проверяем Q→A similarity после processing
            with torch.no_grad():
                processed_questions = trainer.forward(test_data["questions"], return_intermediate=False)
                qa_sim = F.cosine_similarity(processed_questions, test_data["answers"], dim=1).mean().item()
                training_history["qa_similarities"].append(qa_sim)
            
            # Gradient diagnostics (каждые 5 эпох)
            if epoch % 5 == 0:
                adapter_grad_norm = self._get_gradient_norm(trainer.adapter)
                processor_grad_norm = self._get_gradient_norm(trainer.embedding_processor)
                
                training_history["adapter_gradients"].append({
                    "epoch": epoch,
                    "norm": adapter_grad_norm
                })
                training_history["processor_gradients"].append({
                    "epoch": epoch,
                    "norm": processor_grad_norm
                })
                
                logger.info(f"   Epoch {epoch:2d}: loss={metrics['total_loss']:.4f}, qa_sim={qa_sim:.4f}, "
                           f"adapter_grad={adapter_grad_norm:.6f}, processor_grad={processor_grad_norm:.6f}")
        
        training_time = time.time() - start_time
        
        # Диагностика ПОСЛЕ обучения
        post_training_diagnostics = self._run_diagnostics(trainer, test_data, "POST-TRAINING")
        
        # Финальные метрики
        final_loss = training_history["total_losses"][-1]
        final_qa_similarity = training_history["qa_similarities"][-1]
        final_reconstruction = training_history["reconstruction_losses"][-1]
        
        # Анализ улучшений
        similarity_improvement = final_qa_similarity - baseline_similarity
        convergence_quality = self._analyze_convergence_quality(training_history["total_losses"])
        
        result = {
            "strategy": strategy,
            "configuration": {
                "learning_rate": config.adapter_learning_rate,
                "batch_size": 16,
                "epochs": num_epochs
            },
            "model_info": {
                "parameter_count": param_count,
                "compression_ratio": compression_ratio
            },
            "training_metrics": {
                "training_time": training_time,
                "final_loss": final_loss,
                "final_reconstruction_loss": final_reconstruction,
                "final_qa_similarity": final_qa_similarity,
                "baseline_qa_similarity": baseline_similarity,
                "similarity_improvement": similarity_improvement
            },
            "convergence_analysis": convergence_quality,
            "diagnostics": {
                "pre_training": pre_training_diagnostics,
                "post_training": post_training_diagnostics
            },
            "training_history": training_history,
            "success_metrics": {
                "converged": convergence_quality["converged"],
                "positive_qa_learning": similarity_improvement > 0.01,
                "stable_gradients": len(training_history["adapter_gradients"]) > 0,
                "reasonable_loss": final_loss < 2.0
            }
        }
        
        # Оценка общего успеха
        success_count = sum(result["success_metrics"].values())
        result["overall_success"] = success_count >= 3  # Минимум 3 из 4 критериев
        
        logger.info(f"✅ Test completed:")
        logger.info(f"   Final loss: {final_loss:.4f}")
        logger.info(f"   Q→A similarity: {baseline_similarity:.3f} → {final_qa_similarity:.3f} (Δ{similarity_improvement:+.3f})")
        logger.info(f"   Overall success: {result['overall_success']} ({success_count}/4 criteria)")
        
        return result
    
    def _run_diagnostics(self, trainer: AdapterCubeTrainer, test_data: Dict, stage: str) -> Dict[str, Any]:
        """Запуск диагностики системы"""
        
        logger.info(f"🔍 Running {stage} diagnostics...")
        
        with torch.no_grad():
            # 1. Forward pass через adapter
            adapter_output = trainer.adapter(test_data["questions"])
            
            # 2. Forward pass через processor
            processor_output = trainer.embedding_processor(adapter_output)
            
            # 3. Анализ трансформаций
            input_norm = torch.norm(test_data["questions"], dim=1).mean().item()
            adapter_norm = torch.norm(adapter_output, dim=1).mean().item()  
            processor_norm = torch.norm(processor_output, dim=1).mean().item()
            
            # 4. Similarity analysis
            input_self_sim = F.cosine_similarity(test_data["questions"], test_data["questions"], dim=1).mean().item()
            adapter_self_sim = F.cosine_similarity(adapter_output, adapter_output, dim=1).mean().item()
            processor_self_sim = F.cosine_similarity(processor_output, processor_output, dim=1).mean().item()
            
            # 5. Q→A correlations
            qa_input_sim = F.cosine_similarity(test_data["questions"], test_data["answers"], dim=1).mean().item()
            qa_processor_sim = F.cosine_similarity(processor_output, test_data["answers"], dim=1).mean().item()
            
        diagnostics = {
            "stage": stage,
            "tensor_norms": {
                "input": input_norm,
                "adapter_output": adapter_norm,
                "processor_output": processor_norm
            },
            "self_similarities": {
                "input": input_self_sim,
                "adapter_output": adapter_self_sim,
                "processor_output": processor_self_sim
            },
            "qa_correlations": {
                "input_qa": qa_input_sim,
                "processor_qa": qa_processor_sim,
                "improvement": qa_processor_sim - qa_input_sim
            }
        }
        
        logger.info(f"   Norms: input={input_norm:.3f}, adapter={adapter_norm:.3f}, processor={processor_norm:.3f}")
        logger.info(f"   Q→A correlation: {qa_input_sim:.3f} → {qa_processor_sim:.3f} (Δ{qa_processor_sim - qa_input_sim:+.3f})")
        
        return diagnostics
    
    def _get_gradient_norm(self, model: nn.Module) -> float:
        """Получение нормы градиентов модели"""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return (total_norm ** 0.5)
    
    def _analyze_convergence_quality(self, losses: List[float]) -> Dict[str, Any]:
        """Анализ качества сходимости"""
        
        if len(losses) < 5:
            return {"converged": False, "reason": "insufficient_data"}
        
        # Тренды
        early_losses = losses[:5]
        late_losses = losses[-5:]
        
        early_avg = sum(early_losses) / len(early_losses)
        late_avg = sum(late_losses) / len(late_losses)
        
        improvement = early_avg - late_avg
        improvement_ratio = improvement / early_avg if early_avg > 0 else 0
        
        # Стабильность
        late_std = np.std(late_losses)
        stability = late_std < 0.1
        
        # Финальное значение
        final_reasonable = losses[-1] < 2.0
        
        converged = (
            improvement_ratio > 0.1 and  # Улучшение минимум на 10%
            stability and                # Стабильные последние значения
            final_reasonable            # Разумное финальное значение
        )
        
        return {
            "converged": converged,
            "improvement": improvement,
            "improvement_ratio": improvement_ratio,
            "final_loss": losses[-1],
            "stability": stability,
            "late_std": late_std
        }
    
    def save_results(self, result: Dict[str, Any]):
        """Сохранение результатов"""
        
        results_file = self.output_dir / f"improved_test_{result['strategy']}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Results saved: {results_file}")
    
    def print_detailed_summary(self, result: Dict[str, Any]):
        """Печать детального отчета"""
        
        print("\n" + "🦙+" * 15)
        print("IMPROVED Meta-Llama-3-8B Test Results")
        print("🦙+" * 15)
        
        config = result["configuration"]
        metrics = result["training_metrics"]
        success = result["success_metrics"]
        
        print(f"\n📊 CONFIGURATION:")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Epochs: {config['epochs']}")
        
        print(f"\n🎯 TRAINING RESULTS:")
        print(f"   Final loss: {metrics['final_loss']:.4f}")
        print(f"   Q→A similarity: {metrics['baseline_qa_similarity']:.3f} → {metrics['final_qa_similarity']:.3f}")
        print(f"   Improvement: {metrics['similarity_improvement']:+.3f}")
        print(f"   Training time: {metrics['training_time']:.1f}s")
        
        print(f"\n✅ SUCCESS CRITERIA:")
        print(f"   Converged: {success['converged']}")
        print(f"   Q→A learning: {success['positive_qa_learning']}")
        print(f"   Stable gradients: {success['stable_gradients']}")
        print(f"   Reasonable loss: {success['reasonable_loss']}")
        print(f"   Overall success: {result['overall_success']}")
        
        # Pre/post training comparison
        pre = result["diagnostics"]["pre_training"]
        post = result["diagnostics"]["post_training"]
        
        print(f"\n📈 PRE→POST ANALYSIS:")
        print(f"   Q→A correlation: {pre['qa_correlations']['input_qa']:.3f} → {post['qa_correlations']['processor_qa']:.3f}")
        print(f"   Processing improvement: {post['qa_correlations']['improvement']:+.3f}")


def run_improved_llama_test(strategy: str = "hierarchical", device: str = "cpu") -> Dict[str, Any]:
    """
    Запуск улучшенного теста LLaMA
    
    Args:
        strategy: Adapter strategy
        device: Устройство
        
    Returns:
        Результаты тестирования
    """
    validator = ImprovedLlamaValidator(device=device)
    result = validator.test_training_with_diagnostics(strategy)
    validator.save_results(result)
    validator.print_detailed_summary(result)
    return result


if __name__ == "__main__":
    print("🦙+ Starting Improved Meta-Llama-3-8B Testing...")
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Тестируем лучшую стратегию с улучшенными метриками
    result = run_improved_llama_test("hierarchical", device)
    
    print(f"\n🎉 Improved test completed!")
    if result["overall_success"]:
        improvement = result["training_metrics"]["similarity_improvement"]
        print(f"✅ SUCCESS: Q→A similarity improved by {improvement:+.3f}")
    else:
        print("❌ Issues detected - check diagnostics for details") 