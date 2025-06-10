"""
🦙 LLaMA Strategy Optimization Test - Stage 3.1.3  
Практическое тестирование различных adapter стратегий для Meta-Llama-3-8B
Определение оптимальной конфигурации для production использования
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Наши компоненты
from training.embedding_trainer.adapter_integration import AdapterCubeTrainer, AdapterIntegrationConfig
from data.embedding_adapter.universal_adapter import UniversalEmbeddingAdapter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LlamaStrategyOptimizer:
    """
    Оптимизатор стратегий для Meta-Llama-3-8B
    
    Тестирует различные adapter strategies и находит оптимальную конфигурацию
    """
    
    def __init__(self, device: str = "cpu", output_dir: str = "results/llama_strategy"):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Meta-Llama-3-8B параметры
        self.model_name = "Meta-Llama-3-8B"
        self.embedding_dim = 4096
        self.cube_dimensions = (15, 15, 11)
        self.surface_size = 225  # 15*15
        
        # Стратегии для тестирования
        self.strategies_to_test = [
            "learned_linear",
            "hierarchical", 
            "attention_based",
            "autoencoder"
        ]
        
        # Параметры для тестирования
        self.learning_rates = [0.001, 0.0005, 0.0001]
        self.batch_sizes = [4, 8]
        self.test_epochs = 10  # Достаточно для оценки convergence
        
        # Результаты
        self.results = []
        
        logger.info(f"🦙 LlamaStrategyOptimizer initialized:")
        logger.info(f"   Model: {self.model_name} ({self.embedding_dim}D → {self.surface_size}D)")
        logger.info(f"   Compression: {self.embedding_dim/self.surface_size:.1f}x")
        logger.info(f"   Strategies: {self.strategies_to_test}")
        logger.info(f"   Output: {self.output_dir}")
    
    def create_test_data(self, batch_size: int = 20) -> Dict[str, torch.Tensor]:
        """Создание тестовых данных для LLaMA embeddings"""
        
        # Создаем realistic embeddings (нормализованные, как в реальных LLM)
        torch.manual_seed(42)  # Для воспроизводимости
        
        questions = torch.randn(batch_size, self.embedding_dim)
        answers = torch.randn(batch_size, self.embedding_dim)
        
        # Нормализуем как в реальных embeddings
        questions = torch.nn.functional.normalize(questions, dim=1)
        answers = torch.nn.functional.normalize(answers, dim=1)
        
        # Добавляем некоторую корреляцию между Q&A (как в реальных диалогах)
        correlation_strength = 0.3
        answers = answers + correlation_strength * questions
        answers = torch.nn.functional.normalize(answers, dim=1)
        
        return {
            "questions": questions.to(self.device),
            "answers": answers.to(self.device)
        }
    
    def test_single_strategy(self, 
                           strategy: str, 
                           learning_rate: float, 
                           batch_size: int) -> Dict[str, Any]:
        """Тестирование одной стратегии с конкретными параметрами"""
        
        logger.info(f"🧪 Testing: {strategy}, lr={learning_rate}, batch={batch_size}")
        
        start_time = time.time()
        
        try:
            # Создаем trainer конфигурацию
            config = AdapterIntegrationConfig(
                teacher_model=self.model_name,
                teacher_embedding_dim=self.embedding_dim,
                cube_dimensions=self.cube_dimensions,
                surface_strategy="single",
                adapter_strategy=strategy,
                adapter_learning_rate=learning_rate,
                cube_learning_rate=learning_rate * 0.5,  # Cube LR меньше
                joint_training=True,
                use_reconstruction_loss=True,
                reconstruction_weight=0.1
            )
            
            # Создаем trainer
            trainer = AdapterCubeTrainer(config, device=str(self.device))
            trainer.initialize_components()
            
            # Создаем тестовые данные
            test_data = self.create_test_data(batch_size)
            
            # Базовая информация
            param_count = trainer.adapter.get_parameter_count()
            compression_ratio = trainer.adapter.get_compression_ratio()
            
            # Training loop
            losses = []
            reconstruction_losses = []
            cosine_similarities = []
            
            for epoch in range(self.test_epochs):
                # Training step
                metrics = trainer.train_step(
                    test_data["questions"][:batch_size], 
                    test_data["answers"][:batch_size]
                )
                
                losses.append(metrics["total_loss"])
                reconstruction_losses.append(metrics.get("reconstruction_loss", 0.0))
                cosine_similarities.append(metrics.get("cosine_similarity", 0.0))
                
                # Early stopping если loss очень плохой
                if metrics["total_loss"] > 10.0:
                    logger.warning(f"   Early stopping: loss too high ({metrics['total_loss']:.3f})")
                    break
            
            training_time = time.time() - start_time
            
            # Финальная оценка
            final_loss = losses[-1] if losses else 999.0
            final_reconstruction = reconstruction_losses[-1] if reconstruction_losses else 999.0
            final_similarity = cosine_similarities[-1] if cosine_similarities else 0.0
            
            # Анализ convergence
            converged = self._analyze_convergence(losses)
            stability_score = self._calculate_stability(losses)
            quality_score = self._calculate_quality_score(final_loss, final_similarity, final_reconstruction)
            
            result = {
                "strategy": strategy,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "parameter_count": param_count,
                "compression_ratio": compression_ratio,
                "training_time": training_time,
                "final_loss": final_loss,
                "final_reconstruction_loss": final_reconstruction,
                "final_cosine_similarity": final_similarity,
                "converged": converged,
                "stability_score": stability_score,
                "quality_score": quality_score,
                "loss_history": losses,
                "reconstruction_history": reconstruction_losses,
                "similarity_history": cosine_similarities
            }
            
            logger.info(f"   [OK] Result: loss={final_loss:.3f}, similarity={final_similarity:.3f}, converged={converged}")
            
            return result
            
        except Exception as e:
            logger.error(f"   [ERROR] Failed: {e}")
            return {
                "strategy": strategy,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "error": str(e),
                "failed": True
            }
    
    def _analyze_convergence(self, losses: List[float]) -> bool:
        """Анализ сходимости training loss"""
        if len(losses) < 5:
            return False
        
        # Проверяем что loss уменьшается и стабилизируется
        recent_losses = losses[-5:]
        
        # Условие 1: loss должен быть < 2.0 (разумное значение)
        if recent_losses[-1] > 2.0:
            return False
        
        # Условие 2: loss должен быть относительно стабильным (изменения < 10%)
        if len(recent_losses) >= 3:
            avg_loss = sum(recent_losses[-3:]) / 3
            if recent_losses[-1] > avg_loss * 1.1:  # loss растет более чем на 10%
                return False
        
        # Условие 3: общий тренд должен быть downward
        if len(losses) >= 5:
            early_avg = sum(losses[:3]) / 3
            late_avg = sum(losses[-3:]) / 3
            if late_avg >= early_avg:  # Нет улучшения
                return False
        
        return True
    
    def _calculate_stability(self, losses: List[float]) -> float:
        """Расчет stability score (0-1, где 1 = очень стабильно)"""
        if len(losses) < 3:
            return 0.0
        
        # Считаем стандартное отклонение последних losses
        recent_losses = losses[-min(5, len(losses)):]
        if len(recent_losses) < 2:
            return 0.0
        
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((x - mean_loss) ** 2 for x in recent_losses) / len(recent_losses)
        std_dev = variance ** 0.5
        
        # Normalize: stability = 1 / (1 + relative_std)
        relative_std = std_dev / max(mean_loss, 0.001)  # Avoid division by zero
        stability = 1.0 / (1.0 + relative_std)
        
        return min(stability, 1.0)
    
    def _calculate_quality_score(self, loss: float, similarity: float, reconstruction: float) -> float:
        """Расчет общего quality score (0-1, где 1 = отличное качество)"""
        
        # Компоненты качества (все нормализованы к 0-1)
        loss_score = max(0, 1.0 - loss / 2.0)  # loss < 2.0 = good
        similarity_score = max(0, similarity)   # cosine similarity уже 0-1
        reconstruction_score = max(0, 1.0 - reconstruction / 1.0)  # reconstruction < 1.0 = good
        
        # Взвешенная сумма
        quality = (
            0.4 * loss_score +
            0.4 * similarity_score +
            0.2 * reconstruction_score
        )
        
        return min(quality, 1.0)
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Быстрое тестирование (только 2 лучшие стратегии)"""
        
        logger.info(f"[START] Starting quick LLaMA strategy test...")
        
        # Только лучшие стратегии для быстрого тестирования
        quick_strategies = ["learned_linear", "hierarchical"]
        quick_lr = [0.001]  # Только один learning rate
        quick_batch = [8]   # Только один batch size
        
        all_results = []
        
        for strategy in quick_strategies:
            for lr in quick_lr:
                for batch_size in quick_batch:
                    result = self.test_single_strategy(strategy, lr, batch_size)
                    all_results.append(result)
        
        # Анализ результатов
        analysis = self._analyze_all_results(all_results)
        
        # Сохранение результатов
        self._save_results(all_results, analysis)
        
        return analysis
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Запуск полного тестирования всех стратегий"""
        
        logger.info(f"[START] Starting comprehensive LLaMA strategy testing...")
        logger.info(f"   Total combinations: {len(self.strategies_to_test) * len(self.learning_rates) * len(self.batch_sizes)}")
        
        all_results = []
        current_test = 0
        total_tests = len(self.strategies_to_test) * len(self.learning_rates) * len(self.batch_sizes)
        
        for strategy in self.strategies_to_test:
            for lr in self.learning_rates:
                for batch_size in self.batch_sizes:
                    current_test += 1
                    logger.info(f"[DATA] Test {current_test}/{total_tests}")
                    
                    result = self.test_single_strategy(strategy, lr, batch_size)
                    all_results.append(result)
                    
                    # Небольшая пауза между тестами
                    time.sleep(1)
        
        # Анализ результатов
        analysis = self._analyze_all_results(all_results)
        
        # Сохранение результатов
        self._save_results(all_results, analysis)
        
        return analysis
    
    def _analyze_all_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ всех результатов тестирования"""
        
        # Фильтруем успешные результаты
        successful_results = [r for r in results if not r.get("failed", False)]
        converged_results = [r for r in successful_results if r.get("converged", False)]
        
        if not successful_results:
            return {"error": "No successful tests"}
        
        analysis = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "converged_tests": len(converged_results),
                "success_rate": len(successful_results) / len(results),
                "convergence_rate": len(converged_results) / len(successful_results) if successful_results else 0
            },
            "by_strategy": {},
            "best_overall": None,
            "recommendations": {}
        }
        
        # Анализ по стратегиям
        for strategy in self.strategies_to_test:
            strategy_results = [r for r in successful_results if r["strategy"] == strategy]
            strategy_converged = [r for r in strategy_results if r.get("converged", False)]
            
            if strategy_results:
                best_result = max(strategy_results, key=lambda x: x.get("quality_score", 0))
                
                analysis["by_strategy"][strategy] = {
                    "total_tests": len(strategy_results),
                    "converged_tests": len(strategy_converged),
                    "convergence_rate": len(strategy_converged) / len(strategy_results),
                    "best_config": {
                        "learning_rate": best_result["learning_rate"],
                        "batch_size": best_result["batch_size"],
                        "quality_score": best_result.get("quality_score", 0),
                        "final_loss": best_result.get("final_loss", 999),
                        "final_similarity": best_result.get("final_cosine_similarity", 0),
                        "parameter_count": best_result.get("parameter_count", 0),
                        "training_time": best_result.get("training_time", 0)
                    }
                }
        
        # Лучший результат overall
        if converged_results:
            best_overall = max(converged_results, key=lambda x: x.get("quality_score", 0))
            analysis["best_overall"] = {
                "strategy": best_overall["strategy"],
                "learning_rate": best_overall["learning_rate"],
                "batch_size": best_overall["batch_size"],
                "quality_score": best_overall.get("quality_score", 0),
                "final_loss": best_overall.get("final_loss", 999),
                "final_similarity": best_overall.get("final_cosine_similarity", 0),
                "stability_score": best_overall.get("stability_score", 0),
                "parameter_count": best_overall.get("parameter_count", 0),
                "compression_ratio": best_overall.get("compression_ratio", 0),
                "training_time": best_overall.get("training_time", 0)
            }
        
        # Рекомендации
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация рекомендаций на основе анализа"""
        
        recommendations = {
            "production_ready": None,
            "development_suitable": [],
            "fastest_training": None,
            "best_quality": None,
            "most_stable": None
        }
        
        if not analysis.get("by_strategy"):
            return recommendations
        
        strategy_scores = []
        
        for strategy, data in analysis["by_strategy"].items():
            if data["convergence_rate"] > 0:
                best_config = data["best_config"]
                strategy_scores.append({
                    "strategy": strategy,
                    "quality_score": best_config["quality_score"],
                    "convergence_rate": data["convergence_rate"],
                    "training_time": best_config["training_time"],
                    "final_loss": best_config["final_loss"],
                    "final_similarity": best_config["final_similarity"]
                })
        
        if not strategy_scores:
            return recommendations
        
        # Production ready: высокое качество + высокая сходимость
        production_candidates = [s for s in strategy_scores 
                               if s["quality_score"] > 0.7 and s["convergence_rate"] > 0.8]
        if production_candidates:
            recommendations["production_ready"] = max(production_candidates, 
                                                    key=lambda x: x["quality_score"])["strategy"]
        
        # Development suitable: хотя бы сходится
        recommendations["development_suitable"] = [s["strategy"] for s in strategy_scores 
                                                 if s["convergence_rate"] > 0.5]
        
        # Fastest training
        if strategy_scores:
            recommendations["fastest_training"] = min(strategy_scores, 
                                                    key=lambda x: x["training_time"])["strategy"]
        
        # Best quality
        if strategy_scores:
            recommendations["best_quality"] = max(strategy_scores, 
                                                key=lambda x: x["quality_score"])["strategy"]
        
        return recommendations
    
    def _save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Сохранение результатов в файлы"""
        
        # Подробные результаты
        results_file = self.output_dir / "llama_strategy_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Анализ
        analysis_file = self.output_dir / "llama_strategy_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[FOLDER] Results saved:")
        logger.info(f"   Details: {results_file}")
        logger.info(f"   Analysis: {analysis_file}")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Печать краткого отчета в консоль"""
        
        print("\n" + "🦙" * 20)
        print("Meta-Llama-3-8B Strategy Optimization Results")
        print("🦙" * 20)
        
        summary = analysis.get("summary", {})
        print(f"\n[DATA] SUMMARY:")
        print(f"   Tests run: {summary.get('total_tests', 0)}")
        print(f"   Successful: {summary.get('successful_tests', 0)} ({summary.get('success_rate', 0)*100:.1f}%)")
        print(f"   Converged: {summary.get('converged_tests', 0)} ({summary.get('convergence_rate', 0)*100:.1f}%)")
        
        if analysis.get("best_overall"):
            best = analysis["best_overall"]
            print(f"\n[TROPHY] BEST CONFIGURATION:")
            print(f"   Strategy: {best['strategy']}")
            print(f"   Learning rate: {best['learning_rate']}")
            print(f"   Batch size: {best['batch_size']}")
            print(f"   Quality score: {best['quality_score']:.3f}")
            print(f"   Final loss: {best['final_loss']:.3f}")
            print(f"   Cosine similarity: {best['final_similarity']:.3f}")
            print(f"   Parameters: {best['parameter_count']:,}")
            print(f"   Training time: {best['training_time']:.1f}s")
        
        if analysis.get("recommendations"):
            rec = analysis["recommendations"]
            print(f"\n[IDEA] RECOMMENDATIONS:")
            print(f"   [START] Production ready: {rec.get('production_ready', 'None')}")
            print(f"   🛠️  Development suitable: {', '.join(rec.get('development_suitable', []))}")
            print(f"   [FAST] Fastest training: {rec.get('fastest_training', 'None')}")
            print(f"   [STAR] Best quality: {rec.get('best_quality', 'None')}")
        
        print(f"\n[FOLDER] Detailed results saved to: {self.output_dir}")


def run_llama_quick_test(device: str = "cpu") -> Dict[str, Any]:
    """
    Быстрый тест стратегий для LLaMA (только 2 стратегии)
    
    Args:
        device: Устройство для вычислений
        
    Returns:
        Analysis результатов
    """
    optimizer = LlamaStrategyOptimizer(device=device)
    analysis = optimizer.run_quick_test()
    optimizer.print_summary(analysis)
    return analysis


def run_llama_optimization(device: str = "cpu") -> Dict[str, Any]:
    """
    Полный тест стратегий для LLaMA
    
    Args:
        device: Устройство для вычислений
        
    Returns:
        Analysis результатов
    """
    optimizer = LlamaStrategyOptimizer(device=device)
    analysis = optimizer.run_comprehensive_test()
    optimizer.print_summary(analysis)
    return analysis


if __name__ == "__main__":
    print("🦙 Starting Meta-Llama-3-8B Strategy Optimization...")
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CONFIG] Using device: {device}")
    
    # Запускаем быстрый тест сначала
    print("[FAST] Running quick test first...")
    results = run_llama_quick_test(device)
    
    print("\n[OK] Quick test completed!")
    if results.get("best_overall"):
        best = results["best_overall"]
        print(f"[TROPHY] Recommended strategy: {best['strategy']} (quality: {best['quality_score']:.3f})")
    else:
        print("[ERROR] No successful configurations found - check system setup") 