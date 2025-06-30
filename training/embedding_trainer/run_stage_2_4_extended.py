"""
Stage 2.4 Extended: Aggressive Hyperparameter Optimization
Стратегия преодоления регрессии и достижения 50%+ Q→A similarity

Анализ проблемы: Stage 2.4 показал 36.6% vs Stage 2.3 38.4% (-1.8%)
Решение: Расширенная оптимизация с фокусом на proven techniques
"""

import sys
import time
from pathlib import Path
import json
import numpy as np

# Добавляем корневую директорию в path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.embedding_trainer.hyperparameter_optimizer_stage_2_4 import (
    HyperparameterOptimizer,
    HyperparameterConfig
)
from training.embedding_trainer.advanced_training_stage_2_3 import (
    AdvancedTrainingStage23,
    Stage23Config,
    run_stage_2_3_training
)


class ExtendedOptimizationStrategy:
    """
    Расширенная стратегия оптимизации для преодоления регрессии
    
    План:
    1. Baseline validation (воспроизведение Stage 2.3 результата)
    2. Conservative optimization (близко к Stage 2.3 settings)
    3. Aggressive optimization (расширенный search)
    4. Architecture experiments (alternative approaches)
    """
    
    def __init__(self):
        self.baseline_qa_similarity = 0.384  # Stage 2.3 result
        self.target_qa_similarity = 0.50
        self.results_history = []
        
    def run_comprehensive_strategy(self) -> dict:
        """Запуск полной стратегии оптимизации"""
        print("[START] STAGE 2.4 EXTENDED: AGGRESSIVE OPTIMIZATION STRATEGY")
        print("=" * 70)
        print(f"Baseline (Stage 2.3): {self.baseline_qa_similarity:.1%}")
        print(f"Target: {self.target_qa_similarity:.1%}")
        print(f"Required improvement: +{self.target_qa_similarity - self.baseline_qa_similarity:.1%}")
        print()
        
        start_time = time.time()
        
        # Phase 1: Baseline Validation
        print("[INFO] PHASE 1: BASELINE VALIDATION")
        print("-" * 40)
        baseline_result = self._validate_baseline()
        
        # Phase 2: Conservative Optimization
        print("\n[TARGET] PHASE 2: CONSERVATIVE OPTIMIZATION")
        print("-" * 40)
        conservative_result = self._run_conservative_optimization(baseline_result)
        
        # Phase 3: Aggressive Optimization
        print("\n[FAST] PHASE 3: AGGRESSIVE OPTIMIZATION")
        print("-" * 40)
        aggressive_result = self._run_aggressive_optimization(conservative_result)
        
        # Phase 4: Architecture Experiments
        print("\n🏗️ PHASE 4: ARCHITECTURE EXPERIMENTS")
        print("-" * 40)
        architecture_result = self._run_architecture_experiments(aggressive_result)
        
        # Final Analysis
        total_time = time.time() - start_time
        final_analysis = self._generate_final_analysis(total_time)
        
        return final_analysis
    
    def _validate_baseline(self) -> dict:
        """Phase 1: Воспроизведение Stage 2.3 результата"""
        print("[MAGNIFY] Validating Stage 2.3 baseline configuration...")
        
        try:
            # Конфигурация Stage 2.3 (известная рабочая)
            stage_2_3_config = Stage23Config(
                # Параметры из Stage 2.3
                learning_rate=0.0003,          # Из Stage 2.3
                batch_size=6,                  # Из Stage 2.3  
                epochs=15,                     # Из Stage 2.3
                target_pairs=100,              # Из Stage 2.3
                
                # Advanced settings из Stage 2.3
                use_curriculum_learning=True,
                use_triplet_loss=True,
                use_contrastive_loss=True,
                use_multi_teacher=True,
                curriculum_warmup_epochs=5,
                distillation_temperature=3.0,
                
                # Target metrics
                target_qa_similarity=0.50,
                convergence_threshold=0.01,
                validation_patience=5
            )
            
            # Запуск baseline
            print("   [SCIENCE] Running Stage 2.3 baseline configuration...")
            baseline_results = self._run_single_config(stage_2_3_config, "baseline_stage_2_3")
            
            baseline_qa = baseline_results.get('qa_similarity', 0)
            print(f"   [DATA] Baseline result: {baseline_qa:.1%}")
            
            if baseline_qa >= self.baseline_qa_similarity * 0.95:  # 95% of expected
                print("   [OK] Baseline validation SUCCESSFUL")
                return {"config": stage_2_3_config, "qa_similarity": baseline_qa, "status": "SUCCESS"}
            else:
                print(f"   [WARNING] Baseline validation shows degradation: {baseline_qa:.1%} vs expected {self.baseline_qa_similarity:.1%}")
                return {"config": stage_2_3_config, "qa_similarity": baseline_qa, "status": "DEGRADED"}
                
        except Exception as e:
            print(f"   [ERROR] Baseline validation failed: {e}")
            return {"config": None, "qa_similarity": 0, "status": "FAILED", "error": str(e)}
    
    def _run_conservative_optimization(self, baseline_result: dict) -> dict:
        """Phase 2: Консервативная оптимизация близко к baseline"""
        print("[TARGET] Conservative optimization around proven configuration...")
        
        base_config = baseline_result["config"]
        best_result = baseline_result
        
        if baseline_result["status"] != "SUCCESS":
            print("   [WARNING] Baseline failed, using default configuration")
            base_config = Stage23Config()
        
        # Conservative parameter variations (близко к Stage 2.3)
        conservative_variations = [
            # Learning rate variations
            {"learning_rate": 0.0002, "batch_size": 6, "epochs": 15},    # Немного меньше LR
            {"learning_rate": 0.0003, "batch_size": 4, "epochs": 15},    # Меньше batch size
            {"learning_rate": 0.0003, "batch_size": 8, "epochs": 15},    # Больше batch size
            {"learning_rate": 0.0004, "batch_size": 6, "epochs": 15},    # Немного больше LR
            
            # Epochs variations
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 20},    # Больше epochs
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 12},    # Меньше epochs
            
            # Advanced parameters
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 15, "curriculum_warmup_epochs": 3},
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 15, "curriculum_warmup_epochs": 8},
        ]
        
        for i, variation in enumerate(conservative_variations):
            print(f"   [TEST] Conservative experiment {i+1}/{len(conservative_variations)}: {variation}")
            
            # Создание конфигурации
            test_config = Stage23Config(
                learning_rate=variation.get("learning_rate", 0.0003),
                batch_size=variation.get("batch_size", 6),
                epochs=variation.get("epochs", 15),
                target_pairs=100,
                use_curriculum_learning=True,
                use_triplet_loss=True,
                use_contrastive_loss=True,
                use_multi_teacher=True,
                curriculum_warmup_epochs=variation.get("curriculum_warmup_epochs", 5),
                distillation_temperature=3.0,
                target_qa_similarity=0.50
            )
            
            # Запуск эксперимента
            result = self._run_single_config(test_config, f"conservative_{i+1}")
            
            if result.get('qa_similarity', 0) > best_result.get('qa_similarity', 0):
                best_result = {"config": test_config, "qa_similarity": result['qa_similarity']}
                print(f"      [SUCCESS] New best: {result['qa_similarity']:.1%}")
        
        print(f"   [DATA] Conservative phase best: {best_result.get('qa_similarity', 0):.1%}")
        return best_result
    
    def _run_aggressive_optimization(self, conservative_result: dict) -> dict:
        """Phase 3: Агрессивная оптимизация"""
        print("[FAST] Aggressive optimization with expanded search space...")
        
        best_result = conservative_result
        
        # Aggressive parameter combinations
        aggressive_configs = [
            # High learning rates
            {"learning_rate": 0.0005, "batch_size": 4, "epochs": 25, "target_pairs": 150},
            {"learning_rate": 0.0008, "batch_size": 2, "epochs": 20, "target_pairs": 150},
            
            # Large batch sizes
            {"learning_rate": 0.0002, "batch_size": 12, "epochs": 30, "target_pairs": 150},
            {"learning_rate": 0.0001, "batch_size": 16, "epochs": 25, "target_pairs": 150},
            
            # Extended training
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 40, "target_pairs": 200},
            {"learning_rate": 0.0002, "batch_size": 8, "epochs": 35, "target_pairs": 200},
            
            # Advanced curriculum learning
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 20, "curriculum_warmup_epochs": 10, "target_pairs": 150},
            {"learning_rate": 0.0004, "batch_size": 4, "epochs": 25, "curriculum_warmup_epochs": 12, "target_pairs": 150},
            
            # Distillation temperature optimization
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 20, "distillation_temperature": 5.0, "target_pairs": 150},
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 20, "distillation_temperature": 2.0, "target_pairs": 150},
        ]
        
        for i, config_params in enumerate(aggressive_configs):
            print(f"   [START] Aggressive experiment {i+1}/{len(aggressive_configs)}")
            
            test_config = Stage23Config(
                learning_rate=config_params.get("learning_rate", 0.0003),
                batch_size=config_params.get("batch_size", 6),
                epochs=config_params.get("epochs", 15),
                target_pairs=config_params.get("target_pairs", 100),
                use_curriculum_learning=True,
                use_triplet_loss=True,
                use_contrastive_loss=True,
                use_multi_teacher=True,
                curriculum_warmup_epochs=config_params.get("curriculum_warmup_epochs", 5),
                distillation_temperature=config_params.get("distillation_temperature", 3.0),
                target_qa_similarity=0.50
            )
            
            result = self._run_single_config(test_config, f"aggressive_{i+1}")
            
            if result.get('qa_similarity', 0) > best_result.get('qa_similarity', 0):
                best_result = {"config": test_config, "qa_similarity": result['qa_similarity']}
                print(f"      [SUCCESS] NEW BREAKTHROUGH: {result['qa_similarity']:.1%}")
                
                # Early success check
                if result['qa_similarity'] >= self.target_qa_similarity:
                    print(f"      [TROPHY] TARGET ACHIEVED! Stopping optimization.")
                    break
        
        print(f"   [DATA] Aggressive phase best: {best_result.get('qa_similarity', 0):.1%}")
        return best_result
    
    def _run_architecture_experiments(self, aggressive_result: dict) -> dict:
        """Phase 4: Эксперименты с архитектурой"""
        print("🏗️ Architecture experiments for breakthrough performance...")
        
        best_result = aggressive_result
        
        # Architecture variations (пока используем базовую архитектуру)
        # В будущем здесь будут эксперименты с размерами куба
        architecture_configs = [
            # Multi-stage training approach
            {"learning_rate": 0.0001, "batch_size": 4, "epochs": 50, "target_pairs": 250},
            {"learning_rate": 0.0002, "batch_size": 8, "epochs": 45, "target_pairs": 250},
            
            # Ultra high-quality dataset
            {"learning_rate": 0.0003, "batch_size": 6, "epochs": 30, "target_pairs": 300, "quality_threshold": 0.8},
            {"learning_rate": 0.0002, "batch_size": 4, "epochs": 35, "target_pairs": 300, "quality_threshold": 0.9},
        ]
        
        for i, config_params in enumerate(architecture_configs):
            print(f"   🏗️ Architecture experiment {i+1}/{len(architecture_configs)}")
            
            test_config = Stage23Config(
                learning_rate=config_params.get("learning_rate", 0.0003),
                batch_size=config_params.get("batch_size", 6),
                epochs=config_params.get("epochs", 15),
                target_pairs=config_params.get("target_pairs", 100),
                quality_threshold=config_params.get("quality_threshold", 0.6),
                use_curriculum_learning=True,
                use_triplet_loss=True,
                use_contrastive_loss=True,
                use_multi_teacher=True,
                target_qa_similarity=0.50
            )
            
            result = self._run_single_config(test_config, f"architecture_{i+1}")
            
            if result.get('qa_similarity', 0) > best_result.get('qa_similarity', 0):
                best_result = {"config": test_config, "qa_similarity": result['qa_similarity']}
                print(f"      [SUCCESS] ARCHITECTURE BREAKTHROUGH: {result['qa_similarity']:.1%}")
                
                if result['qa_similarity'] >= self.target_qa_similarity:
                    print(f"      [TROPHY] TARGET ACHIEVED! Stopping optimization.")
                    break
        
        print(f"   [DATA] Architecture phase best: {best_result.get('qa_similarity', 0):.1%}")
        return best_result
    
    def _run_single_config(self, config: Stage23Config, experiment_name: str) -> dict:
        """Запуск одного эксперимента"""
        try:
            start_time = time.time()
            
            # Создание training system
            trainer = AdvancedTrainingStage23(config)
            trainer.setup_training_components()
            
            # Создание dataset
            dataset = trainer.create_enhanced_dataset()
            
            # Запуск обучения
            training_results = trainer.run_advanced_training(dataset)
            
            qa_similarity = training_results.get("best_qa_similarity", 0.0)
            training_time = time.time() - start_time
            
            result = {
                "experiment_name": experiment_name,
                "qa_similarity": qa_similarity,
                "training_time": training_time,
                "success": True
            }
            
            # Сохранение результата
            self.results_history.append(result)
            
            print(f"      [OK] {experiment_name}: {qa_similarity:.1%} ({training_time:.1f}s)")
            return result
            
        except Exception as e:
            print(f"      [ERROR] {experiment_name}: Failed - {e}")
            return {"experiment_name": experiment_name, "qa_similarity": 0.0, "success": False, "error": str(e)}
    
    def _generate_final_analysis(self, total_time: float) -> dict:
        """Генерация финального анализа"""
        successful_results = [r for r in self.results_history if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful experiments"}
        
        qa_similarities = [r['qa_similarity'] for r in successful_results]
        best_qa = max(qa_similarities)
        mean_qa = np.mean(qa_similarities)
        
        final_analysis = {
            "total_experiments": len(self.results_history),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(self.results_history),
            
            "best_qa_similarity": best_qa,
            "mean_qa_similarity": mean_qa,
            "target_achieved": best_qa >= self.target_qa_similarity,
            
            "improvement_from_baseline": best_qa - self.baseline_qa_similarity,
            "improvement_from_stage_2_3": best_qa - 0.384,
            
            "total_optimization_time": total_time,
            "experiments_history": self.results_history
        }
        
        # Отображение результатов
        print("\n" + "=" * 70)
        print("[TROPHY] STAGE 2.4 EXTENDED OPTIMIZATION - FINAL RESULTS")
        print("=" * 70)
        
        print(f"[TARGET] Target achieved: {'[OK]' if final_analysis['target_achieved'] else '[ERROR]'}")
        print(f"[TROPHY] Best Q→A similarity: {best_qa:.1%}")
        print(f"[DATA] Mean Q→A similarity: {mean_qa:.1%}")
        print(f"[CHART] Improvement from Stage 2.3: {final_analysis['improvement_from_stage_2_3']:+.1%}")
        print(f"[TEST] Total experiments: {final_analysis['total_experiments']}")
        print(f"[OK] Success rate: {final_analysis['success_rate']:.1%}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        
        if final_analysis['target_achieved']:
            print("\n[SUCCESS] BREAKTHROUGH ACHIEVED! 50%+ Q→A similarity reached!")
            print("[START] Ready for Stage 3.1: End-to-End Integration")
        else:
            gap = self.target_qa_similarity - best_qa
            print(f"\n[WARNING] Target not yet reached. Remaining gap: {gap:.1%}")
            print("[IDEA] Consider additional optimization strategies")
        
        return final_analysis


def main():
    """Запуск расширенной оптимизации"""
    print("[START] STAGE 2.4 EXTENDED: AGGRESSIVE HYPERPARAMETER OPTIMIZATION")
    print("[TARGET] Mission: Overcome regression and achieve 50%+ Q→A similarity!")
    print()
    
    # Создание стратегии
    strategy = ExtendedOptimizationStrategy()
    
    # Запуск
    results = strategy.run_comprehensive_strategy()
    
    # Сохранение результатов
    try:
        results_dir = Path("reports/stage_2_4_extended")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "extended_optimization_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n[FILE] Results saved to: {results_dir}/")
        
    except Exception as e:
        print(f"[WARNING] Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    results = main() 