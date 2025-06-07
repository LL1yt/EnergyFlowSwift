"""
Запуск Stage 2.4: Advanced Hyperparameter Optimization
Финальный рывок к достижению 50%+ Q→A similarity!

Использование:
python run_stage_2_4_optimization.py [--quick] [--target 0.50] [--max-experiments 50]
"""

import sys
import argparse
import time
from pathlib import Path
import json

# Добавляем корневую директорию в path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from training.embedding_trainer.hyperparameter_optimizer_stage_2_4 import (
        HyperparameterOptimizer,
        HyperparameterConfig,
        run_stage_2_4_optimization,
        analyze_optimization_results
    )
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Attempting to run from current directory...")
    # Fallback import
    from hyperparameter_optimizer_stage_2_4 import (
        HyperparameterOptimizer,
        HyperparameterConfig,
        run_stage_2_4_optimization,
        analyze_optimization_results
    )


def main():
    """Главная функция запуска оптимизации"""
    parser = argparse.ArgumentParser(description='Stage 2.4 Hyperparameter Optimization')
    parser.add_argument('--quick', action='store_true', help='Quick mode (сокращенный search)')
    parser.add_argument('--target', type=float, default=0.50, help='Target Q→A similarity')
    parser.add_argument('--max-experiments', type=int, default=20, help='Maximum experiments')
    parser.add_argument('--test-only', action='store_true', help='Test mode (minimal experiments)')
    
    args = parser.parse_args()
    
    print("🚀 STAGE 2.4: ADVANCED HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Target Q→A similarity: {args.target:.1%}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"Quick mode: {'✅' if args.quick else '❌'}")
    print(f"Test mode: {'✅' if args.test_only else '❌'}")
    print()
    
    # Настройка для test mode
    if args.test_only:
        args.max_experiments = 3  # Минимум для тестирования
        args.quick = True
        print("🧪 Test mode: Running minimal experiments for system validation")
    
    start_time = time.time()
    
    try:
        # Запуск оптимизации
        print("🎯 Starting optimization process...")
        results = run_stage_2_4_optimization(
            max_experiments=args.max_experiments,
            target_qa_similarity=args.target,
            quick_mode=args.quick
        )
        
        # Время выполнения
        total_time = time.time() - start_time
        
        # Отображение результатов
        print("\n" + "=" * 60)
        print("📊 STAGE 2.4 OPTIMIZATION RESULTS")
        print("=" * 60)
        
        print(f"🎯 Target achieved: {'✅' if results.get('target_achieved', False) else '❌'}")
        print(f"🏆 Best Q→A similarity: {results.get('best_qa_similarity', 0):.1%}")
        
        improvement = results.get('improvement_from_stage_2_3', 0)
        print(f"📈 Improvement from Stage 2.3: +{improvement:.1%}")
        
        print(f"🧪 Total experiments: {results.get('total_experiments', 0)}")
        print(f"✅ Success rate: {results.get('success_rate', 0):.1%}")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        
        if results.get('target_achieved'):
            print("\n🎉 BREAKTHROUGH ACHIEVED! 50%+ Q→A similarity reached!")
            print("🚀 Ready for Stage 3.1: End-to-End Integration")
        else:
            current_best = results.get('best_qa_similarity', 0)
            remaining_gap = args.target - current_best
            print(f"\n⚠️ Target not yet reached. Remaining gap: {remaining_gap:.1%}")
            print("💡 Consider running with more experiments or adjusting target")
        
        # Сохранение подробного отчета
        save_optimization_report(results, args, total_time)
        
        # Рекомендации для следующих шагов
        print_next_steps_recommendations(results, args)
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        print("🔧 Please check system dependencies and configuration")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def save_optimization_report(results: dict, args: argparse.Namespace, total_time: float):
    """Сохранение подробного отчета оптимизации"""
    try:
        report_dir = Path("reports/stage_2_4")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание comprehensive report
        report = {
            "stage": "2.4",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "target_qa_similarity": args.target,
                "max_experiments": args.max_experiments,
                "quick_mode": args.quick,
                "test_mode": args.test_only
            },
            "results": results,
            "execution_time_seconds": total_time,
            "status": "SUCCESS" if results.get('target_achieved') else "PARTIAL_SUCCESS"
        }
        
        # Сохранение JSON отчета
        with open(report_dir / "optimization_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # Сохранение человеко-читаемого отчета
        with open(report_dir / "optimization_summary.txt", "w", encoding='utf-8') as f:
            f.write(f"Stage 2.4 Hyperparameter Optimization Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Target Q→A similarity: {args.target:.1%}\n")
            f.write(f"Achieved Q→A similarity: {results.get('best_qa_similarity', 0):.1%}\n")
            f.write(f"Target achieved: {'YES' if results.get('target_achieved') else 'NO'}\n")
            f.write(f"Total experiments: {results.get('total_experiments', 0)}\n")
            f.write(f"Execution time: {total_time:.1f} seconds\n\n")
            
            if results.get('convergence_analysis'):
                f.write("Best Configuration Analysis:\n")
                f.write(f"- Best learning rate: {results['convergence_analysis'].get('best_learning_rate', 'N/A')}\n")
                f.write(f"- Best batch size: {results['convergence_analysis'].get('best_batch_size', 'N/A')}\n")
                f.write(f"- Mean convergence epochs: {results['convergence_analysis'].get('mean_convergence_epochs', 'N/A')}\n")
        
        print(f"📄 Detailed report saved to: {report_dir}/")
        
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")


def print_next_steps_recommendations(results: dict, args: argparse.Namespace):
    """Рекомендации для следующих шагов"""
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS RECOMMENDATIONS")
    print("=" * 60)
    
    if results.get('target_achieved'):
        print("🎯 STAGE 2.4 COMPLETE! Ready for next phase:")
        print("1. 📋 Update plan.md with achieved results")
        print("2. 🚀 Begin Stage 3.1: End-to-End Integration")
        print("3. 📊 Prepare comprehensive evaluation in Stage 3.2")
        
    else:
        current_best = results.get('best_qa_similarity', 0)
        target = args.target
        gap = target - current_best
        
        print(f"⚠️ Target not achieved. Current best: {current_best:.1%} (gap: {gap:.1%})")
        print("\nRecommended actions:")
        
        if gap > 0.05:  # >5% gap
            print("1. 🔧 Run extended optimization (more experiments)")
            print("2. 📚 Enhance dataset quality (Stage 2.4.3)")
            print("3. 🏗️ Try alternative architectures (Stage 2.4.4)")
        elif gap > 0.02:  # 2-5% gap  
            print("1. ⚡ Fine-tune loss weights (Stage 2.4.2)")
            print("2. 🎛️ Optimize learning rate scheduling")
            print("3. 📊 Run statistical validation (more runs)")
        else:  # <2% gap
            print("1. 🎲 Current result is very close! Try more runs")
            print("2. 📈 Consider target achieved within margin of error")
            print("3. 🚀 Proceed to Stage 3.1 with current best")
        
        print(f"\n🔄 To continue optimization:")
        print(f"   python run_stage_2_4_optimization.py --max-experiments {args.max_experiments * 2}")


def test_system_readiness():
    """Тестирование готовности системы"""
    print("🧪 Testing system readiness...")
    
    try:
        # Тест импортов
        from training.embedding_trainer.cube_trainer import CubeTrainer
        from training.embedding_trainer.advanced_training_stage_2_3 import AdvancedTrainingStage23
        print("✅ Core components available")
        
        # Тест конфигурации
        config = HyperparameterConfig(max_experiments=1)
        optimizer = HyperparameterOptimizer(config)
        print("✅ Optimizer initialization successful")
        
        print("🎯 System ready for optimization!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔬 STAGE 2.4 HYPERPARAMETER OPTIMIZATION LAUNCHER")
    print("🎯 Goal: Achieve 50%+ Q→A similarity breakthrough!")
    print()
    
    # Тестирование системы
    if not test_system_readiness():
        print("🛑 System not ready. Please check dependencies.")
        sys.exit(1)
    
    # Запуск оптимизации
    exit_code = main()
    sys.exit(exit_code) 