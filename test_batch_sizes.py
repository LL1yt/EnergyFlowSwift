#!/usr/bin/env python3
"""
🧪 Batch Size Testing Script
Сравнивает качество обучения при разных размерах batch
"""

import logging
import time
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_batch_sizes():
    """Тестирует разные размеры batch для сравнения качества"""

    # Размеры batch для тестирования
    batch_sizes = [16, 32, 64, 128, 256]

    # Параметры тестирования
    test_config = {
        "mode": "development",  # Маленькая решетка для быстрого теста
        "dataset_limit": 2000,  # Небольшой датасет
        "epochs": 8,  # Достаточно эпох чтобы увидеть разницу
    }

    logger.info("🧪 Starting Batch Size Comparison Test")
    logger.info(f"   Test config: {test_config}")
    logger.info(f"   Batch sizes to test: {batch_sizes}")
    logger.info("   This will take some time... ⏱️")

    results = []

    for batch_size in batch_sizes:
        logger.info(f"\n🚀 Testing batch_size={batch_size}")
        logger.info("=" * 50)

        try:
            start_time = time.time()

            # Запускаем обучение с текущим batch size
            import subprocess

            cmd = [
                "python",
                "smart_resume_training.py",
                "--mode",
                test_config["mode"],
                "--dataset-limit",
                str(test_config["dataset_limit"]),
                "--additional-epochs",
                str(test_config["epochs"]),
                "--batch-size",
                str(batch_size),
            ]

            logger.info(f"   Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 минут максимум на один тест
            )

            end_time = time.time()
            test_time = end_time - start_time

            if result.returncode == 0:
                # Извлекаем финальную similarity из вывода
                final_similarity = extract_similarity_from_output(result.stdout)

                test_result = {
                    "batch_size": batch_size,
                    "success": True,
                    "final_similarity": final_similarity,
                    "time_minutes": test_time / 60,
                    "stdout": result.stdout[-500:],  # Последние 500 символов
                }

                logger.info(f"✅ Batch {batch_size} completed!")
                logger.info(f"   Final similarity: {final_similarity:.4f}")
                logger.info(f"   Time: {test_time/60:.1f} minutes")

            else:
                test_result = {
                    "batch_size": batch_size,
                    "success": False,
                    "error": result.stderr[:200],
                    "time_minutes": test_time / 60,
                }

                logger.error(f"❌ Batch {batch_size} failed!")
                logger.error(f"   Error: {result.stderr[:200]}...")

            results.append(test_result)

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Batch {batch_size} timed out (>10 minutes)")
            results.append(
                {
                    "batch_size": batch_size,
                    "success": False,
                    "error": "Timeout",
                    "time_minutes": 10.0,
                }
            )
        except Exception as e:
            logger.error(f"💥 Batch {batch_size} crashed: {e}")
            results.append(
                {
                    "batch_size": batch_size,
                    "success": False,
                    "error": str(e),
                    "time_minutes": 0,
                }
            )

    # Анализируем результаты
    analyze_batch_results(results)

    return results


def extract_similarity_from_output(output: str) -> float:
    """Извлекает final similarity из вывода обучения"""
    try:
        for line in output.split("\n"):
            if "final_similarity:" in line:
                parts = line.split("final_similarity:")
                if len(parts) > 1:
                    similarity_str = parts[1].strip()
                    return float(similarity_str)
    except:
        pass
    return 0.0


def analyze_batch_results(results: List[Dict[str, Any]]):
    """Анализирует результаты тестирования разных batch sizes"""

    logger.info("\n" + "=" * 60)
    logger.info("📊 BATCH SIZE COMPARISON RESULTS")
    logger.info("=" * 60)

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        logger.error("❌ No successful tests!")
        return

    # Сортируем по качеству
    successful_results.sort(key=lambda x: x["final_similarity"], reverse=True)

    logger.info("\n🏆 Results ranked by similarity:")
    for i, result in enumerate(successful_results):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        logger.info(
            f"   {rank_emoji} Batch {result['batch_size']:3d}: "
            f"Similarity {result['final_similarity']:.4f} "
            f"({result['time_minutes']:.1f} min)"
        )

    # Анализ трендов
    logger.info("\n📈 Analysis:")

    best_result = successful_results[0]
    worst_result = successful_results[-1]

    logger.info(
        f"   🎯 Best batch size: {best_result['batch_size']} (similarity: {best_result['final_similarity']:.4f})"
    )
    logger.info(
        f"   ❌ Worst batch size: {worst_result['batch_size']} (similarity: {worst_result['final_similarity']:.4f})"
    )

    # Проверяем тренд: ухудшается ли качество с ростом batch size?
    batch_sizes = [r["batch_size"] for r in successful_results]
    similarities = [r["final_similarity"] for r in successful_results]

    if len(successful_results) >= 3:
        # Простая корреляция между batch size и качеством
        large_batches = [r for r in successful_results if r["batch_size"] >= 128]
        small_batches = [r for r in successful_results if r["batch_size"] <= 64]

        if large_batches and small_batches:
            large_avg = sum(r["final_similarity"] for r in large_batches) / len(
                large_batches
            )
            small_avg = sum(r["final_similarity"] for r in small_batches) / len(
                small_batches
            )

            if small_avg > large_avg:
                logger.info(f"   ✅ Confirmed: Small batches (≤64) perform better!")
                logger.info(f"      Small batches avg: {small_avg:.4f}")
                logger.info(f"      Large batches avg: {large_avg:.4f}")
            else:
                logger.info(
                    f"   🤔 Unexpected: Large batches performed better in this test"
                )

    # Рекомендации
    logger.info("\n💡 Recommendations:")
    if best_result["batch_size"] <= 64:
        logger.info(
            f"   ✅ Use batch_size={best_result['batch_size']} for optimal quality"
        )
    else:
        logger.info(
            f"   ⚠️ Best was batch_size={best_result['batch_size']}, but consider 32-64 for stability"
        )

    logger.info(f"   🚫 Avoid very large batch sizes (>128) for this type of task")
    logger.info(f"   ⚡ For speed vs quality, balance around 32-64")


def quick_batch_recommendation():
    """Быстрые рекомендации без тестирования"""

    logger.info("🎯 Quick Batch Size Recommendations for 3D CNN:")
    logger.info("")
    logger.info("📊 For different priorities:")
    logger.info("   🎮 Quick testing:     --batch-size 64")
    logger.info("   🧪 Best quality:      --batch-size 32")
    logger.info("   ⚡ Speed (if fits):   --batch-size 128")
    logger.info("   💾 Memory limited:    --batch-size 16")
    logger.info("")
    logger.info("❌ Avoid:")
    logger.info("   --batch-size 256+   (poor generalization)")
    logger.info("   --batch-size 4096   (terrible for learning!)")
    logger.info("")
    logger.info("🎯 Sweet spot: 32-64 for most scenarios")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Size Testing")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run actual batch size comparison test (takes time!)",
    )
    parser.add_argument(
        "--quick-advice",
        action="store_true",
        help="Show quick recommendations without testing",
    )

    args = parser.parse_args()

    if args.run_test:
        test_batch_sizes()
    elif args.quick_advice:
        quick_batch_recommendation()
    else:
        logger.info("🧪 Batch Size Tester")
        logger.info("")
        logger.info("Options:")
        logger.info("  --quick-advice    Show recommendations")
        logger.info("  --run-test        Run comparison test (20-30 minutes)")
        logger.info("")
        logger.info("Quick start:")
        logger.info("  python test_batch_sizes.py --quick-advice")
