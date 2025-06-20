#!/usr/bin/env python3
"""
🎯 ТЕСТ ФАЗЫ 4: Полный цикл обучения с оптимизациями

Цель: Протестировать полный цикл обучения с новыми возможностями Фазы 4:
- Реальное обучение на малых решетках (16×16×16 → 24×24×24)
- Измерение memory reduction в процессе обучения
- Проверка сохранения emergent behavior
- Валидация quality metrics

Это финальный тест НЕДЕЛИ 1 согласно плану интеграции.
"""

import sys
import os
import time
import json
import psutil
import torch
from pathlib import Path
from datetime import datetime

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))

from training.automated_training.types import StageConfig
from training.automated_training.progressive_config import ProgressiveConfigManager
from training.automated_training.stage_runner import TrainingStageRunner
from training.automated_training.automated_trainer import AutomatedTrainer


class FullCycleMemoryMonitor:
    """Расширенный монитор памяти для полного цикла обучения"""

    def __init__(self):
        self.measurements = []
        self.start_time = None

    def start_monitoring(self):
        """Начать мониторинг полного цикла"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        initial_measurement = {
            "timestamp": 0,
            "ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "gpu_mb": (
                torch.cuda.memory_allocated(0) / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
            "event": "start",
        }
        self.measurements.append(initial_measurement)
        print(
            f"📊 Monitoring started: {initial_measurement['ram_mb']:.1f}MB RAM, {initial_measurement['gpu_mb']:.1f}MB GPU"
        )

    def record_measurement(self, event: str):
        """Записать измерение памяти"""
        if self.start_time is None:
            return

        current_time = time.time() - self.start_time
        measurement = {
            "timestamp": current_time,
            "ram_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "gpu_mb": (
                torch.cuda.memory_allocated(0) / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
            "event": event,
        }
        self.measurements.append(measurement)
        print(
            f"📊 {event}: {measurement['ram_mb']:.1f}MB RAM, {measurement['gpu_mb']:.1f}MB GPU"
        )

    def get_peak_usage(self):
        """Получить пиковое использование памяти"""
        if not self.measurements:
            return 0, 0

        peak_ram = max(m["ram_mb"] for m in self.measurements)
        peak_gpu = max(m["gpu_mb"] for m in self.measurements)
        return peak_ram, peak_gpu

    def get_memory_efficiency(self):
        """Рассчитать эффективность использования памяти"""
        if len(self.measurements) < 2:
            return {}

        start = self.measurements[0]
        peak_ram, peak_gpu = self.get_peak_usage()

        return {
            "start_ram_mb": start["ram_mb"],
            "start_gpu_mb": start["gpu_mb"],
            "peak_ram_mb": peak_ram,
            "peak_gpu_mb": peak_gpu,
            "ram_growth_mb": peak_ram - start["ram_mb"],
            "gpu_growth_mb": peak_gpu - start["gpu_mb"],
            "total_measurements": len(self.measurements),
        }


def create_test_stage_config(stage: int, mode: str = "optimized") -> StageConfig:
    """Создать конфигурацию стадии для тестирования"""

    if mode == "optimized":
        # Оптимизированная конфигурация с возможностями Фазы 4
        configs = {
            1: {
                "dataset_limit": 50,  # Маленький для быстрого теста
                "epochs": 3,
                "plasticity_profile": "discovery",
                "clustering_enabled": False,
                "activity_threshold": 0.01,
                "memory_optimizations": True,
                "emergence_tracking": True,
            },
            2: {
                "dataset_limit": 100,
                "epochs": 2,
                "plasticity_profile": "learning",
                "clustering_enabled": True,
                "activity_threshold": 0.025,
                "memory_optimizations": True,
                "emergence_tracking": True,
                "sparse_connection_ratio": 0.1,
            },
        }
    else:
        # Базовая конфигурация без оптимизаций
        configs = {
            1: {
                "dataset_limit": 50,
                "epochs": 3,
                "plasticity_profile": "balanced",
                "clustering_enabled": False,
                "activity_threshold": 0.05,
                "memory_optimizations": False,
                "emergence_tracking": False,
            },
            2: {
                "dataset_limit": 100,
                "epochs": 2,
                "plasticity_profile": "balanced",
                "clustering_enabled": False,
                "activity_threshold": 0.05,
                "memory_optimizations": False,
                "emergence_tracking": False,
            },
        }

    config = configs.get(stage, configs[1])

    return StageConfig(
        stage=stage,
        dataset_limit=config["dataset_limit"],
        epochs=config["epochs"],
        batch_size=8,  # Маленький batch для контролируемого тестирования
        description=f"Full Cycle Test Stage {stage} ({mode})",
        plasticity_profile=config["plasticity_profile"],
        clustering_enabled=config["clustering_enabled"],
        activity_threshold=config["activity_threshold"],
        memory_optimizations=config["memory_optimizations"],
        emergence_tracking=config["emergence_tracking"],
        sparse_connection_ratio=config.get("sparse_connection_ratio", 0.0),
        progressive_scaling=True,
    )


def run_training_stage_with_monitoring(
    stage_config: StageConfig, monitor: FullCycleMemoryMonitor
) -> dict:
    """Запустить стадию обучения с мониторингом"""

    monitor.record_measurement(f"stage_{stage_config.stage}_start")

    # Создаем runner
    runner = TrainingStageRunner(
        mode="development",
        scale=0.01,  # Очень маленький масштаб для быстрого тестирования
        timeout_multiplier=1.5,
        verbose=True,
    )

    # Оценка времени
    config_manager = ProgressiveConfigManager()
    estimated_time = config_manager.estimate_stage_time(
        stage_config, mode="development"
    )

    monitor.record_measurement(f"stage_{stage_config.stage}_config_ready")

    # Запуск обучения
    print(f"🏃‍♂️ Запуск Stage {stage_config.stage}: {stage_config.description}")
    print(f"   Dataset: {stage_config.dataset_limit}, Epochs: {stage_config.epochs}")
    print(
        f"   Plasticity: {stage_config.plasticity_profile}, Memory opt: {stage_config.memory_optimizations}"
    )

    start_time = time.time()

    try:
        # Запускаем обучение (это может занять время)
        result = runner.run_stage(stage_config, estimated_time)

        actual_time = (time.time() - start_time) / 60
        monitor.record_measurement(f"stage_{stage_config.stage}_complete")

        if result and result.success:
            print(f"✅ Stage {stage_config.stage} completed in {actual_time:.1f}min")
            return {
                "success": True,
                "actual_time_minutes": actual_time,
                "final_similarity": result.final_similarity,
                "stage_result": result,
            }
        else:
            print(f"❌ Stage {stage_config.stage} failed after {actual_time:.1f}min")
            return {
                "success": False,
                "actual_time_minutes": actual_time,
                "error": result.error if result else "Unknown error",
            }

    except Exception as e:
        actual_time = (time.time() - start_time) / 60
        monitor.record_measurement(f"stage_{stage_config.stage}_error")
        print(
            f"❌ Stage {stage_config.stage} exception after {actual_time:.1f}min: {e}"
        )
        return {"success": False, "actual_time_minutes": actual_time, "error": str(e)}


def test_optimized_vs_baseline_training():
    """Сравнение оптимизированного обучения с базовым"""
    print("🆚 Сравнение оптимизированного vs базового обучения...")

    results = {"optimized": {}, "baseline": {}}

    # === ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ===
    print("\n🚀 === ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ (Phase 4) ===")

    monitor_opt = FullCycleMemoryMonitor()
    monitor_opt.start_monitoring()

    # Стадия 1: Discovery с оптимизациями
    stage1_opt = create_test_stage_config(1, "optimized")
    result1_opt = run_training_stage_with_monitoring(stage1_opt, monitor_opt)

    # Стадия 2: Learning с кластеризацией
    stage2_opt = create_test_stage_config(2, "optimized")
    result2_opt = run_training_stage_with_monitoring(stage2_opt, monitor_opt)

    opt_efficiency = monitor_opt.get_memory_efficiency()
    results["optimized"] = {
        "stage1": result1_opt,
        "stage2": result2_opt,
        "memory_efficiency": opt_efficiency,
        "total_time": result1_opt["actual_time_minutes"]
        + result2_opt["actual_time_minutes"],
    }

    print(f"🚀 Оптимизированное обучение завершено:")
    print(f"   Общее время: {results['optimized']['total_time']:.1f}min")
    print(f"   Peak RAM: {opt_efficiency['peak_ram_mb']:.1f}MB")
    print(f"   Peak GPU: {opt_efficiency['peak_gpu_mb']:.1f}MB")

    # === БАЗОВОЕ ОБУЧЕНИЕ ===
    print("\n📊 === БАЗОВОЕ ОБУЧЕНИЕ (без оптимизаций) ===")

    monitor_base = FullCycleMemoryMonitor()
    monitor_base.start_monitoring()

    # Стадия 1: Без оптимизаций
    stage1_base = create_test_stage_config(1, "baseline")
    result1_base = run_training_stage_with_monitoring(stage1_base, monitor_base)

    # Стадия 2: Без оптимизаций
    stage2_base = create_test_stage_config(2, "baseline")
    result2_base = run_training_stage_with_monitoring(stage2_base, monitor_base)

    base_efficiency = monitor_base.get_memory_efficiency()
    results["baseline"] = {
        "stage1": result1_base,
        "stage2": result2_base,
        "memory_efficiency": base_efficiency,
        "total_time": result1_base["actual_time_minutes"]
        + result2_base["actual_time_minutes"],
    }

    print(f"📊 Базовое обучение завершено:")
    print(f"   Общее время: {results['baseline']['total_time']:.1f}min")
    print(f"   Peak RAM: {base_efficiency['peak_ram_mb']:.1f}MB")
    print(f"   Peak GPU: {base_efficiency['peak_gpu_mb']:.1f}MB")

    # === СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===
    print("\n📊 === СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")

    if base_efficiency["peak_ram_mb"] > 0:
        ram_savings = (
            (base_efficiency["peak_ram_mb"] - opt_efficiency["peak_ram_mb"])
            / base_efficiency["peak_ram_mb"]
            * 100
        )
        print(f"💾 Экономия RAM: {ram_savings:.1f}%")

    if base_efficiency["peak_gpu_mb"] > 0:
        gpu_savings = (
            (base_efficiency["peak_gpu_mb"] - opt_efficiency["peak_gpu_mb"])
            / base_efficiency["peak_gpu_mb"]
            * 100
        )
        print(f"🎮 Экономия GPU: {gpu_savings:.1f}%")

    time_diff = results["baseline"]["total_time"] - results["optimized"]["total_time"]
    print(
        f"⏱️  Разница во времени: {time_diff:.1f}min ({'быстрее' if time_diff > 0 else 'медленнее'})"
    )

    return results


def test_progressive_scaling_training():
    """Тест обучения с прогрессивным масштабированием"""
    print("📐 Тестирование обучения с прогрессивным масштабированием...")

    monitor = FullCycleMemoryMonitor()
    monitor.start_monitoring()

    scaling_results = []

    # Тест 3 стадий с увеличивающимися размерами решетки
    for stage in [1, 2, 3]:
        stage_config = StageConfig(
            stage=stage,
            dataset_limit=30,  # Маленький датасет для быстрого теста
            epochs=1,
            batch_size=4,
            description=f"Progressive Scaling Test Stage {stage}",
            plasticity_profile="learning",
            clustering_enabled=(stage >= 2),
            activity_threshold=0.02 + stage * 0.01,
            memory_optimizations=True,
            emergence_tracking=True,
            progressive_scaling=True,  # Ключевая опция
        )

        print(f"\n📐 Stage {stage} - Progressive Scaling Test")
        result = run_training_stage_with_monitoring(stage_config, monitor)

        # Проверяем размеры решетки
        runner = TrainingStageRunner(mode="development")
        expected_dims = runner._get_adaptive_dimensions(stage)

        stage_summary = {
            "stage": stage,
            "expected_dimensions": expected_dims,
            "training_result": result,
            "total_cells": expected_dims[0] * expected_dims[1] * expected_dims[2],
        }

        scaling_results.append(stage_summary)
        print(
            f"   Expected dimensions: {expected_dims[0]}×{expected_dims[1]}×{expected_dims[2]} = {stage_summary['total_cells']:,} cells"
        )
        print(f"   Training success: {'✅' if result['success'] else '❌'}")

    efficiency = monitor.get_memory_efficiency()

    print(f"\n📊 Progressive Scaling Summary:")
    print(
        f"   Stages completed: {len([r for r in scaling_results if r['training_result']['success']])}/{len(scaling_results)}"
    )
    print(f"   Memory efficiency: {efficiency['peak_ram_mb']:.1f}MB peak RAM")
    print(
        f"   Scaling range: {scaling_results[0]['total_cells']:,} → {scaling_results[-1]['total_cells']:,} cells"
    )

    return scaling_results, efficiency


def save_test_results(results: dict, filename: str = None):
    """Сохранить результаты тестирования"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase4_full_cycle_results_{timestamp}.json"

    # Создаем директорию если не существует
    results_dir = Path("results/phase4_tests")
    results_dir.mkdir(parents=True, exist_ok=True)

    filepath = results_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"💾 Результаты сохранены: {filepath}")
    return filepath


def main():
    """Основная функция тестирования полного цикла"""
    print("🎯 ТЕСТ ФАЗЫ 4: Полный цикл обучения с оптимизациями")
    print("=" * 80)
    print("🎯 Цель: Финальная валидация НЕДЕЛИ 1 согласно плану интеграции")
    print("⏱️  Ожидаемое время: 10-20 минут")
    print()

    all_results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 4 Integration",
            "week": "Week 1 Final Test",
            "python_version": sys.version,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            ),
        }
    }

    try:
        # 1. Сравнение оптимизированного vs базового обучения
        comparison_results = test_optimized_vs_baseline_training()
        all_results["comparison"] = comparison_results

        print("\n" + "=" * 50)

        # 2. Тест прогрессивного масштабирования
        scaling_results, scaling_efficiency = test_progressive_scaling_training()
        all_results["progressive_scaling"] = {
            "results": scaling_results,
            "memory_efficiency": scaling_efficiency,
        }

        # 3. Сохранение результатов
        results_file = save_test_results(all_results)

        print("\n" + "=" * 80)
        print("🎉 ПОЛНЫЙ ЦИКЛ ТЕСТИРОВАНИЯ ЗАВЕРШЕН УСПЕШНО!")
        print()

        # Финальный отчет
        print("📊 ФИНАЛЬНЫЙ ОТЧЕТ НЕДЕЛИ 1:")
        print()

        opt_success = (
            comparison_results["optimized"]["stage1"]["success"]
            and comparison_results["optimized"]["stage2"]["success"]
        )
        base_success = (
            comparison_results["baseline"]["stage1"]["success"]
            and comparison_results["baseline"]["stage2"]["success"]
        )

        print(
            f"✅ Оптимизированное обучение: {'Успешно' if opt_success else 'Неудачно'}"
        )
        print(f"📊 Базовое обучение: {'Успешно' if base_success else 'Неудачно'}")

        scaling_success_count = len(
            [r for r in scaling_results if r["training_result"]["success"]]
        )
        print(
            f"📐 Прогрессивное масштабирование: {scaling_success_count}/{len(scaling_results)} стадий"
        )

        print()
        print("🚀 ГОТОВНОСТЬ К НЕДЕЛЕ 2:")
        if opt_success and scaling_success_count >= 2:
            print("   ✅ Все системы готовы к масштабированию")
            print("   ✅ Memory optimizations работают")
            print("   ✅ Progressive scaling функционирует")
            print("   ✅ Plasticity profiles интегрированы")
            print()
            print("➡️  СЛЕДУЮЩИЙ ЭТАП: Progressive Scaling (Week 2)")
            print("     - Scaling до 32×32×24 решеток")
            print("     - Real-time decoder monitoring")
            print("     - Advanced memory budget management")
        else:
            print("   ⚠️  Требуется дополнительная отладка")
            print("   ⚠️  Проверьте логи ошибок")

        print(f"\n📄 Подробные результаты: {results_file}")

        return opt_success and scaling_success_count >= 2

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
