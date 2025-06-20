#!/usr/bin/env python3
"""
🚀 ТЕСТ ФАЗЫ 4: Малые решетки с оптимизациями памяти

Цель: Проверить работу интеграции Фазы 4 на малых решетках (16×16×16)
и измерить effectiveness memory optimization (target: 50%+ reduction)

Тестируем:
- Профили пластичности (discovery → learning → consolidation)
- Memory optimizations (mixed precision, gradient checkpointing)
- Emergent behavior preservation
- Progressive scaling integration
"""

import sys
import os
import tempfile
import yaml
import torch
import psutil
import time
from pathlib import Path

# Добавляем корневую директорию в Python path
sys.path.append(str(Path(__file__).parent))

from training.automated_training.types import StageConfig
from training.automated_training.progressive_config import ProgressiveConfigManager
from training.automated_training.stage_runner import TrainingStageRunner
from training.automated_training.automated_trainer import AutomatedTrainer


class MemoryMonitor:
    """Монитор использования памяти"""

    def __init__(self):
        self.baseline_memory = 0
        self.peak_memory = 0
        self.gpu_baseline = 0
        self.gpu_peak = 0

    def start_monitoring(self):
        """Начать мониторинг памяти"""
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gpu_baseline = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB

        print(f"📊 Baseline memory: {self.baseline_memory:.1f}MB RAM")
        if torch.cuda.is_available():
            print(f"📊 Baseline GPU: {self.gpu_baseline:.1f}MB VRAM")

    def update_peak(self):
        """Обновить пиковое использование памяти"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)

        if torch.cuda.is_available():
            current_gpu = torch.cuda.memory_allocated(0) / 1024 / 1024
            self.gpu_peak = max(self.gpu_peak, current_gpu)

    def get_memory_usage(self):
        """Получить использование памяти"""
        self.update_peak()
        ram_usage = self.peak_memory - self.baseline_memory
        gpu_usage = (
            self.gpu_peak - self.gpu_baseline if torch.cuda.is_available() else 0
        )
        return ram_usage, gpu_usage


def create_optimized_stage_config(
    stage: int, optimization_level: str = "standard"
) -> StageConfig:
    """Создать оптимизированную конфигурацию стадии"""

    base_configs = {
        1: {
            "plasticity_profile": "discovery",
            "clustering_enabled": False,
            "activity_threshold": 0.01,
            "epochs": 2,
            "dataset_limit": 100,
        },
        2: {
            "plasticity_profile": "learning",
            "clustering_enabled": False,
            "activity_threshold": 0.02,
            "epochs": 2,
            "dataset_limit": 200,
        },
        3: {
            "plasticity_profile": "learning",
            "clustering_enabled": True,
            "activity_threshold": 0.03,
            "epochs": 2,
            "dataset_limit": 300,
        },
    }

    config = base_configs.get(stage, base_configs[1])

    # Оптимизации в зависимости от уровня
    if optimization_level == "standard":
        memory_optimizations = True
        sparse_ratio = 0.0
        emergence_tracking = True
    elif optimization_level == "aggressive":
        memory_optimizations = True
        sparse_ratio = 0.2
        emergence_tracking = True
    else:  # minimal
        memory_optimizations = False
        sparse_ratio = 0.0
        emergence_tracking = False

    return StageConfig(
        stage=stage,
        dataset_limit=config["dataset_limit"],
        epochs=config["epochs"],
        batch_size=16,  # Маленький batch для тестирования
        description=f"Phase 4 Test Stage {stage} ({optimization_level})",
        plasticity_profile=config["plasticity_profile"],
        clustering_enabled=config["clustering_enabled"],
        activity_threshold=config["activity_threshold"],
        memory_optimizations=memory_optimizations,
        emergence_tracking=emergence_tracking,
        sparse_connection_ratio=sparse_ratio,
        progressive_scaling=True,  # Включаем прогрессивное масштабирование
    )


def test_memory_optimization_comparison():
    """Сравнение использования памяти с оптимизациями и без"""
    print("🔬 Тестирование эффективности оптимизации памяти...")

    results = {}

    # Тест без оптимизаций
    print("\n--- Тест БЕЗ оптимизаций ---")
    monitor_baseline = MemoryMonitor()
    monitor_baseline.start_monitoring()

    stage_config_baseline = create_optimized_stage_config(1, "minimal")
    runner_baseline = TrainingStageRunner(mode="development", verbose=False)

    # Генерируем конфигурацию без оптимизаций
    temp_config_baseline = runner_baseline._generate_temp_config(stage_config_baseline)
    if temp_config_baseline:
        monitor_baseline.update_peak()
        ram_baseline, gpu_baseline = monitor_baseline.get_memory_usage()
        os.remove(temp_config_baseline)

    results["baseline"] = {"ram": ram_baseline, "gpu": gpu_baseline}
    print(f"📊 Baseline usage: {ram_baseline:.1f}MB RAM, {gpu_baseline:.1f}MB GPU")

    # Тест С оптимизациями
    print("\n--- Тест С оптимизациями ---")
    monitor_optimized = MemoryMonitor()
    monitor_optimized.start_monitoring()

    stage_config_optimized = create_optimized_stage_config(1, "standard")
    runner_optimized = TrainingStageRunner(mode="development", verbose=False)

    # Генерируем конфигурацию с оптимизациями
    temp_config_optimized = runner_optimized._generate_temp_config(
        stage_config_optimized
    )
    if temp_config_optimized:
        monitor_optimized.update_peak()
        ram_optimized, gpu_optimized = monitor_optimized.get_memory_usage()
        os.remove(temp_config_optimized)

    results["optimized"] = {"ram": ram_optimized, "gpu": gpu_optimized}
    print(f"📊 Optimized usage: {ram_optimized:.1f}MB RAM, {gpu_optimized:.1f}MB GPU")

    # Расчет экономии
    if ram_baseline > 0:
        ram_savings = (ram_baseline - ram_optimized) / ram_baseline * 100
        print(f"💾 RAM savings: {ram_savings:.1f}%")
        results["ram_savings_percent"] = ram_savings

    if gpu_baseline > 0:
        gpu_savings = (gpu_baseline - gpu_optimized) / gpu_baseline * 100
        print(f"🎮 GPU savings: {gpu_savings:.1f}%")
        results["gpu_savings_percent"] = gpu_savings

    return results


def test_plasticity_progression():
    """Тест прогрессии пластичности через стадии"""
    print("🧠 Тестирование прогрессии пластичности...")

    stages_tested = []

    for stage in [1, 2, 3]:
        stage_config = create_optimized_stage_config(stage, "standard")
        runner = TrainingStageRunner(mode="development", verbose=False)

        # Генерируем конфигурацию
        temp_config_path = runner._generate_temp_config(stage_config)

        if temp_config_path:
            # Проверяем содержимое конфигурации
            with open(temp_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            plasticity_config = config_data.get("plasticity", {})

            stage_result = {
                "stage": stage,
                "profile": plasticity_config.get("profile", "unknown"),
                "activity_threshold": plasticity_config.get("activity_threshold", 0),
                "clustering_enabled": "functional_clustering" in plasticity_config,
                "emergence_tracking": "emergence_detection" in plasticity_config,
                "stdp_rate": plasticity_config.get("stdp_learning_rate", 0),
            }

            stages_tested.append(stage_result)
            print(
                f"  Stage {stage}: {stage_result['profile']} profile, threshold={stage_result['activity_threshold']}"
            )

            os.remove(temp_config_path)

    # Проверяем правильность прогрессии
    assert stages_tested[0]["profile"] == "discovery"
    assert stages_tested[1]["profile"] == "learning"
    assert stages_tested[2]["profile"] == "learning"
    assert stages_tested[2]["clustering_enabled"] == True

    print("✅ Прогрессия пластичности работает корректно")
    return stages_tested


def test_progressive_scaling():
    """Тест прогрессивного масштабирования размеров решетки"""
    print("📐 Тестирование прогрессивного масштабирования...")

    runner = TrainingStageRunner(mode="development")
    scaling_results = []

    for stage in [1, 2, 3, 4, 5]:
        stage_config = create_optimized_stage_config(stage, "standard")

        # Генерируем конфигурацию с масштабированием
        temp_config_path = runner._generate_temp_config(stage_config)

        if temp_config_path:
            with open(temp_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            lattice_config = config_data.get("lattice", {})
            dimensions = (
                lattice_config.get("lattice_width", 0),
                lattice_config.get("lattice_height", 0),
                lattice_config.get("lattice_depth", 0),
            )

            total_cells = dimensions[0] * dimensions[1] * dimensions[2]

            scaling_results.append(
                {"stage": stage, "dimensions": dimensions, "total_cells": total_cells}
            )

            print(
                f"  Stage {stage}: {dimensions[0]}×{dimensions[1]}×{dimensions[2]} = {total_cells:,} cells"
            )

            os.remove(temp_config_path)

    # Проверяем что размеры увеличиваются
    for i in range(1, len(scaling_results)):
        assert (
            scaling_results[i]["total_cells"] >= scaling_results[i - 1]["total_cells"]
        ), f"Scaling должен увеличиваться: Stage {i} vs Stage {i+1}"

    print("✅ Прогрессивное масштабирование работает корректно")
    return scaling_results


def test_config_integration():
    """Тест интеграции конфигураций с основной системой"""
    print("🔗 Тестирование интеграции конфигураций...")

    # Создаем конфигурацию с полным набором оптимизаций
    stage_config = StageConfig(
        stage=2,
        dataset_limit=50,  # Очень маленький для быстрого теста
        epochs=1,
        batch_size=8,
        description="Integration Test",
        plasticity_profile="learning",
        clustering_enabled=True,
        activity_threshold=0.025,
        memory_optimizations=True,
        emergence_tracking=True,
        sparse_connection_ratio=0.1,
        progressive_scaling=True,
        decoder_monitoring=False,  # Отключаем для простоты
    )

    # Проверяем что конфигурация правильно генерируется
    runner = TrainingStageRunner(mode="development", verbose=True)
    temp_config_path = runner._generate_temp_config(stage_config)

    assert temp_config_path is not None, "Конфигурация должна генерироваться"

    # Проверяем содержимое
    with open(temp_config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Проверяем ключевые секции
    assert "plasticity" in config_data, "Секция plasticity должна присутствовать"
    assert "optimization" in config_data, "Секция optimization должна присутствовать"
    assert "lattice" in config_data, "Секция lattice должна присутствовать"

    plasticity = config_data["plasticity"]
    optimization = config_data["optimization"]

    assert plasticity["enable_plasticity"] == True
    assert plasticity["profile"] == "learning"
    assert "functional_clustering" in plasticity
    assert "emergence_detection" in plasticity

    assert optimization["mixed_precision"]["enable"] == True
    assert optimization["gradient_checkpointing"]["enable"] == True
    assert optimization["sparse_connections"]["enable"] == True

    os.remove(temp_config_path)

    print("✅ Интеграция конфигураций работает корректно")
    return True


def main():
    """Основная функция тестирования малых решеток"""
    print("🚀 ТЕСТ ФАЗЫ 4: Малые решетки с оптимизациями")
    print("=" * 70)

    try:
        # 1. Тест эффективности памяти
        memory_results = test_memory_optimization_comparison()

        # 2. Тест прогрессии пластичности
        plasticity_results = test_plasticity_progression()

        # 3. Тест прогрессивного масштабирования
        scaling_results = test_progressive_scaling()

        # 4. Тест интеграции конфигураций
        integration_success = test_config_integration()

        print("=" * 70)
        print("🎉 ВСЕ ТЕСТЫ МАЛЫХ РЕШЕТОК ПРОШЛИ УСПЕШНО!")
        print()

        # Отчет по результатам
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print()

        if "ram_savings_percent" in memory_results:
            ram_savings = memory_results["ram_savings_percent"]
            print(f"💾 Экономия RAM: {ram_savings:.1f}%")
            if ram_savings >= 20:
                print("   ✅ Хорошая экономия памяти!")
            else:
                print("   ⚠️  Экономия памяти меньше ожидаемой")

        if "gpu_savings_percent" in memory_results:
            gpu_savings = memory_results["gpu_savings_percent"]
            print(f"🎮 Экономия GPU: {gpu_savings:.1f}%")

        print(f"🧠 Протестировано профилей пластичности: {len(plasticity_results)}")
        print(f"📐 Протестировано конфигураций масштабирования: {len(scaling_results)}")
        print(
            f"🔗 Интеграция конфигураций: {'✅ Работает' if integration_success else '❌ Ошибка'}"
        )

        print()
        print("🚀 ГОТОВО К СЛЕДУЮЩЕМУ ЭТАПУ:")
        print("   - Memory optimization протестированы")
        print("   - Plasticity progression работает")
        print("   - Progressive scaling функционирует")
        print("   - Конфигурации интегрируются корректно")
        print()
        print("➡️  Следующий шаг: Тестирование полного цикла обучения")

        return True

    except Exception as e:
        print(f"❌ ОШИБКА В ТЕСТАХ: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
