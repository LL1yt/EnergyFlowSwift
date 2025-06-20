#!/usr/bin/env python3
"""
Golden Middle GatedMLPCell - компромисс между качеством и эффективностью
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from core.cell_prototype.architectures.gmlp_cell_minimal import MinimalGatedMLPCell
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def test_golden_middle_configs():
    """Тестирование конфигураций в диапазоне 15-20K параметров"""

    print("🏆 GOLDEN MIDDLE OPTIMIZATION (15-20K параметров)")
    print("=" * 65)

    golden_configs = [
        # Компромиссные конфигурации с разумным bottleneck
        {
            "bottleneck_dim": 12,
            "hidden_dim": 28,
            "external_input_size": 4,
            "target": "~20K",
        },
        {
            "bottleneck_dim": 10,
            "hidden_dim": 24,
            "external_input_size": 3,
            "target": "~17K",
        },
        {
            "bottleneck_dim": 8,
            "hidden_dim": 20,
            "external_input_size": 2,
            "target": "~15K",
        },
        # Текущие reference points
        {
            "bottleneck_dim": 16,
            "hidden_dim": 32,
            "external_input_size": 4,
            "target": "23K baseline",
        },
        {
            "bottleneck_dim": 6,
            "hidden_dim": 20,
            "external_input_size": 2,
            "target": "10K aggressive",
        },
    ]

    print("Config | bottleneck | hidden | ext | params | compression | time | target")
    print("-" * 75)

    results = []

    for i, config in enumerate(golden_configs):
        base_config = {
            "state_size": 36,
            "neighbor_count": 26,
            "target_params": 15000,  # Средний target для анализа
        }

        # Обновляем конфигурацией
        test_config = base_config.copy()
        test_config.update({k: v for k, v in config.items() if k != "target"})

        try:
            cell = MinimalGatedMLPCell(**test_config)
            actual_params = sum(p.numel() for p in cell.parameters())

            # Тестируем производительность
            batch_size = 4
            neighbor_states = torch.randn(batch_size, 26, 36)
            own_state = torch.randn(batch_size, 36)
            connection_weights = torch.randn(batch_size, 26)
            external_input = torch.randn(batch_size, config["external_input_size"])

            import time

            start_time = time.time()
            output = cell(
                neighbor_states, own_state, connection_weights, external_input
            )
            forward_time = (time.time() - start_time) * 1000

            # Рассчитываем compression ratio
            total_input_size = (
                26 * 36 + 36 + config["external_input_size"]
            )  # neighbor + own + external
            compression_ratio = total_input_size / config["bottleneck_dim"]

            results.append(
                {
                    "config": test_config,
                    "params": actual_params,
                    "time": forward_time,
                    "compression": compression_ratio,
                    "target_name": config["target"],
                }
            )

            print(
                f"  {i+1:2d}   |     {config['bottleneck_dim']:2d}     |   {config['hidden_dim']:2d}   |  {config['external_input_size']:1d}  | {actual_params:6,d} |    {compression_ratio:4.0f}x    | {forward_time:5.1f}ms | {config['target']}"
            )

        except Exception as e:
            print(f"  {i+1:2d}   | ERROR: {e}")

    return results


def analyze_information_capacity(results):
    """Анализ информационной емкости разных конфигураций"""

    print(f"\n🔍 АНАЛИЗ ИНФОРМАЦИОННОЙ ЕМКОСТИ")
    print("=" * 50)

    for result in results:
        config = result["config"]

        print(f"\n📊 {result['target_name']} ({result['params']:,} params):")

        # Входная информация
        neighbor_signals = 26 * 36  # 936
        own_state = 36
        external_input = config["external_input_size"]
        total_input = neighbor_signals + own_state + external_input

        # Bottleneck capacity
        bottleneck_capacity = config["bottleneck_dim"]
        compression_ratio = total_input / bottleneck_capacity

        # Information density
        info_density = result["params"] / total_input  # params per input signal

        print(
            f"   Input signals: {total_input} ({neighbor_signals} neighbors + {own_state} own + {external_input} external)"
        )
        print(f"   Bottleneck capacity: {bottleneck_capacity}")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        print(f"   Information density: {info_density:.1f} params/signal")
        print(f"   Forward time: {result['time']:.1f}ms")

        # Качественная оценка
        if compression_ratio < 50:
            quality = "🟢 Excellent"
        elif compression_ratio < 100:
            quality = "🟡 Good"
        elif compression_ratio < 150:
            quality = "🟠 Acceptable"
        else:
            quality = "🔴 Aggressive"

        print(f"   Information quality: {quality}")


def recommend_optimal_config(results):
    """Рекомендация оптимальной конфигурации"""

    print(f"\n🎯 РЕКОМЕНДАЦИЯ ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ")
    print("=" * 55)

    # Критерии оценки
    for result in results:
        compression = result["compression"]
        params = result["params"]
        time = result["time"]

        # Scoring system
        compression_score = max(0, 100 - compression)  # Меньше compression = лучше
        efficiency_score = max(0, 100 - (params / 250))  # Меньше params = лучше
        speed_score = max(0, 100 - time * 5)  # Быстрее = лучше

        total_score = (
            compression_score * 0.5 + efficiency_score * 0.3 + speed_score * 0.2
        )

        result["score"] = total_score

        print(
            f"{result['target_name']:15s}: compression={compression:4.0f}x, "
            f"params={params:6,d}, time={time:5.1f}ms, score={total_score:5.1f}"
        )

    # Найти лучший
    best_result = max(results, key=lambda x: x["score"])

    print(f"\n🏆 РЕКОМЕНДУЕМАЯ КОНФИГУРАЦИЯ: {best_result['target_name']}")
    print(f"   Parameters: {best_result['params']:,}")
    print(f"   Compression: {best_result['compression']:.0f}x")
    print(f"   Forward time: {best_result['time']:.1f}ms")
    print(f"   Score: {best_result['score']:.1f}/100")

    print(f"\n⚖️ ОБОСНОВАНИЕ:")
    if best_result["compression"] < 100:
        print(f"   ✅ Разумное сжатие информации ({best_result['compression']:.0f}x)")
    else:
        print(f"   ⚠️ Агрессивное сжатие ({best_result['compression']:.0f}x)")

    if best_result["params"] < 25000:
        print(f"   ✅ Эффективное использование параметров")
    else:
        print(f"   ⚠️ Высокое потребление параметров")

    if best_result["time"] < 20:
        print(f"   ✅ Быстрая обработка")
    else:
        print(f"   ⚠️ Медленная обработка")

    return best_result


if __name__ == "__main__":
    # Тестируем golden middle конфигурации
    results = test_golden_middle_configs()

    if results:
        # Анализируем информационную емкость
        analyze_information_capacity(results)

        # Рекомендуем оптимальную
        best_config = recommend_optimal_config(results)

        print(f"\n📋 ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ:")
        config = best_config["config"]
        print(f"   bottleneck_dim: {config['bottleneck_dim']}")
        print(f"   hidden_dim: {config['hidden_dim']}")
        print(f"   external_input_size: {config['external_input_size']}")
        print(f"   total_parameters: {best_config['params']:,}")

        print(
            f"\n💡 ВЫВОД: {'23K baseline лучше 10K aggressive' if best_config['params'] > 15000 else '10K оптимизация достаточна'}"
        )
