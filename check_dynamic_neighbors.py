#!/usr/bin/env python3
"""
Проверка динамических соседей в 3D Cellular Neural Network
=========================================================

Анализирует:
- Максимальный радиус для заданной решетки
- Распределение на local/functional/distant тиры  
- Количество соседей в каждом тире
- Проверяет что legacy 6/26 соседей нигде не осталось
"""

import torch
import math
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple

from new_rebuild.config import SimpleProjectConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_3d_distance(pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
    """Вычисляет евклидово расстояние между двумя 3D позициями"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


def generate_lattice_positions(dimensions: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Генерирует все позиции в 3D решетке"""
    positions = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                positions.append((x, y, z))
    return positions


def analyze_neighbor_distribution(config: SimpleProjectConfig) -> Dict[str, Any]:
    """Анализирует распределение соседей для заданной конфигурации"""
    
    dimensions = config.lattice.dimensions
    max_radius = config.lattice.max_radius
    local_threshold = config.lattice.local_distance_threshold
    functional_threshold = config.lattice.functional_distance_threshold
    distant_threshold = config.lattice.distant_distance_threshold
    
    logger.info(f"📏 Analyzing lattice: {dimensions}")
    logger.info(f"🎯 Max radius: {max_radius:.2f}")
    logger.info(f"🔵 Local threshold: {local_threshold:.2f}")
    logger.info(f"🟡 Functional threshold: {functional_threshold:.2f}")
    logger.info(f"🔴 Distant threshold: {distant_threshold:.2f}")
    
    # Генерируем все позиции
    positions = generate_lattice_positions(dimensions)
    total_cells = len(positions)
    
    # Анализируем несколько образцовых клеток
    sample_positions = [
        (0, 0, 0),                              # Угол
        (dimensions[0]//2, dimensions[1]//2, dimensions[2]//2),  # Центр
        (dimensions[0]-1, dimensions[1]-1, dimensions[2]-1),     # Противоположный угол
    ]
    
    analysis_results = {
        "lattice_info": {
            "dimensions": dimensions,
            "total_cells": total_cells,
            "max_radius": max_radius,
            "thresholds": {
                "local": local_threshold,
                "functional": functional_threshold,
                "distant": distant_threshold
            }
        },
        "sample_analysis": []
    }
    
    for sample_pos in sample_positions:
        if sample_pos[0] < dimensions[0] and sample_pos[1] < dimensions[1] and sample_pos[2] < dimensions[2]:
            neighbor_analysis = analyze_cell_neighbors(sample_pos, positions, config)
            analysis_results["sample_analysis"].append(neighbor_analysis)
    
    return analysis_results


def analyze_cell_neighbors(
    cell_pos: Tuple[int, int, int], 
    all_positions: List[Tuple[int, int, int]], 
    config: SimpleProjectConfig
) -> Dict[str, Any]:
    """Анализирует соседей для одной клетки"""
    
    local_threshold = config.lattice.local_distance_threshold
    functional_threshold = config.lattice.functional_distance_threshold
    
    neighbors = {
        "local": [],
        "functional": [], 
        "distant": []
    }
    
    distances = []
    
    for pos in all_positions:
        if pos == cell_pos:
            continue
            
        distance = calculate_3d_distance(cell_pos, pos)
        distances.append(distance)
        
        if distance <= local_threshold:
            neighbors["local"].append((pos, distance))
        elif distance <= functional_threshold:
            neighbors["functional"].append((pos, distance))
        else:
            neighbors["distant"].append((pos, distance))
    
    # Сортируем по расстоянию
    for tier in neighbors.values():
        tier.sort(key=lambda x: x[1])
    
    return {
        "cell_position": cell_pos,
        "total_neighbors": len(distances),
        "tier_counts": {
            "local": len(neighbors["local"]),
            "functional": len(neighbors["functional"]),
            "distant": len(neighbors["distant"])
        },
        "tier_percentages": {
            "local": len(neighbors["local"]) / len(distances) * 100 if distances else 0,
            "functional": len(neighbors["functional"]) / len(distances) * 100 if distances else 0,
            "distant": len(neighbors["distant"]) / len(distances) * 100 if distances else 0
        },
        "closest_neighbors": {
            "local": neighbors["local"][:5],  # Ближайшие 5 в каждом тире
            "functional": neighbors["functional"][:5],
            "distant": neighbors["distant"][:5]
        },
        "distance_stats": {
            "min": min(distances) if distances else 0,
            "max": max(distances) if distances else 0,
            "avg": sum(distances) / len(distances) if distances else 0
        }
    }


def check_legacy_neighbor_counts(config: SimpleProjectConfig) -> Dict[str, Any]:
    """Проверяет что нигде не осталось legacy значений 6/26 соседей"""
    
    issues = []
    warnings = []
    
    # Проверяем конфигурацию
    if hasattr(config.model, 'neighbor_count'):
        if config.model.neighbor_count == 6:
            issues.append("❌ Found legacy 6-connectivity in model.neighbor_count")
        elif config.model.neighbor_count == 26:
            issues.append("❌ Found legacy 26-connectivity in model.neighbor_count")
        elif config.model.neighbor_count == -1:
            warnings.append("✅ Dynamic neighbor count enabled (neighbor_count = -1)")
        else:
            warnings.append(f"ℹ️ Static neighbor count: {config.model.neighbor_count}")
    
    # Проверяем neighbor settings
    if hasattr(config, 'neighbors') and config.neighbors:
        if hasattr(config.neighbors, 'dynamic_count'):
            if config.neighbors.dynamic_count:
                warnings.append("✅ Dynamic count enabled in NeighborSettings")
            else:
                issues.append("❌ Dynamic count disabled in NeighborSettings")
        
        if hasattr(config.neighbors, 'base_neighbor_count'):
            if config.neighbors.base_neighbor_count in [6, 26]:
                issues.append(f"❌ Found legacy base_neighbor_count: {config.neighbors.base_neighbor_count}")
    
    return {
        "issues": issues,
        "warnings": warnings,
        "legacy_check_passed": len(issues) == 0
    }


def check_lattice_size_consistency(config: SimpleProjectConfig) -> Dict[str, Any]:
    """Проверяет консистентность размера решетки"""
    
    lattice_dims = config.lattice.dimensions
    target_embedding_dim = config.training_embedding.target_embedding_dim
    total_cells = lattice_dims[0] * lattice_dims[1] * lattice_dims[2]
    
    issues = []
    
    # Проверяем соответствие размеров
    expected_target_dim = total_cells // 8  # Примерная формула
    if abs(target_embedding_dim - expected_target_dim) > 10:
        issues.append(f"⚠️ target_embedding_dim ({target_embedding_dim}) may not match lattice size ({total_cells} cells)")
    
    return {
        "lattice_dimensions": lattice_dims,
        "total_cells": total_cells,
        "target_embedding_dim": target_embedding_dim,
        "expected_target_dim": expected_target_dim,
        "consistency_issues": issues,
        "dimensions_consistent": len(issues) == 0
    }


def main():
    """Главная функция проверки динамических соседей"""
    
    print("🧪 DYNAMIC NEIGHBORS ANALYSIS")
    print("=" * 50)
    
    # Загружаем центральную конфигурацию
    logger.info("⚙️ Loading central configuration...")
    config = SimpleProjectConfig()
    
    # 1. Проверяем legacy значения
    print("\n🔍 Checking for legacy neighbor counts...")
    legacy_check = check_legacy_neighbor_counts(config)
    
    for warning in legacy_check["warnings"]:
        print(f"   {warning}")
    for issue in legacy_check["issues"]:
        print(f"   {issue}")
    
    if legacy_check["legacy_check_passed"]:
        print("   ✅ No legacy neighbor counts found")
    else:
        print("   ❌ Legacy neighbor counts detected!")
    
    # 2. Проверяем консистентность размеров
    print("\n📏 Checking lattice size consistency...")
    size_check = check_lattice_size_consistency(config)
    
    print(f"   📐 Lattice: {size_check['lattice_dimensions']} ({size_check['total_cells']} cells)")
    print(f"   🎯 Target embedding dim: {size_check['target_embedding_dim']}")
    print(f"   📊 Expected target dim: ~{size_check['expected_target_dim']}")
    
    for issue in size_check["consistency_issues"]:
        print(f"   {issue}")
    
    if size_check["dimensions_consistent"]:
        print("   ✅ Dimensions are consistent")
    
    # 3. Анализируем распределение соседей
    print(f"\n🎯 Analyzing neighbor distribution for {config.lattice.dimensions}...")
    neighbor_analysis = analyze_neighbor_distribution(config)
    
    print(f"\n📊 NEIGHBOR DISTRIBUTION ANALYSIS:")
    print(f"   📏 Lattice: {neighbor_analysis['lattice_info']['dimensions']}")
    print(f"   🔥 Total cells: {neighbor_analysis['lattice_info']['total_cells']}")
    print(f"   📐 Max radius: {neighbor_analysis['lattice_info']['max_radius']:.2f}")
    
    thresholds = neighbor_analysis['lattice_info']['thresholds']
    print(f"   🔵 Local tier: 0 → {thresholds['local']:.2f}")
    print(f"   🟡 Functional tier: {thresholds['local']:.2f} → {thresholds['functional']:.2f}")
    print(f"   🔴 Distant tier: {thresholds['functional']:.2f} → {thresholds['distant']:.2f}")
    
    # Показываем анализ образцовых клеток
    for i, sample in enumerate(neighbor_analysis['sample_analysis']):
        pos = sample['cell_position']
        counts = sample['tier_counts']
        percentages = sample['tier_percentages']
        
        if i == 0:
            location = "Corner"
        elif i == 1:
            location = "Center"
        else:
            location = "Opposite Corner"
        
        print(f"\n   📍 {location} cell {pos}:")
        print(f"     Total neighbors: {sample['total_neighbors']}")
        print(f"     🔵 Local: {counts['local']} ({percentages['local']:.1f}%)")
        print(f"     🟡 Functional: {counts['functional']} ({percentages['functional']:.1f}%)")
        print(f"     🔴 Distant: {counts['distant']} ({percentages['distant']:.1f}%)")
        
        stats = sample['distance_stats']
        print(f"     📊 Distance range: {stats['min']:.2f} → {stats['max']:.2f} (avg: {stats['avg']:.2f})")
    
    # Сохраняем результаты
    results = {
        "config_check": {
            "legacy_check": legacy_check,
            "size_check": size_check
        },
        "neighbor_analysis": neighbor_analysis,
        "analysis_timestamp": "2025-01-01T00:00:00"  # Placeholder
    }
    
    results_file = Path("dynamic_neighbors_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Финальная оценка
    print(f"\n🏆 FINAL ASSESSMENT:")
    
    all_checks_passed = (
        legacy_check["legacy_check_passed"] and
        size_check["dimensions_consistent"]
    )
    
    if all_checks_passed:
        print("✅ All dynamic neighbor checks passed!")
        print("🚀 System ready for dynamic neighbor computation")
    else:
        print("⚠️ Some issues found - review above")
    
    # Проверяем что тестовый размер (4,4,4) это нормально
    test_dims = neighbor_analysis['sample_analysis'][0] if neighbor_analysis['sample_analysis'] else None
    if test_dims and config.lattice.dimensions != (8, 8, 8):
        print(f"\nℹ️ NOTE: Current lattice size is {config.lattice.dimensions}")
        print("This may be expected for testing purposes")
        print("For real training, ensure lattice.dimensions = (8, 8, 8)")


if __name__ == "__main__":
    main()