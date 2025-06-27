#!/usr/bin/env python3
"""
Отладочный скрипт для диагностики проблемы с неправильными координатами
в MoE spatial optimization
"""

import sys
import torch
import logging

# Добавляем путь к проекту
sys.path.append(".")

from new_rebuild.config import get_project_config
from new_rebuild.core.moe.moe_connection_processor import MoEConnectionProcessor
from new_rebuild.core.lattice.spatial_optimization.moe_spatial_optimizer import (
    create_moe_spatial_optimizer,
)
from new_rebuild.utils.logging import setup_logging


def debug_moe_coords():
    """Отлаживаем проблему с координатами"""

    # Включаем debug логирование
    setup_logging(debug_mode=True)

    print("🔍 ОТЛАДКА ПРОБЛЕМЫ С КООРДИНАТАМИ MoE")
    print("=" * 60)

    # Малая решетка для отладки
    test_dimensions = (5, 5, 5)
    total_cells = test_dimensions[0] * test_dimensions[1] * test_dimensions[2]

    print(f"📐 Тестовые размеры: {test_dimensions} ({total_cells} клеток)")
    print(f"📊 Допустимые индексы: 0 - {total_cells-1}")
    print(f"📊 Допустимые Z-координаты: 0 - {test_dimensions[2]-1}")

    # Получаем конфигурацию
    config = get_project_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🎯 Устройство: {device}")

    # Создаем MoE processor
    print("\n🛠️ Создание MoE processor...")
    moe_processor = MoEConnectionProcessor(
        state_size=config.gnn_state_size,
        lattice_dimensions=test_dimensions,
        neighbor_count=config.max_neighbors,
        enable_cnf=config.enable_cnf,
    )
    moe_processor.to(device)

    # Создаем spatial optimizer
    print("\n🗂️ Создание spatial optimizer...")
    spatial_optimizer = create_moe_spatial_optimizer(
        dimensions=test_dimensions,
        moe_processor=moe_processor,
        device=device,
    )

    # Создаем состояния
    print(f"\n📊 Создание состояний...")
    states = torch.randn(
        total_cells,
        config.gnn_state_size,
        device=device,
        dtype=torch.float32,
    )

    print(f"   States shape: {states.shape}")
    print(f"   States device: {states.device}")

    # Тестируем обработку ОДНОЙ клетки
    print(f"\n🧪 Тестируем обработку одной клетки (idx=0)...")

    try:
        # Обрабатываем только первую клетку
        current_state = states[0].unsqueeze(0)
        empty_neighbors = torch.empty(1, 0, states.shape[1], device=device)

        print(f"   Current state shape: {current_state.shape}")
        print(f"   Calling MoE processor...")

        result = moe_processor(
            current_state=current_state,
            neighbor_states=empty_neighbors,
            cell_idx=0,  # Тестируем с валидным индексом
            neighbor_indices=[],
            spatial_optimizer=spatial_optimizer,
            full_lattice_states=states,
        )

        print(f"✅ MoE processor работает для cell_idx=0")
        print(f"   Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"   Result keys: {result.keys()}")

    except Exception as e:
        print(f"❌ Ошибка в MoE processor: {e}")
        import traceback

        traceback.print_exc()

    # Тестируем spatial optimizer напрямую
    print(f"\n🧪 Тестируем spatial optimizer напрямую...")
    try:
        neighbors = spatial_optimizer.find_neighbors_optimized(
            coords=(0, 0, 0),  # Тестируем с валидными координатами
            radius=config.calculate_adaptive_radius(),
        )
        print(f"✅ Spatial optimizer работает для (0,0,0)")
        print(f"   Найдено соседей: {len(neighbors)}")
        print(f"   Первые 10 соседей: {neighbors[:10]}")

    except Exception as e:
        print(f"❌ Ошибка в spatial optimizer: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_moe_coords()
