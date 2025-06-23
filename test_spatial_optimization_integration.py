#!/usr/bin/env python3
"""
Тест интеграции spatial optimization с MoE архитектурой
=====================================================

Проверяет что безопасный поиск соседей правильно интегрирован
в MoE Connection Processor.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "new_rebuild"))

import torch
from new_rebuild.core.lattice.spatial_optimization.moe_spatial_optimizer import (
    MoESpatialOptimizer,
)
from new_rebuild.core.moe.moe_processor import MoEConnectionProcessor
from new_rebuild.config.project_config import get_project_config
from new_rebuild.utils.logging import get_logger

logger = get_logger(__name__)


def test_moe_spatial_integration():
    """Тестирует интеграцию spatial optimization в MoE архитектуру"""

    logger.info("🧪 Запуск теста интеграции MoE + Spatial Optimization")

    # Создаем небольшую тестовую решетку
    dimensions = (10, 10, 10)
    state_size = 32
    total_cells = 10 * 10 * 10

    # Принудительно отключаем CUDA для тестирования
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Отключаем CUDA

    # Сбрасываем глобальный device manager для принудительного использования CPU
    from new_rebuild.utils.device_manager import reset_device_manager

    reset_device_manager()

    # Создаем spatial optimizer с принудительным CPU
    moe_spatial_optimizer = MoESpatialOptimizer(dimensions)

    # Создаем MoE processor
    moe_processor = MoEConnectionProcessor(
        state_size=state_size, lattice_dimensions=dimensions
    )

    # Убеждаемся что все на CPU
    device = torch.device("cpu")
    moe_processor = moe_processor.to(device)

    # Создаем тестовые данные (на CPU для отладки)
    current_state = torch.randn(state_size, device="cpu")
    full_lattice_states = torch.randn(total_cells, state_size, device="cpu")

    # Тестируем интеграцию - клетка в центре решетки
    center_cell = 555

    # Вызываем MoE processor с spatial optimizer
    result = moe_processor(
        current_state=current_state,
        neighbor_states=torch.empty(
            0, state_size
        ),  # Пустые - будут найдены через spatial optimizer
        cell_idx=center_cell,
        neighbor_indices=[],  # Пустые - будут найдены через spatial optimizer
        spatial_optimizer=moe_spatial_optimizer,
        full_lattice_states=full_lattice_states,
    )

    # Проверяем результат
    assert "new_state" in result
    assert "neighbor_count" in result

    neighbor_count = result["neighbor_count"]
    logger.info(
        f"   ✅ Центральная клетка {center_cell}: найдено {neighbor_count} соседей"
    )

    # Тестируем граничную клетку
    edge_cell = 0
    result = moe_processor(
        current_state=current_state,
        neighbor_states=torch.empty(0, state_size),
        cell_idx=edge_cell,
        neighbor_indices=[],
        spatial_optimizer=moe_spatial_optimizer,
        full_lattice_states=full_lattice_states,
    )

    neighbor_count = result["neighbor_count"]
    logger.info(f"   ✅ Граничная клетка {edge_cell}: найдено {neighbor_count} соседей")

    # Тестируем fallback без spatial optimizer
    result_fallback = moe_processor(
        current_state=current_state,
        neighbor_states=torch.randn(
            5, state_size, device="cpu"
        ),  # Предустановленные соседи на CPU
        cell_idx=center_cell,
        neighbor_indices=[1, 2, 3, 4, 5],
        # spatial_optimizer не передаем - fallback режим
    )

    fallback_count = result_fallback["neighbor_count"]
    logger.info(f"   ✅ Fallback режим: {fallback_count} соседей")

    logger.info("✅ Интеграция MoE + Spatial Optimization работает корректно!")


if __name__ == "__main__":
    try:
        test_moe_spatial_integration()
        print("🎉 Тест интеграции пройден успешно!")
    except Exception as e:
        logger.error(f"❌ Ошибка в тесте интеграции: {e}")
        raise
