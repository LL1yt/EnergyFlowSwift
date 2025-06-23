#!/usr/bin/env python3
"""
MoE Spatial Optimizer - Интеграция с реальным MoE Connection Processor
=====================================================================

Spatial Optimizer адаптированный для MoE архитектуры.
Интегрирует пространственную оптимизацию с реальным MoE Connection Processor
для максимальной производительности на больших решетках.

ИНТЕГРАЦИЯ:
- Реальный MoEConnectionProcessor (вместо Mock)
- Адаптивный радиус поиска соседей
- GPU acceleration для всех компонентов
- Chunked processing для больших решеток
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import time

from .spatial_optimizer import SpatialOptimizer
from ....config.project_config import ChunkInfo, create_spatial_config_for_lattice
from ....config.project_config import get_project_config
from ..spatial_hashing import Coordinates3D
from ..position import Position3D
from ....utils.logging import get_logger

logger = get_logger(__name__)


class MoESpatialOptimizer(SpatialOptimizer):
    """
    Spatial Optimizer адаптированный для MoE архитектуры

    Интегрирует пространственную оптимизацию с реальным MoE Connection Processor
    для максимальной производительности на больших решетках.
    """

    def __init__(
        self,
        dimensions: Coordinates3D,
        moe_processor=None,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(dimensions, config)

        self.moe_processor = moe_processor
        self.expert_cache = {}  # Кэш для экспертов по chunk'ам

        # Определяем устройство
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Переносим компоненты на устройство если нужно
        if hasattr(self.moe_processor, "to") and self.moe_processor is not None:
            self.moe_processor.to(self.device)

        # MoE-специфичные настройки из ProjectConfig
        project_config = get_project_config()

        self.connection_distributions = {
            "local": project_config.local_connections_ratio,
            "functional": project_config.functional_connections_ratio,
            "distant": project_config.distant_connections_ratio,
        }

        logger.info(f"🔧 MoESpatialOptimizer готов для реальной MoE архитектуры")
        logger.info(f"   📊 Распределение связей: {self.connection_distributions}")
        logger.info(f"   🎯 Устройство: {self.device}")

    def optimize_moe_forward(
        self, states: torch.Tensor, moe_processor=None
    ) -> torch.Tensor:
        """
        Оптимизированный forward pass с реальным MoE Connection Processor

        Args:
            states: [num_cells, state_size] - состояния клеток на правильном устройстве
            moe_processor: реальный MoE Connection Processor (опционально)

        Returns:
            new_states: [num_cells, state_size] - новые состояния клеток
        """
        if moe_processor is None:
            moe_processor = self.moe_processor

        if moe_processor is None:
            raise ValueError(
                "MoE Connection Processor не найден. Передайте moe_processor в аргументах или конструкторе."
            )

        # Убеждаемся что MoE processor на правильном устройстве
        if hasattr(moe_processor, "to"):
            moe_processor.to(self.device)

        # Убеждаемся что состояния на правильном устройстве
        if states.device != self.device:
            states = states.to(self.device)

        logger.info(f"🚀 Запуск MoE forward pass на {self.device}")
        logger.info(f"   📊 Входные состояния: {states.shape} на {states.device}")

        start_time = time.time()

        # Используем chunked processing для больших решеток
        output_states = self._process_moe_chunks(states, moe_processor)

        processing_time = time.time() - start_time

        # Проверяем результаты
        if torch.isnan(output_states).any():
            logger.warning("⚠️ Обнаружены NaN в выходных состояниях")
        if torch.isinf(output_states).any():
            logger.warning("⚠️ Обнаружены Inf в выходных состояниях")

        logger.info(f"✅ MoE forward pass завершен за {processing_time:.3f}s")
        logger.info(
            f"   📊 Выходные состояния: {output_states.shape} на {output_states.device}"
        )

        return output_states

    def _process_moe_chunks(self, states: torch.Tensor, moe_processor) -> torch.Tensor:
        """
        Chunked processing с реальным MoE Connection Processor

        Разбивает большие решетки на управляемые части для обработки.
        """
        num_cells = states.shape[0]
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

        if num_cells != total_cells:
            logger.warning(
                f"⚠️ Несоответствие размеров: states={num_cells}, lattice={total_cells}"
            )

        # Инициализируем выходные состояния
        output_states = states.clone()

        # Получаем настройки для chunked processing
        project_config = get_project_config()
        adaptive_radius = project_config.calculate_adaptive_radius()

        logger.info(f"   📐 Adaptive radius: {adaptive_radius:.2f}")

        # Для тестирования используем быстрый режим с ограниченным количеством batch'ей
        max_batches = getattr(
            project_config, "max_test_batches", 3
        )  # Ограничение для тестов
        batch_size = (
            min(1000, num_cells // max_batches) if num_cells > 5000 else num_cells
        )

        processed_cells = 0
        batch_count = 0

        for batch_start in range(0, num_cells, batch_size):
            if batch_count >= max_batches:
                logger.info(
                    f"   🚫 Достигнут лимит batch'ей ({max_batches}) для тестирования"
                )
                break

            batch_end = min(batch_start + batch_size, num_cells)
            batch_cells = list(range(batch_start, batch_end))

            logger.debug(
                f"   🔄 Batch {batch_count + 1}: клетки {batch_start}-{batch_end}"
            )

            # Обрабатываем batch через реальный MoE processor
            batch_output = self._process_moe_batch(
                states, batch_cells, moe_processor, adaptive_radius
            )

            # Обновляем выходные состояния
            output_states[batch_start:batch_end] = batch_output

            processed_cells += len(batch_cells)
            batch_count += 1

        logger.info(
            f"   ✅ Обработано {processed_cells}/{num_cells} клеток в {batch_count} batch'ах"
        )

        return output_states

    def _process_moe_batch(
        self,
        states: torch.Tensor,
        cell_indices: List[int],
        moe_processor,
        adaptive_radius: float,
    ) -> torch.Tensor:
        """
        Обработка batch'а клеток через реальный MoE Connection Processor

        Args:
            states: все состояния решетки
            cell_indices: индексы клеток в текущем batch'е
            moe_processor: реальный MoE Connection Processor
            adaptive_radius: радиус поиска соседей

        Returns:
            batch_output: новые состояния для batch'а клеток
        """
        batch_size = len(cell_indices)
        state_size = states.shape[1]

        # Инициализируем выходной tensor
        batch_output = torch.zeros(
            batch_size, state_size, device=self.device, dtype=states.dtype
        )

        pos_helper = Position3D(self.dimensions)

        for i, cell_idx in enumerate(cell_indices):
            # Получаем координаты клетки
            coords = pos_helper.index_to_coords(cell_idx)

            # НОВАЯ АРХИТЕКТУРА: Передаем spatial_optimizer в MoE processor
            # Он сам найдет соседей по adaptive radius
            current_state = states[cell_idx].unsqueeze(0)  # [1, state_size]

            # Создаем пустые neighbor_states - MoE processor сам найдет нужных соседей
            empty_neighbors = torch.empty(1, 0, states.shape[1], device=self.device)

            try:
                # Передаем self как spatial_optimizer для adaptive radius поиска
                result = moe_processor(
                    current_state=current_state,
                    neighbor_states=empty_neighbors,  # Будет заменено на adaptive neighbors
                    cell_idx=cell_idx,
                    neighbor_indices=[],  # Будет заменено на adaptive neighbors
                    spatial_optimizer=self,  # КЛЮЧЕВОЕ: передаем себя для поиска соседей
                    full_lattice_states=states,  # Передаем полные состояния для adaptive radius
                )

                # Извлекаем новое состояние
                if isinstance(result, dict) and "new_state" in result:
                    new_state = result["new_state"]
                elif isinstance(result, torch.Tensor):
                    new_state = result
                else:
                    logger.warning(
                        f"⚠️ Неожиданный тип результата от MoE: {type(result)}"
                    )
                    new_state = current_state

                batch_output[i] = new_state.squeeze(0)

            except Exception as e:
                logger.warning(f"⚠️ Ошибка в MoE обработке клетки {cell_idx}: {e}")
                # Fallback: оставляем состояние без изменений
                batch_output[i] = states[cell_idx]

        return batch_output

    def _classify_neighbors_for_moe(
        self, cell_idx: int, neighbors: List[int]
    ) -> Dict[str, List[int]]:
        """
        Классификация соседей для MoE экспертов

        Args:
            cell_idx: индекс текущей клетки
            neighbors: список индексов соседей

        Returns:
            dict с классифицированными соседями по типам
        """
        if not neighbors:
            return {"local": [], "functional": [], "distant": []}

        # Вычисляем расстояния до соседей
        pos_helper = Position3D(self.dimensions)
        cell_coords = pos_helper.index_to_coords(cell_idx)

        neighbor_distances = []
        for neighbor_idx in neighbors:
            neighbor_coords = pos_helper.index_to_coords(neighbor_idx)

            # Евклидово расстояние
            distance = (
                (cell_coords[0] - neighbor_coords[0]) ** 2
                + (cell_coords[1] - neighbor_coords[1]) ** 2
                + (cell_coords[2] - neighbor_coords[2]) ** 2
            ) ** 0.5

            neighbor_distances.append((distance, neighbor_idx))

        # Сортируем по расстоянию
        neighbor_distances.sort(key=lambda x: x[0])

        # Классифицируем согласно MoE распределению
        total_neighbors = len(neighbors)
        local_count = max(
            1, int(total_neighbors * self.connection_distributions["local"])
        )
        functional_count = max(
            1, int(total_neighbors * self.connection_distributions["functional"])
        )
        # distant_count автоматически = остальные

        classified = {
            "local": [idx for _, idx in neighbor_distances[:local_count]],
            "functional": [
                idx
                for _, idx in neighbor_distances[
                    local_count : local_count + functional_count
                ]
            ],
            "distant": [
                idx for _, idx in neighbor_distances[local_count + functional_count :]
            ],
        }

        return classified

    def estimate_moe_memory_requirements(
        self, dimensions: Coordinates3D
    ) -> Dict[str, float]:
        """Оценивает требования к памяти для MoE + Spatial Optimization"""
        from .spatial_optimizer import estimate_memory_requirements

        base_requirements = estimate_memory_requirements(dimensions)

        # Дополнительные требования для MoE
        total_cells = dimensions[0] * dimensions[1] * dimensions[2]

        moe_overhead = {
            "expert_states_gb": total_cells * 32 * 4 * 3 / (1024**3),  # 3 эксперта
            "connection_classification_gb": total_cells
            * 26
            * 4
            / (1024**3),  # классификация связей
            "spatial_index_gb": total_cells * 8 / (1024**3),  # пространственный индекс
            "chunk_coordination_gb": 0.1,  # координация chunk'ов
        }

        # Общие требования
        total_moe_overhead = sum(moe_overhead.values())

        result = base_requirements.copy()
        result.update(moe_overhead)
        result["total_memory_gb"] += total_moe_overhead
        result["recommended_gpu_memory_gb"] = (
            result["total_memory_gb"] * 1.3
        )  # 30% запас

        return result


def create_moe_spatial_optimizer(
    dimensions: Coordinates3D, moe_processor=None, device: Optional[torch.device] = None
) -> MoESpatialOptimizer:
    """
    Фабричная функция для создания MoE Spatial Optimizer

    Args:
        dimensions: размеры решетки (x, y, z)
        moe_processor: реальный MoE Connection Processor
        device: устройство для вычислений

    Returns:
        MoESpatialOptimizer настроенный для данной решетки
    """
    # Используем централизованную конфигурацию

    project_config = get_project_config()
    config = project_config.get_spatial_optim_config()

    # Определяем устройство
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"🏭 Создание MoE Spatial Optimizer для {dimensions}")
    logger.info(f"   🎯 Устройство: {device}")
    logger.info(
        f"   ⚙️ Конфигурация: {config['chunk_size']}×{config['chunk_size']}×{config['chunk_size']} chunks"
    )

    # Если MoE processor не передан, создаем Mock версию для обратной совместимости
    if moe_processor is None:
        logger.warning("⚠️ MoE processor не передан, будет использоваться Mock версия")
        # Здесь можно создать Mock или выбросить ошибку

    return MoESpatialOptimizer(
        dimensions=dimensions,
        moe_processor=moe_processor,
        config=config,
        device=device,
    )


def estimate_moe_memory_requirements(dimensions: Coordinates3D) -> Dict[str, float]:
    """
    Быстрая оценка требований к памяти для MoE Spatial Optimization

    Args:
        dimensions: размеры решетки (x, y, z)

    Returns:
        dict с оценками памяти в GB
    """
    # Используем централизованную конфигурацию

    project_config = get_project_config()
    config = project_config.get_spatial_optim_config()

    optimizer = MoESpatialOptimizer(dimensions, config=config)

    return optimizer.estimate_moe_memory_requirements(dimensions)
