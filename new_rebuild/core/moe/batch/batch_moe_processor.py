#!/usr/bin/env python3
"""
Batch MoE Processor
==================

Основной модуль для batch обработки в MoE архитектуре.
Заменяет per-cell обработку на эффективную batch обработку.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import time

from .batch_neighbor_extractor import BatchNeighborExtractor
from .batch_expert_processor import BatchExpertProcessor
from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager

logger = get_logger(__name__)


class BatchMoEProcessor(nn.Module):
    """
    Batch версия MoE процессора для эффективной обработки множества клеток.
    
    Ключевые оптимизации:
    - Batch обработка вместо per-cell
    - Параллельное выполнение экспертов
    - Векторизованные операции с соседями
    - Минимальные аллокации памяти
    """
    
    def __init__(
        self,
        moe_processor,  # Оригинальный MoEConnectionProcessor
        enable_cuda_streams: bool = True,
        profile_performance: bool = False
    ):
        super().__init__()
        
        # Сохраняем ссылку на оригинальный процессор
        self.moe_processor = moe_processor
        
        # Копируем необходимые атрибуты
        self.state_size = moe_processor.state_size
        self.local_expert = moe_processor.local_expert
        self.functional_expert = moe_processor.functional_expert
        self.distant_expert = moe_processor.distant_expert
        self.gating_network = moe_processor.gating_network
        self.connection_classifier = moe_processor.connection_classifier
        self.enable_cnf = moe_processor.enable_cnf
        
        # Device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Batch processing компоненты
        self.neighbor_extractor = BatchNeighborExtractor()
        self.expert_processor = BatchExpertProcessor(
            local_expert=self.local_expert,
            functional_expert=self.functional_expert,
            distant_expert=self.distant_expert,
            state_size=self.state_size,
            enable_cuda_streams=enable_cuda_streams
        )
        
        self.profile_performance = profile_performance
        self.performance_stats = {
            "batch_sizes": [],
            "processing_times_ms": [],
            "neighbor_extraction_ms": [],
            "expert_processing_ms": [],
            "gating_ms": [],
        }
    
    def forward(
        self,
        cell_indices: torch.Tensor,  # [batch_size] - индексы клеток
        full_lattice_states: torch.Tensor,  # [total_cells, state_size]
        external_inputs: Optional[torch.Tensor] = None,  # [batch_size, state_size]
    ) -> torch.Tensor:
        """
        Batch forward pass через MoE
        
        Args:
            cell_indices: индексы клеток для обработки
            full_lattice_states: состояния всех клеток решетки
            external_inputs: внешние входы (опционально)
            
        Returns:
            torch.Tensor: [batch_size, state_size] - новые состояния
        """
        start_time = time.time()
        batch_size = cell_indices.shape[0]
        
        # Проверка размерностей
        if full_lattice_states.dim() != 2:
            raise ValueError(f"Expected 2D full_lattice_states, got {full_lattice_states.dim()}D")
        
        # Убеждаемся что все входные тензоры на правильном устройстве
        device = self.device_manager.get_device()
        cell_indices = cell_indices.to(device)
        full_lattice_states = full_lattice_states.to(device)
        if external_inputs is not None:
            external_inputs = external_inputs.to(device)
        
        # 1. Извлекаем текущие состояния
        current_states = full_lattice_states[cell_indices]  # [batch_size, state_size]
        
        # 2. Извлекаем информацию о соседях для batch'а
        neighbor_start = time.time()
        batch_neighbors = self.neighbor_extractor.extract_batch_neighbors(
            cell_indices=cell_indices,
            connection_classifier=self.connection_classifier,
            full_states=full_lattice_states
        )
        neighbor_time_ms = (time.time() - neighbor_start) * 1000
        
        # 3. Извлекаем состояния соседей для каждого типа
        neighbor_states_dict = {}
        
        if batch_neighbors.local_mask.any():
            neighbor_states_dict["local"] = self.neighbor_extractor.extract_neighbor_states(
                full_states=full_lattice_states,
                neighbor_indices=batch_neighbors.local_indices,
                neighbor_counts=batch_neighbors.local_counts,
                mask=batch_neighbors.local_mask
            )
        
        if batch_neighbors.functional_mask.any():
            neighbor_states_dict["functional"] = self.neighbor_extractor.extract_neighbor_states(
                full_states=full_lattice_states,
                neighbor_indices=batch_neighbors.functional_indices,
                neighbor_counts=batch_neighbors.functional_counts,
                mask=batch_neighbors.functional_mask
            )
        
        if batch_neighbors.distant_mask.any():
            neighbor_states_dict["distant"] = self.neighbor_extractor.extract_neighbor_states(
                full_states=full_lattice_states,
                neighbor_indices=batch_neighbors.distant_indices,
                neighbor_counts=batch_neighbors.distant_counts,
                mask=batch_neighbors.distant_mask
            )
        
        # 4. Обрабатываем через экспертов параллельно
        expert_start = time.time()
        expert_outputs = self.expert_processor.process_batch(
            current_states=current_states,
            neighbor_states_dict=neighbor_states_dict,
            masks={
                "local": batch_neighbors.local_mask,
                "functional": batch_neighbors.functional_mask,
                "distant": batch_neighbors.distant_mask,
            },
            enable_cnf=self.enable_cnf
        )
        expert_time_ms = (time.time() - expert_start) * 1000
        
        # 5. Применяем gating network
        gating_start = time.time()
        
        # Подготавливаем вход для gating network
        if external_inputs is not None:
            gating_input = torch.cat([current_states, external_inputs], dim=-1)
        else:
            gating_input = current_states
        
        # Получаем веса от gating network
        gating_weights = self.gating_network(gating_input)  # [batch_size, 3]
        
        # 6. Комбинируем выходы экспертов
        # Stack expert outputs: [batch_size, 3, state_size]
        expert_stack = torch.stack([
            expert_outputs.local_outputs,
            expert_outputs.functional_outputs,
            expert_outputs.distant_outputs
        ], dim=1)
        
        # Применяем веса: [batch_size, 3, 1] * [batch_size, 3, state_size]
        gating_weights_expanded = gating_weights.unsqueeze(-1)
        weighted_outputs = expert_stack * gating_weights_expanded
        
        # Суммируем взвешенные выходы
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, state_size]
        
        # 7. Residual connection
        final_output = combined_output + current_states
        
        gating_time_ms = (time.time() - gating_start) * 1000
        
        # Профилирование производительности
        if self.profile_performance:
            total_time_ms = (time.time() - start_time) * 1000
            self.performance_stats["batch_sizes"].append(batch_size)
            self.performance_stats["processing_times_ms"].append(total_time_ms)
            self.performance_stats["neighbor_extraction_ms"].append(neighbor_time_ms)
            self.performance_stats["expert_processing_ms"].append(expert_time_ms)
            self.performance_stats["gating_ms"].append(gating_time_ms)
            
            logger.info(
                f"📊 Batch MoE Performance: batch_size={batch_size}, "
                f"total={total_time_ms:.1f}ms, neighbors={neighbor_time_ms:.1f}ms, "
                f"experts={expert_time_ms:.1f}ms, gating={gating_time_ms:.1f}ms"
            )
        
        return final_output
    
    def process_chunk(
        self,
        chunk_indices: torch.Tensor,
        full_lattice_states: torch.Tensor,
        external_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Обрабатывает chunk клеток (совместимость с существующим интерфейсом)
        
        Returns:
            Dict с результатами для совместимости
        """
        new_states = self.forward(
            cell_indices=chunk_indices,
            full_lattice_states=full_lattice_states,
            external_inputs=external_inputs
        )
        
        return {
            "new_states": new_states,
            "cell_indices": chunk_indices,
        }
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Получить сводку производительности"""
        if not self.performance_stats["batch_sizes"]:
            return {}
            
        import numpy as np
        
        return {
            "avg_batch_size": np.mean(self.performance_stats["batch_sizes"]),
            "avg_total_time_ms": np.mean(self.performance_stats["processing_times_ms"]),
            "avg_neighbor_time_ms": np.mean(self.performance_stats["neighbor_extraction_ms"]),
            "avg_expert_time_ms": np.mean(self.performance_stats["expert_processing_ms"]),
            "avg_gating_time_ms": np.mean(self.performance_stats["gating_ms"]),
            "total_batches_processed": len(self.performance_stats["batch_sizes"]),
        }