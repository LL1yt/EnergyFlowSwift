#!/usr/bin/env python3
"""
Batch Expert Processor
=====================

Модуль для параллельной обработки экспертов в batch режиме.
Использует CUDA streams для максимальной утилизации GPU.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import time

from ....utils.logging import get_logger
from ....utils.device_manager import get_device_manager

logger = get_logger(__name__)


@dataclass
class BatchExpertOutput:
    """Результаты обработки экспертами"""
    local_outputs: torch.Tensor  # [batch_size, state_size]
    functional_outputs: torch.Tensor  # [batch_size, state_size]
    distant_outputs: torch.Tensor  # [batch_size, state_size]
    processing_time_ms: float


class BatchExpertProcessor:
    """
    Обрабатывает batch'и через экспертов параллельно.
    Оптимизирован для GPU с использованием CUDA streams.
    """
    
    def __init__(
        self,
        local_expert: nn.Module,
        functional_expert: nn.Module,
        distant_expert: nn.Module,
        state_size: int,
        enable_cuda_streams: bool = True
    ):
        self.local_expert = local_expert
        self.functional_expert = functional_expert
        self.distant_expert = distant_expert
        self.state_size = state_size
        
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # CUDA streams для параллельной обработки
        self.enable_cuda_streams = enable_cuda_streams and self.device_manager.is_cuda()
        if self.enable_cuda_streams:
            self.local_stream = torch.cuda.Stream()
            self.functional_stream = torch.cuda.Stream()
            self.distant_stream = torch.cuda.Stream()
            
    def process_batch(
        self,
        current_states: torch.Tensor,  # [batch_size, state_size]
        neighbor_states_dict: Dict[str, torch.Tensor],  # Состояния соседей по типам
        masks: Dict[str, torch.Tensor],  # Маски для каждого типа
        enable_cnf: bool = True
    ) -> BatchExpertOutput:
        """
        Обрабатывает batch через всех экспертов
        
        Args:
            current_states: текущие состояния клеток
            neighbor_states_dict: словарь с состояниями соседей для каждого типа
            masks: маски указывающие какие клетки имеют соседей каждого типа
            enable_cnf: использовать ли distant expert (CNF)
            
        Returns:
            BatchExpertOutput с результатами всех экспертов
        """
        start_time = time.time()
        batch_size = current_states.shape[0]
        
        # Инициализируем выходы нулями
        local_outputs = torch.zeros_like(current_states)
        functional_outputs = torch.zeros_like(current_states)
        distant_outputs = torch.zeros_like(current_states)
        
        if self.enable_cuda_streams:
            # Параллельная обработка с CUDA streams
            self._process_parallel(
                current_states, neighbor_states_dict, masks,
                local_outputs, functional_outputs, distant_outputs,
                enable_cnf
            )
        else:
            # Последовательная обработка (для CPU или отладки)
            self._process_sequential(
                current_states, neighbor_states_dict, masks,
                local_outputs, functional_outputs, distant_outputs,
                enable_cnf
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchExpertOutput(
            local_outputs=local_outputs,
            functional_outputs=functional_outputs,
            distant_outputs=distant_outputs,
            processing_time_ms=processing_time_ms
        )
    
    def _process_parallel(
        self,
        current_states: torch.Tensor,
        neighbor_states_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        local_outputs: torch.Tensor,
        functional_outputs: torch.Tensor,
        distant_outputs: torch.Tensor,
        enable_cnf: bool
    ):
        """Параллельная обработка с использованием CUDA streams"""
        
        # LOCAL expert processing
        with torch.cuda.stream(self.local_stream):
            if masks["local"].any() and "local" in neighbor_states_dict:
                local_mask = masks["local"]
                local_states = current_states[local_mask]
                local_neighbors = neighbor_states_dict["local"]
                
                local_result = self._call_expert(
                    self.local_expert, local_states, local_neighbors
                )
                local_outputs[local_mask] = local_result
        
        # FUNCTIONAL expert processing
        with torch.cuda.stream(self.functional_stream):
            if masks["functional"].any() and "functional" in neighbor_states_dict:
                functional_mask = masks["functional"]
                functional_states = current_states[functional_mask]
                functional_neighbors = neighbor_states_dict["functional"]
                
                functional_result = self._call_expert(
                    self.functional_expert, functional_states, functional_neighbors
                )
                functional_outputs[functional_mask] = functional_result
        
        # DISTANT expert processing
        if enable_cnf:
            with torch.cuda.stream(self.distant_stream):
                if masks["distant"].any() and "distant" in neighbor_states_dict:
                    distant_mask = masks["distant"]
                    distant_states = current_states[distant_mask]
                    distant_neighbors = neighbor_states_dict["distant"]
                    
                    distant_result = self._call_expert(
                        self.distant_expert, distant_states, distant_neighbors
                    )
                    distant_outputs[distant_mask] = distant_result
        
        # Синхронизация всех streams
        torch.cuda.synchronize()
    
    def _process_sequential(
        self,
        current_states: torch.Tensor,
        neighbor_states_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        local_outputs: torch.Tensor,
        functional_outputs: torch.Tensor,
        distant_outputs: torch.Tensor,
        enable_cnf: bool
    ):
        """Последовательная обработка (для CPU или отладки)"""
        
        # LOCAL expert
        if masks["local"].any() and "local" in neighbor_states_dict:
            local_mask = masks["local"]
            local_states = current_states[local_mask]
            local_neighbors = neighbor_states_dict["local"]
            
            local_result = self._call_expert(
                self.local_expert, local_states, local_neighbors
            )
            local_outputs[local_mask] = local_result
        
        # FUNCTIONAL expert
        if masks["functional"].any() and "functional" in neighbor_states_dict:
            functional_mask = masks["functional"]
            functional_states = current_states[functional_mask]
            functional_neighbors = neighbor_states_dict["functional"]
            
            functional_result = self._call_expert(
                self.functional_expert, functional_states, functional_neighbors
            )
            functional_outputs[functional_mask] = functional_result
        
        # DISTANT expert
        if enable_cnf and masks["distant"].any() and "distant" in neighbor_states_dict:
            distant_mask = masks["distant"]
            distant_states = current_states[distant_mask]
            distant_neighbors = neighbor_states_dict["distant"]
            
            distant_result = self._call_expert(
                self.distant_expert, distant_states, distant_neighbors
            )
            distant_outputs[distant_mask] = distant_result
    
    def _call_expert(
        self,
        expert: nn.Module,
        states: torch.Tensor,
        neighbor_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Вызывает эксперта с обработкой различных форматов вывода
        """
        # Убеждаемся что все тензоры на правильном устройстве
        states = states.to(self.device)
        neighbor_states = neighbor_states.to(self.device)
        
        # Обрабатываем каждый элемент batch'а отдельно для совместимости с текущими экспертами
        batch_size = states.shape[0]
        results = []
        
        for i in range(batch_size):
            # Извлекаем состояние одной клетки
            single_state = states[i:i+1]  # Сохраняем batch dimension [1, state_size]
            
            # Извлекаем соседей для этой клетки
            if neighbor_states.dim() == 3:
                single_neighbors = neighbor_states[i]  # [max_neighbors, state_size]
                # Убираем padding (нулевые соседи)
                non_zero_mask = single_neighbors.abs().sum(dim=-1) > 0
                if non_zero_mask.any():
                    single_neighbors = single_neighbors[non_zero_mask]
                else:
                    # Если нет соседей, создаем фиктивного
                    single_neighbors = torch.zeros(1, self.state_size, device=self.device)
            else:
                single_neighbors = neighbor_states
            
            # Вызываем эксперта для одной клетки
            single_result = expert(single_state, single_neighbors)
            
            # Обрабатываем результат
            if isinstance(single_result, dict):
                single_result = single_result.get("output", single_result.get("new_state", single_state))
            
            # Убеждаемся что результат имеет правильную форму
            if single_result.dim() > 1:
                single_result = single_result.squeeze(0)  # Убираем batch dimension
                
            results.append(single_result)
        
        # Стекаем результаты обратно в batch
        result = torch.stack(results, dim=0)  # [batch_size, state_size]
        
        # Убеждаемся что размерности правильные
        if result.dim() > 2:
            result = result.view(-1, self.state_size)
            
        return result