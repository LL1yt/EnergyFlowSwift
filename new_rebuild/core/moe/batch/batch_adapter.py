#!/usr/bin/env python3
"""
Batch Processing Adapter
=======================

Адаптер для переключения между per-cell и batch обработкой.
Позволяет легко интегрировать batch оптимизации в существующий код.
"""

import torch
from typing import Dict, Optional, Any, List
import time

from .batch_moe_processor import BatchMoEProcessor
from ....utils.logging import get_logger
from ....config import get_project_config

logger = get_logger(__name__)


class BatchProcessingAdapter:
    """
    Адаптер для переключения между per-cell и batch режимами.
    
    Позволяет:
    - Плавный переход на batch обработку
    - Fallback на per-cell при необходимости
    - A/B тестирование производительности
    - Постепенное внедрение оптимизаций
    """
    
    def __init__(
        self,
        moe_processor,  # Оригинальный MoEConnectionProcessor
        enable_batch: bool = True,
        batch_size_threshold: int = 4,  # Минимальный размер для batch обработки
        enable_profiling: bool = True,
        fallback_on_error: bool = True
    ):
        self.moe_processor = moe_processor
        self.enable_batch = enable_batch
        self.batch_size_threshold = batch_size_threshold
        self.enable_profiling = enable_profiling
        self.fallback_on_error = fallback_on_error
        
        # Создаем batch процессор если включен
        if self.enable_batch:
            try:
                self.batch_processor = BatchMoEProcessor(
                    moe_processor=moe_processor,
                    profile_performance=enable_profiling
                )
                logger.info("✅ Batch processing adapter initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize batch processor: {e}")
                self.enable_batch = False
                self.batch_processor = None
        else:
            self.batch_processor = None
            
        # Статистика batch производительности
        self.performance_stats = {
            "batch_calls": 0,
            "batch_time_ms": 0.0,
            "cells_processed": 0,
        }
    
    # УДАЛЕН: should_use_batch - всегда используем batch обработку
    
    def process_cells(
        self,
        cell_indices: List[int],
        full_lattice_states: torch.Tensor,
        external_inputs: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Обрабатывает набор клеток, автоматически выбирая режим
        
        Args:
            cell_indices: список индексов клеток
            full_lattice_states: состояния всех клеток
            external_inputs: внешние входы (опционально)
            
        Returns:
            Dict[cell_idx -> new_state] для каждой клетки
        """
        # ТОЛЬКО BATCH ОБРАБОТКА - убрана fallback логика
        logger.debug_spatial(f"🔄 Batch processing {len(cell_indices)} cells")
        return self._process_batch(cell_indices, full_lattice_states, external_inputs)
    
    def _process_batch(
        self,
        cell_indices: List[int],
        full_lattice_states: torch.Tensor,
        external_inputs: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """Batch обработка клеток"""
        start_time = time.time()
        
        try:
            # Конвертируем индексы в тензор (правильный способ)
            if isinstance(cell_indices, torch.Tensor):
                indices_tensor = cell_indices.detach().clone().to(
                    device=full_lattice_states.device, dtype=torch.long
                )
            else:
                indices_tensor = torch.tensor(
                    cell_indices,
                    device=full_lattice_states.device,
                    dtype=torch.long
                )
            
            # Вызываем batch процессор
            new_states = self.batch_processor.forward(
                cell_indices=indices_tensor,
                full_lattice_states=full_lattice_states,
                external_inputs=external_inputs
            )
            
            # Конвертируем результат в словарь
            result = {}
            for i, cell_idx in enumerate(cell_indices):
                result[cell_idx] = new_states[i]
            
            # Обновляем статистику
            elapsed_ms = (time.time() - start_time) * 1000
            self.performance_stats["batch_calls"] += 1
            self.performance_stats["batch_time_ms"] += elapsed_ms
            self.performance_stats["cells_processed"] += len(cell_indices)
            
            if self.enable_profiling:
                logger.debug(
                    f"🚀 Batch processed {len(cell_indices)} cells in {elapsed_ms:.1f}ms "
                    f"({elapsed_ms/len(cell_indices):.2f}ms per cell)"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            logger.error(f"   cell_indices: {cell_indices}")
            logger.error(f"   full_lattice_states shape: {full_lattice_states.shape}")
            # УБРАЛИ FALLBACK - сразу выбрасываем ошибку для отладки
            raise
    
    # УДАЛЕН: _process_per_cell метод - используем только batch обработку
    
    def get_batch_performance(self) -> Dict[str, Any]:
        """Получить статистику batch обработки"""
        stats = self.performance_stats.copy()
        
        # Добавляем статистику от batch процессора
        if self.batch_processor:
            stats["batch_processor_stats"] = self.batch_processor.get_performance_summary()
        
        return stats
    
    def set_batch_enabled(self, enabled: bool):
        """Включить/выключить batch обработку"""
        self.enable_batch = enabled and self.batch_processor is not None
        logger.info(f"Batch processing {'enabled' if self.enable_batch else 'disabled'}")