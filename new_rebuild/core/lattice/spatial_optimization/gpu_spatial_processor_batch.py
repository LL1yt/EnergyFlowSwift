#!/usr/bin/env python3
"""
GPU Spatial Processor with Batch Support
=======================================

Расширение GPUSpatialProcessor с поддержкой batch обработки.
Использует модульную batch архитектуру для оптимизации производительности.
"""

import torch
import time
from typing import Dict, Optional, Callable, Any

from .gpu_spatial_processor import GPUSpatialProcessor
from ...moe.batch import BatchProcessingAdapter
from ....utils.logging import get_logger
from ....config import get_project_config

logger = get_logger(__name__)


class GPUSpatialProcessorBatch(GPUSpatialProcessor):
    """
    GPU Spatial Processor с поддержкой batch обработки chunk'ов.
    
    Наследует от оригинального GPUSpatialProcessor и добавляет:
    - Batch обработку chunk'ов вместо per-cell
    - Автоматическое переключение между режимами
    - Профилирование производительности
    """
    
    def __init__(
        self,
        dimensions: tuple,
        moe_processor,
        enable_batch: bool = True,
        batch_threshold: int = 4,
        profile_performance: bool = True
    ):
        # Инициализируем родительский класс
        super().__init__(dimensions, moe_processor)
        
        # Создаем batch адаптер
        self.batch_adapter = BatchProcessingAdapter(
            moe_processor=moe_processor,
            enable_batch=enable_batch,
            batch_size_threshold=batch_threshold,
            enable_profiling=profile_performance,
            fallback_on_error=True
        )
        
        self.enable_batch = enable_batch
        self.performance_history = []
        
        logger.info(
            f"🚀 GPUSpatialProcessorBatch initialized: "
            f"dimensions={dimensions}, batch_enabled={enable_batch}"
        )
    
    def _process_chunk_with_function(
        self,
        chunk_info,
        all_states: torch.Tensor,
        processor_fn: callable,
        updates: dict = None
    ) -> str:
        """
        Переопределяем обработку chunk'а для использования batch processing
        """
        try:
            # Performance tracking
            chunk_start_time = time.time()
            
            # DEBUG: Логируем размерности
            logger.debug(f"🔧 BATCH CHUNK PROCESSING: all_states shape {all_states.shape}")
            logger.debug(f"🔧 CHUNK {chunk_info.chunk_id}: {len(chunk_info.cell_indices)} cells")
            
            # Проверяем что chunk_info.cell_indices не пустой
            if not chunk_info.cell_indices:
                logger.warning(f"⚠️ Пустой chunk {chunk_info.chunk_id} - пропускаем")
                return f"Chunk {chunk_info.chunk_id} skipped (empty)"
            
            # Определяем, использовать ли batch обработку
            num_cells = len(chunk_info.cell_indices)
            use_batch = self.batch_adapter.should_use_batch(num_cells)
            
            if use_batch:
                # BATCH PROCESSING PATH
                logger.debug(f"🚀 Using BATCH processing for chunk {chunk_info.chunk_id} with {num_cells} cells")
                
                # Обрабатываем весь chunk как batch
                new_states_dict = self.batch_adapter.process_cells(
                    cell_indices=chunk_info.cell_indices,
                    full_lattice_states=all_states,
                    external_inputs=None
                )
                
                # Конвертируем результат в тензор
                indices = torch.tensor(chunk_info.cell_indices, device=self.device, dtype=torch.long)
                new_states = torch.stack([new_states_dict[idx] for idx in chunk_info.cell_indices])
                
                # Сохраняем обновления
                if updates is not None:
                    updates[tuple(chunk_info.cell_indices)] = new_states
                else:
                    # Apply updates directly if no updates dict provided
                    if all_states.dim() == 3:
                        all_states[:, indices, :] = new_states.unsqueeze(0).expand(all_states.shape[0], -1, -1)
                    else:
                        all_states[indices] = new_states
                
            else:
                # Fallback to original per-cell processing
                logger.debug(f"🐌 Using per-cell processing for chunk {chunk_info.chunk_id} with {num_cells} cells")
                return super()._process_chunk_with_function(chunk_info, all_states, processor_fn, updates)
            
            # Измеряем производительность
            chunk_time_ms = (time.time() - chunk_start_time) * 1000
            cells_per_ms = num_cells / chunk_time_ms if chunk_time_ms > 0 else 0
            
            # Сохраняем метрики
            self.performance_history.append({
                "chunk_id": chunk_info.chunk_id,
                "num_cells": num_cells,
                "processing_time_ms": chunk_time_ms,
                "cells_per_ms": cells_per_ms,
                "used_batch": use_batch
            })
            
            logger.info(
                f"✅ Chunk {chunk_info.chunk_id} processed: "
                f"{num_cells} cells in {chunk_time_ms:.1f}ms "
                f"({cells_per_ms:.1f} cells/ms) using {'BATCH' if use_batch else 'per-cell'}"
            )
            
            return f"Chunk {chunk_info.chunk_id} processed successfully"
            
        except Exception as e:
            logger.error(f"❌ Error processing chunk {chunk_info.chunk_id}: {e}")
            logger.exception("Detailed error:")
            
            # Try fallback to original processing
            if self.batch_adapter.fallback_on_error:
                logger.info("⚠️ Attempting fallback to original processing")
                return super()._process_chunk_with_function(chunk_info, all_states, processor_fn, updates)
            else:
                raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получить отчет о производительности"""
        if not self.performance_history:
            return {"status": "No performance data available"}
        
        import numpy as np
        
        # Разделяем по типу обработки
        batch_runs = [p for p in self.performance_history if p["used_batch"]]
        percell_runs = [p for p in self.performance_history if not p["used_batch"]]
        
        report = {
            "total_chunks_processed": len(self.performance_history),
            "batch_chunks": len(batch_runs),
            "percell_chunks": len(percell_runs),
        }
        
        if batch_runs:
            batch_times = [p["processing_time_ms"] for p in batch_runs]
            batch_cells = [p["num_cells"] for p in batch_runs]
            batch_rates = [p["cells_per_ms"] for p in batch_runs]
            
            report["batch_stats"] = {
                "avg_time_ms": np.mean(batch_times),
                "avg_cells": np.mean(batch_cells),
                "avg_cells_per_ms": np.mean(batch_rates),
                "total_cells": sum(batch_cells),
                "total_time_ms": sum(batch_times),
            }
        
        if percell_runs:
            percell_times = [p["processing_time_ms"] for p in percell_runs]
            percell_cells = [p["num_cells"] for p in percell_runs]
            percell_rates = [p["cells_per_ms"] for p in percell_runs]
            
            report["percell_stats"] = {
                "avg_time_ms": np.mean(percell_times),
                "avg_cells": np.mean(percell_cells),
                "avg_cells_per_ms": np.mean(percell_rates),
                "total_cells": sum(percell_cells),
                "total_time_ms": sum(percell_times),
            }
        
        # Сравнение производительности
        if batch_runs and percell_runs:
            batch_rate = report["batch_stats"]["avg_cells_per_ms"]
            percell_rate = report["percell_stats"]["avg_cells_per_ms"]
            report["speedup"] = batch_rate / percell_rate if percell_rate > 0 else 0
        
        # Добавляем статистику от адаптера
        report["adapter_stats"] = self.batch_adapter.get_performance_comparison()
        
        return report
    
    def set_batch_enabled(self, enabled: bool):
        """Включить/выключить batch обработку"""
        self.enable_batch = enabled
        self.batch_adapter.set_batch_enabled(enabled)
        logger.info(f"Batch processing {'enabled' if enabled else 'disabled'}")
    
    def clear_performance_history(self):
        """Очистить историю производительности"""
        self.performance_history.clear()
        logger.info("Performance history cleared")